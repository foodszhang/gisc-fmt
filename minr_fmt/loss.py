import torch
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim


class ScatterLightLoss(nn.Module):
    def __init__(
        self,
        # 散射校正损失参数
        init_scatter_weight=1.0,  # 初始权重（训练初期）
        target_scatter_weight=0.5,  # 目标权重（训练后期）
        start_decay_epoch=200,  # 开始衰减的epoch
        decay_epochs=100,  # 衰减持续epoch（200→300逐步衰减）
        # SparseLightLoss参数
        pos_weight=150.0,
        sparse_weight=0.01,
        lambda_dice=0.3,  # Soft Dice loss 权重
    ):
        super().__init__()
        self.init_scatter_weight = init_scatter_weight
        self.target_scatter_weight = target_scatter_weight
        self.start_decay_epoch = start_decay_epoch
        self.decay_epochs = decay_epochs
        self.current_epoch = 0  # 需外部传入当前epoch

        # 初始化子损失
        self.sparse_light_loss = SparseLightLoss(pos_weight, sparse_weight, lambda_dice=lambda_dice)
        self.l1_loss = nn.L1Loss()

    def update_epoch(self, epoch):
        """训练循环中调用，更新当前epoch（用于权重调度）"""
        self.current_epoch = epoch

    def get_dynamic_scatter_weight(self):
        """根据当前epoch计算动态散射校正权重"""
        if self.current_epoch < self.start_decay_epoch:
            # 训练初期：保持初始权重（优先校正）
            return self.init_scatter_weight
        elif self.current_epoch < self.start_decay_epoch + self.decay_epochs:
            # 衰减阶段：线性降低权重
            decay_ratio = (self.current_epoch - self.start_decay_epoch) / self.decay_epochs
            return self.init_scatter_weight - decay_ratio * (
                self.init_scatter_weight - self.target_scatter_weight
            )
        else:
            # 训练后期：保持目标权重（优先光源预测）
            return self.target_scatter_weight

    def scatter_correction_loss(self, pred_scatter: dict, gt_scatter: dict):
        """散射校正损失（L1+SSIM，均为可微）"""

        def ssim_torch(x, y, data_range=1.0, window_size=11, K1=0.01, K2=0.03, eps=1e-6):
            # x, y: [B, C, H, W] or [B, H, W], float32, 0~1
            if x.dim() == 3:
                x = x.unsqueeze(1)
                y = y.unsqueeze(1)
            B, C, H, W = x.shape

            pad = window_size // 2
            window = torch.ones((1, C, window_size, window_size), device=x.device, dtype=x.dtype)
            window = window / (window_size * window_size)

            mu_x = F.conv2d(x, window, padding=pad, groups=C)
            mu_y = F.conv2d(y, window, padding=pad, groups=C)

            sigma_x = F.conv2d(x * x, window, padding=pad, groups=C) - mu_x**2
            sigma_y = F.conv2d(y * y, window, padding=pad, groups=C) - mu_y**2
            sigma_xy = F.conv2d(x * y, window, padding=pad, groups=C) - mu_x * mu_y

            L = data_range
            C1 = (K1 * L) ** 2
            C2 = (K2 * L) ** 2

            # 加 eps 防止分母为 0 或负数
            sigma_x = torch.clamp(sigma_x, min=0.0)
            sigma_y = torch.clamp(sigma_y, min=0.0)
            sigma_xy = torch.clamp(sigma_xy, min=-torch.sqrt(sigma_x * sigma_y + eps))

            numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
            denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
            denominator = denominator + eps  # 防 division by zero

            ssim_map = numerator / denominator

            # 限制数值范围，防止极端 outlier
            ssim_map = torch.clamp(ssim_map, min=-1.0, max=1.0)

            return ssim_map.mean(dim=[1, 2, 3])  # [B]

        total_scatter_loss = 0.0
        for angle in pred_scatter.keys():
            pred = pred_scatter[angle]
            gt = gt_scatter[str(angle)]
            l1 = self.l1_loss(pred, gt)
            # SSIM损失（可微）
            # ssim_loss = 1 - ssim_torch(pred, gt).mean()
            # total_scatter_loss += l1 + ssim_loss
            total_scatter_loss += l1
        return total_scatter_loss / len(pred_scatter)

    def forward(self, pred_scatter, gt_scatter, pred_density, gt_density):
        # 1. 获取动态权重
        scatter_weight = self.get_dynamic_scatter_weight()

        # 2. 计算各部分损失
        scatter_loss = self.scatter_correction_loss(pred_scatter, gt_scatter)
        light_loss = self.sparse_light_loss(pred_density, gt_density)

        # 3. 加权组合总损失
        total_loss = scatter_weight * scatter_loss + light_loss

        # 返回损失及当前权重（便于监控）
        return {
            "total_loss": total_loss,
            "scatter_loss": scatter_loss,
            "light_loss": light_loss,
            "scatter_weight": scatter_weight,  # 监控权重变化
        }


class SparseLightLoss(nn.Module):
    """针对稀疏光源的损失函数（抑制背景，增强光源区域权重）"""

    def __init__(
        self,
        pos_weight=150.0,
        sparse_weight=0.01,
        attn_weight=0.1,
        lambda_dice=0.3,
        # Focal Tversky loss 参数
        alpha: float = 0.6,
        beta: float = 0.4,
        gamma: float = 1.33,
        lambda_tv: float = 0.3,
    ):
        super().__init__()
        self.pos_weight = pos_weight  # 正样本（光源）权重
        self.sparse_weight = sparse_weight  # 稀疏性约束权重
        self.attn_weight = attn_weight  # 注意力约束权重

        # 兼容保留：lambda_dice 已弃用（原 soft dice 权重）；若用户仅调了 lambda_dice，则映射到 lambda_tv。
        self.lambda_dice = lambda_dice
        self.lambda_tv = lambda_tv if not (lambda_tv == 0.3 and lambda_dice != 0.3) else lambda_dice

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

    def forward(self, pred_density, gt_density):
        """
        pred_density: [B, N, 1] 或 [B, N] 预测密度（logits，未 sigmoid）
        gt_density: [B, N, 1] 或 [B, N] 真实密度（0/1）
        """
        # 对齐 shape
        if pred_density.dim() == 3 and pred_density.size(-1) == 1:
            pred_density = pred_density.squeeze(-1)
        if gt_density.dim() == 3 and gt_density.size(-1) == 1:
            gt_density = gt_density.squeeze(-1)
        gt_density = gt_density.float()

        # 1) 加权 BCE（保持原行为）
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_density,
            gt_density,
            weight=gt_density * (self.pos_weight - 1) + 1,
        )

        # 2) 稀疏正则：概率域 + 只惩罚背景（压假阳性体积）
        p = torch.sigmoid(pred_density)
        sparse_loss = self.sparse_weight * torch.mean(p * (1.0 - gt_density))

        # 3) Focal Tversky loss（按样本分别算，再 batch 平均）
        eps = 1e-6
        TP_b = (p * gt_density).sum(dim=1)
        FP_b = (p * (1.0 - gt_density)).sum(dim=1)
        FN_b = ((1.0 - p) * gt_density).sum(dim=1)

        TI_b = (TP_b + eps) / (TP_b + self.alpha * FP_b + self.beta * FN_b + eps)
        tversky_loss_b = 1.0 - TI_b
        focal_tversky_loss_b = torch.clamp(tversky_loss_b, 0.0, 1.0).pow(self.gamma)
        focal_tversky_loss = focal_tversky_loss_b.mean()

        # 4) 总损失
        total_loss = bce_loss + sparse_loss + self.lambda_tv * focal_tversky_loss
        return total_loss


def dice_coefficient(pred, target, threshold=0.5, eps=1e-8):
    """
    计算三维体素二分类的Dice系数

    参数:
        pred: 模型输出的预测值，shape为(B, D, H, W)，通常是sigmoid后的概率值
        target: 真实标签，shape为(B, D, H, W)，值为0或1
        threshold: 二值化阈值，默认0.5
        eps: 防止分母为0的微小值

    返回:
        批次的平均Dice系数（ scalar ）
    """
    # 1. 预测值二值化（二分类）
    pred_bin = (pred >= threshold).float()  # 转换为0/1的float类型

    # 2. 计算交（intersection）和并（union的分子部分）
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))  # 对D、H、W维度求和，保留批次维度(B,)
    pred_sum = pred_bin.sum(dim=(1, 2, 3))  # 预测正类总和 (B,)
    target_sum = target.sum(dim=(1, 2, 3))  # 真实正类总和 (B,)

    # 3. 计算每个样本的Dice系数，再求批次平均
    dice_per_sample = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    return dice_per_sample.mean()  # 返回批次平均Dice


def compute_dice(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    if union == 0:
        return 1.0  # 均无光源时视为完全匹配
    return 2 * intersection / (union + 1e-8)
