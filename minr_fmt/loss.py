import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxProjectionLightLoss(nn.Module):
    def __init__(
        self,
        # Auxiliary projection / descatter supervision parameters
        init_scatter_weight=1.0,  # legacy name; auxiliary projection initial weight
        target_scatter_weight=0.5,  # legacy name; auxiliary projection target weight
        start_decay_epoch=200,  # 开始衰减的epoch
        decay_epochs=100,  # 衰减持续epoch（200→300逐步衰减）
        # SparseLightLoss参数
        pos_weight=150.0,
        sparse_weight=0.01,
        lambda_dice=0.3,  # Soft Dice loss 权重
        use_tversky: bool = True,
        tversky_alpha: float = 0.6,
        tversky_beta: float = 0.4,
        tversky_gamma: float = 1.33,
        tversky_weight: float | None = None,
        light_weight: float = 1.0,
    ):
        super().__init__()
        self.init_aux_projection_weight = init_scatter_weight
        self.target_aux_projection_weight = target_scatter_weight
        self.start_decay_epoch = start_decay_epoch
        self.decay_epochs = decay_epochs
        self.current_epoch = 0  # 需外部传入当前epoch
        self.light_weight = float(light_weight)

        # 初始化子损失
        self.sparse_light_loss = SparseLightLoss(
            pos_weight,
            sparse_weight,
            lambda_dice=lambda_dice,
            use_tversky=use_tversky,
            alpha=tversky_alpha,
            beta=tversky_beta,
            gamma=tversky_gamma,
            lambda_tv=lambda_dice if tversky_weight is None else tversky_weight,
        )
        self.l1_loss = nn.L1Loss()

    def update_epoch(self, epoch):
        """训练循环中调用，更新当前epoch（用于权重调度）"""
        self.current_epoch = epoch

    def get_dynamic_aux_projection_weight(self):
        """Compute the epoch-scheduled auxiliary projection weight."""
        if self.current_epoch < self.start_decay_epoch:
            # 训练初期：保持初始权重（优先校正）
            return self.init_aux_projection_weight
        elif self.current_epoch < self.start_decay_epoch + self.decay_epochs:
            # 衰减阶段：线性降低权重
            decay_ratio = (self.current_epoch - self.start_decay_epoch) / self.decay_epochs
            return self.init_aux_projection_weight - decay_ratio * (
                self.init_aux_projection_weight - self.target_aux_projection_weight
            )
        else:
            # 训练后期：保持目标权重（优先光源预测）
            return self.target_aux_projection_weight

    def get_dynamic_scatter_weight(self):
        """Backward-compatible alias for older diagnostic scripts."""
        return self.get_dynamic_aux_projection_weight()

    def aux_projection_loss(self, pred_aux: dict | None, target_aux: dict | None, device):
        """Auxiliary projection supervision loss.

        When descatter targets are unavailable, the auxiliary branch is left unsupervised
        and contributes a zero loss. This keeps new datasets without descatter labels
        compatible without training the branch against the raw input projection.
        """
        if not pred_aux or target_aux is None:
            return torch.zeros((), dtype=torch.float32, device=device)

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

        total_aux_loss = torch.zeros((), dtype=torch.float32, device=device)
        matched = 0
        for angle in pred_aux.keys():
            key = str(angle)
            if key not in target_aux:
                continue
            pred = pred_aux[angle]
            gt = target_aux[key]
            l1 = self.l1_loss(pred, gt)
            # SSIM损失（可微）
            # ssim_loss = 1 - ssim_torch(pred, gt).mean()
            # total_aux_loss += l1 + ssim_loss
            total_aux_loss = total_aux_loss + l1
            matched += 1
        if matched == 0:
            return torch.zeros((), dtype=torch.float32, device=device)
        return total_aux_loss / matched

    def scatter_correction_loss(self, pred_scatter: dict, gt_scatter: dict):
        """Backward-compatible alias for older diagnostic scripts."""
        device = next(iter(pred_scatter.values())).device
        return self.aux_projection_loss(pred_scatter, gt_scatter, device)

    def forward(self, pred_aux, target_aux, pred_density, gt_density):
        # 1. 获取动态权重
        aux_weight = self.get_dynamic_aux_projection_weight()

        # 2. 计算各部分损失
        aux_loss = self.aux_projection_loss(pred_aux, target_aux, pred_density.device)
        if self.light_weight == 0.0:
            light_loss = torch.zeros((), dtype=pred_density.dtype, device=pred_density.device)
        else:
            light_loss = self.sparse_light_loss(pred_density, gt_density)

        # 3. 加权组合总损失
        total_loss = aux_weight * aux_loss + self.light_weight * light_loss

        # 返回损失及当前权重（便于监控）
        return {
            "total_loss": total_loss,
            "aux_projection_loss": aux_loss,
            "light_loss": light_loss,
            "aux_projection_weight": aux_weight,
            "light_weight": torch.as_tensor(
                self.light_weight, dtype=torch.float32, device=pred_density.device
            ),
        }


# Backward-compatible public name used by older scripts/configs.
ScatterLightLoss = AuxProjectionLightLoss


class SparseLightLoss(nn.Module):
    """针对稀疏光源的损失函数（抑制背景，增强光源区域权重）"""

    def __init__(
        self,
        pos_weight=150.0,
        sparse_weight=0.01,
        attn_weight=0.1,
        lambda_dice=0.3,
        use_tversky: bool = True,
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
        self.use_tversky = bool(use_tversky)

        # 兼容保留：lambda_dice 已弃用（原 soft dice 权重）。
        # 若用户仅调了 lambda_dice，则映射到 lambda_tv。
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
        focal_tversky_loss = torch.zeros((), dtype=pred_density.dtype, device=pred_density.device)
        if self.use_tversky and self.lambda_tv > 0:
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


class VoxelReconstructionLoss(nn.Module):
    """Voxel-domain reconstruction loss for projection-to-volume baselines.

    This is intentionally model-agnostic: adapted baselines expose logits in
    ``pred_voxel`` and the LightningModule supplies the same FMT-SimGen voxel GT.
    """

    def __init__(
        self,
        pos_weight=2.0,
        sparse_weight=0.05,
        lambda_dice=0.5,
        use_tversky: bool = True,
        tversky_alpha: float = 0.6,
        tversky_beta: float = 0.4,
        tversky_gamma: float = 1.33,
        tversky_weight: float | None = None,
    ):
        super().__init__()
        self.sparse_light_loss = SparseLightLoss(
            pos_weight=pos_weight,
            sparse_weight=sparse_weight,
            lambda_dice=lambda_dice,
            use_tversky=use_tversky,
            alpha=tversky_alpha,
            beta=tversky_beta,
            gamma=tversky_gamma,
            lambda_tv=lambda_dice if tversky_weight is None else tversky_weight,
        )
        self.l1_loss = nn.L1Loss()

    def forward(self, pred_voxel, target_voxel, aux_outputs=None):
        if pred_voxel.dim() == 5 and pred_voxel.size(1) == 1:
            pred = pred_voxel[:, 0]
        else:
            pred = pred_voxel
        if target_voxel.dim() == 5 and target_voxel.size(1) == 1:
            target = target_voxel[:, 0]
        else:
            target = target_voxel
        target = target.to(device=pred.device, dtype=pred.dtype)

        if pred.shape[1:] != target.shape[1:]:
            raise ValueError(
                "Voxel prediction and target shapes differ. "
                f"pred={tuple(pred.shape)}, target={tuple(target.shape)}. "
                "Use full-volume output, crop the target explicitly, paste ROI predictions, "
                "or mesh_to_voxel before metrics/loss."
            )

        flat_pred = pred.reshape(pred.shape[0], -1, 1)
        flat_target = target.reshape(target.shape[0], -1, 1)
        voxel_weight = 1.0
        loss_type = "default"
        dice_weight = None
        if isinstance(aux_outputs, dict):
            voxel_weight = float(aux_outputs.get("voxel_loss_weight", 1.0))
            loss_type = str(aux_outputs.get("voxel_loss_type", "default")).lower()
            if "dice_weight" in aux_outputs:
                dice_weight = float(aux_outputs["dice_weight"])

        if loss_type == "mse":
            prob = torch.sigmoid(pred)
            mse_loss = F.mse_loss(prob, target)
            if dice_weight is None:
                dice_weight = 0.0
            prob_flat = prob.reshape(prob.shape[0], -1)
            target_flat = target.reshape(target.shape[0], -1)
            intersection = (prob_flat * target_flat).sum(dim=1)
            dice = (2.0 * intersection + 1e-6) / (
                prob_flat.sum(dim=1) + target_flat.sum(dim=1) + 1e-6
            )
            dice_loss = 1.0 - dice.mean()
            light_loss = dice_loss
            rec_l1 = mse_loss
            total = voxel_weight * (mse_loss + float(dice_weight) * dice_loss)
        else:
            light_loss = self.sparse_light_loss(flat_pred, flat_target)
            rec_l1 = self.l1_loss(torch.sigmoid(pred), target)
            total = voxel_weight * (light_loss + rec_l1)

        extra_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
        if isinstance(aux_outputs, dict):
            for key in ("projection_loss", "perceptual_loss", "distribution_loss", "stn_loss"):
                value = aux_outputs.get(key)
                if torch.is_tensor(value):
                    extra_loss = extra_loss + value
        total = total + extra_loss
        return {
            "total_loss": total,
            "voxel_light_loss": light_loss,
            "voxel_l1_loss": rec_l1,
            "voxel_aux_loss": extra_loss,
        }


class MorphologyAwareDensityLoss(nn.Module):
    """Probability-domain SSQ-FMT density loss with optional sampled SDF supervision."""

    def __init__(
        self,
        lambda_sdf: float = 0.0,
        tau_s: float = 3.0,
        boundary_weight: float = 1.0,
        pos_weight: float = 1.0,
        dice_weight: float = 0.0,
        sparse_weight: float = 0.0,
    ):
        super().__init__()
        self.lambda_sdf = float(lambda_sdf)
        self.tau_s = float(tau_s)
        self.boundary_weight = float(boundary_weight)
        self.pos_weight = float(pos_weight)
        self.dice_weight = float(dice_weight)
        self.sparse_weight = float(sparse_weight)

    @staticmethod
    def _sample_gt(gt_voxels: torch.Tensor, points_ijk: torch.Tensor) -> torch.Tensor:
        idx = points_ijk.round().long()
        x = idx[..., 0].clamp(0, gt_voxels.shape[1] - 1)
        y = idx[..., 1].clamp(0, gt_voxels.shape[2] - 1)
        z = idx[..., 2].clamp(0, gt_voxels.shape[3] - 1)
        batch = torch.arange(gt_voxels.shape[0], device=gt_voxels.device)[:, None]
        return gt_voxels[batch, x, y, z].unsqueeze(-1)

    def forward(
        self,
        pred_density: torch.Tensor,
        target_density: torch.Tensor,
        aux_outputs: dict | None = None,
        *,
        gt_voxels: torch.Tensor | None = None,
        points_ijk: torch.Tensor | None = None,
        sdf_targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        target_density = target_density.to(device=pred_density.device, dtype=pred_density.dtype)
        pred = pred_density.clamp(0.0, 1.0)
        if aux_outputs is not None and torch.is_tensor(aux_outputs.get("measurement_supported")):
            support = aux_outputs["measurement_supported"].to(
                device=pred.device, dtype=pred.dtype
            ).unsqueeze(-1)
        else:
            support = torch.ones_like(pred)
        density_err = F.smooth_l1_loss(pred, target_density, reduction="none")
        density_weight = 1.0 + target_density.clamp(0.0, 1.0) * (self.pos_weight - 1.0)
        density_weight = density_weight * support
        density_loss = (density_err * density_weight).sum() / density_weight.sum().clamp_min(1e-8)
        eps = 1e-6
        pred_flat = pred.squeeze(-1)
        target_flat = target_density.squeeze(-1).clamp(0.0, 1.0)
        support_flat = support.squeeze(-1)
        intersection = (pred_flat * target_flat * support_flat).sum(dim=1)
        dice = (2.0 * intersection + eps) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + eps)
        denom = (pred_flat * support_flat).sum(dim=1) + (target_flat * support_flat).sum(dim=1)
        dice = (2.0 * intersection + eps) / (denom + eps)
        valid_samples = support_flat.sum(dim=1) > 0
        dice_loss = 1.0 - dice[valid_samples].mean() if valid_samples.any() else pred.sum() * 0.0
        sparse_num = (pred * (1.0 - target_density.clamp(0.0, 1.0)) * support).sum()
        sparse_loss = sparse_num / support.sum().clamp_min(1e-8)
        total = density_loss + self.dice_weight * dice_loss + self.sparse_weight * sparse_loss
        sdf_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
        if (
            self.lambda_sdf > 0.0
            and aux_outputs is not None
            and "sdf" in aux_outputs
            and sdf_targets is not None
        ):
            sdf_target = sdf_targets.to(device=pred.device, dtype=pred.dtype)
            sdf_pred = aux_outputs["sdf"].to(dtype=pred.dtype)
            weight = 1.0 + self.boundary_weight * (sdf_target.abs() < 0.25).to(dtype=pred.dtype)
            err = F.smooth_l1_loss(sdf_pred, sdf_target, reduction="none") * weight * support
            sdf_loss = err.sum() / (weight * support).sum().clamp_min(1e-8)
            total = total + self.lambda_sdf * sdf_loss
        unsupported_query_ratio = 1.0 - support.mean()
        unsupported_positive_gt_ratio = (
            ((1.0 - support) * (target_density > 0).to(dtype=pred.dtype)).sum()
            / (target_density > 0).to(dtype=pred.dtype).sum().clamp_min(1e-8)
        )
        return {
            "total_loss": total,
            "density_loss": density_loss,
            "dice_loss": dice_loss,
            "sparse_loss": sparse_loss,
            "sdf_loss": sdf_loss,
            "unsupported_query_ratio": unsupported_query_ratio,
            "unsupported_positive_gt_ratio": unsupported_positive_gt_ratio,
        }


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
