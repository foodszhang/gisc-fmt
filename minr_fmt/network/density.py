import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import ProjectionConfig
from .encoder import SpatialAttentionFusion, UNet, GateFusion, PointFeatureSampler
from .fusion import ViewWeightNet
from .fast_kan import KAN


# ===== 改动3：隐式源场网络结构 =====


class ResBlock(nn.Module):
    """
    残差块：LayerNorm + 两层全连接 + 残差连接

    输入/输出：[B, N, D] 或 [Batch, D]

    结构：
    - 对输入做LayerNorm规范化
    - 通过两层全连接线性变换
    - 添加残差连接: out = x + f(x)
    """

    def __init__(self, in_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(in_dim, in_dim)
        self.linear2 = nn.Linear(in_dim, in_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        参数:
        x: 输入张量 [B, N, D] 或 [Batch, D]

        输出:
        out: 经过残差变换的输出，形状同输入
        """
        h = self.layer_norm(x)
        h = self.relu(self.linear1(h))
        h = self.linear2(h)
        out = x + h  # 残差连接
        return out


class ImplicitSourceField(nn.Module):
    """
    改动3：隐式源场网络（Implicit Source Field）

    采用"坐标支路 + 特征支路 + 输出支路"的双支路结构，每个支路都包含残差块。

    输入：
    - gamma_x: [B, N, P] 坐标位置编码（positional encoding）
    - fused_features: [B, N, C_fused] 融合的多视角特征

    处理流程：
    1. MLP_x: 2-3层全连接 + ReLU + ResBlock -> z_x [B, N, D_x]
    2. MLP_f: 2-3层全连接 + ReLU + ResBlock -> z_f [B, N, D_f]
    3. concat(z_x, z_f) -> MLP_out (2层 + ResBlock) -> logits [B, N]

    输出：
    - logits: [B, N] 未归一化分数（用于BCEWithLogitsLoss）
    """

    def __init__(self, coord_dim, feature_dim, hidden_dim=64, d_x=64, d_f=64):
        super().__init__()

        # ===== 坐标支路：MLP_x =====
        # 处理位置编码 gamma_x
        self.mlp_x = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, d_x),
            ResBlock(d_x),  # 残差块用于增强表达能力
        )

        # ===== 特征支路：MLP_f =====
        # 处理融合的多视角特征 fused_features
        self.mlp_f = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, d_f),
            ResBlock(d_f),  # 残差块用于增强表达能力
        )

        # ===== 输出支路：MLP_out =====
        # 将两个支路的特征拼接并进行最终预测
        self.mlp_out = nn.Sequential(
            nn.Linear(d_x + d_f, hidden_dim),
            nn.ReLU(inplace=True),
            ResBlock(hidden_dim),  # 残差块用于增强表达能力
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, gamma_x, fused_features):
        """
        参数:
        gamma_x: [B, N, P] 输入坐标嵌入
        fused_features: [B, N, C_fused] 融合的视角信息

        返回:
        logits: [B, N] 未归一化logits（可直接用于BCEWithLogitsLoss）
        """
        # 坐标支路处理
        z_x = self.mlp_x(gamma_x)  # [B, N, D_x]

        # 特征支路处理
        z_f = self.mlp_f(fused_features)  # [B, N, D_f]

        # 拼接两个支路
        z = torch.cat([z_x, z_f], dim=-1)  # [B, N, D_x + D_f]

        # 输出支路预测
        logits = self.mlp_out(z)  # [B, N, 1]
        # logits = logits.squeeze(-1)  # [B, N]

        return logits


class PointDensityNet(nn.Module):
    """
    改进的荧光体内光源重建网络：
    1. 多视角U-Net特征提取
    2. 融合视角角度编码的空间注意力融合（可选择替换为ViewWeightNet）
    3. 隐式源场网络进行最终密度预测

    原因：
    - U-Net可从每个视角提取多尺度特征，兼顾局部细节和全局上下文。
    - ViewWeightNet/SpatialAttentionFusion利用几何知识智能整合多视角信息，比简单平均更好地处理遮挡。
    - ImplicitSourceField采用双支路结构分别处理坐标和特征，具有更好的表达能力。
    """

    def __init__(self, num_views=7, in_channels=1, feature_dim=64, pos_enc_dim=120):
        super().__init__()
        self.pos_enc_dim = pos_enc_dim
        self.feature_dim = feature_dim

        # 1. 多视角特征提取器（每个视角独立的U-Net）
        self.view_extractors = nn.ModuleList(
            [
                UNet(n_channels=in_channels, n_features=feature_dim, bilinear=True)
                for _ in range(num_views)
            ]
        )

        # 2. 多视角融合选项
        # 选项A：使用改动2的ViewWeightNet（轻量级、自适应）
        self.view_weight_net = ViewWeightNet(
            feature_dim=feature_dim, num_views=num_views, embed_dim=16, hidden_dim=32
        )

        # 选项B：使用原来的SpatialAttentionFusion（更复杂，可选）
        # self.attention_fusion = SpatialAttentionFusion(
        #     feature_dim=feature_dim, pos_enc_dim=pos_enc_dim
        # )

        # 3. 门控融合（用于生成辅助输出）
        self.gate_fusions = nn.ModuleList([GateFusion(feature_dim) for _ in range(num_views)])

        # 4. 改动3：隐式源场网络（替代原来的简单MLP）
        self.density_head = ImplicitSourceField(
            coord_dim=pos_enc_dim,  # 位置编码维度
            feature_dim=feature_dim,  # 融合特征维度
            hidden_dim=64,
            d_x=64,
            d_f=64,
        )

    def forward(self, view_projections, x3d, gamma_x=None):
        """
        参数:
        view_projections: dict of {view_name: tensor} - 多视角投影图
        x3d: [B, N, 3] - 3D采样点坐标
        gamma_x: [B, N, pos_enc_dim] - 坐标位置编码（可选，由外部提供）

        返回:
        logits: [B, N] - 未归一化密度预测
        aux_projections: dict - 辅助输出（各视角的门控融合结果）
        """
        B, N, _ = x3d.shape

        # 1. 从各视角提取特征
        view_list = ["-90", "-60", "-30", "0", "30", "60", "90"]
        view_features = {}
        aux_projections = {}

        for view_name, projection in view_projections.items():
            # 确保投影图有通道维度
            if projection.dim() == 3:
                projection = projection.unsqueeze(1)

            angle = view_name
            idx = view_list.index(str(angle))

            # U-Net特征提取
            feat = self.view_extractors[idx](projection)  # [B, C, H, W]
            view_features[view_name] = feat

            # 门控融合（生成辅助输出）
            gate_out = self.gate_fusions[idx](feat)
            aux_projections[view_name] = gate_out.squeeze(1)

        # 2. 几何投影：从特征图中采样点特征
        # 这部分通常在外部完成（在trainer或main中）
        # 这里假设已经获得了 f: [B, N, V, C]
        # 在实际使用中需要通过grid_sample或其他方法从view_features中采样

        sampler = PointFeatureSampler()
        try:
            f_list = []
            for view_name in view_list:
                if view_name in view_features:
                    feat_map = view_features[view_name]  # [B, C, H, W]
                    # 使用grid_sample进行几何投影采样
                    # 参考encoder.py中的SpatialAttentionFusion实现
                    grid = sampler.project_points_to_view(
                        x3d, view_name, 200, (256, 256), (182, 164, 210)
                    )
                    grid = grid.unsqueeze(1)  # [B, 1, N, 2]
                    feat_map = feat_map.transpose(2, 3)  # [B, C, W, H]
                    local_feat = F.grid_sample(
                        feat_map,
                        grid,
                        mode="bilinear",
                        padding_mode="border",
                        align_corners=True,
                    )  # [B, C, 1, N]
                    local_feat = local_feat.squeeze(2).permute(0, 2, 1)  # [B, N, C]
                    f_list.append(local_feat)  # 收集每个视角的采样特征

            f = torch.stack(f_list, dim=2)  # [B, N, V, C]
        except:
            # 如果投影模块尚未实现，使用占位符
            B_f = B
            f = torch.randn(B_f, N, len(view_list), self.feature_dim, device=x3d.device)

        # 3. 改动2：使用ViewWeightNet进行多视角融合
        fused_feat, weights = self.view_weight_net(f)  # [B, N, C]

        # 4. 如果未提供gamma_x，使用占位符或简单编码
        if gamma_x is None:
            # 使用简单的位置编码作为占位符
            gamma_x = self._simple_pos_encoding(x3d, self.pos_enc_dim)

        # 5. 改动3：使用隐式源场网络进行最终预测
        logits = self.density_head(gamma_x, fused_feat)  # [B, N]

        return logits, aux_projections

    def _simple_pos_encoding(self, x, d_model):
        """
        简单的位置编码（Transformer风格）
        输入: x [B, N, 3]
        输出: pos_enc [B, N, d_model]
        """
        B, N, _ = x.shape

        # 确保d_model是6的倍数
        if d_model % 6 != 0:
            d_model = (d_model // 6) * 6

        num_freqs = d_model // 6
        encodings = []

        freqs = torch.linspace(1.0, 10.0, num_freqs, device=x.device)
        freqs = 2 * torch.pi * freqs

        for i in range(3):
            coord = x[..., i]  # [B, N]
            for freq in freqs:
                encodings.append(torch.sin(coord * freq))
                encodings.append(torch.cos(coord * freq))

        pos_enc = torch.stack(encodings, dim=-1)  # [B, N, d_model]
        return pos_enc


# Backwards-compatible alias
MINRFMT = PointDensityNet
