import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
import numpy as np
from .base import ProjectionConfig
from ..utils.utils import index_2d
from ..utils.cam import project_points_to_camera, VolumeProjector


class PointFeatureSampler:
    """
    点特征采样器：从各个视图的特征图中采样点特征
    """

    def project_points_to_view(
        self,
        points: torch.Tensor,
        view_name: str,
        camera_distance: int,
        detector_size: tuple[int],
        voxel_shape: tuple[int],
    ):
        """将3D点投影到2D视图坐标, 并且归一化到(-1,1)"""

        # 坐标转换
        voxel_shape = torch.as_tensor(voxel_shape, device=points.device, dtype=points.dtype)
        points = points * (voxel_shape - 1)
        points = points - voxel_shape / 2.0 + 0.5
        proj, dep = project_points_to_camera(points, int(view_name), camera_distance, detector_size)

        detector_size_t = torch.as_tensor(detector_size, device=points.device, dtype=points.dtype)
        proj_coords = proj / detector_size_t * 2

        return proj_coords


def _get_norm_layer(num_channels, norm_type="group"):
    """自适应选择 norm layer，确保通道数可整除 groups 数"""
    if norm_type == "group":
        # 自动找一个合适的 groups 数
        for groups in [32, 16, 8, 4, 2, 1]:
            if num_channels % groups == 0:
                return nn.GroupNorm(groups, num_channels)
        return nn.GroupNorm(1, num_channels)
    elif norm_type == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)
    else:
        return nn.BatchNorm2d(num_channels)


class DoubleConv(nn.Module):
    """U-Net基础模块：两次卷积+GN/IN+ReLU（小batch稳定版本）

    Args:
        in_channels, out_channels, mid_channels: 通道数
        norm_type: 'group'(GroupNorm)/'instance'(InstanceNorm)/'batch'(BatchNorm2d)，默认group
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, norm_type="group"):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            _get_norm_layer(mid_channels, norm_type),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _get_norm_layer(out_channels, norm_type),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """U-Net下采样模块：MaxPool+DoubleConv

    Args:
        norm_type: 'group'/'instance'/'batch'，传递给 DoubleConv
    """

    def __init__(self, in_channels, out_channels, norm_type="group"):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, norm_type=norm_type)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """U-Net上采样模块：双线性插值/转置卷积+特征拼接+DoubleConv

    Args:
        norm_type: 'group'/'instance'/'batch'，传递给 DoubleConv
    """

    def __init__(self, in_channels, out_channels, bilinear=True, norm_type="group"):
        super().__init__()
        # 双线性插值（无参数，计算量小）或转置卷积（有参数，可能更精确）
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm_type=norm_type)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels, norm_type=norm_type)

    def forward(self, x1, x2):
        # x1: 上采样输入（低分辨率）；x2: 跳跃连接输入（高分辨率）
        x1 = self.up(x1)
        # 处理尺寸不匹配（边缘对齐）
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)  # 通道维度拼接
        return self.conv(x)


class OutConv(nn.Module):
    """U-Net输出卷积：调整通道数"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """标准U-Net实现（支持多尺度特征输出，用于特征提取）

    Args:
        norm_type: 'group'/'instance'/'batch'，默认 'group' 以支持小batch稳定性
    """

    def __init__(self, n_channels, n_features=64, bilinear=False, norm_type="group"):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_features = n_features  # 基础通道数
        self.bilinear = bilinear
        self.norm_type = norm_type

        # 编码器（下采样）
        self.inc = DoubleConv(n_channels, n_features, norm_type=norm_type)
        self.down1 = Down(n_features, n_features * 2, norm_type=norm_type)
        self.down2 = Down(n_features * 2, n_features * 4, norm_type=norm_type)
        self.down3 = Down(n_features * 4, n_features * 8, norm_type=norm_type)
        factor = 2 if bilinear else 1
        self.down4 = Down(n_features * 8, n_features * 16 // factor, norm_type=norm_type)

        # 解码器（上采样）
        self.up1 = Up(n_features * 16, n_features * 8 // factor, bilinear, norm_type=norm_type)
        self.up2 = Up(n_features * 8, n_features * 4 // factor, bilinear, norm_type=norm_type)
        self.up3 = Up(n_features * 4, n_features * 2 // factor, bilinear, norm_type=norm_type)
        self.up4 = Up(n_features * 2, n_features, bilinear, norm_type=norm_type)

        # 多尺度特征融合（保留输入分辨率的特征图）
        self.feature_fusion = nn.Sequential(
            DoubleConv(n_features, n_features, norm_type=norm_type),
            OutConv(n_features, n_features),  # 输出与输入通道数一致（便于后续融合）
        )

    def forward(self, x):
        """输入单视图投影图，输出同分辨率特征图（多尺度融合）"""
        # 编码器特征
        x1 = self.inc(x)  # [B, 64, H, W]
        x2 = self.down1(x1)  # [B, 128, H/2, W/2]
        x3 = self.down2(x2)  # [B, 256, H/4, W/4]
        x4 = self.down3(x3)  # [B, 512, H/8, W/8]
        x5 = self.down4(x4)  # [B, 512, H/16, W/16]（若bilinear=True）

        # 解码器特征
        x = self.up1(x5, x4)  # [B, 256, H/8, W/8]
        x = self.up2(x, x3)  # [B, 128, H/4, W/4]
        x = self.up3(x, x2)  # [B, 64, H/2, W/2]
        x = self.up4(x, x1)  # [B, 64, H, W]

        # 融合并输出与输入同分辨率的特征图
        feature_map = self.feature_fusion(x)  # [B, 64, H, W]
        return feature_map


class DualBranchUNetEncoder(nn.Module):
    """
    改动1：双分支U-Net编码器
    从多视角输入提取特征，包括Source和Background两个分支。

    输入: Y_views [B, V, 1, H, W]
    输出: F_src, F_bg, Y_des
    """

    def __init__(self, n_channels=1, n_features=64, c_src=64, c_bg=32, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_features = n_features
        self.c_src = c_src
        self.c_bg = c_bg
        self.bilinear = bilinear

        # 获取两个UNet的编码器和解码器
        src_unet = UNet(n_channels=n_channels, n_features=n_features, bilinear=bilinear)
        bg_unet = UNet(n_channels=n_channels, n_features=n_features, bilinear=bilinear)

        # 共用编码器
        self.inc = src_unet.inc
        self.down1 = src_unet.down1
        self.down2 = src_unet.down2
        self.down3 = src_unet.down3
        self.down4 = src_unet.down4

        # Source分支解码器
        self.src_up1 = src_unet.up1
        self.src_up2 = src_unet.up2
        self.src_up3 = src_unet.up3
        self.src_up4 = src_unet.up4
        self.src_out = OutConv(n_features, c_src)

        # Background分支解码器
        self.bg_up1 = bg_unet.up1
        self.bg_up2 = bg_unet.up2
        self.bg_up3 = bg_unet.up3
        self.bg_up4 = bg_unet.up4
        self.bg_out = OutConv(n_features, c_bg)

        # 去散射head
        self.descatter_head = nn.Sequential(
            DoubleConv(c_bg, max(c_bg // 2, 16)),
            OutConv(max(c_bg // 2, 16), 1),
        )

    def forward(self, Y_views):
        """
        输入：Y_views [B, V, 1, H, W]
        输出：F_src [B, V, C_src, H, W], F_bg [B, V, C_bg, H, W], Y_des [B, V, 1, H, W]
        """
        B, V, C, H, W = Y_views.shape
        Y_views_flat = Y_views.reshape(B * V, C, H, W)

        # 共用编码器
        x1 = self.inc(Y_views_flat)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Source分支
        src_x = self.src_up1(x5, x4)
        src_x = self.src_up2(src_x, x3)
        src_x = self.src_up3(src_x, x2)
        src_x = self.src_up4(src_x, x1)
        F_src_flat = self.src_out(src_x)

        # Background分支
        bg_x = self.bg_up1(x5, x4)
        bg_x = self.bg_up2(bg_x, x3)
        bg_x = self.bg_up3(bg_x, x2)
        bg_x = self.bg_up4(bg_x, x1)
        F_bg_flat = self.bg_out(bg_x)

        # 去散射
        Y_des_flat = self.descatter_head(F_bg_flat)

        # Reshape回[B, V, ...]
        F_src = F_src_flat.reshape(B, V, self.c_src, H, W)
        F_bg = F_bg_flat.reshape(B, V, self.c_bg, H, W)
        Y_des = Y_des_flat.reshape(B, V, 1, H, W)

        return F_src, F_bg, Y_des


class PositionalEncoding3D(nn.Module):
    """3D位置编码（增强空间位置区分性，官方Transformer风格实现）"""

    def __init__(self, d_model, max_coords=1):
        super().__init__()
        self.d_model = d_model
        self.max_coords = max_coords
        assert d_model % 6 == 0, "d_model must be divisible by 6"
        self.num_freqs = d_model // 6

    def forward(self, x):
        """
        x: [B, N, 3] 3D点坐标
        输出: [B, N, d_model]
        """
        x = x / self.max_coords * 2 - 1
        B, N, _ = x.shape
        freqs = torch.linspace(1.0, 10.0, self.num_freqs, device=x.device)
        freqs = 2 * np.pi * freqs

        encodings = []
        for i in range(3):
            coord = x[..., i]
            for freq in freqs:
                encodings.append(torch.sin(coord * freq))
                encodings.append(torch.cos(coord * freq))

        pos_enc = torch.stack(encodings, dim=-1)
        return pos_enc


class ViewAngleEncoder(nn.Module):
    """将视角角度编码为高维向量"""

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, angles_deg):
        """
        参数: angles_deg [num_views]
        返回: embeddings [num_views, embed_dim]
        """
        rads = torch.deg2rad(angles_deg)
        s = torch.sin(rads)
        c = torch.cos(rads)
        x = torch.stack([s, c], dim=-1)
        return self.fc(x)


class SpatialAttentionFusion(nn.Module):
    """改进的空间注意力融合"""

    def __init__(self, feature_dim=64, pos_enc_dim=120, kernel_size=9):
        super().__init__()
        self.kernel_size = kernel_size
        self.feature_dim = feature_dim
        self.pos_encoder = PositionalEncoding3D(d_model=pos_enc_dim)
        self.sampler = PointFeatureSampler()
        self.view_encoder = ViewAngleEncoder(embed_dim=feature_dim)

        self.query_proj = nn.Linear(pos_enc_dim + feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim * 2, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)

        self.layer_norm_content = nn.LayerNorm(feature_dim)
        self.layer_norm_angle = nn.LayerNorm(feature_dim)
        self.layer_norm_query = nn.LayerNorm(feature_dim)

        self.num_heads = 4
        self.head_dim = feature_dim // self.num_heads

        self.cross_view_attn = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=self.num_heads, batch_first=True
        )

        self.output_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, x3d, view_features):
        """
        x3d: [B, N, 3]
        view_features: dict {view_name: feature_map}
        """
        B, N, _ = x3d.shape
        pos_enc = self.pos_encoder(x3d)

        local_features_list = []
        view_angles = []

        view_list = ["-90", "-60", "-30", "0", "30", "60", "90"]

        for view_name in view_list:
            feat_map = view_features[view_name]
            grid = self.sampler.project_points_to_view(
                x3d, view_name, 200, (256, 256), (182, 164, 210)
            )
            grid = grid.unsqueeze(1)

            feat_map = feat_map.transpose(2, 3)
            local_feat = F.grid_sample(
                feat_map,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            local_feat = local_feat.squeeze(2).permute(0, 2, 1)
            local_features_list.append(local_feat)
            view_angles.append(float(view_name))

        stack_features = torch.stack(local_features_list, dim=2)

        angles_tensor = torch.tensor(view_angles, device=x3d.device, dtype=torch.float32)
        view_embeddings = self.view_encoder(angles_tensor)
        view_embeddings = view_embeddings.view(1, 1, len(view_list), -1).expand(B, N, -1, -1)

        normed_stack = self.layer_norm_content(stack_features)
        normed_angle = self.layer_norm_angle(view_embeddings)

        features_with_pos = torch.cat([normed_stack, normed_angle], dim=-1)

        V = stack_features.size(2)
        cross_input = normed_stack.reshape(B * N, V, -1)
        cross_feat, _ = self.cross_view_attn(cross_input, cross_input, cross_input)
        cross_feat = cross_feat.reshape(B, N, V, -1)

        enhanced_stack = normed_stack + cross_feat

        global_context = enhanced_stack.mean(dim=2)
        query_input = torch.cat([pos_enc, global_context], dim=-1)
        query = self.query_proj(query_input)
        query = self.layer_norm_query(query)
        query = query.unsqueeze(2)

        key = self.key_proj(features_with_pos)

        attn_scores = (query * key).sum(dim=-1) / (self.feature_dim**0.5)
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)

        value = self.value_proj(enhanced_stack)
        fused_feat = (value * attn_weights).sum(dim=2)

        fused_feat = fused_feat + self.output_proj(fused_feat)

        return fused_feat


class GateFusion(nn.Module):
    """简化的门控融合模块"""

    def __init__(self, in_channels=64, out_channels=1):
        super().__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, unet_feat):
        """
        Args: unet_feat [B, 64, H, W]
        Returns: output [B, out_channels, H, W]
        """
        channel_weight = self.channel_gate(unet_feat)
        feat = unet_feat * channel_weight

        spatial_weight = self.spatial_gate(feat)
        feat = feat * spatial_weight

        return self.out_conv(feat)
