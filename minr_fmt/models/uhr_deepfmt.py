"""
UHR-DeepFMT: 3D U-Net based voxel network for NIR-II FMT reconstruction

核心架构：3D编码器-解码器 + 双池化SE块 + 膨胀卷积上采样
所有参数从config对象提取，无硬编码默认值
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config_extractor import ConfigExtractor


# ===== 核心模块 =====


class DualPoolingSEBlock(nn.Module):
    """双池化 + SE块：用于skip连接融合"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc1 = nn.Linear(channels, max(channels // reduction, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(max(channels // reduction, 1), channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C = x.size(0), x.size(1)
        avg_feat = self.avg_pool(x)
        max_feat = self.max_pool(x)
        pooled = (avg_feat + max_feat) / 2.0
        pooled = pooled.view(B, C)
        se = self.fc1(pooled)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        return x * se.view(B, C, 1, 1, 1)


class ConvBlock3D(nn.Module):
    """3D卷积块：Conv3D -> BatchNorm -> ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size, padding=padding, dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DoubleConv3D(nn.Module):
    """双3D卷积块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            ConvBlock3D(in_channels, out_channels),
            ConvBlock3D(out_channels, out_channels),
        )

    def forward(self, x):
        return self.double_conv(x)


class DilatedConvUpsample(nn.Module):
    """膨胀卷积上采样模块"""

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.dilated_conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=3, padding=2, dilation=2,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dilated_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return F.interpolate(
            x, scale_factor=self.scale_factor,
            mode="trilinear", align_corners=False
        )


class Encoder3D(nn.Module):
    """3D编码器：逐级下采样"""

    def __init__(self, in_channels, base_channels, num_levels):
        super().__init__()
        self.num_levels = num_levels
        self.down_paths = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_ch = in_channels
        for i in range(num_levels):
            out_ch = base_channels * (2**i)
            self.down_paths.append(DoubleConv3D(in_ch, out_ch))
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            in_ch = out_ch

    def forward(self, x):
        features = []
        for i in range(self.num_levels):
            x = self.down_paths[i](x)
            features.append(x)
            x = self.pools[i](x)
        return features


class BottleneckBlock(nn.Module):
    """Bottleneck块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x):
        return self.double_conv(x)


class Decoder3D(nn.Module):
    """3D解码器：逐级上采样 + skip融合"""

    def __init__(self, base_channels, num_levels, out_channels=1):
        super().__init__()
        self.num_levels = num_levels
        self.up_paths = nn.ModuleList()
        self.se_blocks = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        for i in range(num_levels):
            level_idx = num_levels - 1 - i
            in_ch = base_channels * (2 ** (level_idx + 1))
            out_ch = base_channels * (2**level_idx)

            self.up_paths.append(DilatedConvUpsample(in_ch, out_ch, scale_factor=2))
            self.se_blocks.append(DualPoolingSEBlock(out_ch, reduction=16))
            self.skip_convs.append(nn.Sequential(
                nn.Conv3d(out_ch * 2, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
            ))

        self.final_conv = nn.Sequential(
            DoubleConv3D(base_channels, base_channels),
            nn.Conv3d(base_channels, out_channels, kernel_size=1, bias=True),
        )

    def forward(self, x, skip_features):
        for i in range(self.num_levels):
            x = self.up_paths[i](x)
            skip = skip_features[i]

            if skip.shape[2:] != x.shape[2:]:
                skip = F.interpolate(
                    skip, size=x.shape[2:],
                    mode="trilinear", align_corners=False
                )

            skip = self.se_blocks[i](skip)
            x = torch.cat([x, skip], dim=1)
            x = self.skip_convs[i](x)

        return self.final_conv(x)


# ===== 主网络 =====


class UHRDeepFMT3DUNet(nn.Module):
    """3D U-Net体素重建网络
    
    架构流程：
    1. 投影到3D体积（自动处理H/W不匹配）
    2. 3D编码器下采样
    3. Bottleneck
    4. 3D解码器上采样（带SE融合）
    5. 输出密度预测
    """

    def __init__(self, config):
        """
        Args:
            config: 配置对象，包含所有网络参数
        """
        super().__init__()
        
        if config is None:
            raise ValueError("config 参数为必需项，不能为 None")

        # 提取参数
        net_params = ConfigExtractor.extract_network_params(config)
        geo_params = ConfigExtractor.extract_geometry_config(config)
        uhr_params = ConfigExtractor.extract_uhr_deepfmt_config(config)

        # 网络参数
        num_views = net_params["num_views"]
        base_channels = uhr_params["base_channels"]
        num_levels = uhr_params["num_levels"]
        
        # 几何参数
        self.camera_distance = geo_params["camera_distance"]
        self.detector_size = geo_params["detector_size"]
        self.global_voxel_shape = geo_params["global_voxel_shape"]
        self.config = config
        
        # ROI体积参数：UHR 只回归 ROI（由 data.voxel_ranges 决定）
        vr = config.data.voxel_ranges
        self.roi_x = int(vr.x[1] - vr.x[0])
        self.roi_y = int(vr.y[1] - vr.y[0])
        self.roi_z = int(vr.z[1] - vr.z[0])
        self.voxel_depth = self.roi_z
        
        # 网络参数
        self.in_channels = num_views
        self.base_channels = base_channels
        self.num_levels = num_levels

        # 3D U-Net组件
        self.encoder = Encoder3D(num_views, base_channels, num_levels)

        bottleneck_in = base_channels * (2 ** (num_levels - 1))
        bottleneck_out = base_channels * (2**num_levels)
        self.bottleneck = BottleneckBlock(bottleneck_in, bottleneck_out)

        self.decoder = Decoder3D(base_channels, num_levels, out_channels=1)
        # Return logits; loss uses BCEWithLogits.
        self.output_activation = nn.Identity()

    def _reshape_projections_to_3d(self, projections_dict, target_hw=None):
        """将2D投影转换为3D体积张量
        
        自动处理相机H/W与ROI H/W的不匹配
        """
        view_order = ["-90", "-60", "-30", "0", "30", "60", "90"]

        proj_list = []
        for view_name in view_order:
            proj = projections_dict[view_name]
            if proj.dim() == 3:
                if target_hw is not None:
                    B, H_cam, W_cam = proj.shape
                    H_target, W_target = target_hw

                    if H_cam != H_target or W_cam != W_target:
                        proj = proj.unsqueeze(1)
                        proj = F.interpolate(
                            proj, size=(H_target, W_target),
                            mode="bilinear", align_corners=False,
                        )
                        proj = proj.squeeze(1)

                proj_list.append(proj)
            else:
                raise ValueError(f"投影shape应为[B,H,W]，得到{proj.shape}")

        volume = torch.stack(proj_list, dim=1)
        B, V, H, W = volume.shape
        D = self.voxel_depth
        volume_3d = volume.unsqueeze(2).repeat(1, 1, D, 1, 1)

        return volume_3d

    def forward(self, projections_dict, points=None, target_proj_hw=None, **kwargs):
        """
        Args:
            projections_dict: dict {view_name: [B, H, W]}
            points: [B, N, 3] (可选)
            target_proj_hw: (H, W) ROI尺寸

        Returns:
            output: [B, N, 1] or [B, D*H*W, 1]
            aux_output: dict
        """
        # 转换投影到3D（默认对齐 ROI 的 x/y 尺寸）
        if target_proj_hw is None and (
            points is not None
            or str(getattr(self.config.model, "output_type", "query")).lower() == "voxel"
        ):
            target_proj_hw = (self.roi_x, self.roi_y)
        x = self._reshape_projections_to_3d(projections_dict, target_hw=target_proj_hw)

        # 编码-Bottleneck-解码
        skip_features = self.encoder(x)
        x = skip_features[-1]
        x = self.bottleneck(x)
        skip_features_reversed = skip_features[::-1]
        output = self.decoder(x, skip_features_reversed)

        # 激活和重塑
        output = self.output_activation(output)
        if str(getattr(self.config.model, "output_type", "query")).lower() == "voxel":
            # Decoder is [B,1,z,x,y]; FMT-SimGen GT is indexed [x,y,z].
            pred_voxel = output.permute(0, 1, 3, 4, 2).contiguous()
            aux_output = {k: v for k, v in projections_dict.items()}
            return {"pred_voxel": pred_voxel, "aux_outputs": aux_output}
        B, _, D, H, W = output.shape
        output_flat = output.view(B, -1, 1)

        # 点级预测（如提供points）：从 ROI 体素里采样 points 对应位置
        if points is not None:
            # points are normalized in global voxel coordinates (x,y,z in [0,1])
            vr = self.config.data.voxel_ranges
            global_shape = torch.tensor(self.global_voxel_shape, device=points.device, dtype=points.dtype)
            pts_global = torch.round(points * (global_shape - 1)).long()  # [B,N,3] as (x,y,z)

            x0, y0, z0 = int(vr.x[0]), int(vr.y[0]), int(vr.z[0])
            xi = (pts_global[..., 0] - x0).clamp(0, H - 1)  # H maps to ROI-x
            yi = (pts_global[..., 1] - y0).clamp(0, W - 1)  # W maps to ROI-y
            zi = (pts_global[..., 2] - z0).clamp(0, D - 1)  # D maps to ROI-z

            vol = output[:, 0]  # [B,D,H,W]
            b_idx = torch.arange(B, device=points.device).view(B, 1).expand_as(zi)
            output_flat = vol[b_idx, zi, xi, yi].unsqueeze(-1)

        aux_output = {k: v for k, v in projections_dict.items()}
        return output_flat, aux_output


# 向后兼容性
UHRDeepFMT3DUNetV2 = UHRDeepFMT3DUNet
