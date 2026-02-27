from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoder import get_encoder


class DirectVoxelDecoder(nn.Module):
    """
    方式1: 直接2D特征还原3D体素密度
    """

    def __init__(
        self,
        input_feature_dim=256,
        target_voxel_size: Tuple[int, int, int] = (64, 64, 64),
    ):
        super().__init__()
        self.target_voxel_size = target_voxel_size

        # 2D特征到3D体素的投影
        self.feature_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
        )

        # 动态计算初始3D尺寸
        init_size = self._compute_initial_size(target_voxel_size)
        self.init_3d_features = nn.Linear(
            2048, 128 * init_size[0] * init_size[1] * init_size[2]
        )
        self.init_channels = 128

        # 构建动态3D解码器
        self.decoder_3d = self._build_3d_decoder(init_size, target_voxel_size)

        # 密度预测头
        self.density_head = nn.Sequential(
            nn.Conv3d(16, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 1, 1),
            nn.Sigmoid(),
        )

    def _compute_initial_size(self, target_size):
        """计算合适的初始3D尺寸"""
        # 找到最小的2的幂次尺寸
        min_dim = min(target_size)
        init_dim = max(4, 2 ** int(np.log2(min_dim) - 3))  # 至少4，最多缩小8倍
        return (init_dim, init_dim, init_dim)

    def _build_3d_decoder(self, init_size, target_size):
        """动态构建3D上采样路径"""
        decoder_blocks = nn.ModuleList()
        current_size = init_size
        current_channels = self.init_channels

        # 逐步上采样直到达到目标尺寸
        while current_size != target_size:
            # 计算下一个尺寸（逐步接近目标尺寸）
            next_size = []
            for curr, target in zip(current_size, target_size):
                if curr * 2 <= target:
                    next_size.append(curr * 2)
                else:
                    next_size.append(target)
            next_size = tuple(next_size)

            # 计算缩放因子
            scale_factors = tuple(n / c for n, c in zip(next_size, current_size))

            # 构建上采样块
            out_channels = max(16, current_channels // 2)
            decoder_blocks.append(
                self._upsample_3d_block(current_channels, out_channels, scale_factors)
            )

            current_size = next_size
            current_channels = out_channels

        return decoder_blocks

    def _upsample_3d_block(self, in_channels, out_channels, scale_factors):
        return nn.Sequential(
            nn.Upsample(
                scale_factor=scale_factors, mode="trilinear", align_corners=False
            ),
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, fused_2d_features: torch.Tensor) -> torch.Tensor:
        batch_size = fused_2d_features.shape[0]

        # 投影到3D空间
        projected = self.feature_projection(fused_2d_features)
        initial_3d = self.init_3d_features(projected)
        initial_3d = initial_3d.view(
            batch_size,
            self.init_channels,
            *self._compute_initial_size(self.target_voxel_size),
        )

        # 3D解码
        x = initial_3d
        for decoder_block in self.decoder_3d:
            x = decoder_block(x)

        # 最终密度预测
        density_field = self.density_head(x)

        return density_field


class PointSamplingDecoder(nn.Module):
    """
    基于融合特征的点采样解码器：使用CrossViewFusion的输出特征进行3D密度回归
    """

    def __init__(
        self, fused_feature_dim: int = 256, mlp_hidden_dims: list[int] | None = None
    ):
        super().__init__()
        self.fused_feature_dim = fused_feature_dim

        # Instant NGP 哈希编码器

        self.hash_encoder = get_encoder("hashgrid")
        hash_encoding_dim = self.hash_encoder.output_dim

        # MLP输入维度 = 哈希编码特征 + 融合特征
        mlp_input_dim = hash_encoding_dim + fused_feature_dim

        if mlp_hidden_dims is None:
            mlp_hidden_dims = [512, 256, 128, 64, 32]

        # 构建MLP网络
        mlp_layers = []
        current_dim = mlp_input_dim

        for hidden_dim in mlp_hidden_dims:
            mlp_layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                ]
            )
            current_dim = hidden_dim

        # 输出层
        mlp_layers.extend([nn.Linear(current_dim, 1), nn.Sigmoid()])

        self.mlp = nn.Sequential(*mlp_layers)

        print(f"FusionBasedPointSamplingDecoder:")
        print(f"  Hash encoding dim: {hash_encoding_dim}")
        print(f"  Fused feature dim: {fused_feature_dim}")
        print(f"  MLP input dim: {mlp_input_dim}")
        print(
            f"  MLP structure: {mlp_input_dim} -> {' -> '.join(map(str, mlp_hidden_dims))} -> 1"
        )

    def forward(
        self, fused_features: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            fused_features: [B, C, H, W] 跨视图融合的特征图
            points: [B, N, 3] 3D点坐标（归一化到[0,1]）
        Returns:
            densities: [B, N, 1] 每个点的密度值
        """
        batch_size, num_points, _ = points.shape

        # 1. 使用Instant NGP哈希编码处理3D坐标
        hash_features = self.hash_encoder(points)  # [B, N, hash_dim]

        # 2. 从融合特征图中采样对应点的特征
        point_fused_features = self._sample_fused_features(
            fused_features, points
        )  # [B, N, fused_feature_dim]

        # 3. 拼接哈希编码特征和融合特征
        combined_input = torch.cat(
            [hash_features, point_fused_features], dim=-1
        )  # [B, N, hash_dim + fused_feature_dim]

        # 4. 通过MLP预测密度
        densities = self.mlp(combined_input)  # [B, N, 1]

        return densities

    def _sample_fused_features(
        self, fused_features: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:
        """
        从融合特征图中采样点特征
        使用正交投影将3D点映射到2D特征图
        """
        batch_size, num_points, _ = points.shape
        _, feat_dim, feat_h, feat_w = fused_features.shape

        # 方法1: 使用正交投影（XY平面投影）
        # 假设融合特征图对应的是XY平面的视图
        proj_coords = points[:, :, :2].clone()  # 使用XY坐标 [B, N, 2]

        # 方法2: 多平面投影（可选，提供更多几何信息）
        # 可以同时从多个平面投影并拼接特征
        xy_features = self._project_to_plane(fused_features, points, "xy")
        xz_features = self._project_to_plane(fused_features, points, "xz")
        yz_features = self._project_to_plane(fused_features, points, "yz")
        point_fused_features = torch.cat(
            [xy_features, xz_features, yz_features], dim=-1
        )

        # 缩放坐标到特征图尺寸
        proj_coords[:, :, 0] = proj_coords[:, :, 0] * (feat_w - 1)  # X -> 宽度
        proj_coords[:, :, 1] = proj_coords[:, :, 1] * (feat_h - 1)  # Y -> 高度

        # 转换到[-1, 1]范围（grid_sample要求的范围）
        proj_coords[:, :, 0] = 2.0 * proj_coords[:, :, 0] / (feat_w - 1) - 1.0
        proj_coords[:, :, 1] = 2.0 * proj_coords[:, :, 1] / (feat_h - 1) - 1.0

        # 双线性插值采样特征
        point_features = F.grid_sample(
            fused_features,
            proj_coords.unsqueeze(2).unsqueeze(2),  # [B, N, 1, 1, 2]
            mode="bilinear",
            align_corners=True,
            padding_mode="border",  # 边界处理
        )

        # 调整形状: [B, C, N, 1, 1] -> [B, N, C]
        point_features = point_features.squeeze(-1).squeeze(-1).transpose(1, 2)

        return point_features

    def _project_to_plane(
        self, features: torch.Tensor, points: torch.Tensor, plane: str
    ) -> torch.Tensor:
        """投影到指定平面"""
        batch_size, num_points, _ = points.shape
        _, feat_dim, feat_h, feat_w = features.shape

        if plane == "d1":
            proj_coords = points[:, :, :2]  # [B, N, 2]
        elif plane == "d2":
            proj_coords = points[:, :, :2]  # [B, N, 2]
        elif plane == "d3":
            proj_coords = points[:, :, 1:]  # [B, N, 2]
        elif plane == "d4":
            proj_coords = points[:, :, 1:]  # [B, N, 2]
        else:
            raise ValueError(f"Unknown view name: {view_name}")

        # 坐标转换
        proj_coords = proj_coords.clone()
        proj_coords[:, :, 0] = proj_coords[:, :, 0] * (feat_w - 1)
        proj_coords[:, :, 1] = proj_coords[:, :, 1] * (feat_h - 1)

        proj_coords[:, :, 0] = 2.0 * proj_coords[:, :, 0] / (feat_w - 1) - 1.0
        proj_coords[:, :, 1] = 2.0 * proj_coords[:, :, 1] / (feat_h - 1) - 1.0

        # 采样特征
        plane_features = (
            F.grid_sample(
                features,
                proj_coords.unsqueeze(2).unsqueeze(2),
                mode="bilinear",
                align_corners=True,
                padding_mode="border",
            )
            .squeeze(-1)
            .squeeze(-1)
            .transpose(1, 2)
        )

        return plane_features
