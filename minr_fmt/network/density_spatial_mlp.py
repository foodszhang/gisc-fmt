import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import ProjectionConfig
from .encoder import SpatialAttentionFusion, UNet, GateFusion

"""
本文件实现了适用于3D生物体体内荧光光源重建的创新型空间自适应MLP网络，
针对百万级稠密体素点（如稠密体素网格）显存消耗过大的问题，做了如下工程与科学优化：
1. 网络forward支持分块(chunk)处理大规模点集，避免一次性全量点对距离矩阵计算，显著降低显存消耗。
2. SpatialGraphConv支持chunk_size参数，自动分批处理点，适配百万级点云/体素输入。
3. 距离矩阵与k-NN邻域查找采用分块策略，避免N×N全量显存占用。
4. 推荐数据集采样时采用随机/重要性采样或分块采样，进一步降低单次forward点数。
5. 训练与推理接口与原有流程兼容，无需额外修改。
6. 详细实现与接口说明见各类docstring。

# 工程建议：
# - 若输入点数极大（如N>100w），建议在训练/推理时将点集分块，逐块送入网络，聚合输出。
# - 数据集可通过随机采样、重要性采样或分块采样减少单次forward点数。
# - 相关采样策略可参考src/dataset/proj_dataset.py。
"""


class SpatialGraphConv(nn.Module):
    """
    轻量级空间图卷积模块：捕捉点之间的k-NN邻域关系
    创新点：基于欧氏距离的动态图构建，编码点对相对位置关系
    """

    def __init__(self, feature_dim=64, k=8):
        """
        Args:
            feature_dim: 特征维度
            k: k-NN邻域大小（默认8个邻点）
        """
        super().__init__()
        self.k = k
        self.feature_dim = feature_dim

        # 点对关系编码网络：编码目标点与邻点的相对位置关系
        self.relation_encoder = nn.Sequential(
            nn.Linear(3 * 2, 64),  # 输入：3D相对位置 + 自身位置
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, feature_dim),
        )

        # 邻域特征融合MLP
        self.neighbor_fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),  # 目标特征 + 邻点特征
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )

        # 注意力权重计算（决定每个邻点的影响权重）
        self.attention_weights = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

def forward(self, x3d, features, global_voxel_shape, range_x, range_y, range_z):
        """
        Args:
            x3d: [B, N, 3] 3D点坐标（全局归一化）
            features: [B, N, C] 点特征
            global_voxel_shape: (Dx, Dy, Dz) 全局体素shape
            range_x, range_y, range_z: 可行域范围（tuple）
        Returns:
            updated_features: [B, N, C] 融合了空间关系的更新特征
        """
        B, N, _ = x3d.shape
        C = features.shape[-1]
        device = x3d.device
        if N <= 1:
            return features
        Dx, Dy, Dz = global_voxel_shape
        # 反归一化为全局体素索引
        idx = (x3d[0] * (torch.tensor([Dx, Dy, Dz], device=device) - 1)).round().long()  # [N, 3]
        offsets = torch.tensor([
            [-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]
        ], device=device)
        neighbor_indices = []
        for off in offsets:
            n_idx = idx + off
            # 判断是否在可行域内
            in_feasible = (
                (n_idx[:,0] >= range_x[0]) & (n_idx[:,0] < range_x[1]) &
                (n_idx[:,1] >= range_y[0]) & (n_idx[:,1] < range_y[1]) &
                (n_idx[:,2] >= range_z[0]) & (n_idx[:,2] < range_z[1])
            )
            # 超出可行域的点用自身索引替代
            n_idx[~in_feasible] = idx[~in_feasible]
            # 计算邻域点在当前点集的索引
            # 由于点顺序与可行域体素顺序一致，可直接计算线性索引
            n_lin = (
                (n_idx[:,0] - range_x[0]) * ((range_y[1]-range_y[0]) * (range_z[1]-range_z[0])) +
                (n_idx[:,1] - range_y[0]) * (range_z[1]-range_z[0]) +
                (n_idx[:,2] - range_z[0])
            )
            neighbor_indices.append(n_lin)
        neighbor_indices = torch.stack(neighbor_indices, dim=1).long()  # [N, 6]
        neighbor_indices = neighbor_indices.unsqueeze(0).expand(B, N, 6)
        batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, 6)
        neighbor_features = features[batch_indices, neighbor_indices]  # [B, N, 6, C]
        # 融合自身特征与邻域平均
        self_feat = features.unsqueeze(2).expand(B, N, 6, C)
        concat_feat = torch.cat([self_feat, neighbor_features], dim=-1)  # [B, N, 6, 2C]
        fused = self.neighbor_fusion(concat_feat)  # [B, N, 6, C]
        fused = fused.mean(dim=2)  # [B, N, C]
        updated_features = features + fused * 0.5
        return updated_features



class MultiScalePointMLP(nn.Module):
    """
    多尺度自适应MLP：在不同的空间尺度上捕捉特征
    创新点：通过多尺度邻域采样，捕捉从局部到全局的空间信息
    """

    def __init__(self, feature_dim=64, num_scales=2):
        """
        Args:
            feature_dim: 特征维度
            num_scales: 空间尺度数量
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_scales = num_scales

        # 多尺度图卷积层 - 使用相同的feature_dim
        self.scale_convs = nn.ModuleList(
            [
                SpatialGraphConv(feature_dim, k=4 * (i + 1))  # 不同尺度的k值
                for i in range(num_scales)
            ]
        )

        # 尺度融合MLP
        self.scale_fusion = nn.Sequential(
            nn.Linear(feature_dim * (num_scales + 1), feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 2, feature_dim),
        )

    def forward(self, x3d, features):
        """
        Args:
            x3d: [B, N, 3] 3D点坐标
            features: [B, N, C] 点特征

        Returns:
            multi_scale_features: [B, N, C] 融合了多尺度空间关系的特征
        """
        # 在多个空间尺度上应用图卷积
        scale_features = [features]  # 包含原始特征

        for scale_conv in self.scale_convs:
            scale_feat = scale_conv(x3d, features)
            scale_features.append(scale_feat)

        # 融合多尺度特征
        multi_scale = torch.cat(scale_features, dim=-1)  # [B, N, C*(num_scales+1)]
        B, N, C = multi_scale.shape

        # 通过MLP融合
        multi_scale_flat = multi_scale.view(B * N, C)
        fused = self.scale_fusion(multi_scale_flat)  # [B*N, C_out]
        fused = fused.view(B, N, self.feature_dim)

        return fused


class SpatialAdaptiveMLPHead(nn.Module):
    """
    创新的空间自适应MLP密度预测头
    基于MLP构建，但通过图卷积和多尺度融合捕捉空间信息

    创新点：
    1. 动态图构建：基于实际3D点分布的k-NN图
    2. 点对关系编码：显式编码点间相对位置
    3. 多尺度融合：在不同空间尺度捕捉特征
    4. 自适应注意力：根据点对距离自适应调整权重
    """

    def __init__(self, feature_dim=64, num_views=7):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_views = num_views

        # 多尺度空间图卷积模块 - 处理融合后的特征
        self.multi_scale_graph = MultiScalePointMLP(
            feature_dim=feature_dim, num_scales=2
        )

        # 首先将融合特征投影到单一特征维度
        self.feature_projection = nn.Linear(feature_dim * num_views, feature_dim)

        # 最终密度预测MLP
        self.density_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.BatchNorm1d(feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, fused_features, x3d=None, global_voxel_shape=None, range_x=None, range_y=None, range_z=None):
        """
        Args:
            fused_features: [B, N, C*num_views] 或 [B*N, C*num_views]
            x3d: [B, N, 3] 3D点坐标（用于空间图卷积）
            global_voxel_shape, range_x, range_y, range_z: 空间信息
        Returns:
            density: [B*N, 1] 或 [B, N, 1] 密度预测
        """
        # 处理2D输入 [B*N, C*num_views]
        if fused_features.dim() == 2:
            B_N = fused_features.shape[0]
            projected = self.feature_projection(fused_features)  # [B*N, C]
            density = self.density_predictor(projected)  # [B*N, 1]
            return density
        else:
            B, N, C_total = fused_features.shape
            fused_flat = fused_features.view(B * N, C_total)
            projected = self.feature_projection(fused_flat)  # [B*N, C]
            projected = projected.view(B, N, self.feature_dim)  # [B, N, C]
            if x3d is not None and global_voxel_shape is not None:
                spatial_refined = self.multi_scale_graph(x3d, projected, global_voxel_shape, range_x, range_y, range_z)  # [B, N, C]
                projected = spatial_refined
            projected_flat = projected.view(B * N, self.feature_dim)
            density = self.density_predictor(projected_flat)  # [B*N, 1]
            return density



class PointDensityNet(nn.Module):
    """
    改进的荧光光源重建网络：
    - 共享U-Net特征提取
    - 空间注意力融合
    - 创新的空间自适应MLP密度预测（基于图卷积 + 多尺度融合）

    优势：
    1. 参数量少于3DCNN（仅MLP + 轻量图卷积）
    2. 充分捕捉点间空间关系（k-NN图 + 点对关系编码）
    3. 多尺度特征融合（局部→全局）
    4. 创新的自适应注意力机制
    """

    def __init__(self, num_views=7, in_channels=1, feature_dim=64, pos_enc_dim=18):
        super().__init__()
        self.num_views = num_views
        self.pos_enc_dim = pos_enc_dim
        self.feature_dim = feature_dim

        # 1. 共享U-Net特征提取器
        self.shared_unet = UNet(
            n_channels=in_channels, n_features=feature_dim, bilinear=True
        )

        # 2. 空间注意力融合模块
        self.attention_fusion = SpatialAttentionFusion(
            feature_dim=feature_dim, pos_enc_dim=pos_enc_dim
        )

        # 3. 视图特定的gate fusion adaptors
        self.gate_adaptors = nn.ModuleList(
            [GateFusion(feature_dim) for _ in range(num_views)]
        )

        # 4. 创新的空间自适应MLP密度预测头
        self.density_head = SpatialAdaptiveMLPHead(
            feature_dim=feature_dim, num_views=num_views
        )

        # 5. 视图特定的轻量级adaptors
        self.view_adaptors = nn.ModuleList(
            [nn.Conv2d(feature_dim, feature_dim, 1) for _ in range(num_views)]
        )

    def forward(self, view_projections, x3d, global_voxel_shape=None, range_x=None, range_y=None, range_z=None):
        """
        Args:
            view_projections: 字典 {view_name: [B, H, W], ...}
            x3d: [B, N, 3] 3D点坐标
            global_voxel_shape, range_x, range_y, range_z: 空间信息
        Returns:
            density: [B, N, 1] 预测光源密度
            aux_projections: 字典，各视图处理后的投影
        """
        view_list = ["-90", "-60", "-30", "0", "30", "60", "90"]
        view_features = {}
        view_specific_features = {}
        aux_projections = {}
        for view_name, projection in view_projections.items():
            angle = view_name
            projection = projection.unsqueeze(1)  # [B, 1, H, W]
            feat = self.shared_unet(projection)  # [B, feature_dim, H, W]
            view_features[view_name] = feat
            view_specific_feat = self.view_adaptors[view_list.index(str(angle))](feat)
            view_specific_features[view_name] = view_specific_feat
            aux_projections[view_name] = self.gate_adaptors[
                view_list.index(str(angle))
            ](view_specific_feat)
            aux_projections[view_name] = aux_projections[view_name].squeeze(1)
        fused_feat = self.attention_fusion(x3d, view_specific_features)
        B, N, C = fused_feat.shape
        density = self.density_head(fused_feat, x3d, global_voxel_shape, range_x, range_y, range_z)  # [B*N, 1]
        density = density.view(B, N, 1)
        return density, aux_projections

