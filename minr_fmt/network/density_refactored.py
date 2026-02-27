import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import ProjectionConfig
from .encoder import SpatialAttentionFusion, UNet, GateFusion


class Conv3DBlock(nn.Module):
    """3D卷积基础块: Conv3d + BatchNorm + ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Density3DCNNHead(nn.Module):
    """基于3D CNN的密度预测头 - 考虑点之间的空间关系"""

    def __init__(self, feature_dim=64, num_views=7):
        super().__init__()
        # 将多视图特征重新组织为3D体素表示
        # 输入: [B*N, C*num_views] -> 输出: [B*N, 1]
        self.feature_dim = feature_dim
        self.num_views = num_views

        # 3D卷积分支: 捕捉空间依赖关系
        self.conv3d_branch = nn.Sequential(
            Conv3DBlock(feature_dim, feature_dim * 2, kernel_size=3, padding=1),
            nn.MaxPool3d(kernel_size=(2,1,1), stride=(2,1,1)),
            Conv3DBlock(feature_dim * 2, feature_dim * 4, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool3d(1),  # 全局池化获得固定大小的特征
        )

        # 最终密度预测
        self.density_predictor = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim * 2),
            nn.BatchNorm1d(feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, fused_features):
        """
        fused_features: [B*N, C*num_views] 融合后的多视图特征
        输出: [B*N, 1] 密度预测
        """
        B_N = fused_features.shape[0]
        C = self.feature_dim

        # 重新组织为3D张量用于3D卷积
        # [B*N, C*num_views] -> [B*N, C, num_views, 1, 1]
        feat_3d = fused_features.view(B_N, C, self.num_views, 1, 1)

        # 应用3D卷积分支捕捉视图间的空间关系
        feat_3d_proc = self.conv3d_branch(feat_3d)  # [B*N, C*4, 1, 1, 1]
        feat_3d_proc = feat_3d_proc.view(B_N, -1)  # 展平为 [B*N, C*4]

        # 最终密度预测
        density = self.density_predictor(feat_3d_proc)  # [B*N, 1]
        return density


class ViewAdaptor(nn.Module):
    """
    视图特定的轻量级adaptor - 从共享特征生成视图特定的输出
    现在采用Conv2d以适配UNet输出的[B, feature_dim, H, W]张量，避免Linear层shape不匹配问题。
    """

    def __init__(self, feature_dim=64):
        super().__init__()
        self.adaptor = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
        )

    def forward(self, x):
        # x: [B, feature_dim, H, W]
        return self.adaptor(x)


class PointDensityNet(nn.Module):
    """整体荧光光源重建网络：共享U-Net特征提取 + 空间注意力融合 + 3D CNN密度预测 + 视图特定adaptor

    架构改进:
    1. 共享U-Net: 用单个U-Net替代7个独立U-Net，大幅减少参数量
    2. 3D CNN密度头: 替代MLP以考虑点之间的空间关系
    3. 轻量级adaptors: 为每个视图提供特定的输出处理而不增加太多复杂性
    """

    def __init__(self, num_views=7, in_channels=1, feature_dim=64, pos_enc_dim=18):
        super().__init__()
        self.num_views = num_views
        self.pos_enc_dim = pos_enc_dim
        self.feature_dim = feature_dim

        # 1. 共享U-Net特征提取器（所有视图共用一个U-Net以减少网络复杂性）
        self.shared_unet = UNet(
            n_channels=in_channels, n_features=feature_dim, bilinear=True
        )

        # 2. 空间注意力融合模块
        self.attention_fusion = SpatialAttentionFusion(
            feature_dim=feature_dim, pos_enc_dim=pos_enc_dim
        )

        # 3. 视图特定的gate fusion adaptors（轻量级）
        self.gate_adaptors = nn.ModuleList(
            [GateFusion(feature_dim) for _ in range(num_views)]
        )

        # 4. 基于3D CNN的密度预测头（考虑点之间的空间关系）
        self.density_head = Density3DCNNHead(
            feature_dim=feature_dim, num_views=num_views
        )

        # 5. 视图特定的轻量级adaptors（用于进一步优化视图特定输出）
        self.view_adaptors = nn.ModuleList(
            [ViewAdaptor(feature_dim) for _ in range(num_views)]
        )

    def forward(self, view_projections, x3d):
        """
        view_projections: 字典，每个元素为[B, in_channels, H_i, W_i]（多视图投影图）
        x3d: [B, N, 3] 3D点坐标
        输出:
            density: [B, N, 1] 预测光源密度
            view_projections_processed: 字典，各视图处理后的投影（用于损失计算）
        """
        view_list = ["-90", "-60", "-30", "0", "30", "60", "90"]
        view_features = {}
        view_specific_features = {}
        views_no_projections = {}

        # 1. 使用共享U-Net提取所有视图的特征（减少参数量）
        for view_name, projection in view_projections.items():
            angle = view_name
            # 添加通道维度用于U-Net输入
            projection = projection.unsqueeze(1)  # [B, 1, H, W]
            # 共享U-Net处理
            feat = self.shared_unet(projection)  # [B, feature_dim, H, W]
            view_features[view_name] = feat
            # 2. 先用视图特定的adaptor处理共享特征，得到视图独立特征
            view_specific_feat = self.view_adaptors[view_list.index(str(angle))](feat)
            view_specific_features[view_name] = view_specific_feat
            # 3. gate adaptor处理视图独立特征，得到投影重建
            views_no_projections[view_name] = self.gate_adaptors[
                view_list.index(str(angle))
            ](view_specific_feat)
            views_no_projections[view_name] = views_no_projections[view_name].squeeze(1)
        # 4. 空间注意力融合 - 融合所有视图特征 [B, N, C*num_views]
        fused_feat = self.attention_fusion(x3d, view_specific_features)
        # 5. 基于3D CNN预测光源密度（考虑点之间的空间关系）
        B, N, C = fused_feat.shape
        # reshape为[B*N, C*num_views]用于3D CNN处理
        total_feat = fused_feat.reshape(B * N, C)
        density = self.density_head(total_feat).view(B, N, 1)
        # 说明：先用view_adaptors处理UNet特征，得到各视图独立特征，再用gate_adaptors处理，最后整体融合做3dcnn回归。
        return density, views_no_projections
