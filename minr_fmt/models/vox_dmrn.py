"""
VoxDMRN: 单视图2D到3D体素重建网络

核心架构：DCNN特征提取 + MLP头
所有参数从config对象提取，无硬编码默认值
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config_extractor import ConfigExtractor


# ===== 核心模块 =====


class ResidualBlock(nn.Module):
    """残差块：两层卷积 + 残差连接"""

    def __init__(self, in_channels, out_channels, stride=1, use_bn=True):
        super().__init__()
        self.use_bn = use_bn
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1,
            bias=not use_bn
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1,
            bias=not use_bn
        )
        
        if use_bn:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.skip_conv = None
        if stride != 1 or in_channels != out_channels:
            self.skip_conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, stride=stride,
                bias=not use_bn
            )
            if use_bn:
                self.skip_bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        
        if self.skip_conv is not None:
            residual = self.skip_conv(residual)
            if self.use_bn:
                residual = self.skip_bn(residual)
        
        out = out + residual
        return F.relu(out, inplace=True)


class ConvStage(nn.Module):
    """卷积阶段：残差块组 + 最大池化"""

    def __init__(self, in_channels, out_channels, num_blocks=2, use_bn=True):
        super().__init__()
        
        self.blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride=1, use_bn=use_bn)
        )
        
        for i in range(1, num_blocks):
            self.blocks.add_module(
                f'block_{i}',
                ResidualBlock(out_channels, out_channels, stride=1, use_bn=use_bn)
            )
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.blocks(x)
        return self.pool(x)


class VoxDMRNDCNN(nn.Module):
    """DCNN特征提取主干（阶段S1）"""

    def __init__(self, in_channels, base_channels, num_stages, blocks_per_stage, use_bn):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.use_bn = use_bn
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, base_channels,
                kernel_size=7, stride=1, padding=3,
                bias=not use_bn
            ),
            nn.BatchNorm2d(base_channels) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        
        self.stages = nn.ModuleList()
        for i in range(num_stages):
            in_ch = base_channels * (2 ** i)
            out_ch = base_channels * (2 ** (i + 1))
            self.stages.append(
                ConvStage(in_ch, out_ch, num_blocks=blocks_per_stage, use_bn=use_bn)
            )
    
    def forward(self, x):
        x = self.initial_conv(x)
        intermediate_features = []
        
        for stage in self.stages:
            intermediate_features.append(x)
            x = stage(x)
        
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        feat = x.flatten(1)
        
        return feat, intermediate_features


class VoxDMRNMLP(nn.Module):
    """MLP头用于体积预测（阶段S2）"""

    def __init__(self, input_dim, hidden_dims, output_dim, use_bn=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims if isinstance(hidden_dims, (list, tuple)) else [hidden_dims]
        self.output_dim = output_dim
        self.use_bn = use_bn
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, feat):
        return self.mlp(feat)


# ===== 主网络 =====


class VoxDMRN(nn.Module):
    """单视图2D到3D体素重建网络
    
    架构流程：
    1. 单视图投影输入
    2. DCNN特征提取
    3. MLP密度预测
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
        vox_params = ConfigExtractor.extract_vox_dmrn_config(config)
        
        # 网络参数
        in_channels = net_params["in_channels"]
        base_channels = vox_params["base_channels"]
        num_stages = 4  # 标准配置
        blocks_per_stage = 2  # 标准配置
        output_dim = vox_params["output_dim"]
        
        # 几何参数
        self.camera_distance = geo_params["camera_distance"]
        self.detector_size = geo_params["detector_size"]
        self.global_voxel_shape = geo_params["global_voxel_shape"]
        self.config = config
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.output_dim = output_dim

        # DCNN主干
        self.dcnn = VoxDMRNDCNN(
            in_channels=in_channels,
            base_channels=base_channels,
            num_stages=num_stages,
            blocks_per_stage=blocks_per_stage,
            use_bn=True
        )
        
        dcnn_output_dim = base_channels * (2 ** num_stages)

        # MLP头
        self.mlp = VoxDMRNMLP(
            input_dim=dcnn_output_dim,
            hidden_dims=[256, 128],
            output_dim=output_dim,
            use_bn=False
        )
    
    def forward(self, projections_dict, points=None):
        """
        Args:
            projections_dict: dict {view_name: [B, H, W]}
            points: [B, N, 3] (可选)

        Returns:
            output: [B, N, 1] or [B, D*H*W, 1]
            aux_output: dict
        """
        # 使用0°投影
        img_0deg = projections_dict["0"].unsqueeze(1)

        # 特征提取
        feat, _ = self.dcnn(img_0deg)

        # Dense prediction over ROI grid (flattened)
        logits = self.mlp(feat)  # [B, ROI_X*ROI_Y*ROI_Z]

        if points is not None:
            # Points are normalized in global voxel coordinates (x,y,z in [0,1]).
            # Convert to global indices, then map into ROI index space using voxel_ranges.
            B, N, _ = points.shape
            shape = torch.tensor(self.global_voxel_shape, device=points.device, dtype=points.dtype)
            idx = torch.round(points * (shape - 1)).long()  # [B, N, 3] as (x,y,z)

            vr = self.config.data.voxel_ranges
            x0, y0, z0 = int(vr.x[0]), int(vr.y[0]), int(vr.z[0])
            rx = int(vr.x[1] - vr.x[0])
            ry = int(vr.y[1] - vr.y[0])
            rz = int(vr.z[1] - vr.z[0])

            xi = (idx[..., 0] - x0).clamp(0, rx - 1)
            yi = (idx[..., 1] - y0).clamp(0, ry - 1)
            zi = (idx[..., 2] - z0).clamp(0, rz - 1)

            lin = xi * (ry * rz) + yi * rz + zi  # [B, N]
            flat = logits.view(B, -1)
            output = flat.gather(1, lin).unsqueeze(-1)  # [B, N, 1]
        else:
            B = logits.shape[0]
            output = logits.view(B, -1, 1)

        aux_output = {k: v for k, v in projections_dict.items()}
        return output, aux_output
