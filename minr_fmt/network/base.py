import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from typing import List, Dict, Tuple, Optional, Union
import numpy as np


class ProjectionConfig:
    """投影面配置 - 支持灵活尺寸"""

    def __init__(self, projection_sizes: dict[str, tuple[int, int]]):
        self.projection_sizes = projection_sizes
        self.projection_channels = 1  # 单通道投影图
        self.views = list(projection_sizes.keys())


# 测试代码
def test_both_regression_types():
    """测试两种回归方式"""

    # 定义投影面尺寸（示例）
    projection_sizes = {
        "xy": (180, 300),
        "xz": (300, 180),
        "yz": (208, 300),
        "zy": (300, 208),
    }

    print("Testing both regression types:")

    # 测试直接体素回归
    print("\n1. Direct Voxel Regression:")
    direct_model = create_density_net(
        projection_sizes=projection_sizes,
        regression_type="direct",
        target_voxel_size=(64, 64, 64),  # 可以设置为任意尺寸
        encoder_type="unet",
    )

    # 创建输入
    projections = {
        name: torch.randn(2, 1, h, w) for name, (h, w) in projection_sizes.items()
    }

    with torch.no_grad():
        density_field = direct_model(projections)

    print(f"Input projections: {[(k, v.shape) for k, v in projections.items()]}")
    print(f"Output density field: {density_field.shape}")

    # 测试点采样回归
    print("\n2. Point Sampling Regression:")
    point_model = create_density_net(
        projection_sizes=projection_sizes,
        regression_type="point_sampling",
        encoder_type="unet",
    )

    # 创建采样点 (在[0,1]范围内的3D坐标)
    points = torch.rand(2, 1000, 3)  # [B, N, 3]

    with torch.no_grad():
        densities = point_model(projections, points)

    print(f"Input points: {points.shape}")
    print(f"Output densities: {densities.shape}")


if __name__ == "__main__":
    test_both_regression_types()
