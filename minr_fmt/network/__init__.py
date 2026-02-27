"""
Network factory function for FMT reconstruction models

Primary networks:
- minr_fmt: Multi-view Implicit Network for 3D Reconstruction
- uhr: UHR-DeepFMT (3D U-Net based voxel network)
- vox_dmrn: VoxDMRN (single-view 2D to 3D)

Legacy networks (backward compatible):
- density, density_spatial_mlp, density_3dcnn: Original point-based networks
"""

from .density import PointDensityNet
from .density_spatial_mlp import PointDensityNet as PointDensityNetSpatialMLP
from .density_refactored import PointDensityNet as PointDensityNet3DCNN

from ..models.uhr_deepfmt import UHRDeepFMT3DUNet
from ..models.vox_dmrn import VoxDMRN
# Note: MINRFMT import removed to avoid circular import
# It will be imported lazily in get_network() when needed


def get_network(
    name, num_views=7, proj_hw=None, output_dim=None, voxel_depth=34, voxel_hw=(100, 40)
):
    """
    Get network by name.

    Primary networks (recommended):
        minr_fmt: Multi-view Implicit Network (uses 0° projection + multi-view fusion)
        uhr: UHR-DeepFMT (3D U-Net, uses 7 views)
        vox_dmrn: VoxDMRN (single-view 2D to 3D)

    Legacy networks (for backward compatibility):
        density: Original PointDensityNet
        density_spatial_mlp: Point density with spatial MLP
        density_3dcnn: Point density with 3D CNN

    Args:
        name: network name (str)
        num_views: number of input views (int, default 7)
        proj_hw: (H, W) tuple for UHR model (optional)
        output_dim: output dimension for VoxDMRN (optional)
        voxel_depth: depth of output volume (optional)
        voxel_hw: (H, W) of output volume (optional)

    Returns:
        model: initialized PyTorch network

    Raises:
        Exception: if network name is not recognized
    """

    # ===== Primary Networks =====
    if name == "minr_fmt":
        """GISC-FMT: Multi-view Implicit Network for 3D Reconstruction"""
        # from ..models.minr_fmt import PointDensityNet as MINRFMT  # Lazy import to avoid circular dependency
        # return MINRFMT(
        #     num_views=num_views,
        #     in_channels=1,
        #     feature_dim=64,
        #     pos_enc_dim=120
        # )
        from .density import (
            PointDensityNet as MINRFMT,
        )  # Lazy import to avoid circular dependency

        return MINRFMT(
            num_views=num_views, in_channels=1, feature_dim=64, pos_enc_dim=60
        )

    elif name == "uhr":
        """UHR-DeepFMT: 3D U-Net based voxel network"""
        return UHRDeepFMT3DUNet(in_channels=num_views, target_proj_hw=proj_hw)

    elif name == "vox_dmrn":
        """VoxDMRN: Single-view 2D to 3D reconstruction network"""
        if output_dim is None:
            output_dim = voxel_depth * voxel_hw[0] * voxel_hw[1]
        return VoxDMRN(
            in_channels=1,
            base_channels=32,
            num_stages=4,
            blocks_per_stage=2,
            mlp_hidden_dims=[256, 128],
            output_dim=output_dim,
            use_bn=True,
            voxel_depth=voxel_depth,
            voxel_hw=voxel_hw,
        )

    # ===== Legacy Networks (for backward compatibility) =====
    elif name == "density":
        """Legacy: Original PointDensityNet"""
        return PointDensityNet(num_views=num_views)

    elif name == "density_spatial_mlp":
        """Legacy: PointDensityNet with spatial MLP"""
        return PointDensityNetSpatialMLP(num_views=num_views)

    elif name == "density_3dcnn":
        """Legacy: PointDensityNet with 3D CNN"""
        return PointDensityNet3DCNN(num_views=num_views)

    else:
        available = [
            "minr_fmt",
            "uhr",
            "vox_dmrn",
            "density",
            "density_spatial_mlp",
            "density_3dcnn",
        ]
        raise Exception(
            f"Unsupported network: '{name}'\nAvailable networks: {', '.join(available)}"
        )


# For backward compatibility with old import patterns
__all__ = ["get_network", "PointDensityNet"]
