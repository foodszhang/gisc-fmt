"""Model factory for SSQ-FMT and comparison baselines."""

from typing import Any, Optional


class ModelFactory:
    """Create reconstruction models from Hydra/OmegaConf configuration."""

    @staticmethod
    def create_minr_fmt_model(config: Optional[Any] = None, **kwargs):
        """Legacy alias kept for older configs."""
        return ModelFactory.create_gisc_fmt_model(config=config, **kwargs)

    @staticmethod
    def create_uhr_deepfmt_model(config: Optional[Any] = None, **kwargs):
        from .models.uhr_deepfmt import UHRDeepFMT3DUNet

        if config is None:
            raise ValueError("UHR-DeepFMT 模型需要配置对象")
        return UHRDeepFMT3DUNet(config=config)

    @staticmethod
    def create_vox_dmrn_model(config: Optional[Any] = None, **kwargs):
        from .models.vox_dmrn import VoxDMRN

        if config is None:
            raise ValueError("VoxDMRN 模型需要配置对象")
        return VoxDMRN(config=config)

    @staticmethod
    def create_voxel_baseline_model(model_type: str, config: Optional[Any] = None, **kwargs):
        if config is None:
            raise ValueError("Voxel baseline 模型需要配置对象")
        from .models.native_grid_baselines import (
            NativeGridCNN3DBaseline,
            NativeGridTransUNet3DBaseline,
        )
        from .models.voxel_baselines import (
            D2RecSTAdapted,
            DSPGNAdapted,
            FEM2VoxUNet,
            FMTReconNetAdapted,
            GenericVoxelBaseline,
            MAPPGANAdapted,
            PGDPNNAdapted,
            Stage1InterpolationBaseline,
            TwoStageDeepFMTAdapted,
        )

        registry = {
            "map_pgan": MAPPGANAdapted,
            "d2_recst": D2RecSTAdapted,
            "two_stage_deepfmt": TwoStageDeepFMTAdapted,
            "fmt_reconnet": FMTReconNetAdapted,
            "pgdpnn": PGDPNNAdapted,
            "pgd_pnn": PGDPNNAdapted,
            "pgd-pnn": PGDPNNAdapted,
            "dspgn": DSPGNAdapted,
            "fem2vox_unet": FEM2VoxUNet,
            "stage1_unet": FEM2VoxUNet,
            "stage1_interpolation": Stage1InterpolationBaseline,
            # Controlled baselines now reconstruct on a memory-safe native grid
            # and use a fixed interpolation to the common reference grid.
            "cnn3d_baseline": NativeGridCNN3DBaseline,
            "transunet3d_baseline": NativeGridTransUNet3DBaseline,
        }
        cls = registry.get(model_type, GenericVoxelBaseline)
        return cls(config)

    @staticmethod
    def create_fem_baseline_model(model_type: str, config: Optional[Any] = None, **kwargs):
        if config is None:
            raise ValueError("FEM baseline 模型需要配置对象")
        from .models.fem_baselines import (
            ElasticNetFEM,
            FEMCoarseBaseline,
            FEMToVoxelBaseline,
            FISTAFEM,
            GAICNLikeFEM,
            L1FEM,
            StOMPFEM,
            TikhonovFEM,
        )

        registry = {
            "fem_coarse": FEMCoarseBaseline,
            "fem_to_voxel": FEMToVoxelBaseline,
            "stage1_fem": FEMCoarseBaseline,
            "stage1_to_voxel": FEMToVoxelBaseline,
            "tikhonov_fem": TikhonovFEM,
            "l1_fem": L1FEM,
            "elasticnet_fem": ElasticNetFEM,
            "fista_fem": FISTAFEM,
            "stomp_fem": StOMPFEM,
            "gaicn": GAICNLikeFEM,
        }
        return registry[model_type](config)

    @staticmethod
    def create_gisc_fmt_model(config: Optional[Any] = None, **kwargs):
        """Create the legacy GISC-FMT model for backward-compatible ablations only."""
        from .models.gisc_multisource import GISCFMT

        if config is None:
            raise ValueError("GISC-FMT 模型需要配置对象")
        return GISCFMT(config=config)

    @staticmethod
    def create_ssq_fmt_model(config: Optional[Any] = None, **kwargs):
        """Create the current SSQ-FMT point model."""
        from .models.ssq_fmt import SSQFMT

        if config is None:
            raise ValueError("SSQ-FMT 模型需要配置对象")
        return SSQFMT(config=config)

    @staticmethod
    def create_model(model_type: str, config: Optional[Any] = None, **kwargs):
        if config is None:
            raise ValueError("模型创建需要配置对象")

        model_type = model_type.lower()
        if model_type == "ssq_fmt":
            return ModelFactory.create_ssq_fmt_model(config, **kwargs)
        if model_type in {
            "gisc_fmt",
            "minr_fmt",
            "point_cqr",
            "fixed_footprint_cqr",
            "depth_footprint_cqr",
            "unconstrained_adaptive_cqr",
        }:
            return ModelFactory.create_gisc_fmt_model(config, **kwargs)
        if model_type == "uhr_deepfmt":
            return ModelFactory.create_uhr_deepfmt_model(config, **kwargs)
        if model_type == "vox_dmrn":
            return ModelFactory.create_vox_dmrn_model(config, **kwargs)
        if model_type == "pah2t_former":
            from .models.pah2t_former import PAH2TFormer

            return PAH2TFormer(config)
        if model_type in {
            "map_pgan",
            "d2_recst",
            "two_stage_deepfmt",
            "fmt_reconnet",
            "pgdpnn",
            "pgd_pnn",
            "pgd-pnn",
            "dspgn",
            "fem2vox_unet",
            "stage1_unet",
            "stage1_interpolation",
            "cnn3d_baseline",
            "transunet3d_baseline",
        }:
            return ModelFactory.create_voxel_baseline_model(model_type, config, **kwargs)
        if model_type in {
            "fem_coarse",
            "fem_to_voxel",
            "stage1_fem",
            "stage1_to_voxel",
            "tikhonov_fem",
            "l1_fem",
            "elasticnet_fem",
            "fista_fem",
            "stomp_fem",
            "gaicn",
        }:
            return ModelFactory.create_fem_baseline_model(model_type, config, **kwargs)
        raise ValueError(f"Unknown model type: {model_type}")
