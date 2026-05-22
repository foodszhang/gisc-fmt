"""
模型工厂 - 从配置初始化模型

提供便捷方法从配置文件创建各种模型，
同时传递几何和网络参数给模型内部的组件。

所有模型特定参数都从配置中提取，不再有硬编码默认值。
"""

from typing import Any, Optional, Dict
from .config_extractor import ConfigExtractor


class ModelFactory:
    """模型工厂 - 支持配置初始化"""
    
    @staticmethod
    def create_minr_fmt_model(config: Optional[Any] = None, **kwargs):
        """Legacy alias kept for older configs."""
        return ModelFactory.create_gisc_fmt_model(config=config, **kwargs)
    
    @staticmethod
    def create_uhr_deepfmt_model(config: Optional[Any] = None, **kwargs):
        """
        创建 UHR-DeepFMT 模型
        
        参数:
            config: 配置对象/字典 (必需)
            **kwargs: 忽略（为了兼容性）
        
        返回:
            UHRDeepFMT 模型实例
        """
        from .models.uhr_deepfmt import UHRDeepFMT3DUNet
        
        if config is None:
            raise ValueError("UHR-DeepFMT 模型需要配置对象")
        
        # 创建模型 - 所有参数都从config中提取
        model = UHRDeepFMT3DUNet(config=config)
        
        return model
    
    @staticmethod
    def create_vox_dmrn_model(config: Optional[Any] = None, **kwargs):
        """
        创建 VoxDMRN 模型
        
        参数:
            config: 配置对象/字典 (必需)
            **kwargs: 忽略（为了兼容性）
        
        返回:
            VoxDMRN 模型实例
        """
        from .models.vox_dmrn import VoxDMRN
        
        if config is None:
            raise ValueError("VoxDMRN 模型需要配置对象")
        
        # 创建模型 - 所有参数都从config中提取
        model = VoxDMRN(config=config)
        
        return model

    @staticmethod
    def create_voxel_baseline_model(model_type: str, config: Optional[Any] = None, **kwargs):
        """Create an adapted voxel-domain baseline."""
        if config is None:
            raise ValueError("Voxel baseline 模型需要配置对象")
        from .models.voxel_baselines import (
            D2RecSTAdapted,
            DSPGNAdapted,
            FEM2VoxUNet,
            FMTReconNetAdapted,
            GenericVoxelBaseline,
            MAPPGANAdapted,
            PGDPNNAdapted,
            TwoStageDeepFMTAdapted,
        )

        registry = {
            "map_pgan": MAPPGANAdapted,
            "d2_recst": D2RecSTAdapted,
            "two_stage_deepfmt": TwoStageDeepFMTAdapted,
            "fmt_reconnet": FMTReconNetAdapted,
            "pgdpnn": PGDPNNAdapted,
            "dspgn": DSPGNAdapted,
            "fem2vox_unet": FEM2VoxUNet,
        }
        cls = registry.get(model_type, GenericVoxelBaseline)
        return cls(config)
    
    @staticmethod
    def create_gisc_fmt_model(config: Optional[Any] = None, **kwargs):
        """Create GISC-FMT model."""
        from .models.minr_fmt import GISCFMT

        if config is None:
            raise ValueError("GISC-FMT 模型需要配置对象")

        return GISCFMT(config=config)

    @staticmethod
    def create_model(model_type: str, config: Optional[Any] = None, **kwargs):
        """
        通用模型创建接口
        
        参数:
            model_type: 模型类型 ("gisc_fmt", "uhr_deepfmt", "vox_dmrn")
            config: 配置对象/字典 (必需)
            **kwargs: 忽略（为了兼容性）
        
        返回:
            对应的模型实例
            
        Raises:
            ValueError: 如果配置为None或模型类型未知
        """
        if config is None:
            raise ValueError("模型创建需要配置对象")
        
        model_type = model_type.lower()
        
        if model_type in {
            "gisc_fmt",
            "minr_fmt",
            "point_cqr",
            "fixed_footprint_cqr",
            "depth_footprint_cqr",
            "unconstrained_adaptive_cqr",
        }:
            # Prefer the new name; keep legacy alias for older configs.
            return ModelFactory.create_gisc_fmt_model(config, **kwargs)
        elif model_type == "uhr_deepfmt":
            return ModelFactory.create_uhr_deepfmt_model(config, **kwargs)
        elif model_type == "vox_dmrn":
            return ModelFactory.create_vox_dmrn_model(config, **kwargs)
        elif model_type in {
            "map_pgan",
            "d2_recst",
            "two_stage_deepfmt",
            "fmt_reconnet",
            "pgdpnn",
            "dspgn",
            "fem2vox_unet",
        }:
            return ModelFactory.create_voxel_baseline_model(model_type, config, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
