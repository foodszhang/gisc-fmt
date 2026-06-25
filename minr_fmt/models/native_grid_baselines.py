"""Memory-safe voxel baselines for TMI comparison experiments.

Controlled models reconstruct on a configurable native voxel grid and use a
fixed, parameter-free trilinear mapping to the common reference grid. Adapted
literature proxies keep their existing heads but move expensive volume operations
to their declared internal grids.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .uhr_deepfmt import UHRDeepFMT3DUNet
from .voxel_baselines import (
    ConvBlock3d,
    D2RecSTAdapted,
    DSPGNAdapted,
    MAPPGANAdapted,
    SurfaceVolumeBuilder,
    TransformerBottleneck3D,
    VNet3D,
)


def _tuple3(value, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if value is None:
        return default
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(int(v) for v in value)
        if len(values) != 3:
            raise ValueError(f"Expected a 3-D shape, got {values}")
        return values
    scalar = int(value)
    return (scalar, scalar, scalar)


def _reference_shape(config) -> tuple[int, int, int]:
    geometry = getattr(config.model, "geometry", None)
    shape = getattr(geometry, "global_voxel_shape", None) if geometry is not None else None
    if shape is not None:
        return _tuple3(shape, (190, 200, 104))
    vr = config.data.voxel_ranges
    return (
        int(vr.x[1] - vr.x[0]),
        int(vr.y[1] - vr.y[0]),
        int(vr.z[1] - vr.z[0]),
    )


class _NativeGridMixin:
    """Shared native-grid and fixed-reference-output handling."""

    def _init_native_grid(self, config, params) -> None:
        self.config = config
        self.reference_shape = _reference_shape(config)
        self.native_shape = _tuple3(
            getattr(params, "native_output_shape", None),
            (64, 64, 32),
        )
        if any(v <= 0 for v in self.native_shape):
            raise ValueError(f"native_output_shape must be positive, got {self.native_shape}")
        if any(v <= 0 for v in self.reference_shape):
            raise ValueError(f"reference output shape must be positive, got {self.reference_shape}")
        self.return_reference_grid = bool(getattr(params, "return_reference_grid", True))

    def _to_reference_grid(self, native_logits: torch.Tensor) -> torch.Tensor:
        if not self.return_reference_grid or native_logits.shape[2:] == self.reference_shape:
            return native_logits
        return F.interpolate(
            native_logits,
            size=self.reference_shape,
            mode="trilinear",
            align_corners=False,
        )

    def _aux_outputs(self, native_logits: torch.Tensor) -> dict:
        return {
            "method_fidelity": "controlled",
            "output_space": "common_reference_grid" if self.return_reference_grid else "native_grid",
            "native_output_shape": self.native_shape,
            "reference_output_shape": self.reference_shape,
            "fixed_output_resampling": bool(self.return_reference_grid),
            "train_on_native_grid": True,
            "native_pred_voxel": native_logits,
        }


class NativeGridCNN3DBaseline(_NativeGridMixin, nn.Module):
    """Controlled 3-D CNN reconstructed on a memory-safe native grid."""

    output_type = "voxel"

    def __init__(self, config):
        super().__init__()
        params = getattr(config.model, "cnn3d_baseline", {})
        self._init_native_grid(config, params)
        base = int(getattr(params, "base_channels", 8))
        self.builder = SurfaceVolumeBuilder(self.native_shape, config.data.view_angles)
        self.net = VNet3D(1, base, 1)

    def forward(self, projections, *args, **kwargs):
        native_logits = self.net(self.builder(projections))
        pred = self._to_reference_grid(native_logits)
        aux = self._aux_outputs(native_logits)
        aux["native_prediction_shape"] = tuple(int(v) for v in native_logits.shape[2:])
        return {"pred_voxel": pred, "aux_outputs": aux}


class NativeGridTransUNet3DBaseline(_NativeGridMixin, nn.Module):
    """Controlled 3-D CNN/Transformer baseline on a native grid."""

    output_type = "voxel"

    def __init__(self, config):
        super().__init__()
        params = getattr(config.model, "transunet3d_baseline", {})
        self._init_native_grid(config, params)
        base = int(getattr(params, "base_channels", 8))
        heads = int(getattr(params, "num_heads", 4))
        max_tokens = int(getattr(params, "max_tokens", 512))

        self.builder = SurfaceVolumeBuilder(self.native_shape, config.data.view_angles)
        self.enc1 = ConvBlock3d(1, base)
        self.down1 = nn.MaxPool3d(2, ceil_mode=True)
        self.enc2 = ConvBlock3d(base, base * 2)
        self.down2 = nn.MaxPool3d(2, ceil_mode=True)
        self.enc3 = ConvBlock3d(base * 2, base * 4)
        self.trans = TransformerBottleneck3D(base * 4, heads, max_tokens)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock3d(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock3d(base * 2, base)
        self.out = nn.Conv3d(base, 1, 1)

    @staticmethod
    def _match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[2:] == ref.shape[2:]:
            return x
        return F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)

    def forward(self, projections, *args, **kwargs):
        x = self.builder(projections)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.trans(self.enc3(self.down2(e2)))
        d2 = self._match(self.up2(e3), e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self._match(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        native_logits = self.out(d1)
        if native_logits.shape[2:] != self.native_shape:
            native_logits = F.interpolate(
                native_logits,
                size=self.native_shape,
                mode="trilinear",
                align_corners=False,
            )
        pred = self._to_reference_grid(native_logits)
        aux = self._aux_outputs(native_logits)
        aux["native_prediction_shape"] = tuple(int(v) for v in native_logits.shape[2:])
        return {"pred_voxel": pred, "aux_outputs": aux}


class NativeGridUHRDeepFMTProxy(UHRDeepFMT3DUNet):
    """Memory-safe UHR-inspired proxy with native-grid training supervision.

    This wrapper does not claim paper-faithful dual sampling. It only prevents the
    existing 3-D SE-UNet proxy from upsampling its one-channel output during
    training and exposes the native logits to ``VoxelReconstructionLoss``.
    """

    output_type = "voxel"

    def forward(self, projections_dict, points=None, target_proj_hw=None, **kwargs):
        should_map_to_reference = bool(self.upsample_to_full)
        self.upsample_to_full = False
        try:
            out = super().forward(
                projections_dict,
                points=points,
                target_proj_hw=target_proj_hw,
                **kwargs,
            )
        finally:
            self.upsample_to_full = should_map_to_reference

        if not isinstance(out, dict) or "pred_voxel" not in out:
            return out

        native_logits = out["pred_voxel"]
        pred_voxel = native_logits
        if should_map_to_reference and not self.training:
            pred_voxel = F.interpolate(
                native_logits,
                size=self.full_output_shape,
                mode="trilinear",
                align_corners=False,
            )

        aux = dict(out.get("aux_outputs", {}))
        aux.update(
            {
                "method_fidelity": "architecture_proxy",
                "output_space": (
                    "common_reference_grid"
                    if should_map_to_reference and not self.training
                    else "native_grid"
                ),
                "native_output_shape": tuple(int(v) for v in native_logits.shape[2:]),
                "reference_output_shape": tuple(int(v) for v in self.full_output_shape),
                "fixed_output_resampling": bool(should_map_to_reference and not self.training),
                "train_on_native_grid": True,
                "native_pred_voxel": native_logits,
            }
        )
        return {"pred_voxel": pred_voxel, "aux_outputs": aux}


class _NativeSurfaceProxyMixin:
    """Move projection-to-volume lifting to the proxy's internal grid."""

    def _use_internal_surface_grid(self, config) -> None:
        internal_shape = tuple(int(v) for v in self.internal_shape)
        self.surface_builder = SurfaceVolumeBuilder(internal_shape, config.data.view_angles)


class NativeGridMAPPGANAdapted(_NativeSurfaceProxyMixin, MAPPGANAdapted):
    """Memory-safe MAP-PGAN-inspired proxy; not the paper's WGAN-GP training."""

    def __init__(self, config):
        super().__init__(config)
        self._use_internal_surface_grid(config)


class NativeGridD2RecSTAdapted(_NativeSurfaceProxyMixin, D2RecSTAdapted):
    """Memory-safe D2-RecST-inspired proxy; perceptual/adversarial domains remain absent."""

    def __init__(self, config):
        super().__init__(config)
        self._use_internal_surface_grid(config)


class NativeGridDSPGNAdapted(_NativeSurfaceProxyMixin, DSPGNAdapted):
    """Memory-safe DSPGN-inspired proxy; not a system-matrix FEM graph implementation."""

    def __init__(self, config):
        super().__init__(config)
        self._use_internal_surface_grid(config)
