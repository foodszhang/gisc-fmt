import torch
from omegaconf import OmegaConf

from minr_fmt.models.minr_fmt import GISCFMT


def _make_cfg(enable_background: bool, guidance_enable: bool, guidance_mode: str):
    # Provide ALL required keys (no code-side defaults).
    return OmegaConf.create(
        {
            "model": {
                "name": "gisc_fmt",
                "num_views": 7,
                "in_channels": 1,
                "feature_dim": 64,
                "pos_enc_dim": 60,
                "geometry": {
                    "camera_distance": 200.0,
                    "detector_size": [256, 256],
                    "global_voxel_shape": [182, 164, 210],
                },
                "gisc_fmt": {
                    "use_adapter": True,
                    "adapter_strategy": "bottleneck_decoder",
                    "adapter_mode": "per_view",
                    "adapter_cond_dim": 4,
                    "adapter_r_ratio": 8,
                    "norm_type": "group",
                    "implicit_field_hidden_dim": 64,
                    "implicit_field_d_x": 60,
                    "implicit_field_d_f": 64,
                    "view_weight_embed_dim": 16,
                    "view_weight_hidden_dim": 32,
                    "multiscale": {
                        "view_embed_dim": 16,
                        "depth_dim": 16,
                        "cross_view_hidden_dim": 64,
                        "scale_hidden_dim": 64,
                        "tau": 1.0,
                        "depth_max": 200.0,
                    },
                    "background": {
                        "enable_background": enable_background,
                        "head": {"hidden_dim": 64, "d_x": 60, "d_f": 64},
                        "guidance": {
                            "enable": guidance_enable,
                            "dim": 16,
                            "mode": guidance_mode,
                            "hidden_dim": 64,
                            "scale": 1.0,
                            "gate_init_bias": 4.0,
                        },
                    },
                },
            },
            "data": {"view_angles": [-90, -60, -30, 0, 30, 60, 90]},
        }
    )


def test_no_background_backward_compatible():
    cfg = _make_cfg(enable_background=False, guidance_enable=False, guidance_mode="film")
    net = GISCFMT(config=cfg)

    B, V, H, W = 2, 7, 32, 32
    N = 64
    proj = torch.randn(B * V, 1, H, W)
    x3d = torch.rand(B, N, 3)

    out = net(proj, x3d)
    assert isinstance(out, tuple)
    assert len(out) == 2
    logits, aux = out
    assert logits.shape == (B, N, 1)
    assert isinstance(aux, dict)
    assert torch.isfinite(logits).all()


def test_background_branch_with_film_guidance_backward_ok():
    cfg = _make_cfg(enable_background=True, guidance_enable=True, guidance_mode="film")
    net = GISCFMT(config=cfg)

    B, V, H, W = 2, 7, 32, 32
    N = 64
    G = 16
    proj = torch.randn(B * V, 1, H, W)
    x3d = torch.rand(B, N, 3)
    bg_guidance = torch.randn(B, G)

    logits, aux, extra = net(proj, x3d, bg_guidance=bg_guidance)
    assert logits.shape == (B, N, 1)
    assert "background_logits" in extra
    assert extra["background_logits"].shape == (B, N, 1)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(extra["background_logits"]).all()

    loss = logits.mean() + extra["background_logits"].mean()
    loss.backward()
