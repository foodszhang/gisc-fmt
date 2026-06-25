import torch

from minr_fmt.network.ssq_encoder import SharedMultiscaleSurfaceEncoder


def test_multiscale_encoder_shapes_and_gradients():
    encoder = SharedMultiscaleSurfaceEncoder(
        in_channels=1,
        stem_channels=8,
        stage_channels=(8, 16, 24, 32),
        stage_blocks=(1, 1, 1, 1),
        pyramid_channels=12,
    )
    x = torch.randn(2, 3, 1, 32, 32, requires_grad=True)
    out = encoder(x)
    assert out["full"].shape == (2, 3, 12, 32, 32)
    assert out["half"].shape == (2, 3, 12, 16, 16)
    assert out["quarter"].shape == (2, 3, 12, 8, 8)
    loss = out["full"].mean() + out["half"].mean() + out["quarter"].mean()
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())


def test_multiscale_encoder_view_sharing():
    encoder = SharedMultiscaleSurfaceEncoder(
        in_channels=1,
        stem_channels=8,
        stage_channels=(8, 16, 24, 32),
        stage_blocks=(1, 1, 1, 1),
        pyramid_channels=12,
    )
    image = torch.randn(1, 1, 1, 32, 32)
    x = image.repeat(1, 2, 1, 1, 1)
    out = encoder(x)["full"]
    assert torch.allclose(out[:, 0], out[:, 1], atol=1e-6)
