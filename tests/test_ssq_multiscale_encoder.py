import torch

from minr_fmt.network.ssq_encoder import SharedResidualUNetPyramidEncoder


def test_residual_unet_pyramid_encoder_shape_and_gradients():
    encoder = SharedResidualUNetPyramidEncoder(
        in_channels=1,
        stem_channels=8,
        stage_channels=(8, 16, 24, 32),
        stage_blocks=(1, 1, 1, 1),
        pyramid_channels=12,
        output_channels=14,
    )
    x = torch.randn(2, 3, 1, 32, 32, requires_grad=True)
    out = encoder(x)
    assert out.shape == (2, 3, 14, 32, 32)
    loss = out.mean()
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())


def test_residual_unet_pyramid_encoder_view_sharing():
    encoder = SharedResidualUNetPyramidEncoder(
        in_channels=1,
        stem_channels=8,
        stage_channels=(8, 16, 24, 32),
        stage_blocks=(1, 1, 1, 1),
        pyramid_channels=12,
        output_channels=14,
    )
    image = torch.randn(1, 1, 1, 32, 32)
    x = image.repeat(1, 2, 1, 1, 1)
    out = encoder(x)
    assert torch.allclose(out[:, 0], out[:, 1], atol=1e-6)


def test_pyramid_internal_scales_contribute_to_fused_feature():
    encoder = SharedResidualUNetPyramidEncoder(
        in_channels=1,
        stem_channels=8,
        stage_channels=(8, 16, 24, 32),
        stage_blocks=(1, 1, 1, 1),
        pyramid_channels=12,
        output_channels=14,
    )
    x = torch.randn(1, 1, 1, 32, 32, requires_grad=True)
    out = encoder(x)
    out.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.proj1.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.proj2.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.proj3.parameters())
