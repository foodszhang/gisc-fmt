"""Candidate-conditioned continuous density decoders."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from minr_fmt.network.ssq_diagnostics import ResidualMLPBlock, initialize_probability_head


class FourierPositionEncoding(nn.Module):
    def __init__(self, num_frequencies: int = 8):
        super().__init__()
        freqs = 2.0 ** torch.arange(int(num_frequencies), dtype=torch.float32)
        self.register_buffer("freqs", freqs, persistent=False)
        self.out_dim = 3 + 2 * 3 * int(num_frequencies)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        freqs = self.freqs.to(device=coords.device, dtype=coords.dtype)
        angles = coords[..., None, :] * freqs.view(*([1] * (coords.dim() - 1)), -1, 1) * math.pi
        sincos = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-2)
        return torch.cat([coords, sincos.flatten(start_dim=-2)], dim=-1)


class _ImplicitDecoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        pos_dim: int,
        hidden_dim: int = 128,
        positive_ratio: float = 0.03,
        query_chunk_size: int = 4096,
        checkpoint_decoder: bool = False,
        return_logits: bool = False,
        zero_init_output: bool = False,
    ):
        super().__init__()
        self.coord = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.SiLU(),
            ResidualMLPBlock(hidden_dim),
            ResidualMLPBlock(hidden_dim),
        )
        self.feat = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            ResidualMLPBlock(hidden_dim),
            ResidualMLPBlock(hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, 192),
            nn.SiLU(),
            ResidualMLPBlock(192),
            ResidualMLPBlock(192),
            ResidualMLPBlock(192),
            nn.LayerNorm(192),
            nn.Linear(192, 1),
        )
        initialize_probability_head(self.fusion, positive_ratio)
        if zero_init_output:
            nn.init.zeros_(self.fusion[-1].weight)
            nn.init.zeros_(self.fusion[-1].bias)
        self.query_chunk_size = int(query_chunk_size)
        self.checkpoint_decoder = bool(checkpoint_decoder)
        self.return_logits = bool(return_logits)

    def forward(self, z: torch.Tensor, encoded: torch.Tensor) -> torch.Tensor:
        n = z.shape[1]
        chunk = self.query_chunk_size
        if chunk > 0 and n > chunk:
            parts = []
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                parts.append(self._forward_impl(z[:, start:end], encoded[:, start:end]))
            return torch.cat(parts, dim=1)
        return self._forward_impl(z, encoded)

    def _forward_impl(self, z: torch.Tensor, encoded: torch.Tensor) -> torch.Tensor:
        use_checkpoint = (
            self.checkpoint_decoder
            and self.training
            and (z.requires_grad or encoded.requires_grad)
        )
        if use_checkpoint:
            coord = checkpoint(self.coord, encoded, use_reentrant=False)
            feat = checkpoint(self.feat, z, use_reentrant=False)
            logits = checkpoint(
                self.fusion, torch.cat([coord, feat], dim=-1), use_reentrant=False
            )
        else:
            coord = self.coord(encoded)
            feat = self.feat(z)
            logits = self.fusion(torch.cat([coord, feat], dim=-1))
        return logits if self.return_logits else torch.sigmoid(logits)


class CompensationDensityDecoder(_ImplicitDecoder):
    pass


class SharedDensityLogitDecoder(_ImplicitDecoder):
    """Decode the complete shared quotient directly to an unconstrained density logit."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, return_logits=True)


class CandidateDensityDecoder(_ImplicitDecoder):
    def forward(self, z: torch.Tensor, encoded_relative: torch.Tensor) -> torch.Tensor:
        return super().forward(z, encoded_relative)


class CandidateDensityResidualDecoder(_ImplicitDecoder):
    """Candidate-local logit correction initialized as an exact identity branch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, return_logits=True, zero_init_output=True)

    def forward(self, z: torch.Tensor, encoded_relative: torch.Tensor) -> torch.Tensor:
        return super().forward(z, encoded_relative)


class SourceHypothesisResidualDecoder(_ImplicitDecoder):
    """Zero-initialized candidate quotient residual logit head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, return_logits=True, zero_init_output=True)

    def forward(self, context: torch.Tensor, encoded_relative: torch.Tensor) -> torch.Tensor:
        return super().forward(context, encoded_relative)


class CandidateFieldCalibrator(nn.Module):
    """Candidate-centric residual field initialized as an exact no-op."""

    def __init__(self, context_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim + 4, hidden_dim),
            nn.SiLU(),
            ResidualMLPBlock(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, context: torch.Tensor, relative: torch.Tensor) -> torch.Tensor:
        radius = relative.square().sum(dim=-1, keepdim=True).sqrt()
        return self.net(torch.cat([context, relative, radius], dim=-1))


class _MorphologyDecoder(nn.Module):
    def __init__(self, feature_dim: int, pos_dim: int, hidden_dim: int):
        super().__init__()
        self.coord = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.SiLU(),
            ResidualMLPBlock(hidden_dim),
        )
        self.feat = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            ResidualMLPBlock(hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            ResidualMLPBlock(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, encoded: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat([self.coord(encoded), self.feat(z)], dim=-1))


class CompensationMorphologyDecoder(_MorphologyDecoder):
    pass


class CandidateMorphologyDecoder(_MorphologyDecoder):
    pass
