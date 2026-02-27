import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import math


class ViewWeightNet(nn.Module):
    """Point-wise multi-view fusion with adaptive weights.

    Input:
        f: [B, N, V, C]

    Output:
        fused_features: [B, N, C]
        weights: [B, N, V]

    Changes vs previous version:
    - Predict logits then softmax(score/tau) instead of sigmoid+manual normalize.
    - Fuse as mean + sum_v w_v * (f_v - f_mean) to reduce detail loss from averaging.
    - MLP input adds |f_v - f_mean| as a complementary cue.
    """

    def __init__(self, feature_dim, num_views=7, embed_dim=16, hidden_dim=32, tau: float = 1.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_views = num_views
        self.embed_dim = embed_dim
        self.tau = float(tau)

        # Learnable view embedding e^(v) ∈ R^{embed_dim}
        self.view_embeddings = nn.Parameter(torch.randn(num_views, embed_dim))
        nn.init.normal_(self.view_embeddings, std=0.02)

        # score_mlp outputs raw logits (no sigmoid)
        # input: [f_mean, f_v, |f_v - f_mean|, e_v] -> 3C + embed_dim
        self.score_mlp = nn.Sequential(
            nn.Linear(feature_dim * 3 + embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, f: torch.Tensor):
        """Args:
        f: [B, N, V, C]
        """
        B, N, V, C = f.shape

        # mean feature per point
        f_mean = f.mean(dim=2, keepdim=True)  # [B, N, 1, C]
        f_mean_exp = f_mean.expand(-1, -1, V, -1)  # [B, N, V, C]

        # per-view embedding
        view_emb = self.view_embeddings.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)

        # complementary cue: |f_v - f_mean|
        diff_abs = (f - f_mean_exp).abs()

        combined = torch.cat([f_mean_exp, f, diff_abs, view_emb], dim=-1)  # [B, N, V, 3C+E]

        # logits -> softmax over views
        scores = self.score_mlp(combined).squeeze(-1)  # [B, N, V]
        tau = max(self.tau, 1e-6)
        scores = scores / tau
        scores = scores - scores.amax(dim=-1, keepdim=True)  # stability
        weights = F.softmax(scores, dim=-1)

        # fuse: mean + sum_v w_v * residual_v
        residual = f - f_mean_exp  # [B, N, V, C]
        fused_features = f_mean.squeeze(2) + (residual * weights.unsqueeze(-1)).sum(dim=2)

        return fused_features, weights


class CrossViewResidualFusion(nn.Module):
    """Per-scale cross-view residual completion fusion.

    Args:
        feature_dim: C
        num_views: V

    Inputs:
        g: [B, N, V, C]
        depth: [B, N, 1] (approx 0~200)

    Outputs:
        h: [B, N, C]
        weights: [B, N, V]
    """

    def __init__(
        self,
        feature_dim: int,
        num_views: int,
        embed_dim: int = 16,
        depth_dim: int = 16,
        hidden_dim: int = 64,
        tau: float = 1.0,
        depth_max: float = 200.0,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_views = int(num_views)
        self.embed_dim = int(embed_dim)
        self.depth_dim = int(depth_dim)
        self.hidden_dim = int(hidden_dim)
        self.tau = float(tau)
        self.depth_max = float(depth_max)

        self.view_embeddings = nn.Parameter(torch.randn(num_views, embed_dim))
        nn.init.normal_(self.view_embeddings, std=0.02)

        self.depth_mlp = nn.Sequential(
            nn.Linear(1, depth_dim),
            nn.ReLU(inplace=True),
            nn.Linear(depth_dim, depth_dim),
        )

        in_dim = feature_dim * 3 + 1 + embed_dim + depth_dim
        self.score_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, g: torch.Tensor, depth: torch.Tensor):
        # g: [B, N, V, C]
        # depth: [B, N, 1]
        B, N, V, C = g.shape

        mean = g.mean(dim=2)  # [B, N, C]
        mean_exp = mean.unsqueeze(2).expand(-1, -1, V, -1)  # [B, N, V, C]

        delta = g - mean_exp  # [B, N, V, C]
        delta_abs = delta.abs()  # [B, N, V, C]

        # cos_sim: [B, N, V, 1]
        cos_sim = F.cosine_similarity(g, mean_exp, dim=-1, eps=1e-6).unsqueeze(-1)

        # view embedding: [B, N, V, E]
        view_emb = self.view_embeddings.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)

        # depth embedding: depth->[0,1] then [B,N,D] -> [B,N,V,D]
        depth_norm = (depth / self.depth_max).clamp(0.0, 1.0)
        depth_emb = self.depth_mlp(depth_norm).unsqueeze(2).expand(-1, -1, V, -1)

        # score input: [mean, g_v, |delta|, cos_sim, view_emb, depth_emb]
        score_in = torch.cat([mean_exp, g, delta_abs, cos_sim, view_emb, depth_emb], dim=-1)
        scores = self.score_mlp(score_in).squeeze(-1)  # [B, N, V]

        tau = max(self.tau, 1e-6)
        scores = scores / tau
        scores = scores - scores.amax(dim=-1, keepdim=True)
        weights = F.softmax(scores, dim=-1)

        h = mean + (delta * weights.unsqueeze(-1)).sum(dim=2)  # [B, N, C]
        return h, weights


class ScaleFusionNet(nn.Module):
    """Depth-conditioned fusion across 3 scales.

    Inputs:
        h1,h2,h3: [B, N, C]
        depth: [B, N, 1]

    Outputs:
        h: [B, N, C]
        a: [B, N, 3] (softmax weights over scales)
    """

    def __init__(
        self,
        feature_dim: int,
        depth_dim: int = 16,
        hidden_dim: int = 64,
        depth_max: float = 200.0,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.depth_dim = int(depth_dim)
        self.depth_max = float(depth_max)

        self.depth_mlp = nn.Sequential(
            nn.Linear(1, depth_dim),
            nn.ReLU(inplace=True),
            nn.Linear(depth_dim, depth_dim),
        )

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 3 + depth_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, h1: torch.Tensor, h2: torch.Tensor, h3: torch.Tensor, depth: torch.Tensor):
        # h*: [B, N, C], depth: [B, N, 1]
        depth_norm = (depth / self.depth_max).clamp(0.0, 1.0)
        depth_emb = self.depth_mlp(depth_norm)  # [B, N, D]

        x = torch.cat([h1, h2, h3, depth_emb], dim=-1)  # [B, N, 3C+D]
        a_logits = self.mlp(x)  # [B, N, 3]
        a_logits = a_logits - a_logits.amax(dim=-1, keepdim=True)
        a = F.softmax(a_logits, dim=-1)  # [B, N, 3]

        h = a[..., 0:1] * h1 + a[..., 1:2] * h2 + a[..., 2:3] * h3
        return h, a


class BackgroundGuidanceInteraction(nn.Module):
    """Guidance vector interaction for the Background branch.

    Input:
        x: [B, N, C]
        g: [B, G] or [B, N, G]

    Output:
        x_out: [B, N, C]
        stats: dict of debug scalars (tensors)
    """

    def __init__(
        self,
        feature_dim: int,
        guidance_dim: int,
        hidden_dim: int,
        mode: str,
        scale: float,
        gate_init_bias: float,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.guidance_dim = int(guidance_dim)
        self.hidden_dim = int(hidden_dim)
        self.mode = str(mode)
        self.scale = float(scale)
        self.gate_init_bias = float(gate_init_bias)

        if self.mode == "film":
            self.mlp = nn.Sequential(
                nn.Linear(self.guidance_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, 2 * self.feature_dim),
            )
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
        elif self.mode == "gated":
            self.g_proj = nn.Linear(self.guidance_dim, self.feature_dim)
            self.gate_mlp = nn.Sequential(
                nn.Linear(self.feature_dim + self.guidance_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.feature_dim),
            )
            nn.init.zeros_(self.gate_mlp[-1].weight)
            nn.init.constant_(self.gate_mlp[-1].bias, self.gate_init_bias)
        else:
            raise ValueError(f"Unknown guidance interaction mode: {self.mode}")

    def forward(self, x: torch.Tensor, g: torch.Tensor):
        B, N, C = x.shape
        if g.ndim == 2:
            g_bn = g.unsqueeze(1).expand(-1, N, -1)  # [B, N, G]
            g_global = g
        elif g.ndim == 3:
            if g.shape[0] != B or g.shape[1] != N:
                raise ValueError(f"bg_guidance must be [B,G] or [B,N,G], got {tuple(g.shape)}")
            g_bn = g
            g_global = g.mean(dim=1)
        else:
            raise ValueError(f"bg_guidance must be [B,G] or [B,N,G], got {tuple(g.shape)}")

        stats: dict[str, torch.Tensor] = {}

        if self.mode == "film":
            gb = self.mlp(g_bn).view(B, N, 2 * C)
            gamma, beta = gb.chunk(2, dim=-1)
            stats["gamma_mean"] = gamma.mean()
            stats["beta_mean"] = beta.mean()
            x_int = x * (1.0 + gamma) + beta
        else:  # gated
            x_pool = x.mean(dim=1)  # [B, C]
            gate_in = torch.cat([x_pool, g_global], dim=-1)
            gate = torch.sigmoid(self.gate_mlp(gate_in)).unsqueeze(1)  # [B, 1, C]
            g_proj = self.g_proj(g_bn)  # [B, N, C]
            stats["gate_mean"] = gate.mean()
            x_int = gate * x + (1.0 - gate) * g_proj

        # scale: keep identity when scale=0
        x_out = x + self.scale * (x_int - x)
        return x_out, stats
