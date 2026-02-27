import torch
import torch.nn as nn
import torch.nn.functional as F


class KernelAttentionLayer(nn.Module):
    """轻量级Kernel Attention层，替代传统MLP以捕捉全局空间语义"""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0, "feature_dim must divide evenly by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [B*N, dim]  # 输入特征向量
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(x.shape[0], self.num_heads, self.head_dim).transpose(1, 2) for t in qkv]
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn_logits, dim=-1)
        attended = torch.matmul(attn, v)
        attended = attended.transpose(1, 2).reshape(x.shape[0], -1)
        return self.proj(attended)


class ViewKernelAggregator(nn.Module):
    """在视图维度上应用可学习卷积核，强化多视图空间关系"""

    def __init__(self, feature_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        # 1D卷积在视图轴上滑动，groups=1以允许跨通道交互
        self.kernel_conv = nn.Conv1d(
            feature_dim, feature_dim, kernel_size, padding=padding, bias=False
        )
        self.act = nn.GELU()

    def forward(self, view_features):
        """view_features: [B*N, num_views, feature_dim]  # 多视图特征输入"""
        x = view_features.permute(0, 2, 1)
        x = self.kernel_conv(x)
        x = self.act(x)
        return x.mean(dim=-1)


class KanDensityHead(nn.Module):
    """KAN密度头： kernel conv + attention + 指向性输出，替代简单MLP"""

    def __init__(self, feature_dim=64, num_views=7, pos_enc_dim=18, heads=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_views = num_views
        self.pos_enc_dim = pos_enc_dim
        self.aggregator = ViewKernelAggregator(feature_dim)
        self.pos_projection = nn.Linear(pos_enc_dim, feature_dim)
        self.attention = KernelAttentionLayer(feature_dim, num_heads=heads)

        self.output_mlp = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, fused_features, pos_enc=None):
        B, N, _ = fused_features.shape
        features = fused_features.view(B * N, self.num_views, self.feature_dim)
        aggregated = self.aggregator(features)
        if pos_enc is not None:
            pos = pos_enc.view(B * N, -1)
            pos = self.pos_projection(pos)
            aggregated = aggregated + pos
        attended = self.attention(aggregated)
        density = self.output_mlp(attended).view(B, N, 1)
        return density
