import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import to_2tuple, trunc_normal_

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TemporalMambaBlock(nn.Module):
    """A lightweight Mamba-style temporal mixer for sequence modeling.

    This is intentionally simple and dependency-free. It operates on the
    time axis of shape (B, T, D) and uses a learnable decay to scan through
    the sequence in a recurrent fashion, which is enough as a prototype for
    temporal-axis replacement inside PredFormer.

    Uses a numerically stable recurrent scan that avoids division by
    decay^i, which causes fp16 overflow in the original implementation.
    The scan loop over T is a few iterations (T ≤ 10 in video prediction)
    and its overhead is negligible compared to the spatial attention blocks.
    """

    def __init__(self, dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        # decay_logit ≈ 3.0 → sigmoid ≈ 0.95, a sensible starting point
        # for a temporal forget gate (retains 95% of past state per step).
        self.decay_logit = nn.Parameter(torch.ones(1, 1, dim) * 3.0)

    @torch.compiler.disable  # keep the loop eager — it's short and avoids compile issues
    def _scan(self, x):
        """Numerically stable recurrent scan: h_t = decay * h_{t-1} + x_t.

        Implemented as a simple for-loop over the time axis.  T is
        typically 4–10 in video prediction so the Python-level loop is
        not a bottleneck.  Critically, this avoids the fp16-hostile
        "divide by decay^i" formulation of the original.
        """
        b, t, d = x.shape
        # Bound decay in (0, 1) via sigmoid.  At init decay ≈ 0.95.
        decay = torch.sigmoid(self.decay_logit)  # (1, 1, d)

        h = torch.empty_like(x)
        h_t = x[:, 0]                             # h_0 = x_0
        h[:, 0] = h_t
        for i in range(1, t):
            h_t = decay * h_t + x[:, i]           # h_i = decay * h_{i-1} + x_i
            h[:, i] = h_t
        return h

    def forward(self, x):
        residual = x
        x = self.norm(x)
        gate, value = self.in_proj(x).chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        value = torch.tanh(value)
        x = gate * value
        # Cast to fp32 for the scan to guarantee stability even under AMP,
        # then cast back to the original dtype.
        x = self._scan(x.float()).to(dtype=residual.dtype)
        x = self.dropout(self.out_proj(x))
        return residual + x
     
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class MultiScaleSpatialMixer(nn.Module):
    """Dual-scale spatial mixing block for PredFormer.

    Replaces the single-scale ``GatedTransformer`` in a spatial branch
    with a parallel coarse + fine pyramid:

    - **Coarse**:  spatial stride-2 conv  →  GatedTransformer on ¼ tokens
                  →  bilinear upsample back to original resolution
    - **Fine**:    GatedTransformer on full-resolution tokens (same as original)

    A per-channel learnable gate fuses the two streams followed by a
    lightweight FFN so the fused features can redistribute.

    The coarse branch adds negligible FLOPs because attention over ¼
    the tokens costs ~1/16 of the fine branch.
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim,
                 dropout=0., attn_dropout=0., drop_path=0.1, merge_size=2):
        super().__init__()
        self.merge_size = merge_size

        # ---- Coarse branch ----
        self.spatial_reduce = nn.Conv2d(dim, dim,
                                        kernel_size=merge_size,
                                        stride=merge_size)
        self.coarse_transformer = GatedTransformer(
            dim, depth, heads, dim_head, mlp_dim,
            dropout, attn_dropout, drop_path)

        # ---- Fine branch (same as original spatial block) ----
        self.fine_transformer = GatedTransformer(
            dim, depth, heads, dim_head, mlp_dim,
            dropout, attn_dropout, drop_path)

        # ---- Per-channel learnable fusion gate ----
        self.gate = nn.Parameter(torch.zeros(1, 1, dim))  # init ≈ 0.5 after sigmoid

        # ---- Post-fusion lightweight FFN ----
        self.post_norm = nn.LayerNorm(dim)
        self.post_ffn = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x, H, W):
        """Forward pass.

        Args:
            x:  [B, H*W, D]  spatial tokens
            H:  int, patch-grid height
            W:  int, patch-grid width
        Returns:
            [B, H*W, D]
        """
        B, N, D = x.shape

        # ---- Fine branch ----
        x_fine = self.fine_transformer(x)                     # (B, N, D)

        # ---- Coarse branch ----
        x_c = (x.reshape(B, H, W, D)                          # (B, H, W, D)
                .permute(0, 3, 1, 2))                          # (B, D, H, W)
        x_c = self.spatial_reduce(x_c)                         # (B, D, Hc, Wc)
        _, _, Hc, Wc = x_c.shape
        x_c = (x_c.flatten(2)                                  # (B, D, Hc*Wc)
                .transpose(1, 2))                               # (B, Nc, D)
        x_c = self.coarse_transformer(x_c)                     # (B, Nc, D)

        # Upsample coarse back to fine resolution
        x_c = (x_c.reshape(B, Hc, Wc, D)
                .permute(0, 3, 1, 2))                          # (B, D, Hc, Wc)
        x_c = F.interpolate(x_c, size=(H, W),
                            mode='bilinear', align_corners=False)  # (B, D, H, W)
        x_c = (x_c.flatten(2)                                  # (B, D, N)
                .transpose(1, 2))                               # (B, N, D)

        # ---- Gated fusion ----
        gate = torch.sigmoid(self.gate)                        # (1, 1, D)
        x = gate * x_c + (1.0 - gate) * x_fine                # (B, N, D)

        # ---- Post-fusion FFN ----
        x = self.post_norm(x)
        x = x + self.post_ffn(x)
        return x

