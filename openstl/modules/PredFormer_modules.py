import torch
from torch import nn, einsum
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
    """

    def __init__(self, dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.decay = nn.Parameter(torch.ones(1, 1, dim) * 0.5)

    def _scan(self, x):
        """Vectorized scan: h_t = decay * h_{t-1} + x_t, with h_0 = x_0.

        Uses the identity:
          h_t = sum_{i=0}^{t} decay^{t-i} * x_i
              = decay^t * cumsum( x_i / decay^i )

        This avoids the Python for-loop and runs entirely with fused CUDA ops.
        """
        b, t, d = x.shape
        decay = self.decay.squeeze()  # (d,)
        decay = decay.clamp(min=1e-3, max=1.0 - 1e-4)

        idx = torch.arange(t, device=x.device, dtype=x.dtype)  # (t,)
        decay_pow = decay.unsqueeze(0).pow(idx.unsqueeze(1))    # (t, d)

        # x_i / decay^i
        x_div = x / decay_pow.unsqueeze(0).clamp(min=1e-3)      # (b, t, d)

        # cumsum + re-weight
        h = torch.cumsum(x_div, dim=1) * decay_pow.unsqueeze(0)  # (b, t, d)
        return h

    def forward(self, x):
        residual = x
        x = self.norm(x)
        gate, value = self.in_proj(x).chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        value = torch.tanh(value)
        x = gate * value
        x = self._scan(x)
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

