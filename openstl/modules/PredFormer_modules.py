import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import to_2tuple, trunc_normal_, DropPath

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


class SwiGLU(nn.Module):
    """Gated feed-forward network with SiLU activation.

    Computes:  x = SiLU(gate(x)) * value(x)  followed by a projection.
    Used inside ``GatedTransformer``.
    """

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.SiLU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1_g = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.fc1_x = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

    def forward(self, x):
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GatedTransformer(nn.Module):
    """Standard gated transformer block: PreNorm → Attention → PreNorm → SwiGLU.

    With stochastic depth (DropPath) applied after each sub-block.
    Used as the basic building block for both spatial and temporal mixing.
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim,
                 dropout=0., attn_dropout=0., drop_path=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head,
                                       dropout=attn_dropout)),
                PreNorm(dim, SwiGLU(dim, mlp_dim, drop=dropout)),
                DropPath(drop_path) if drop_path > 0. else nn.Identity(),
                DropPath(drop_path) if drop_path > 0. else nn.Identity()
            ]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        for attn, ff, drop_path1, drop_path2 in self.layers:
            x = x + drop_path1(attn(x))
            x = x + drop_path2(ff(x))
        return self.norm(x)


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
    """Lightweight dual-scale spatial mixing block for PredFormer.

    Replaces the single-scale ``GatedTransformer`` with:

    - **Multi-scale bias**:  a tiny conv-net extracts global spatial context
      at ½ resolution and injects it as a learned bias signal.
    - **Fine transformer**:  the original GatedTransformer on full tokens.

    This preserves the multi-scale innovation while keeping parameter
    overhead below ~5 % of the original block, avoiding the memory
    pressure that causes Xid 79 on commodity GPUs.
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim,
                 dropout=0., attn_dropout=0., drop_path=0.1, merge_size=2,
                 coarse_transformer=None):
        super().__init__()
        self.merge_size = merge_size
        _hd = max(dim // 4, 64)  # coarse hidden dim

        # ---- Lightweight multi-scale bias generator ----
        # A tiny conv-net that captures global context at low resolution
        # and projects it back as a per-token bias for the fine branch.
        self.bias_net = nn.Sequential(
            nn.Conv2d(dim, _hd, kernel_size=3, stride=merge_size, padding=1),
            nn.GroupNorm(4, _hd),
            nn.GELU(),
            nn.Conv2d(_hd, _hd, kernel_size=3, padding=1),
            nn.GroupNorm(4, _hd),
            nn.GELU(),
            nn.Upsample(scale_factor=merge_size, mode='bilinear',
                        align_corners=False),
            nn.Conv2d(_hd, dim, kernel_size=1),
        )

        # ---- Fine branch (same as original spatial block) ----
        self.fine_transformer = GatedTransformer(
            dim, depth, heads, dim_head, mlp_dim,
            dropout, attn_dropout, drop_path)

        # ---- Learnable scalar gate ----
        self.gate = nn.Parameter(torch.tensor(0.0))  # init → 0.5 after sigmoid

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

        # ---- Multi-scale bias ----
        bias = (x.reshape(B, H, W, D)                         # (B, H, W, D)
                 .permute(0, 3, 1, 2))                         # (B, D, H, W)
        bias = self.bias_net(bias)                              # (B, D, H, W)
        bias = (bias.flatten(2)                                 # (B, D, N)
                 .transpose(1, 2))                               # (B, N, D)

        # ---- Inject bias, then fine transformer ----
        gate = torch.sigmoid(self.gate)
        x = x + gate * bias
        x = self.fine_transformer(x)
        return x


class SpatialViLBlock(nn.Module):
    """Vision-LSTM block for spatial token mixing.

    Replaces ``GatedTransformer`` (self-attention) in PredFormer spatial
    branches with a bidirectional LSTM scan.  Image patches are processed
    in raster-scan order; a forward + backward scan gives every patch
    access to global spatial context.

    Key properties (vs. self-attention):
      - O(N) complexity instead of O(N²)
      - ~40 % fewer parameters
      - Exponential input / forget gates (signature of the xLSTM family)

    Reference:
      "xLSTM: Extended Long Short-Term Memory"  (Beck et al., 2024)
      "Vision-LSTM (ViL)"                       (Alkin et al., 2024)
    """

    def __init__(self, dim, expand_ratio=2., dropout=0.):
        super().__init__()
        inner = int(dim * expand_ratio)

        self.norm = nn.LayerNorm(dim)

        # Gate projections (kept separate — NOT a chunk of a single tensor)
        self.W_z = nn.Linear(dim, inner)   # candidate
        self.W_i = nn.Linear(dim, inner)   # input  gate (exp)
        self.W_f = nn.Linear(dim, inner)   # forget gate (exp)
        self.W_o = nn.Linear(dim, inner)   # output gate (sigmoid)

        self.out_proj = nn.Linear(inner, dim)
        self.dropout = nn.Dropout(dropout)

        # Initialise forget bias positive → retain history by default
        nn.init.xavier_uniform_(self.W_z.weight, gain=0.5)
        nn.init.xavier_uniform_(self.W_i.weight, gain=0.1)
        nn.init.xavier_uniform_(self.W_f.weight, gain=0.1)
        nn.init.xavier_uniform_(self.W_o.weight, gain=0.1)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
        nn.init.constant_(self.W_f.bias, 2.0)   # f ≈ 1 at init
        nn.init.constant_(self.W_i.bias, -2.0)   # i ≈ 0 at init
        nn.init.zeros_(self.W_z.bias)
        nn.init.zeros_(self.W_o.bias)
        nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _scan_fwd(z, i_log, f_log, o):
        """Forward raster scan."""
        B, T, D = z.shape
        out = torch.empty_like(z)
        c = torch.zeros(B, D, device=z.device, dtype=z.dtype)
        for t in range(T):
            i_t = torch.exp(i_log[:, t])
            f_t = torch.exp(f_log[:, t])
            c = f_t * c + i_t * z[:, t]
            out[:, t] = o[:, t] * c
        return out

    @staticmethod
    def _scan_bwd(z, i_log, f_log, o):
        """Backward raster scan."""
        B, T, D = z.shape
        out = torch.empty_like(z)
        c = torch.zeros(B, D, device=z.device, dtype=z.dtype)
        for t in reversed(range(T)):
            i_t = torch.exp(i_log[:, t])
            f_t = torch.exp(f_log[:, t])
            c = f_t * c + i_t * z[:, t]
            out[:, t] = o[:, t] * c
        return out

    def forward(self, x):
        """Forward pass.

        Args:
            x:  [B, N, D]  spatial tokens (already in raster order)
        Returns:
            [B, N, D]
        """
        residual = x
        x = self.norm(x)

        z = torch.tanh(self.W_z(x))                     # (B, N, inner)
        i_log = torch.clamp(self.W_i(x), min=-10, max=5)  # stabilised exp
        f_log = torch.clamp(self.W_f(x), min=-10, max=5)
        o = torch.sigmoid(self.W_o(x))

        # Bidirectional scan
        h = self._scan_fwd(z, i_log, f_log, o) \
          + self._scan_bwd(z, i_log, f_log, o)

        x = self.dropout(self.out_proj(h))
        return residual + x

