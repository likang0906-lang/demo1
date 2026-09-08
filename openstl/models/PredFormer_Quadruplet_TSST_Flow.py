"""PredFormer Quadruplet TSST with Flow-Guided Temporal Attention.

A lightweight correlation-based ``FlowEstimator`` computes soft optical
flow between consecutive frames in patch-embedding space.  The flow
magnitude then biases the temporal attention: frames whose content is
fast-moving at a given patch position receive reduced attention weight,
while static content (persistent across frames) is attended more.

This injects a motion prior directly into the attention mechanism --
no extra tokens, no feature warping.
"""

import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
from openstl.modules import GatedTransformer, FlowEstimator


class PredFormerLayer(nn.Module):
    """Quadruplet TSST layer with flow-biased temporal attention.

    Factorization order:  Time -> Space -> Space -> Time
    - Temporal branches:  ``GatedTransformer`` with flow-magnitude bias
    - Spatial branches:   ``GatedTransformer`` unchanged
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim,
                 dropout=0., attn_dropout=0., drop_path=0.1):
        super(PredFormerLayer, self).__init__()

        # Temporal mixers -- flow-guided full attention
        self.ts_temporal = GatedTransformer(
            dim, depth, heads, dim_head, mlp_dim,
            dropout, attn_dropout, drop_path)
        self.st_temporal = GatedTransformer(
            dim, depth, heads, dim_head, mlp_dim,
            dropout, attn_dropout, drop_path)

        # Spatial mixers -- standard full attention
        self.ts_spatial = GatedTransformer(
            dim, depth, heads, dim_head, mlp_dim,
            dropout, attn_dropout, drop_path)
        self.st_spatial = GatedTransformer(
            dim, depth, heads, dim_head, mlp_dim,
            dropout, attn_dropout, drop_path)

        # Per-block learnable flow-bias strength
        self.gamma_ts = nn.Parameter(torch.tensor(0.5))
        self.gamma_st = nn.Parameter(torch.tensor(0.5))

    def _temporal_bias(self, flow_mag, b, n, t, gamma):
        """Build (B*N, 1, 1, T) additive attention bias.

        High motion magnitude at frame s → content at this patch position
        is transient → reduce attention from all frames to frame s.
        """
        # flow_mag: (B, T, N) → (B*N, T)
        m = rearrange(flow_mag, 'b t n -> (b n) t')
        bias = -gamma * m                              # (B*N, T)
        return bias[:, None, None, :]                  # (B*N, 1, 1, T)

    def forward(self, x, flow_mag):
        b, t, n, _ = x.shape

        # ---- Time-Space branch ----
        # ts-t: flow-guided temporal attention
        x = rearrange(x, 'b t n d -> b n t d')
        x = rearrange(x, 'b n t d -> (b n) t d')
        bias = self._temporal_bias(flow_mag, b, n, t, self.gamma_ts)
        x = self.ts_temporal(x, bias=bias)

        # ts-s: spatial attention
        x = rearrange(x, '(b n) t d -> b n t d', b=b)
        x = rearrange(x, 'b n t d -> b t n d')
        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.ts_spatial(x)
        x = rearrange(x, '(b t) n d -> b t n d', b=b)

        # ---- Space-Time branch ----
        # st-s: spatial attention
        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.st_spatial(x)

        # st-t: flow-guided temporal attention
        x = rearrange(x, '(b t) ... -> b t ...', b=b)
        x = x.permute(0, 2, 1, 3)
        x = rearrange(x, 'b n t d -> (b n) t d')
        bias = self._temporal_bias(flow_mag, b, n, t, self.gamma_st)
        x = self.st_temporal(x, bias=bias)

        x = rearrange(x, '(b n) t d -> b n t d', b=b)
        x = rearrange(x, 'b n t d -> b t n d', b=b)
        return x


def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return rearrange(pe, '... -> 1 ...')


class PredFormer_Model(nn.Module):
    """PredFormer Quadruplet TSST with flow-guided temporal attention.

    Flow is estimated once per input clip from the content patch
    embeddings (before positional encoding) and reuses the same motion
    prior across all layers.
    """

    def __init__(self, model_config, **kwargs):
        super().__init__()
        self.image_height = model_config['height']
        self.image_width = model_config['width']
        self.patch_size = model_config['patch_size']
        self.num_patches_h = self.image_height // self.patch_size
        self.num_patches_w = self.image_width // self.patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.num_frames_in = model_config['pre_seq']
        self.dim = model_config['dim']
        self.num_channels = model_config['num_channels']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.attn_dropout = model_config['attn_dropout']
        self.drop_path = model_config['drop_path']
        self.scale_dim = model_config['scale_dim']
        self.Ndepth = model_config['Ndepth']
        self.depth = model_config['depth']

        # Optional flow-specific knobs
        self.flow_search_range = model_config.get('flow_search_range', 3)

        assert self.image_height % self.patch_size == 0
        assert self.image_width % self.patch_size == 0
        self.patch_dim = self.num_channels * self.patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)',
                      p1=self.patch_size, p2=self.patch_size),
            nn.Linear(self.patch_dim, self.dim),
        )

        self.flow_estimator = FlowEstimator(
            self.dim, search_range=self.flow_search_range)

        self.pos_embedding = nn.Parameter(
            sinusoidal_embedding(self.num_frames_in * self.num_patches, self.dim),
            requires_grad=False
        ).view(1, self.num_frames_in, self.num_patches, self.dim)

        self.blocks = nn.ModuleList([
            PredFormerLayer(
                self.dim, self.depth, self.heads, self.dim_head,
                self.dim * self.scale_dim,
                self.dropout, self.attn_dropout, self.drop_path,
            )
            for _ in range(self.Ndepth)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_channels * self.patch_size ** 2)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = self.to_patch_embedding(x)

        # Estimate motion prior from content embeddings (pre-positional)
        _, flow_mag = self.flow_estimator(x, self.num_patches_h, self.num_patches_w)
        # flow_mag: (B, T, N)

        x = x + self.pos_embedding.to(x.device)

        for blk in self.blocks:
            x = blk(x, flow_mag)

        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.view(B, T, self.num_patches_h, self.num_patches_w,
                   C, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, T, C, H, W)
        return x
