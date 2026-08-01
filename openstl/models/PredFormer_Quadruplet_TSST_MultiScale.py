import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
from openstl.modules import MultiScaleSpatialMixer, GatedTransformer


class PredFormerLayer(nn.Module):
    """Quadruplet TSST layer with multi-scale spatial mixing.

    Factorization order:  Time → Space → Space → Time
    - Temporal branches: standard GatedTransformer (full attention)
    - Spatial branches:  MultiScaleSpatialMixer (coarse + fine pyramid)
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim,
                 H_patch, W_patch,
                 dropout=0., attn_dropout=0., drop_path=0.1, merge_size=2):
        super(PredFormerLayer, self).__init__()

        self.H_patch = H_patch
        self.W_patch = W_patch

        # Temporal mixers — standard full attention (T is small)
        self.ts_temporal_transformer = GatedTransformer(
            dim, depth, heads, dim_head, mlp_dim,
            dropout, attn_dropout, drop_path)
        self.st_temporal_transformer = GatedTransformer(
            dim, depth, heads, dim_head, mlp_dim,
            dropout, attn_dropout, drop_path)

        # Spatial mixers — multi-scale (coarse + fine pyramid)
        self.ts_space_transformer = MultiScaleSpatialMixer(
            dim, depth, heads, dim_head, mlp_dim,
            dropout, attn_dropout, drop_path, merge_size)
        self.st_space_transformer = MultiScaleSpatialMixer(
            dim, depth, heads, dim_head, mlp_dim,
            dropout, attn_dropout, drop_path, merge_size)

    def forward(self, x):
        b, t, n, _ = x.shape

        # ---- Time-Space branch ----
        # ts-t: temporal attention over time axis
        x = rearrange(x, 'b t n d -> b n t d')
        x = rearrange(x, 'b n t d -> (b n) t d')
        x = self.ts_temporal_transformer(x)

        # ts-s: multi-scale spatial mixing
        x = rearrange(x, '(b n) t d -> b n t d', b=b)
        x = rearrange(x, 'b n t d -> b t n d')
        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.ts_space_transformer(x, self.H_patch, self.W_patch)
        x = rearrange(x, '(b t) n d -> b t n d', b=b)

        # ---- Space-Time branch ----
        # st-s: multi-scale spatial mixing
        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.st_space_transformer(x, self.H_patch, self.W_patch)

        # st-t: temporal attention over time axis
        x = rearrange(x, '(b t) ... -> b t ...', b=b)
        x = x.permute(0, 2, 1, 3)
        x = rearrange(x, 'b n t d -> (b n) t d')
        x = self.st_temporal_transformer(x)

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
    """PredFormer Quadruplet TSST with Multi-Scale Spatial Mixing.

    Identical to the original TSST except spatial ``GatedTransformer``
    blocks are replaced with ``MultiScaleSpatialMixer``, which processes
    patches at two scales (coarse + fine) and fuses via a learned gate.
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
        self.num_classes = self.num_channels
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.attn_dropout = model_config['attn_dropout']
        self.drop_path = model_config['drop_path']
        self.scale_dim = model_config['scale_dim']
        self.Ndepth = model_config['Ndepth']
        self.depth = model_config['depth']

        assert self.image_height % self.patch_size == 0, \
            'Image height must be divisible by the patch size.'
        assert self.image_width % self.patch_size == 0, \
            'Image width must be divisible by the patch size.'
        self.patch_dim = self.num_channels * self.patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)',
                      p1=self.patch_size, p2=self.patch_size),
            nn.Linear(self.patch_dim, self.dim),
        )

        self.pos_embedding = nn.Parameter(
            sinusoidal_embedding(self.num_frames_in * self.num_patches, self.dim),
            requires_grad=False
        ).view(1, self.num_frames_in, self.num_patches, self.dim)

        self.blocks = nn.ModuleList([
            PredFormerLayer(
                self.dim, self.depth, self.heads, self.dim_head,
                self.dim * self.scale_dim,
                self.num_patches_h, self.num_patches_w,
                self.dropout, self.attn_dropout, self.drop_path,
                merge_size=2,
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
        x = x + self.pos_embedding.to(x.device)

        for blk in self.blocks:
            x = blk(x)

        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.view(B, T, self.num_patches_h, self.num_patches_w,
                   C, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, T, C, H, W)
        return x
