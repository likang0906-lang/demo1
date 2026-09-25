"""PredFormer Quadruplet TSST with Overlapped Convolutional Tokenization.

Replaces the ViT-style non-overlapping patch embedding with a ``CNNStem``:
a stack of stride-2 3x3 convs that downsamples each frame by
``patch_size``.  The token grid (and therefore the encoder cost) is
identical to the original, but every token now sees an overlapping
receptive field ~2x the patch size, eliminating hard patch boundaries
and the grid artifacts they induce in predicted frames.

Everything downstream -- positional encoding, the TSST encoder, and the
MLP head -- is unchanged.
"""

import torch
from torch import nn
from einops import rearrange
from openstl.modules import GatedTransformer, CNNStem


class PredFormerLayer(nn.Module):
    """Quadruplet TSST layer — identical to the original PredFormer."""

    def __init__(self, dim, depth, heads, dim_head, mlp_dim,
                 dropout=0., attn_dropout=0., drop_path=0.1):
        super(PredFormerLayer, self).__init__()

        self.ts_temporal_transformer = GatedTransformer(
            dim, depth, heads, dim_head, mlp_dim,
            dropout, attn_dropout, drop_path)
        self.ts_space_transformer = GatedTransformer(
            dim, depth, heads, dim_head, mlp_dim,
            dropout, attn_dropout, drop_path)
        self.st_space_transformer = GatedTransformer(
            dim, depth, heads, dim_head, mlp_dim,
            dropout, attn_dropout, drop_path)
        self.st_temporal_transformer = GatedTransformer(
            dim, depth, heads, dim_head, mlp_dim,
            dropout, attn_dropout, drop_path)

    def forward(self, x):
        b, t, n, _ = x.shape

        # ts-t branch: temporal attention
        x = rearrange(x, 'b t n d -> b n t d')
        x = rearrange(x, 'b n t d -> (b n) t d')
        x = self.ts_temporal_transformer(x)

        # ts-s branch: spatial attention
        x = rearrange(x, '(b n) t d -> b n t d', b=b)
        x = rearrange(x, 'b n t d -> b t n d')
        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.ts_space_transformer(x)
        x = rearrange(x, '(b t) n d -> b t n d', b=b)

        # st-s branch: spatial attention
        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.st_space_transformer(x)

        # st-t branch: temporal attention
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
    """PredFormer Quadruplet TSST with CNN-stem tokenization.

    Only the patch embedding is replaced; the encoder, positional
    encoding, and MLP head are identical to the original TSST.
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

        assert self.image_height % self.patch_size == 0, \
            'Image height must be divisible by the patch size.'
        assert self.image_width % self.patch_size == 0, \
            'Image width must be divisible by the patch size.'

        # ---- CNN stem: overlapped tokenization, same grid as patchify ----
        # downsample = patch_size → token count identical to the original
        self.stem = CNNStem(self.num_channels, self.dim,
                            downsample=self.patch_size)

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

        # CNN stem over every frame
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.stem(x)                                          # (B*T, dim, Hp, Wp)
        x = rearrange(x, '(b t) d h w -> b t (h w) d', b=B)       # (B, T, N, D)

        x = x + self.pos_embedding.to(x.device)

        for blk in self.blocks:
            x = blk(x)

        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.view(B, T, self.num_patches_h, self.num_patches_w,
                   C, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, T, C, H, W)
        return x
