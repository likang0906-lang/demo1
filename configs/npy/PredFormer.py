# Config for custom per-sample .npy datasets (dataname='npy').
#
# Adjust `height` / `width` / `num_channels` / `pre_seq` / `after_seq`
# to match your data.  Each .npy file must contain a single sample of
# shape (T, C, H, W) with T = pre_seq + after_seq frames.

method = 'PredFormer'

model_config = {
    # image h w c — MUST match your .npy data
    'height': 64,
    'width': 64,
    'num_channels': 1,
    # video length in and out
    'pre_seq': 10,
    'after_seq': 10,
    # patch size
    'patch_size': 8,
    'dim': 256,
    'heads': 8,
    'dim_head': 32,
    # dropout
    'dropout': 0.0,
    'attn_dropout': 0.0,
    'drop_path': 0.0,
    'scale_dim': 4,
    # depth: per-block depth; Ndepth: number of stacked PredFormerLayer
    'depth': 1,
    'Ndepth': 6,
}
