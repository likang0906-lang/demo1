# Config for custom per-sample .npy datasets (dataname='npy').
#
# Adjust `height` / `width` / `num_channels` / `pre_seq` / `after_seq`
# to match your data.  Each .npy file must contain a single sample of
# shape (T, C, H, W) with T = pre_seq + after_seq frames.

method = 'PredFormer'

model_config = {
    # image h w c — MUST match your .npy data
    'height': 128,
    'width': 128,
    'num_channels': 3,
    # video length in and out: 16 input frames -> 16 predicted frames
    'pre_seq': 16,
    'after_seq': 16,
    # patch size: 128 / 8 = 16x16 = 256 spatial tokens per frame
    'patch_size': 8,
    'dim': 256,
    'heads': 8,
    'dim_head': 32,
    # dropout (RGB real-world data benefits from regularization)
    'dropout': 0.1,
    'attn_dropout': 0.1,
    'drop_path': 0.1,
    'scale_dim': 4,
    # depth: per-block depth; Ndepth: number of stacked PredFormerLayer
    'depth': 1,
    'Ndepth': 4,
}
