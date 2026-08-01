import os
import sys
import torch

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


def test_temporal_mamba_model_forward_shape():
    from openstl.models.PredFormer_Quadruplet_TSST_Mamba import PredFormer_Model

    cfg = {
        'height': 64,
        'width': 64,
        'num_channels': 1,
        'pre_seq': 10,
        'after_seq': 10,
        'patch_size': 8,
        'dim': 128,
        'heads': 4,
        'dim_head': 32,
        'dropout': 0.0,
        'attn_dropout': 0.0,
        'drop_path': 0.0,
        'scale_dim': 4,
        'depth': 1,
        'Ndepth': 2,
    }

    model = PredFormer_Model(cfg)
    model.eval()
    x = torch.randn(2, 10, 1, 64, 64)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, 10, 1, 64, 64)
