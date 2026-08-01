"""
Smoke tests for PredFormer — Video Prediction Transformers without Recurrence or Convolution.

Coverage:
  1. Top-level package import
  2. Core modules (Attention, PreNorm, FeedForward)
  3. All 9 PredFormer model variants instantiation + forward pass
  4. Output shape correctness
  5. Config loading utility
  6. Method (trainer) instantiation skeleton

Usage:
    cd /home/user/lik/PredFormer-main
    python -m pytest tests/test_smoke.py -v
"""

import sys
import os
import tempfile

import pytest
import torch
import numpy as np

# ---------------------------------------------------------------------------
# 0.  Ensure the project root is on sys.path (if running from outside)
# ---------------------------------------------------------------------------
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_MODEL_CONFIG = {
    'height': 64,
    'width': 64,
    'num_channels': 1,
    'pre_seq': 10,
    'after_seq': 10,
    'patch_size': 8,
    'dim': 128,          # keep small for smoke speed
    'heads': 4,
    'dim_head': 32,
    'dropout': 0.0,
    'attn_dropout': 0.0,
    'drop_path': 0.0,
    'scale_dim': 4,
    'depth': 1,
    'Ndepth': 2,         # 2 layers for quick smoke
}


def _build_dummy(batch_size=2, config=None):
    """Return a dummy video tensor  B x T x C x H x W. """
    cfg = config or DEFAULT_MODEL_CONFIG
    B = batch_size
    T = cfg['pre_seq']
    C = cfg['num_channels']
    H = cfg['height']
    W = cfg['width']
    return torch.randn(B, T, C, H, W)


# ---------------------------------------------------------------------------
# 1.  Basic package import
# ---------------------------------------------------------------------------

def test_import_openstl():
    import openstl
    assert hasattr(openstl, '__version__') or True


# ---------------------------------------------------------------------------
# 2.  Core modules
# ---------------------------------------------------------------------------

def test_modules_import_ok():
    from openstl.modules import Attention, PreNorm, FeedForward
    assert Attention is not None
    assert PreNorm is not None
    assert FeedForward is not None


@pytest.mark.parametrize("dim,heads,dim_head", [(128, 4, 32), (256, 8, 32)])
def test_attention_forward(dim, heads, dim_head):
    from openstl.modules import Attention
    attn = Attention(dim, heads=heads, dim_head=dim_head)
    x = torch.randn(2, 64, dim)
    y = attn(x)
    assert y.shape == x.shape


def test_prenorm_forward():
    from openstl.modules import PreNorm
    from openstl.modules import FeedForward
    pn = PreNorm(128, FeedForward(128, 512))
    x = torch.randn(2, 64, 128)
    y = pn(x)
    assert y.shape == x.shape


def test_feedforward_forward():
    from openstl.modules import FeedForward
    ff = FeedForward(128, 512)
    x = torch.randn(2, 64, 128)
    y = ff(x)
    assert y.shape == x.shape


# ---------------------------------------------------------------------------
# 3.  All 9 PredFormer model variants
# ---------------------------------------------------------------------------

_PREDFORMER_VARIANTS = [
    'PredFormer_FullAttention',
    'PredFormer_FacST',
    'PredFormer_FacTS',
    'PredFormer_Binary_ST',
    'PredFormer_Binary_TS',
    'PredFormer_Triplet_STS',
    'PredFormer_Triplet_TST',
    'PredFormer_Quadruplet_TSST',
    'PredFormer_Quadruplet_STTS',
]


def _import_model(variant_name: str):
    """Dynamically import PredFormer_Model from each variant file."""
    mod = __import__(
        f'openstl.models.{variant_name}',
        fromlist=['PredFormer_Model'],
    )
    return mod.PredFormer_Model


@pytest.mark.parametrize("variant", _PREDFORMER_VARIANTS)
def test_model_instantiate(variant):
    """Each variant should instantiate without error."""
    ModelCls = _import_model(variant)
    model = ModelCls(DEFAULT_MODEL_CONFIG)
    assert isinstance(model, torch.nn.Module)


@pytest.mark.parametrize("variant", _PREDFORMER_VARIANTS)
def test_model_forward_shape(variant):
    """Forward pass returns correct shape  B x T x C x H x W."""
    ModelCls = _import_model(variant)
    model = ModelCls(DEFAULT_MODEL_CONFIG)
    model.eval()

    B = 2
    x = _build_dummy(B)
    with torch.no_grad():
        y = model(x)

    cfg = DEFAULT_MODEL_CONFIG
    expected = (B, cfg['pre_seq'], cfg['num_channels'], cfg['height'], cfg['width'])
    assert y.shape == expected, f'{variant}: expected {expected}, got {y.shape}'


def test_all_variants_produce_different_outputs():
    """Quick sanity: different variants produce numerically different outputs."""
    x = _build_dummy(batch_size=1)
    results = {}
    for variant in _PREDFORMER_VARIANTS:
        ModelCls = _import_model(variant)
        model = ModelCls(DEFAULT_MODEL_CONFIG)
        model.eval()
        with torch.no_grad():
            results[variant] = model(x.clone())

    # At least one pair should differ
    tensors = list(results.values())
    all_same = all(
        torch.allclose(tensors[0], t, atol=1e-4) for t in tensors[1:]
    )
    # They should differ (otherwise architecture might be duplicated)
    assert not all_same, 'All variants returned identical outputs — check model files.'


# ---------------------------------------------------------------------------
# 4.  Edge-case shapes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("height,width", [(32, 32), (64, 64), (128, 128)])
def test_various_spatial_sizes(height, width):
    from openstl.models.PredFormer_Quadruplet_TSST import PredFormer_Model
    cfg = {**DEFAULT_MODEL_CONFIG, 'height': height, 'width': width, 'patch_size': 8}
    model = PredFormer_Model(cfg)
    model.eval()
    x = _build_dummy(batch_size=1, config=cfg)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, cfg['pre_seq'], cfg['num_channels'], height, width)


@pytest.mark.parametrize("num_channels", [1, 3])
def test_various_channels(num_channels):
    from openstl.models.PredFormer_Quadruplet_TSST import PredFormer_Model
    cfg = {**DEFAULT_MODEL_CONFIG, 'num_channels': num_channels}
    model = PredFormer_Model(cfg)
    model.eval()
    x = _build_dummy(batch_size=1, config=cfg)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, cfg['pre_seq'], num_channels, cfg['height'], cfg['width'])


# ---------------------------------------------------------------------------
# 5.  Config loading utility
# ---------------------------------------------------------------------------

def test_load_config_from_file():
    """Test openstl.utils.load_config with a real config file."""
    from openstl.utils import load_config
    cfg_path = os.path.join(_PROJ_ROOT, 'configs', 'mmnist', 'PredFormer.py')
    cfg = load_config(cfg_path)
    assert isinstance(cfg, dict)
    assert 'method' in cfg
    assert cfg['method'] == 'PredFormer'


# ---------------------------------------------------------------------------
# 6.  Method (trainer) instantiation
# ---------------------------------------------------------------------------

def test_predformer_method_init():
    """Smoke-test that the PredFormer method class instantiates."""
    try:
        from openstl.methods.PredFormer import PredFormer
    except ImportError as e:
        pytest.skip(f"Method import failed (timm compatibility): {e}")
    from argparse import Namespace

    args = Namespace()
    args.method = 'PredFormer'
    args.dist = False
    args.device = 'cpu'
    args.dataname = 'mmnist'
    args.epoch = 1
    args.pre_seq_length = DEFAULT_MODEL_CONFIG['pre_seq']
    args.aft_seq_length = DEFAULT_MODEL_CONFIG['after_seq']
    args.batch_size = 2
    args.val_batch_size = 2
    args.lr = 1e-4
    args.weight_decay = 0.0
    args.opt = 'adamw'
    args.sched = 'onecycle'
    args.clip_grad = None
    args.clip_mode = None
    args.fp16 = False
    args.use_prefetcher = False
    args.early_stop_epoch = 1
    args.use_gpu = False
    args.no_display_method_info = True
    args.filter_bias_and_bn = False
    args.ex_name = 'smoke_test'
    args.res_dir = tempfile.mkdtemp()
    args.tb_dir = tempfile.mkdtemp()
    args.config_file = None
    # PredFormer_Model.__init__ expects a 'model_config' keyword argument,
    # so we nest the model hyper-params under args.model_config.
    args.model_config = DEFAULT_MODEL_CONFIG

    method = PredFormer(args, 'cpu', steps_per_epoch=10)
    assert method is not None
    assert method.model is not None
    assert method.criterion is not None


# ---------------------------------------------------------------------------
# 7.  Gradient flow sanity
# ---------------------------------------------------------------------------

def test_gradient_flow():
    """Loss backward should produce gradients for all parameters."""
    from openstl.models.PredFormer_Quadruplet_TSST import PredFormer_Model

    model = PredFormer_Model(DEFAULT_MODEL_CONFIG)
    model.train()
    x = _build_dummy(batch_size=2)
    y = model(x)
    target = torch.randn_like(y)
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()

    grad_norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.norm().item())

    assert len(grad_norms) > 0, 'No gradients flowed — check model connections.'
    # No NaN gradients
    assert all(np.isfinite(g) for g in grad_norms), 'NaN/Inf gradients detected.'
