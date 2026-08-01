import os

# Default remains the original model so existing training scripts keep working.
# Set PREDFORMER_MODEL before launching training to choose a variant:
#   unset / default      → Quadruplet TSST  (original, full attention)
#   mamba                → Quadruplet TSST with Temporal Mamba
#   multiscale           → Quadruplet TSST with Multi-Scale Spatial Mixing
_model_env = os.environ.get('PREDFORMER_MODEL', '').lower()
if _model_env == 'mamba':
    from .PredFormer_Quadruplet_TSST_Mamba import PredFormer_Model
elif _model_env == 'multiscale':
    from .PredFormer_Quadruplet_TSST_MultiScale import PredFormer_Model
else:
    from .PredFormer_Quadruplet_TSST import PredFormer_Model

__all__ = [
    'PredFormer_Model'
]