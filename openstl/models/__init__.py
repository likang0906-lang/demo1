import os

# Default remains the original model so existing training scripts keep working.
# Set PREDFORMER_MODEL=mamba before launching training to use the temporal-Mamba variant.
if os.environ.get('PREDFORMER_MODEL', '').lower() == 'mamba':
    from .PredFormer_Quadruplet_TSST_Mamba import PredFormer_Model
else:
    from .PredFormer_Quadruplet_TSST import PredFormer_Model

__all__ = [
    'PredFormer_Model'
]