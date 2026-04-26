"""
TruthLens model package initialization.

Exports all model components for easy importing:
    from models import DualInputDetector, SpatialBranch, FFTBranch, DCTBranch, FusionHead
"""

from .spatial_branch import SpatialBranch, SpatialBranchWithAttention
from .frequency_branch import FFTBranch, DCTBranch
from .fusion_head import FusionHead, SimpleFusionHead, WeightedFusionHead
from .dual_input_detector import DualInputDetector, DualInputDetectorLightweight

__all__ = [
    'SpatialBranch',
    'SpatialBranchWithAttention',
    'FFTBranch',
    'DCTBranch',
    'FusionHead',
    'SimpleFusionHead',
    'WeightedFusionHead',
    'DualInputDetector',
    'DualInputDetectorLightweight',
]
