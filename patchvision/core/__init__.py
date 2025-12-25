"""
PatchVision Core - Zero dependencies, maximum efficiency
"""
from .patches import PatchFactory
from .projections import TokenProjector
from .processors import InferenceEngine

__version__ = "1.0.0"
__all__ = ['PatchFactory', 'TokenProjector', 'InferenceEngine']