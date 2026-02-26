"""
Stereo Matchers - A unified interface for multiple stereo matching models.

Available matchers:
- LoFTRMatcher: Local Feature TRansformer (via Kornia)
- RoMaMatcher: Robust Dense Feature Matching
- DKMMatcher: Dense Kernelized Feature Matching

Usage:
    from stereo_matchers import LoFTRMatcher, RoMaMatcher, DKMMatcher
    
    matcher = LoFTRMatcher(device='cuda')
    results = matcher.match(img1, img2)
"""

from .base import BaseMatcher, MatchResult
from .loftr import LoFTRMatcher
from .roma import RoMaMatcher
from .dkm import DKMMatcher
from .benchmark import ModelBenchmark
from .viz import visualize_matches, compare_models, plot_confidence_histogram
from .utils import load_image, ensure_tensor, ensure_numpy, get_device

__version__ = "0.1.0"
__all__ = [
    "BaseMatcher",
    "MatchResult",
    "LoFTRMatcher",
    "RoMaMatcher",
    "DKMMatcher",
    "ModelBenchmark",
    "visualize_matches",
    "compare_models",
    "plot_confidence_histogram",
    "load_image",
    "ensure_tensor",
    "ensure_numpy",
    "get_device",
]
