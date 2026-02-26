"""
Stereo Camera Inference Comparison - Refactored Package

This package contains modular components for comparing segmentation results
between IPSL and ECTL stereo camera systems.
"""

__version__ = "1.0.0"
__author__ = "STEREOSTUDYIPSL"

from . import config
from . import model_loader
from . import data_utils
from . import image_processing
from . import inference
from . import flight_projection
from . import visualization

__all__ = [
    'config',
    'model_loader',
    'data_utils',
    'image_processing',
    'inference',
    'flight_projection',
    'visualization',
]
