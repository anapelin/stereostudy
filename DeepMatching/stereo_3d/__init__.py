"""
Stereo 3D Altitude Estimation Module

This module provides tools for estimating 3D altitudes of atmospheric features
(clouds, contrails) from stereo sky camera imagery using fisheye cameras.

Key components:
- FisheyeCameraModel: Fisheye projection for hemispherical sky cameras
- CoordinateTransforms: WGS84/ECEF/ENU coordinate conversions
- AtmosphericTriangulator: Core triangulation for long-baseline stereo
- AltitudeResult: Result container with filtering and statistics
- Quality filtering and uncertainty estimation utilities
- Visualization tools for fisheye images and 3D point clouds
"""

from .results import AltitudeResult
from .coordinates import CoordinateTransforms
from .fisheye_model import FisheyeCameraModel, CameraCalibration, FisheyeProjection
from .triangulation import AtmosphericTriangulator, CameraParams, create_triangulator_ipsl_ectl
from .quality import (
    filter_altitude_estimates, 
    QualityMetrics, 
    FilterConfig,
    FILTER_STRICT,
    FILTER_MODERATE,
    FILTER_RELAXED,
)
from .uncertainty import estimate_altitude_uncertainty
from .visualization import (
    visualize_matches_with_altitude,
    create_3d_pointcloud_view,
    plot_altitude_histogram,
    plot_altitude_cross_section,
    plot_quality_diagnostics,
    plot_fisheye_overlay,
    create_summary_figure,
)

__all__ = [
    # Core classes
    'AltitudeResult',
    'CoordinateTransforms',
    'FisheyeCameraModel',
    'CameraCalibration',
    'FisheyeProjection',
    'AtmosphericTriangulator',
    'CameraParams',
    'create_triangulator_ipsl_ectl',
    # Filtering and uncertainty
    'filter_altitude_estimates',
    'QualityMetrics',
    'FilterConfig',
    'FILTER_STRICT',
    'FILTER_MODERATE',
    'FILTER_RELAXED',
    'estimate_altitude_uncertainty',
    # Visualization
    'visualize_matches_with_altitude',
    'create_3d_pointcloud_view',
    'plot_altitude_histogram',
    'plot_altitude_cross_section',
    'plot_quality_diagnostics',
    'plot_fisheye_overlay',
    'create_summary_figure',
]

__version__ = '0.1.0'
