"""
Uncertainty estimation for altitude triangulation.

This module provides methods for estimating and propagating
uncertainties in the altitude estimation process.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class UncertaintyModel:
    """
    Configuration for uncertainty estimation model.
    
    Attributes:
        pixel_error_std: Standard deviation of pixel localization (pixels)
        angular_resolution: Angular resolution (radians per pixel)
        baseline: Baseline distance between cameras (meters)
        systematic_error: Additional systematic error (meters)
    """
    pixel_error_std: float = 2.0  # pixels
    angular_resolution: float = np.radians(90) / 512  # ~0.003 rad/pixel
    baseline: float = 15000.0  # meters
    systematic_error: float = 100.0  # meters


def estimate_altitude_uncertainty(
    altitudes: np.ndarray,
    triangulation_angles: np.ndarray,
    elevations: np.ndarray,
    baseline: float = 15000.0,
    pixel_error_std: float = 2.0,
    angular_resolution: Optional[float] = None,
    image_radius_pixels: float = 512.0
) -> np.ndarray:
    """
    Estimate altitude uncertainty using geometric error propagation.
    
    The altitude uncertainty depends on several factors:
    
    1. **Angular error**: Random error in pixel localization translates
       to angular error in ray direction.
       
    2. **Triangulation angle**: Small angles (nearly parallel rays)
       amplify angular errors into large altitude errors.
       
    3. **Elevation angle**: Low elevation (looking near horizon)
       increases sensitivity to errors.
       
    4. **Altitude itself**: Higher altitude means longer ray paths,
       amplifying angular errors.
    
    The simplified model is:
        σ_h ≈ (h / B) * (σ_θ / sin(α)) * f(elevation)
    
    Where:
        h = altitude
        B = baseline
        σ_θ = angular error
        α = triangulation angle
        f(el) = elevation-dependent factor
    
    Args:
        altitudes: Estimated altitudes (meters)
        triangulation_angles: Angles between viewing rays (degrees)
        elevations: Minimum elevation angles from cameras (degrees)
        baseline: Distance between cameras (meters)
        pixel_error_std: Pixel localization error (pixels)
        angular_resolution: Angular resolution (radians/pixel).
                           If None, computed from image_radius_pixels.
        image_radius_pixels: Radius of fisheye image in pixels
        
    Returns:
        uncertainties: Altitude uncertainty estimates (meters)
    """
    # Convert to radians
    tri_angles_rad = np.radians(triangulation_angles)
    elevations_rad = np.radians(elevations)
    
    # Angular resolution
    if angular_resolution is None:
        # For 180° FOV fisheye: 90° / radius_pixels
        angular_resolution = np.radians(90) / image_radius_pixels
    
    # Angular error from pixel localization
    angular_error = pixel_error_std * angular_resolution  # radians
    
    # Triangulation geometry factor
    # At small angles, error is amplified by 1/sin(angle)
    sin_tri = np.sin(tri_angles_rad)
    sin_tri = np.maximum(sin_tri, 0.05)  # Avoid division by very small values
    geometry_factor = 1.0 / sin_tri
    
    # Elevation factor
    # Low elevation increases uncertainty
    sin_el = np.sin(elevations_rad)
    sin_el = np.maximum(sin_el, 0.1)  # Avoid extreme values
    elevation_factor = 1.0 / sin_el
    
    # Base uncertainty
    # Dimensional analysis: (m) * (rad) / (unitless) = meters
    # We need an additional factor for the altitude-to-baseline ratio
    altitude_ratio = np.abs(altitudes) / baseline
    
    base_uncertainty = np.abs(altitudes) * angular_error * geometry_factor
    
    # Apply elevation factor (moderate effect)
    uncertainty = base_uncertainty * np.sqrt(elevation_factor)
    
    # Add baseline uncertainty from both cameras (approximately √2 factor)
    uncertainty *= np.sqrt(2)
    
    # Minimum uncertainty floor (systematic errors, calibration, etc.)
    min_uncertainty = 50.0  # meters
    uncertainty = np.maximum(uncertainty, min_uncertainty)
    
    return uncertainty


def estimate_horizontal_uncertainty(
    altitudes: np.ndarray,
    triangulation_angles: np.ndarray,
    elevations: np.ndarray,
    baseline: float = 15000.0,
    pixel_error_std: float = 2.0,
    image_radius_pixels: float = 512.0
) -> np.ndarray:
    """
    Estimate horizontal position uncertainty.
    
    Horizontal uncertainty is generally smaller than vertical for
    atmospheric triangulation with hemispherical sky cameras.
    
    Returns:
        horizontal_uncertainties: East-North position uncertainty (meters)
    """
    # Horizontal error is approximately proportional to altitude
    # and inversely proportional to baseline
    angular_resolution = np.radians(90) / image_radius_pixels
    angular_error = pixel_error_std * angular_resolution
    
    # For horizontal, main factor is angular error times slant range
    slant_range = altitudes / np.maximum(np.sin(np.radians(elevations)), 0.1)
    
    horizontal_uncertainty = slant_range * angular_error * np.sqrt(2)
    
    # Minimum floor
    horizontal_uncertainty = np.maximum(horizontal_uncertainty, 20.0)
    
    return horizontal_uncertainty


def propagate_calibration_uncertainty(
    altitudes: np.ndarray,
    azimuth_error: float = 1.0,  # degrees
    elevation_bias: float = 0.5,  # degrees
    baseline: float = 15000.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate additional uncertainty from camera calibration errors.
    
    Calibration errors create systematic biases in altitude estimates.
    
    Args:
        altitudes: Estimated altitudes (meters)
        azimuth_error: Uncertainty in camera azimuth orientation (degrees)
        elevation_bias: Potential bias in elevation angle (degrees)
        baseline: Distance between cameras (meters)
        
    Returns:
        azimuth_contribution: Altitude uncertainty from azimuth error
        elevation_contribution: Altitude bias from elevation error
    """
    # Azimuth error creates horizontal displacement, which affects
    # altitude through the triangulation geometry
    azimuth_rad = np.radians(azimuth_error)
    azimuth_contribution = np.abs(altitudes) * azimuth_rad * (baseline / np.abs(altitudes + 1e-6))
    azimuth_contribution = np.minimum(azimuth_contribution, np.abs(altitudes) * 0.5)
    
    # Elevation bias creates direct altitude bias
    # For small angles: Δh ≈ h * Δel / tan(el) ≈ h * Δel (for high elevation)
    elevation_rad = np.radians(elevation_bias)
    elevation_contribution = np.abs(altitudes) * elevation_rad
    
    return azimuth_contribution, elevation_contribution


def compute_confidence_weighted_altitude(
    altitudes: np.ndarray,
    uncertainties: np.ndarray,
    confidence: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """
    Compute confidence-weighted mean altitude and uncertainty.
    
    Uses inverse-variance weighting with optional confidence scores.
    
    Args:
        altitudes: Array of altitude estimates
        uncertainties: Array of uncertainty estimates
        confidence: Optional match confidence scores
        
    Returns:
        weighted_mean: Weighted mean altitude
        combined_uncertainty: Combined uncertainty estimate
    """
    if len(altitudes) == 0:
        return np.nan, np.nan
    
    # Variance weights
    var_weights = 1.0 / (uncertainties**2 + 1e-6)
    
    # Combine with confidence if available
    if confidence is not None:
        weights = var_weights * confidence
    else:
        weights = var_weights
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # Weighted mean
    weighted_mean = np.sum(weights * altitudes)
    
    # Combined uncertainty (inverse-variance weighted)
    combined_variance = 1.0 / np.sum(var_weights)
    combined_uncertainty = np.sqrt(combined_variance)
    
    return weighted_mean, combined_uncertainty


def compute_monte_carlo_uncertainty(
    keypoints0: np.ndarray,
    keypoints1: np.ndarray,
    triangulator,
    n_samples: int = 100,
    pixel_noise_std: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate uncertainty using Monte Carlo simulation.
    
    Perturbs keypoint positions and re-triangulates to estimate
    uncertainty empirically.
    
    Args:
        keypoints0: Original keypoints from camera 1 (Nx2)
        keypoints1: Original keypoints from camera 2 (Nx2)
        triangulator: AtmosphericTriangulator instance
        n_samples: Number of Monte Carlo samples
        pixel_noise_std: Standard deviation of pixel noise to add
        
    Returns:
        altitude_mean: Mean altitude from samples
        altitude_std: Standard deviation of altitude from samples
        altitude_samples: All samples (n_samples x N array)
    """
    n_points = len(keypoints0)
    altitude_samples = np.zeros((n_samples, n_points))
    
    for i in range(n_samples):
        # Add random noise to keypoints
        noise0 = np.random.randn(*keypoints0.shape) * pixel_noise_std
        noise1 = np.random.randn(*keypoints1.shape) * pixel_noise_std
        
        perturbed0 = keypoints0 + noise0
        perturbed1 = keypoints1 + noise1
        
        # Triangulate
        result = triangulator.triangulate_matches(perturbed0, perturbed1)
        altitude_samples[i] = result.altitudes
    
    altitude_mean = np.mean(altitude_samples, axis=0)
    altitude_std = np.std(altitude_samples, axis=0)
    
    return altitude_mean, altitude_std, altitude_samples


def estimate_theoretical_precision(
    altitude: float,
    triangulation_angle: float,
    baseline: float = 15000.0,
    angular_precision: float = 0.001  # radians (~0.06°)
) -> float:
    """
    Calculate theoretical altitude precision for given geometry.
    
    This is the best-case precision assuming perfect calibration
    and only random angular errors.
    
    Args:
        altitude: Target altitude (meters)
        triangulation_angle: Angle between viewing rays (degrees)
        baseline: Camera baseline (meters)
        angular_precision: Angular measurement precision (radians)
        
    Returns:
        theoretical_precision: Best-case altitude precision (meters)
    """
    tri_rad = np.radians(triangulation_angle)
    sin_tri = np.sin(tri_rad)
    
    if sin_tri < 0.01:
        return np.inf
    
    # Simple geometric model
    precision = altitude * angular_precision / sin_tri * np.sqrt(2)
    
    return precision
