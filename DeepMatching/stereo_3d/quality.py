"""
Quality assessment and filtering for altitude estimates.

This module provides tools for evaluating the quality of altitude
estimates and filtering out unreliable results.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

from .results import AltitudeResult


@dataclass
class QualityMetrics:
    """
    Quality metrics summary for altitude estimation.
    
    Provides statistics about the triangulation quality and
    helps diagnose issues with the estimation.
    """
    total_points: int
    filtered_points: int
    pass_rate: float
    
    # Altitude statistics (filtered)
    altitude_mean: float
    altitude_std: float
    altitude_median: float
    altitude_min: float
    altitude_max: float
    
    # Quality metrics (filtered)
    tri_angle_mean: float
    tri_angle_median: float
    miss_distance_mean: float
    miss_distance_median: float
    uncertainty_mean: float
    uncertainty_median: float
    
    # Filter breakdown (how many rejected by each criterion)
    rejected_by_altitude: int = 0
    rejected_by_tri_angle: int = 0
    rejected_by_elevation: int = 0
    rejected_by_miss_distance: int = 0
    rejected_by_uncertainty: int = 0
    rejected_by_confidence: int = 0
    
    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            f"Quality Metrics Summary",
            f"{'='*40}",
            f"Points: {self.filtered_points}/{self.total_points} passed ({self.pass_rate*100:.1f}%)",
            f"",
            f"Altitude (filtered):",
            f"  Mean: {self.altitude_mean:.0f} ± {self.altitude_std:.0f} m",
            f"  Median: {self.altitude_median:.0f} m",
            f"  Range: {self.altitude_min:.0f} - {self.altitude_max:.0f} m",
            f"",
            f"Geometry:",
            f"  Triangulation angle: {self.tri_angle_mean:.1f}° (mean), {self.tri_angle_median:.1f}° (median)",
            f"  Miss distance: {self.miss_distance_mean:.1f} m (mean), {self.miss_distance_median:.1f} m (median)",
            f"  Uncertainty: {self.uncertainty_mean:.0f} m (mean), {self.uncertainty_median:.0f} m (median)",
        ]
        
        if any([self.rejected_by_altitude, self.rejected_by_tri_angle, 
                self.rejected_by_elevation, self.rejected_by_miss_distance,
                self.rejected_by_uncertainty, self.rejected_by_confidence]):
            lines.append("")
            lines.append("Rejection breakdown:")
            if self.rejected_by_altitude > 0:
                lines.append(f"  By altitude range: {self.rejected_by_altitude}")
            if self.rejected_by_tri_angle > 0:
                lines.append(f"  By triangulation angle: {self.rejected_by_tri_angle}")
            if self.rejected_by_elevation > 0:
                lines.append(f"  By elevation: {self.rejected_by_elevation}")
            if self.rejected_by_miss_distance > 0:
                lines.append(f"  By miss distance: {self.rejected_by_miss_distance}")
            if self.rejected_by_uncertainty > 0:
                lines.append(f"  By uncertainty: {self.rejected_by_uncertainty}")
            if self.rejected_by_confidence > 0:
                lines.append(f"  By confidence: {self.rejected_by_confidence}")
        
        return '\n'.join(lines)


@dataclass
class FilterConfig:
    """Configuration for quality filtering."""
    # Triangulation angle bounds (degrees)
    min_tri_angle: float = 5.0
    max_tri_angle: float = 80.0
    
    # Elevation bounds from either camera (degrees)
    min_elevation: float = 10.0
    max_elevation: float = 90.0
    
    # Altitude bounds (meters ASL)
    min_altitude: float = 500.0
    max_altitude: float = 20000.0
    
    # Ray miss distance threshold (meters)
    max_miss_distance: float = 1000.0
    
    # Uncertainty threshold (meters)
    max_uncertainty: float = 5000.0
    
    # Match confidence threshold
    min_confidence: float = 0.3
    
    def to_dict(self) -> Dict:
        return {
            'min_tri_angle': self.min_tri_angle,
            'max_tri_angle': self.max_tri_angle,
            'min_elevation': self.min_elevation,
            'max_elevation': self.max_elevation,
            'min_altitude': self.min_altitude,
            'max_altitude': self.max_altitude,
            'max_miss_distance': self.max_miss_distance,
            'max_uncertainty': self.max_uncertainty,
            'min_confidence': self.min_confidence
        }


# Preset filter configurations
FILTER_STRICT = FilterConfig(
    min_tri_angle=15.0,
    max_tri_angle=65.0,
    min_elevation=20.0,
    min_altitude=1000.0,
    max_altitude=15000.0,
    max_miss_distance=200.0,
    max_uncertainty=1000.0,
    min_confidence=0.5
)

FILTER_MODERATE = FilterConfig(
    min_tri_angle=10.0,
    max_tri_angle=70.0,
    min_elevation=15.0,
    min_altitude=500.0,
    max_altitude=18000.0,
    max_miss_distance=500.0,
    max_uncertainty=3000.0,
    min_confidence=0.3
)

FILTER_RELAXED = FilterConfig(
    min_tri_angle=5.0,
    max_tri_angle=80.0,
    min_elevation=10.0,
    min_altitude=200.0,
    max_altitude=25000.0,
    max_miss_distance=1000.0,
    max_uncertainty=5000.0,
    min_confidence=0.2
)


def filter_altitude_estimates(
    result: AltitudeResult,
    config: Optional[FilterConfig] = None,
    min_tri_angle: float = 10.0,
    max_tri_angle: float = 70.0,
    min_elevation: float = 10.0,
    min_altitude: float = 500.0,
    max_altitude: float = 20000.0,
    max_miss_distance: float = 500.0,
    max_uncertainty: float = 5000.0,
    min_confidence: float = 0.3,
    return_metrics: bool = False
) -> Tuple[AltitudeResult, np.ndarray] | Tuple[AltitudeResult, np.ndarray, QualityMetrics]:
    """
    Filter altitude estimates based on quality criteria.
    
    Points are rejected if they fail ANY of the criteria:
    - Triangulation angle outside [min_tri_angle, max_tri_angle]
    - Elevation angle from either camera below min_elevation
    - Altitude outside [min_altitude, max_altitude]
    - Ray miss distance above max_miss_distance
    - Altitude uncertainty above max_uncertainty
    - Match confidence below min_confidence
    
    Args:
        result: AltitudeResult to filter
        config: FilterConfig with all parameters (overrides individual params)
        min_tri_angle: Minimum triangulation angle (degrees)
        max_tri_angle: Maximum triangulation angle (degrees)
        min_elevation: Minimum elevation angle from both cameras (degrees)
        min_altitude: Minimum altitude ASL (meters)
        max_altitude: Maximum altitude ASL (meters)
        max_miss_distance: Maximum ray miss distance (meters)
        max_uncertainty: Maximum altitude uncertainty (meters)
        min_confidence: Minimum match confidence
        return_metrics: If True, also return QualityMetrics
        
    Returns:
        filtered_result: AltitudeResult with only passing points
        mask: Boolean array (True = passed)
        metrics: QualityMetrics (only if return_metrics=True)
    """
    if config is not None:
        min_tri_angle = config.min_tri_angle
        max_tri_angle = config.max_tri_angle
        min_elevation = config.min_elevation
        min_altitude = config.min_altitude
        max_altitude = config.max_altitude
        max_miss_distance = config.max_miss_distance
        max_uncertainty = config.max_uncertainty
        min_confidence = config.min_confidence
    
    n = result.num_points
    if n == 0:
        if return_metrics:
            metrics = QualityMetrics(
                total_points=0, filtered_points=0, pass_rate=0.0,
                altitude_mean=np.nan, altitude_std=np.nan, altitude_median=np.nan,
                altitude_min=np.nan, altitude_max=np.nan,
                tri_angle_mean=np.nan, tri_angle_median=np.nan,
                miss_distance_mean=np.nan, miss_distance_median=np.nan,
                uncertainty_mean=np.nan, uncertainty_median=np.nan
            )
            return result, np.array([], dtype=bool), metrics
        return result, np.array([], dtype=bool)
    
    # Track rejections
    rejected_altitude = np.zeros(n, dtype=bool)
    rejected_tri_angle = np.zeros(n, dtype=bool)
    rejected_elevation = np.zeros(n, dtype=bool)
    rejected_miss = np.zeros(n, dtype=bool)
    rejected_uncertainty = np.zeros(n, dtype=bool)
    rejected_confidence = np.zeros(n, dtype=bool)
    
    # Build mask from criteria
    mask = np.ones(n, dtype=bool)
    
    # Triangulation angle
    tri_mask = (result.triangulation_angles >= min_tri_angle) & \
               (result.triangulation_angles <= max_tri_angle)
    rejected_tri_angle = ~tri_mask
    mask &= tri_mask
    
    # Elevation from both cameras
    el_mask = (result.elevations_cam1 >= min_elevation) & \
              (result.elevations_cam2 >= min_elevation)
    rejected_elevation = ~el_mask
    mask &= el_mask
    
    # Altitude range
    alt_mask = (result.altitudes >= min_altitude) & \
               (result.altitudes <= max_altitude)
    rejected_altitude = ~alt_mask
    mask &= alt_mask
    
    # Miss distance
    miss_mask = result.ray_miss_distances <= max_miss_distance
    rejected_miss = ~miss_mask
    mask &= miss_mask
    
    # Uncertainty
    unc_mask = result.uncertainties <= max_uncertainty
    rejected_uncertainty = ~unc_mask
    mask &= unc_mask
    
    # Confidence (if available)
    if result.confidence is not None:
        conf_mask = result.confidence >= min_confidence
        rejected_confidence = ~conf_mask
        mask &= conf_mask
    
    # Filter result
    filtered = result.filter(mask)
    
    if not return_metrics:
        return filtered, mask
    
    # Compute quality metrics
    if filtered.num_points > 0:
        metrics = QualityMetrics(
            total_points=n,
            filtered_points=filtered.num_points,
            pass_rate=filtered.num_points / n,
            altitude_mean=float(np.mean(filtered.altitudes)),
            altitude_std=float(np.std(filtered.altitudes)),
            altitude_median=float(np.median(filtered.altitudes)),
            altitude_min=float(np.min(filtered.altitudes)),
            altitude_max=float(np.max(filtered.altitudes)),
            tri_angle_mean=float(np.mean(filtered.triangulation_angles)),
            tri_angle_median=float(np.median(filtered.triangulation_angles)),
            miss_distance_mean=float(np.mean(filtered.ray_miss_distances)),
            miss_distance_median=float(np.median(filtered.ray_miss_distances)),
            uncertainty_mean=float(np.mean(filtered.uncertainties)),
            uncertainty_median=float(np.median(filtered.uncertainties)),
            rejected_by_altitude=int(rejected_altitude.sum()),
            rejected_by_tri_angle=int(rejected_tri_angle.sum()),
            rejected_by_elevation=int(rejected_elevation.sum()),
            rejected_by_miss_distance=int(rejected_miss.sum()),
            rejected_by_uncertainty=int(rejected_uncertainty.sum()),
            rejected_by_confidence=int(rejected_confidence.sum())
        )
    else:
        metrics = QualityMetrics(
            total_points=n,
            filtered_points=0,
            pass_rate=0.0,
            altitude_mean=np.nan,
            altitude_std=np.nan,
            altitude_median=np.nan,
            altitude_min=np.nan,
            altitude_max=np.nan,
            tri_angle_mean=np.nan,
            tri_angle_median=np.nan,
            miss_distance_mean=np.nan,
            miss_distance_median=np.nan,
            uncertainty_mean=np.nan,
            uncertainty_median=np.nan,
            rejected_by_altitude=int(rejected_altitude.sum()),
            rejected_by_tri_angle=int(rejected_tri_angle.sum()),
            rejected_by_elevation=int(rejected_elevation.sum()),
            rejected_by_miss_distance=int(rejected_miss.sum()),
            rejected_by_uncertainty=int(rejected_uncertainty.sum()),
            rejected_by_confidence=int(rejected_confidence.sum())
        )
    
    return filtered, mask, metrics


def identify_outliers(
    result: AltitudeResult,
    altitude_percentile: float = 95.0,
    miss_distance_percentile: float = 95.0
) -> np.ndarray:
    """
    Identify statistical outliers in altitude estimates.
    
    Uses percentile-based outlier detection for altitude and
    miss distance.
    
    Args:
        result: AltitudeResult to analyze
        altitude_percentile: Percentile threshold for altitude outliers
        miss_distance_percentile: Percentile threshold for miss distance
        
    Returns:
        outlier_mask: Boolean array (True = outlier)
    """
    n = result.num_points
    if n < 10:
        return np.zeros(n, dtype=bool)
    
    outlier_mask = np.zeros(n, dtype=bool)
    
    # Altitude outliers (both tails)
    low_pct = (100 - altitude_percentile) / 2
    high_pct = 100 - low_pct
    alt_low = np.percentile(result.altitudes, low_pct)
    alt_high = np.percentile(result.altitudes, high_pct)
    outlier_mask |= (result.altitudes < alt_low) | (result.altitudes > alt_high)
    
    # Miss distance outliers (upper tail only)
    miss_thresh = np.percentile(result.ray_miss_distances, miss_distance_percentile)
    outlier_mask |= result.ray_miss_distances > miss_thresh
    
    return outlier_mask


def compute_quality_score(result: AltitudeResult) -> np.ndarray:
    """
    Compute a quality score for each altitude estimate.
    
    Higher score = better quality. Combines multiple factors:
    - Triangulation angle (optimal around 30-50°)
    - Match confidence
    - Elevation angle
    - Ray miss distance
    - Uncertainty
    
    Args:
        result: AltitudeResult to score
        
    Returns:
        scores: N array of quality scores (0-1)
    """
    n = result.num_points
    if n == 0:
        return np.array([])
    
    # Triangulation angle score (bell curve peaking at 40°)
    optimal_angle = 40.0
    angle_sigma = 20.0
    angle_score = np.exp(-((result.triangulation_angles - optimal_angle) / angle_sigma)**2)
    
    # Elevation score (higher is better, sigmoid)
    el_min = np.minimum(result.elevations_cam1, result.elevations_cam2)
    elevation_score = 1.0 / (1.0 + np.exp(-(el_min - 30) / 10))
    
    # Miss distance score (lower is better, exponential decay)
    miss_score = np.exp(-result.ray_miss_distances / 500)
    
    # Uncertainty score (lower is better, exponential decay)
    uncertainty_score = np.exp(-result.uncertainties / 2000)
    
    # Confidence score (if available)
    if result.confidence is not None:
        conf_score = result.confidence
    else:
        conf_score = np.ones(n)
    
    # Combine scores (geometric mean)
    scores = (angle_score * elevation_score * miss_score * 
              uncertainty_score * conf_score) ** 0.2
    
    return scores


def suggest_filter_params(
    result: AltitudeResult,
    target_pass_rate: float = 0.3
) -> FilterConfig:
    """
    Suggest filter parameters to achieve target pass rate.
    
    Analyzes the distribution of quality metrics and suggests
    thresholds that would result in approximately the target
    pass rate.
    
    Args:
        result: AltitudeResult to analyze
        target_pass_rate: Desired fraction of points to keep (0-1)
        
    Returns:
        FilterConfig with suggested parameters
    """
    n = result.num_points
    if n == 0:
        return FILTER_MODERATE
    
    # Use percentiles to estimate thresholds
    keep_pct = target_pass_rate * 100
    reject_pct = 100 - keep_pct
    
    # For symmetric filtering, use both tails
    low_pct = reject_pct / 4
    high_pct = 100 - reject_pct / 4
    
    suggested = FilterConfig(
        min_tri_angle=float(np.percentile(result.triangulation_angles, low_pct)),
        max_tri_angle=float(np.percentile(result.triangulation_angles, high_pct)),
        min_elevation=float(np.percentile(
            np.minimum(result.elevations_cam1, result.elevations_cam2), low_pct)),
        min_altitude=float(np.percentile(result.altitudes, low_pct)),
        max_altitude=float(np.percentile(result.altitudes, high_pct)),
        max_miss_distance=float(np.percentile(result.ray_miss_distances, high_pct)),
        max_uncertainty=float(np.percentile(result.uncertainties, high_pct)),
        min_confidence=0.3  # Keep default
    )
    
    # Apply sanity bounds
    suggested.min_tri_angle = max(5.0, suggested.min_tri_angle)
    suggested.max_tri_angle = min(85.0, suggested.max_tri_angle)
    suggested.min_elevation = max(5.0, suggested.min_elevation)
    suggested.min_altitude = max(100.0, suggested.min_altitude)
    suggested.max_altitude = min(25000.0, suggested.max_altitude)
    
    return suggested
