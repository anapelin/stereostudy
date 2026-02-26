"""
Result containers for altitude estimation.

This module defines dataclasses for storing and manipulating
altitude estimation results from stereo triangulation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json


@dataclass
class AltitudeResult:
    """
    Container for altitude estimation results from stereo triangulation.
    
    This class holds the 3D positions, quality metrics, and uncertainties
    for triangulated atmospheric features (clouds, contrails, etc.).
    
    Attributes:
        altitudes: Nx1 array of altitudes above sea level in meters
        points_3d_geo: Nx3 array of (latitude, longitude, altitude) in degrees/meters
        points_3d_enu: Nx3 array of (east, north, up) in meters relative to reference
        triangulation_angles: Nx1 array of angles between viewing rays in degrees
        ray_miss_distances: Nx1 array of closest approach distances in meters
        uncertainties: Nx1 array of estimated altitude uncertainties in meters
        elevations_cam1: Nx1 array of elevation angles from camera 1 in degrees
        elevations_cam2: Nx1 array of elevation angles from camera 2 in degrees
        azimuths_cam1: Nx1 array of azimuth angles from camera 1 in degrees
        azimuths_cam2: Nx1 array of azimuth angles from camera 2 in degrees
        reference_lat: Reference latitude for ENU coordinates
        reference_lon: Reference longitude for ENU coordinates
        reference_height: Reference height for ENU coordinates
        
    Example:
        >>> result = triangulator.triangulate_matches(matches)
        >>> print(f"Mean altitude: {result.altitudes.mean():.0f} m")
        >>> filtered = result.filter(result.triangulation_angles > 20)
    """
    # Core 3D results
    altitudes: np.ndarray  # meters ASL
    points_3d_geo: np.ndarray  # (lat, lon, alt)
    points_3d_enu: np.ndarray  # (east, north, up) meters
    
    # Quality metrics
    triangulation_angles: np.ndarray  # degrees
    ray_miss_distances: np.ndarray  # meters
    uncertainties: np.ndarray  # meters
    
    # Viewing geometry
    elevations_cam1: np.ndarray  # degrees
    elevations_cam2: np.ndarray  # degrees
    azimuths_cam1: np.ndarray  # degrees  
    azimuths_cam2: np.ndarray  # degrees
    
    # Reference point for ENU
    reference_lat: float = 0.0
    reference_lon: float = 0.0
    reference_height: float = 0.0
    
    # Optional: store original pixel coordinates
    keypoints0: Optional[np.ndarray] = None
    keypoints1: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    
    @property
    def num_points(self) -> int:
        """Number of triangulated points."""
        return len(self.altitudes)
    
    @property
    def altitude_stats(self) -> Dict[str, float]:
        """Get altitude statistics."""
        if len(self.altitudes) == 0:
            return {'count': 0, 'mean': np.nan, 'std': np.nan, 
                    'min': np.nan, 'max': np.nan, 'median': np.nan}
        return {
            'count': len(self.altitudes),
            'mean': float(np.mean(self.altitudes)),
            'std': float(np.std(self.altitudes)),
            'min': float(np.min(self.altitudes)),
            'max': float(np.max(self.altitudes)),
            'median': float(np.median(self.altitudes))
        }
    
    @property
    def quality_stats(self) -> Dict[str, float]:
        """Get quality metric statistics."""
        if len(self.altitudes) == 0:
            return {}
        return {
            'mean_tri_angle': float(np.mean(self.triangulation_angles)),
            'median_tri_angle': float(np.median(self.triangulation_angles)),
            'mean_miss_distance': float(np.mean(self.ray_miss_distances)),
            'median_miss_distance': float(np.median(self.ray_miss_distances)),
            'mean_uncertainty': float(np.mean(self.uncertainties)),
            'median_uncertainty': float(np.median(self.uncertainties))
        }
    
    def filter(self, mask: np.ndarray) -> 'AltitudeResult':
        """
        Filter results by boolean mask.
        
        Args:
            mask: Boolean array of length num_points
            
        Returns:
            New AltitudeResult with only points where mask is True
        """
        mask = np.asarray(mask).astype(bool)
        
        return AltitudeResult(
            altitudes=self.altitudes[mask],
            points_3d_geo=self.points_3d_geo[mask],
            points_3d_enu=self.points_3d_enu[mask],
            triangulation_angles=self.triangulation_angles[mask],
            ray_miss_distances=self.ray_miss_distances[mask],
            uncertainties=self.uncertainties[mask],
            elevations_cam1=self.elevations_cam1[mask],
            elevations_cam2=self.elevations_cam2[mask],
            azimuths_cam1=self.azimuths_cam1[mask],
            azimuths_cam2=self.azimuths_cam2[mask],
            reference_lat=self.reference_lat,
            reference_lon=self.reference_lon,
            reference_height=self.reference_height,
            keypoints0=self.keypoints0[mask] if self.keypoints0 is not None else None,
            keypoints1=self.keypoints1[mask] if self.keypoints1 is not None else None,
            confidence=self.confidence[mask] if self.confidence is not None else None
        )
    
    def filter_by_altitude(self, min_alt: float = 0, max_alt: float = np.inf) -> 'AltitudeResult':
        """Filter to points within altitude range (meters ASL)."""
        mask = (self.altitudes >= min_alt) & (self.altitudes <= max_alt)
        return self.filter(mask)
    
    def filter_by_triangulation_angle(self, min_angle: float = 0, max_angle: float = 90) -> 'AltitudeResult':
        """Filter to points with triangulation angle in range (degrees)."""
        mask = (self.triangulation_angles >= min_angle) & (self.triangulation_angles <= max_angle)
        return self.filter(mask)
    
    def filter_by_miss_distance(self, max_distance: float) -> 'AltitudeResult':
        """Filter to points with ray miss distance below threshold (meters)."""
        mask = self.ray_miss_distances <= max_distance
        return self.filter(mask)
    
    def filter_by_uncertainty(self, max_uncertainty: float) -> 'AltitudeResult':
        """Filter to points with uncertainty below threshold (meters)."""
        mask = self.uncertainties <= max_uncertainty
        return self.filter(mask)
    
    def filter_by_elevation(self, min_elevation: float = 0, max_elevation: float = 90) -> 'AltitudeResult':
        """Filter by elevation angle from either camera (degrees)."""
        mask = (
            (self.elevations_cam1 >= min_elevation) & (self.elevations_cam1 <= max_elevation) &
            (self.elevations_cam2 >= min_elevation) & (self.elevations_cam2 <= max_elevation)
        )
        return self.filter(mask)
    
    def filter_by_confidence(self, min_confidence: float) -> 'AltitudeResult':
        """Filter by match confidence (if available)."""
        if self.confidence is None:
            return self
        mask = self.confidence >= min_confidence
        return self.filter(mask)
    
    def apply_quality_filter(
        self,
        min_tri_angle: float = 10.0,
        max_tri_angle: float = 70.0,
        min_elevation: float = 10.0,
        min_altitude: float = 500.0,
        max_altitude: float = 20000.0,
        max_miss_distance: float = 500.0,
        max_uncertainty: float = 5000.0,
        min_confidence: float = 0.3
    ) -> 'AltitudeResult':
        """
        Apply comprehensive quality filtering.
        
        Args:
            min_tri_angle: Minimum triangulation angle (degrees)
            max_tri_angle: Maximum triangulation angle (degrees)
            min_elevation: Minimum elevation angle from both cameras (degrees)
            min_altitude: Minimum altitude ASL (meters)
            max_altitude: Maximum altitude ASL (meters)
            max_miss_distance: Maximum ray miss distance (meters)
            max_uncertainty: Maximum altitude uncertainty (meters)
            min_confidence: Minimum match confidence
            
        Returns:
            Filtered AltitudeResult
        """
        mask = np.ones(self.num_points, dtype=bool)
        
        # Triangulation angle
        mask &= (self.triangulation_angles >= min_tri_angle)
        mask &= (self.triangulation_angles <= max_tri_angle)
        
        # Elevation
        mask &= (self.elevations_cam1 >= min_elevation)
        mask &= (self.elevations_cam2 >= min_elevation)
        
        # Altitude range
        mask &= (self.altitudes >= min_altitude)
        mask &= (self.altitudes <= max_altitude)
        
        # Miss distance
        mask &= (self.ray_miss_distances <= max_miss_distance)
        
        # Uncertainty
        mask &= (self.uncertainties <= max_uncertainty)
        
        # Confidence (if available)
        if self.confidence is not None:
            mask &= (self.confidence >= min_confidence)
        
        return self.filter(mask)
    
    def get_geographic_bounds(self) -> Dict[str, float]:
        """Get geographic bounding box of points."""
        if self.num_points == 0:
            return {}
        return {
            'lat_min': float(np.min(self.points_3d_geo[:, 0])),
            'lat_max': float(np.max(self.points_3d_geo[:, 0])),
            'lon_min': float(np.min(self.points_3d_geo[:, 1])),
            'lon_max': float(np.max(self.points_3d_geo[:, 1])),
            'alt_min': float(np.min(self.altitudes)),
            'alt_max': float(np.max(self.altitudes))
        }
    
    def summary(self) -> str:
        """Generate a text summary of the results."""
        lines = [
            f"{'='*60}",
            f"ALTITUDE ESTIMATION RESULTS",
            f"{'='*60}",
            f"Total points: {self.num_points}",
            f"",
            f"Altitude Statistics:",
        ]
        
        stats = self.altitude_stats
        if stats['count'] > 0:
            lines.extend([
                f"  Range: {stats['min']:.0f} - {stats['max']:.0f} m ASL",
                f"  Mean:  {stats['mean']:.0f} ± {stats['std']:.0f} m",
                f"  Median: {stats['median']:.0f} m",
            ])
        else:
            lines.append("  No valid points")
        
        lines.append("")
        lines.append("Quality Metrics:")
        
        qstats = self.quality_stats
        if qstats:
            lines.extend([
                f"  Triangulation angle: {qstats['mean_tri_angle']:.1f}° (mean), {qstats['median_tri_angle']:.1f}° (median)",
                f"  Ray miss distance: {qstats['mean_miss_distance']:.1f} m (mean), {qstats['median_miss_distance']:.1f} m (median)",
                f"  Uncertainty: {qstats['mean_uncertainty']:.0f} m (mean), {qstats['median_uncertainty']:.0f} m (median)",
            ])
        
        lines.append(f"{'='*60}")
        
        return '\n'.join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for JSON serialization)."""
        return {
            'num_points': self.num_points,
            'altitude_stats': self.altitude_stats,
            'quality_stats': self.quality_stats,
            'geographic_bounds': self.get_geographic_bounds(),
            'reference': {
                'lat': self.reference_lat,
                'lon': self.reference_lon,
                'height': self.reference_height
            }
        }
    
    def save_csv(self, filepath: str) -> None:
        """Save results to CSV file."""
        import pandas as pd
        
        data = {
            'latitude': self.points_3d_geo[:, 0],
            'longitude': self.points_3d_geo[:, 1],
            'altitude_m': self.altitudes,
            'east_m': self.points_3d_enu[:, 0],
            'north_m': self.points_3d_enu[:, 1],
            'up_m': self.points_3d_enu[:, 2],
            'tri_angle_deg': self.triangulation_angles,
            'miss_distance_m': self.ray_miss_distances,
            'uncertainty_m': self.uncertainties,
            'elevation_cam1_deg': self.elevations_cam1,
            'elevation_cam2_deg': self.elevations_cam2,
            'azimuth_cam1_deg': self.azimuths_cam1,
            'azimuth_cam2_deg': self.azimuths_cam2,
        }
        
        if self.keypoints0 is not None:
            data['pixel_x_cam1'] = self.keypoints0[:, 0]
            data['pixel_y_cam1'] = self.keypoints0[:, 1]
        if self.keypoints1 is not None:
            data['pixel_x_cam2'] = self.keypoints1[:, 0]
            data['pixel_y_cam2'] = self.keypoints1[:, 1]
        if self.confidence is not None:
            data['confidence'] = self.confidence
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
    
    @classmethod
    def load_csv(cls, filepath: str) -> 'AltitudeResult':
        """Load results from CSV file."""
        import pandas as pd
        df = pd.read_csv(filepath)
        
        return cls(
            altitudes=df['altitude_m'].values,
            points_3d_geo=df[['latitude', 'longitude', 'altitude_m']].values,
            points_3d_enu=df[['east_m', 'north_m', 'up_m']].values,
            triangulation_angles=df['tri_angle_deg'].values,
            ray_miss_distances=df['miss_distance_m'].values,
            uncertainties=df['uncertainty_m'].values,
            elevations_cam1=df['elevation_cam1_deg'].values,
            elevations_cam2=df['elevation_cam2_deg'].values,
            azimuths_cam1=df['azimuth_cam1_deg'].values,
            azimuths_cam2=df['azimuth_cam2_deg'].values,
            keypoints0=df[['pixel_x_cam1', 'pixel_y_cam1']].values if 'pixel_x_cam1' in df else None,
            keypoints1=df[['pixel_x_cam2', 'pixel_y_cam2']].values if 'pixel_x_cam2' in df else None,
            confidence=df['confidence'].values if 'confidence' in df else None
        )
