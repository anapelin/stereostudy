"""
Atmospheric triangulation for long-baseline stereo sky cameras.

This module implements stereo triangulation for atmospheric observations
where:
- Cameras are separated by 10-20+ km (long baseline)
- Targets are at 1-15+ km altitude (clouds, contrails)
- Viewing rays may not intersect perfectly (closest approach used)
- Fisheye lens geometry must be handled correctly
"""

import numpy as np
from typing import Tuple, Optional, Dict, Union
from dataclasses import dataclass
import logging

from .coordinates import CoordinateTransforms
from .fisheye_model import FisheyeCameraModel, CameraCalibration
from .results import AltitudeResult

logger = logging.getLogger(__name__)


@dataclass 
class CameraParams:
    """Simple container for camera parameters."""
    lat: float  # degrees
    lon: float  # degrees
    height: float  # meters ASL
    image_width: int
    image_height: int
    fov_degrees: float = 180.0
    azimuth_offset: float = 0.0  # degrees
    name: str = "camera"


class AtmosphericTriangulator:
    """
    Triangulator for atmospheric feature altitude estimation.
    
    This class handles the complete pipeline from matched pixel coordinates
    to 3D geographic positions and altitudes, accounting for:
    
    - Fisheye camera projection (180° hemispherical FOV)
    - Long baseline between cameras (~15 km)
    - ECEF coordinate transforms
    - Closest approach for non-intersecting rays
    - Uncertainty estimation
    
    Args:
        cam1_params: Parameters for camera 1 (CameraParams or dict)
        cam2_params: Parameters for camera 2 (CameraParams or dict)
        
    Example:
        >>> cam1 = CameraParams(lat=48.7132, lon=2.207, height=177.5,
        ...                     image_width=1024, image_height=1024)
        >>> cam2 = CameraParams(lat=48.6005, lon=2.3468, height=90,
        ...                     image_width=1024, image_height=1024)
        >>> triangulator = AtmosphericTriangulator(cam1, cam2)
        >>> altitude_result = triangulator.triangulate_matches(match_result)
    """
    
    def __init__(
        self,
        cam1_params: Union[CameraParams, Dict],
        cam2_params: Union[CameraParams, Dict]
    ):
        """Initialize triangulator with two camera configurations."""
        # Convert dicts to CameraParams if needed
        if isinstance(cam1_params, dict):
            cam1_params = CameraParams(**cam1_params)
        if isinstance(cam2_params, dict):
            cam2_params = CameraParams(**cam2_params)
        
        self.cam1_params = cam1_params
        self.cam2_params = cam2_params
        
        # Create fisheye camera models
        self.cam1_model = FisheyeCameraModel(
            image_width=cam1_params.image_width,
            image_height=cam1_params.image_height,
            fov_degrees=cam1_params.fov_degrees,
            azimuth_offset=cam1_params.azimuth_offset
        )
        self.cam2_model = FisheyeCameraModel(
            image_width=cam2_params.image_width,
            image_height=cam2_params.image_height,
            fov_degrees=cam2_params.fov_degrees,
            azimuth_offset=cam2_params.azimuth_offset
        )
        
        # Camera positions in ECEF
        self.cam1_pos_ecef = np.array(CoordinateTransforms.geographic_to_ecef(
            cam1_params.lat, cam1_params.lon, cam1_params.height
        ))
        self.cam2_pos_ecef = np.array(CoordinateTransforms.geographic_to_ecef(
            cam2_params.lat, cam2_params.lon, cam2_params.height
        ))
        
        # Reference point for local ENU (midpoint between cameras)
        self.ref_lat = (cam1_params.lat + cam2_params.lat) / 2
        self.ref_lon = (cam1_params.lon + cam2_params.lon) / 2
        self.ref_height = (cam1_params.height + cam2_params.height) / 2
        
        # Rotation matrices for each camera (ENU to ECEF)
        self.R1_enu_to_ecef = CoordinateTransforms.get_rotation_matrix_enu_to_ecef(
            cam1_params.lat, cam1_params.lon
        )
        self.R2_enu_to_ecef = CoordinateTransforms.get_rotation_matrix_enu_to_ecef(
            cam2_params.lat, cam2_params.lon
        )
        
        # Calculate baseline
        self.baseline_3d, self.baseline_horizontal, self.baseline_azimuth = \
            CoordinateTransforms.calculate_baseline(
                cam1_params.lat, cam1_params.lon, cam1_params.height,
                cam2_params.lat, cam2_params.lon, cam2_params.height
            )
        
        logger.info(f"Triangulator initialized:")
        logger.info(f"  Camera 1: {cam1_params.name} ({cam1_params.lat:.4f}°, {cam1_params.lon:.4f}°, {cam1_params.height:.1f}m)")
        logger.info(f"  Camera 2: {cam2_params.name} ({cam2_params.lat:.4f}°, {cam2_params.lon:.4f}°, {cam2_params.height:.1f}m)")
        logger.info(f"  Baseline: {self.baseline_horizontal/1000:.2f} km horizontal, azimuth {self.baseline_azimuth:.1f}°")
    
    def triangulate_matches(
        self,
        keypoints0: np.ndarray,
        keypoints1: np.ndarray,
        confidence: Optional[np.ndarray] = None
    ) -> AltitudeResult:
        """
        Triangulate 3D positions from matched pixel coordinate pairs.
        
        Args:
            keypoints0: Nx2 array of pixel coordinates in camera 1
            keypoints1: Nx2 array of pixel coordinates in camera 2
            confidence: Optional N array of match confidence scores
            
        Returns:
            AltitudeResult with 3D positions, altitudes, and quality metrics
        """
        keypoints0 = np.atleast_2d(keypoints0)
        keypoints1 = np.atleast_2d(keypoints1)
        n_points = len(keypoints0)
        
        if n_points == 0:
            return self._empty_result()
        
        logger.debug(f"Triangulating {n_points} matched points...")
        
        # Step 1: Convert pixels to azimuth/elevation angles
        az1, el1 = self.cam1_model.pixel_to_angles(keypoints0)
        az2, el2 = self.cam2_model.pixel_to_angles(keypoints1)
        
        # Step 2: Convert to ray directions in local ENU
        rays1_enu = self.cam1_model.angles_to_ray_direction(az1, el1)
        rays2_enu = self.cam2_model.angles_to_ray_direction(az2, el2)
        
        # Step 3: Transform ray directions to ECEF
        rays1_ecef = (self.R1_enu_to_ecef @ rays1_enu.T).T  # Nx3
        rays2_ecef = (self.R2_enu_to_ecef @ rays2_enu.T).T  # Nx3
        
        # Normalize ray directions
        rays1_ecef = rays1_ecef / np.linalg.norm(rays1_ecef, axis=1, keepdims=True)
        rays2_ecef = rays2_ecef / np.linalg.norm(rays2_ecef, axis=1, keepdims=True)
        
        # Step 4: Find closest approach points for each ray pair
        points_ecef, miss_distances, t1_vals, t2_vals = self._intersect_rays_batch(
            self.cam1_pos_ecef, rays1_ecef,
            self.cam2_pos_ecef, rays2_ecef
        )
        
        # Step 5: Calculate triangulation angles
        tri_angles = self._calculate_triangulation_angles(rays1_ecef, rays2_ecef)
        
        # Step 6: Convert ECEF points to geographic coordinates
        lats, lons, alts = CoordinateTransforms.ecef_to_geographic(
            points_ecef[:, 0], points_ecef[:, 1], points_ecef[:, 2]
        )
        points_3d_geo = np.column_stack([lats, lons, alts])
        
        # Step 7: Convert to local ENU relative to reference
        e, n, u = CoordinateTransforms.ecef_to_enu(
            points_ecef[:, 0], points_ecef[:, 1], points_ecef[:, 2],
            self.ref_lat, self.ref_lon, self.ref_height
        )
        points_3d_enu = np.column_stack([e, n, u])
        
        # Step 8: Estimate uncertainties
        uncertainties = self._estimate_uncertainties(
            alts, tri_angles, miss_distances, el1, el2
        )
        
        return AltitudeResult(
            altitudes=alts,
            points_3d_geo=points_3d_geo,
            points_3d_enu=points_3d_enu,
            triangulation_angles=np.degrees(tri_angles),
            ray_miss_distances=miss_distances,
            uncertainties=uncertainties,
            elevations_cam1=np.degrees(el1),
            elevations_cam2=np.degrees(el2),
            azimuths_cam1=np.degrees(az1),
            azimuths_cam2=np.degrees(az2),
            reference_lat=self.ref_lat,
            reference_lon=self.ref_lon,
            reference_height=self.ref_height,
            keypoints0=keypoints0,
            keypoints1=keypoints1,
            confidence=confidence
        )
    
    def triangulate_from_match_result(self, match_result) -> AltitudeResult:
        """
        Convenience method to triangulate from a MatchResult object.
        
        Args:
            match_result: MatchResult from stereo_matchers (has keypoints0, keypoints1, confidence)
            
        Returns:
            AltitudeResult
        """
        return self.triangulate_matches(
            match_result.keypoints0,
            match_result.keypoints1,
            match_result.confidence
        )
    
    def _intersect_rays_batch(
        self,
        origin1: np.ndarray,
        dirs1: np.ndarray,
        origin2: np.ndarray,
        dirs2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Find closest approach between pairs of 3D rays (batch version).
        
        Uses the formula for closest points on two skew lines.
        For lines: P1(t) = O1 + t*D1 and P2(s) = O2 + s*D2
        
        Args:
            origin1: 3-element array, origin of all rays from camera 1
            dirs1: Nx3 array of unit direction vectors from camera 1
            origin2: 3-element array, origin of all rays from camera 2
            dirs2: Nx3 array of unit direction vectors from camera 2
            
        Returns:
            intersection_points: Nx3 array (midpoint of closest approach)
            miss_distances: N array (distance between rays at closest approach)
            t1: N array (parameter along ray 1 at closest point)
            t2: N array (parameter along ray 2 at closest point)
        """
        n_points = len(dirs1)
        
        # Vector from origin1 to origin2
        w0 = origin1 - origin2  # 3-element
        
        # Precompute dot products (batch)
        a = np.einsum('ij,ij->i', dirs1, dirs1)  # |D1|^2 = 1 for unit vectors
        b = np.einsum('ij,ij->i', dirs1, dirs2)  # D1 · D2
        c = np.einsum('ij,ij->i', dirs2, dirs2)  # |D2|^2 = 1 for unit vectors
        d = np.einsum('ij,j->i', dirs1, w0)      # D1 · w0
        e = np.einsum('ij,j->i', dirs2, w0)      # D2 · w0
        
        # Denominator (nearly zero when rays are parallel)
        denom = a * c - b * b
        
        # Handle parallel rays
        parallel_mask = np.abs(denom) < 1e-10
        denom = np.where(parallel_mask, 1.0, denom)  # Avoid division by zero
        
        # Parameters at closest approach
        t1 = (b * e - c * d) / denom
        t2 = (a * e - b * d) / denom
        
        # For parallel rays, use projection onto line
        t1 = np.where(parallel_mask, 0.0, t1)
        t2 = np.where(parallel_mask, -e / c, t2)
        
        # Ensure t1, t2 are positive (point should be in front of camera)
        # Negative t means the point is behind the camera
        t1 = np.maximum(t1, 0)
        t2 = np.maximum(t2, 0)
        
        # Closest points on each ray
        p1 = origin1 + t1[:, np.newaxis] * dirs1  # Nx3
        p2 = origin2 + t2[:, np.newaxis] * dirs2  # Nx3
        
        # Midpoint and miss distance
        intersection_points = (p1 + p2) / 2
        miss_distances = np.linalg.norm(p1 - p2, axis=1)
        
        return intersection_points, miss_distances, t1, t2
    
    def _calculate_triangulation_angles(
        self,
        rays1: np.ndarray,
        rays2: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the angle between pairs of viewing rays.
        
        The triangulation angle affects the accuracy of altitude estimation:
        - Small angles (<10°): Poor geometry, large altitude uncertainty
        - Optimal (20-50°): Good geometry for altitude estimation
        - Large angles (>70°): May indicate matching errors
        
        Args:
            rays1: Nx3 unit direction vectors from camera 1
            rays2: Nx3 unit direction vectors from camera 2
            
        Returns:
            angles: N array of angles in radians
        """
        # Dot product of unit vectors gives cosine of angle
        cos_angles = np.einsum('ij,ij->i', rays1, rays2)
        # Clip to handle numerical errors
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        angles = np.arccos(cos_angles)
        return angles
    
    def _estimate_uncertainties(
        self,
        altitudes: np.ndarray,
        tri_angles: np.ndarray,
        miss_distances: np.ndarray,
        el1: np.ndarray,
        el2: np.ndarray,
        pixel_error_std: float = 2.0
    ) -> np.ndarray:
        """
        Estimate altitude uncertainty using geometric error propagation.
        
        The uncertainty depends on:
        - Triangulation angle (small angles = high uncertainty)
        - Altitude (higher = more uncertainty per angular error)
        - Pixel localization error
        - Elevation angle (low elevation = high uncertainty)
        
        Args:
            altitudes: Estimated altitudes in meters
            tri_angles: Triangulation angles in radians
            miss_distances: Ray miss distances in meters
            el1, el2: Elevation angles from each camera (radians)
            pixel_error_std: Standard deviation of pixel localization (pixels)
            
        Returns:
            uncertainties: Altitude uncertainty in meters
        """
        # Angular resolution (radians per pixel)
        # For 180° FOV and ~512 pixel radius: 90° / 512 ≈ 0.003 rad/pixel
        angular_res = np.radians(90) / (self.cam1_model.r_max)
        angular_error = pixel_error_std * angular_res  # radians
        
        # Geometric dilution of precision
        # At small triangulation angles, error is amplified
        sin_tri = np.sin(tri_angles)
        sin_tri = np.maximum(sin_tri, 0.01)  # Avoid division by zero
        
        # Base uncertainty from angular error and geometry
        # Simplified model: uncertainty ∝ altitude * angular_error / sin(tri_angle)
        base_uncertainty = np.abs(altitudes) * angular_error / sin_tri
        
        # Additional uncertainty from elevation (low elevation = high uncertainty)
        min_el = np.minimum(el1, el2)
        elevation_factor = 1.0 / np.maximum(np.sin(min_el), 0.1)
        
        # Add contribution from ray miss distance (indicates matching error)
        miss_contribution = miss_distances * 2.0  # meters
        
        # Combine uncertainties (quadrature sum)
        uncertainties = np.sqrt(
            (base_uncertainty * elevation_factor)**2 + 
            miss_contribution**2
        )
        
        return uncertainties
    
    def _empty_result(self) -> AltitudeResult:
        """Return empty result for zero matches."""
        empty_1d = np.array([])
        empty_2d = np.zeros((0, 2))
        empty_3d = np.zeros((0, 3))
        
        return AltitudeResult(
            altitudes=empty_1d,
            points_3d_geo=empty_3d,
            points_3d_enu=empty_3d,
            triangulation_angles=empty_1d,
            ray_miss_distances=empty_1d,
            uncertainties=empty_1d,
            elevations_cam1=empty_1d,
            elevations_cam2=empty_1d,
            azimuths_cam1=empty_1d,
            azimuths_cam2=empty_1d,
            reference_lat=self.ref_lat,
            reference_lon=self.ref_lon,
            reference_height=self.ref_height,
            keypoints0=empty_2d,
            keypoints1=empty_2d,
            confidence=None
        )
    
    def get_expected_altitude_range(
        self,
        target_altitudes: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calculate expected triangulation angles for given target altitudes.
        
        Useful for understanding what geometry to expect.
        
        Args:
            target_altitudes: Array of altitudes to analyze (meters ASL)
            
        Returns:
            Dictionary with expected angles and uncertainties
        """
        target_altitudes = np.atleast_1d(target_altitudes)
        
        # Simple model: assume target is at midpoint between cameras
        mid_lat = (self.cam1_params.lat + self.cam2_params.lat) / 2
        mid_lon = (self.cam1_params.lon + self.cam2_params.lon) / 2
        
        results = {
            'altitudes': target_altitudes,
            'tri_angles': [],
            'uncertainties': []
        }
        
        for alt in target_altitudes:
            # Viewing angles from each camera to the target
            e1, n1, u1 = CoordinateTransforms.geographic_to_enu(
                mid_lat, mid_lon, alt,
                self.cam1_params.lat, self.cam1_params.lon, self.cam1_params.height
            )
            e2, n2, u2 = CoordinateTransforms.geographic_to_enu(
                mid_lat, mid_lon, alt,
                self.cam2_params.lat, self.cam2_params.lon, self.cam2_params.height
            )
            
            # Convert to unit vectors
            v1 = np.array([e1, n1, u1])
            v2 = np.array([e2, n2, u2])
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            
            # Triangulation angle
            tri_angle = np.arccos(np.clip(np.dot(v1, v2), -1, 1))
            results['tri_angles'].append(np.degrees(tri_angle))
            
            # Elevation angles
            el1 = np.arcsin(u1 / np.sqrt(e1**2 + n1**2 + u1**2))
            el2 = np.arcsin(u2 / np.sqrt(e2**2 + n2**2 + u2**2))
            
            # Estimated uncertainty
            angular_res = np.radians(90) / self.cam1_model.r_max
            sin_tri = max(np.sin(tri_angle), 0.01)
            uncertainty = alt * (2.0 * angular_res) / sin_tri
            results['uncertainties'].append(uncertainty)
        
        results['tri_angles'] = np.array(results['tri_angles'])
        results['uncertainties'] = np.array(results['uncertainties'])
        
        return results
    
    @property
    def info(self) -> str:
        """Get triangulator configuration info."""
        lines = [
            "AtmosphericTriangulator Configuration:",
            f"  Camera 1: {self.cam1_params.name}",
            f"    Position: ({self.cam1_params.lat:.4f}°, {self.cam1_params.lon:.4f}°, {self.cam1_params.height:.1f}m)",
            f"    Image: {self.cam1_params.image_width}x{self.cam1_params.image_height}",
            f"  Camera 2: {self.cam2_params.name}",
            f"    Position: ({self.cam2_params.lat:.4f}°, {self.cam2_params.lon:.4f}°, {self.cam2_params.height:.1f}m)",
            f"    Image: {self.cam2_params.image_width}x{self.cam2_params.image_height}",
            f"  Baseline: {self.baseline_horizontal/1000:.2f} km (horizontal), {self.baseline_azimuth:.1f}° azimuth",
            f"  Reference: ({self.ref_lat:.4f}°, {self.ref_lon:.4f}°, {self.ref_height:.1f}m)"
        ]
        return '\n'.join(lines)


# Convenience function for quick setup
def create_triangulator_ipsl_ectl(
    image_shape: Tuple[int, int],
    ipsl_azimuth_offset: float = 0.0,
    ectl_azimuth_offset: float = 0.0
) -> AtmosphericTriangulator:
    """
    Create triangulator for SIRTA-IPSL and ECTL camera pair.
    
    Args:
        image_shape: (height, width) of images
        ipsl_azimuth_offset: Azimuth calibration offset for IPSL camera (degrees)
        ectl_azimuth_offset: Azimuth calibration offset for ECTL camera (degrees)
        
    Returns:
        Configured AtmosphericTriangulator
    """
    cam1 = CameraParams(
        name="SIRTA-IPSL",
        lat=48.7132,
        lon=2.207,
        height=177.5,
        image_width=image_shape[1],
        image_height=image_shape[0],
        azimuth_offset=ipsl_azimuth_offset
    )
    
    cam2 = CameraParams(
        name="ECTL",
        lat=48.600518087374105,
        lon=2.3467954996250784,
        height=90.0,
        image_width=image_shape[1],
        image_height=image_shape[0],
        azimuth_offset=ectl_azimuth_offset
    )
    
    return AtmosphericTriangulator(cam1, cam2)
