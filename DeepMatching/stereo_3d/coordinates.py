"""
Coordinate transformations for geographic, ECEF, and local ENU coordinates.

This module handles conversions between:
- WGS84 geographic coordinates (latitude, longitude, height)
- ECEF (Earth-Centered Earth-Fixed) Cartesian coordinates
- ENU (East-North-Up) local tangent plane coordinates

All transformations use the WGS84 ellipsoid parameters.
"""

import numpy as np
from typing import Tuple, Union
from dataclasses import dataclass


@dataclass
class WGS84:
    """WGS84 ellipsoid parameters."""
    # Semi-major axis (equatorial radius) in meters
    a: float = 6378137.0
    # Semi-minor axis (polar radius) in meters
    b: float = 6356752.314245
    # Flattening
    f: float = 1 / 298.257223563
    # First eccentricity squared
    e2: float = 0.00669437999014
    # Second eccentricity squared
    ep2: float = 0.00673949674228


class CoordinateTransforms:
    """
    Handle conversions between WGS84, ECEF, and local ENU coordinates.
    
    This class provides static methods for coordinate transformations
    commonly needed in atmospheric remote sensing and stereo vision.
    
    Examples:
        >>> # Convert camera position to ECEF
        >>> x, y, z = CoordinateTransforms.geographic_to_ecef(48.7132, 2.207, 177.5)
        
        >>> # Convert ECEF point to local ENU relative to camera
        >>> e, n, u = CoordinateTransforms.ecef_to_enu(x_pt, y_pt, z_pt, 48.7132, 2.207, 177.5)
    """
    
    wgs84 = WGS84()
    
    @staticmethod
    def geographic_to_ecef(
        lat: Union[float, np.ndarray],
        lon: Union[float, np.ndarray],
        height: Union[float, np.ndarray]
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Convert WGS84 geographic coordinates to ECEF Cartesian coordinates.
        
        Args:
            lat: Latitude in degrees (-90 to 90)
            lon: Longitude in degrees (-180 to 180)
            height: Height above WGS84 ellipsoid in meters
            
        Returns:
            x, y, z: ECEF coordinates in meters
            
        Notes:
            ECEF coordinate system:
            - Origin: Earth's center of mass
            - X-axis: Points to 0° latitude, 0° longitude (intersection of equator and prime meridian)
            - Y-axis: Points to 0° latitude, 90° E longitude
            - Z-axis: Points to North Pole (90° N latitude)
        """
        a = CoordinateTransforms.wgs84.a
        e2 = CoordinateTransforms.wgs84.e2
        
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # Radius of curvature in the prime vertical
        N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
        
        x = (N + height) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N + height) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (N * (1 - e2) + height) * np.sin(lat_rad)
        
        return x, y, z
    
    @staticmethod
    def ecef_to_geographic(
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
        tol: float = 1e-12,
        max_iter: int = 10
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Convert ECEF Cartesian coordinates to WGS84 geographic coordinates.
        
        Uses Bowring's iterative method for accurate conversion.
        
        Args:
            x, y, z: ECEF coordinates in meters
            tol: Convergence tolerance for iterative solution
            max_iter: Maximum iterations
            
        Returns:
            lat: Latitude in degrees
            lon: Longitude in degrees
            height: Height above WGS84 ellipsoid in meters
        """
        a = CoordinateTransforms.wgs84.a
        b = CoordinateTransforms.wgs84.b
        e2 = CoordinateTransforms.wgs84.e2
        ep2 = CoordinateTransforms.wgs84.ep2
        
        # Longitude is straightforward
        lon_rad = np.arctan2(y, x)
        
        # Distance from z-axis
        p = np.sqrt(x**2 + y**2)
        
        # Initial estimate of latitude using spherical approximation
        lat_rad = np.arctan2(z, p * (1 - e2))
        
        # Iterative refinement (Bowring's method)
        for _ in range(max_iter):
            N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
            lat_rad_new = np.arctan2(z + e2 * N * np.sin(lat_rad), p)
            
            if np.all(np.abs(lat_rad_new - lat_rad) < tol):
                lat_rad = lat_rad_new
                break
            lat_rad = lat_rad_new
        
        # Calculate height
        N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
        
        # Avoid division by zero near poles
        cos_lat = np.cos(lat_rad)
        if np.isscalar(cos_lat):
            if np.abs(cos_lat) > 1e-10:
                height = p / cos_lat - N
            else:
                height = np.abs(z) / np.abs(np.sin(lat_rad)) - N * (1 - e2)
        else:
            height = np.where(
                np.abs(cos_lat) > 1e-10,
                p / cos_lat - N,
                np.abs(z) / np.abs(np.sin(lat_rad)) - N * (1 - e2)
            )
        
        lat = np.degrees(lat_rad)
        lon = np.degrees(lon_rad)
        
        return lat, lon, height
    
    @staticmethod
    def ecef_to_enu(
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
        lat_ref: float,
        lon_ref: float,
        height_ref: float
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Convert ECEF coordinates to local East-North-Up (ENU) coordinates.
        
        Args:
            x, y, z: Point(s) in ECEF coordinates (meters)
            lat_ref: Reference point latitude (degrees)
            lon_ref: Reference point longitude (degrees)
            height_ref: Reference point height (meters)
            
        Returns:
            e: East coordinate(s) in meters
            n: North coordinate(s) in meters
            u: Up coordinate(s) in meters
            
        Notes:
            ENU coordinate system:
            - Origin: Reference point
            - E-axis: Points East
            - N-axis: Points North
            - U-axis: Points Up (perpendicular to ellipsoid)
        """
        # Reference point in ECEF
        x_ref, y_ref, z_ref = CoordinateTransforms.geographic_to_ecef(
            lat_ref, lon_ref, height_ref
        )
        
        # Difference vector in ECEF
        dx = x - x_ref
        dy = y - y_ref
        dz = z - z_ref
        
        # Rotation matrix from ECEF to ENU
        lat_rad = np.radians(lat_ref)
        lon_rad = np.radians(lon_ref)
        
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)
        
        # ENU = R * ECEF_diff
        # R rotates ECEF difference to local ENU
        e = -sin_lon * dx + cos_lon * dy
        n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
        u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
        
        return e, n, u
    
    @staticmethod
    def enu_to_ecef(
        e: Union[float, np.ndarray],
        n: Union[float, np.ndarray],
        u: Union[float, np.ndarray],
        lat_ref: float,
        lon_ref: float,
        height_ref: float
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Convert local ENU coordinates to ECEF coordinates.
        
        Args:
            e: East coordinate(s) in meters
            n: North coordinate(s) in meters
            u: Up coordinate(s) in meters
            lat_ref: Reference point latitude (degrees)
            lon_ref: Reference point longitude (degrees)
            height_ref: Reference point height (meters)
            
        Returns:
            x, y, z: ECEF coordinates in meters
        """
        # Reference point in ECEF
        x_ref, y_ref, z_ref = CoordinateTransforms.geographic_to_ecef(
            lat_ref, lon_ref, height_ref
        )
        
        # Rotation matrix from ENU to ECEF (transpose of ECEF to ENU)
        lat_rad = np.radians(lat_ref)
        lon_rad = np.radians(lon_ref)
        
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)
        
        # ECEF_diff = R^T * ENU
        dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
        dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
        dz = cos_lat * n + sin_lat * u
        
        x = x_ref + dx
        y = y_ref + dy
        z = z_ref + dz
        
        return x, y, z
    
    @staticmethod
    def geographic_to_enu(
        lat: Union[float, np.ndarray],
        lon: Union[float, np.ndarray],
        height: Union[float, np.ndarray],
        lat_ref: float,
        lon_ref: float,
        height_ref: float
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Convert geographic coordinates directly to local ENU.
        
        Convenience function combining geographic_to_ecef and ecef_to_enu.
        """
        x, y, z = CoordinateTransforms.geographic_to_ecef(lat, lon, height)
        return CoordinateTransforms.ecef_to_enu(x, y, z, lat_ref, lon_ref, height_ref)
    
    @staticmethod
    def enu_to_geographic(
        e: Union[float, np.ndarray],
        n: Union[float, np.ndarray],
        u: Union[float, np.ndarray],
        lat_ref: float,
        lon_ref: float,
        height_ref: float
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Convert local ENU coordinates to geographic coordinates.
        
        Convenience function combining enu_to_ecef and ecef_to_geographic.
        """
        x, y, z = CoordinateTransforms.enu_to_ecef(e, n, u, lat_ref, lon_ref, height_ref)
        return CoordinateTransforms.ecef_to_geographic(x, y, z)
    
    @staticmethod
    def calculate_baseline(
        lat1: float, lon1: float, height1: float,
        lat2: float, lon2: float, height2: float
    ) -> Tuple[float, float, float]:
        """
        Calculate the baseline vector between two cameras.
        
        Args:
            lat1, lon1, height1: First camera position (degrees, degrees, meters)
            lat2, lon2, height2: Second camera position
            
        Returns:
            distance: 3D distance between cameras in meters
            horizontal_distance: Horizontal (ground) distance in meters
            azimuth: Azimuth from camera 1 to camera 2 in degrees (0=N, 90=E)
        """
        # Get ENU of camera 2 relative to camera 1
        e, n, u = CoordinateTransforms.geographic_to_enu(
            lat2, lon2, height2,
            lat1, lon1, height1
        )
        
        distance = np.sqrt(e**2 + n**2 + u**2)
        horizontal_distance = np.sqrt(e**2 + n**2)
        azimuth = np.degrees(np.arctan2(e, n)) % 360
        
        return distance, horizontal_distance, azimuth
    
    @staticmethod
    def get_rotation_matrix_ecef_to_enu(lat: float, lon: float) -> np.ndarray:
        """
        Get the 3x3 rotation matrix from ECEF to ENU coordinates.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            
        Returns:
            R: 3x3 rotation matrix
        """
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)
        
        R = np.array([
            [-sin_lon, cos_lon, 0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
        ])
        
        return R
    
    @staticmethod
    def get_rotation_matrix_enu_to_ecef(lat: float, lon: float) -> np.ndarray:
        """
        Get the 3x3 rotation matrix from ENU to ECEF coordinates.
        
        This is the transpose of the ECEF-to-ENU rotation matrix.
        """
        return CoordinateTransforms.get_rotation_matrix_ecef_to_enu(lat, lon).T
