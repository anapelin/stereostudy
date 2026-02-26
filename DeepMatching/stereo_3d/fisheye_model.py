"""
Fisheye camera model for hemispherical sky cameras.

This module implements the projection model for Reuniwatt Sky Cam Vision
and similar 180° fisheye cameras used for all-sky imaging.

Supported projection models:
- Equidistant (r = f * theta) - most common for sky cameras
- Equisolid angle (r = 2f * sin(theta/2))
- Stereographic (r = 2f * tan(theta/2))
- Orthographic (r = f * sin(theta))

Where:
- r: radial distance from image center in pixels
- f: focal length parameter
- theta: zenith angle (angle from optical axis)
"""

import numpy as np
from typing import Tuple, Union, Optional, Literal
from dataclasses import dataclass, field
from enum import Enum


class FisheyeProjection(Enum):
    """Fisheye projection models."""
    EQUIDISTANT = "equidistant"      # r = f * theta (linear)
    EQUISOLID = "equisolid"          # r = 2f * sin(theta/2)
    STEREOGRAPHIC = "stereographic"  # r = 2f * tan(theta/2)
    ORTHOGRAPHIC = "orthographic"    # r = f * sin(theta)


@dataclass
class FisheyeCameraModel:
    """
    Fisheye camera model for hemispherical sky cameras.
    
    This model handles the conversion between pixel coordinates and
    angular coordinates (azimuth, elevation) for fisheye lenses.
    
    The camera is assumed to be pointing at zenith (straight up), so:
    - Image center = zenith (elevation = 90°)
    - Image edge = horizon (elevation = 0°)
    - Radial distance from center encodes zenith angle
    - Angular position around center encodes azimuth
    
    Args:
        image_width: Image width in pixels
        image_height: Image height in pixels
        fov_degrees: Field of view in degrees (180 for full hemisphere)
        projection: Fisheye projection model type
        cx: Principal point x (defaults to image center)
        cy: Principal point y (defaults to image center)
        azimuth_offset: Rotation offset in degrees (0 = image up is North)
        azimuth_clockwise: If True, azimuth increases clockwise in image
        
    Attributes:
        r_max: Maximum radius in pixels (from center to edge of FOV)
        
    Example:
        >>> cam = FisheyeCameraModel(1024, 1024, fov_degrees=180)
        >>> az, el = cam.pixel_to_angles(np.array([[512, 512]]))  # Center = zenith
        >>> print(f"Elevation at center: {np.degrees(el[0]):.1f}°")  # Should be 90°
    """
    image_width: int
    image_height: int
    fov_degrees: float = 180.0
    projection: FisheyeProjection = FisheyeProjection.EQUIDISTANT
    cx: Optional[float] = None  # Principal point x
    cy: Optional[float] = None  # Principal point y
    azimuth_offset: float = 0.0  # Degrees to add to azimuth (camera rotation)
    azimuth_clockwise: bool = True  # Standard for looking up at sky
    
    def __post_init__(self):
        """Initialize derived parameters."""
        # Default principal point to image center
        if self.cx is None:
            self.cx = self.image_width / 2
        if self.cy is None:
            self.cy = self.image_height / 2
        
        # Maximum radius (from center to FOV edge)
        # For a circular fisheye, this is typically half the smaller dimension
        self.r_max = min(self.image_width, self.image_height) / 2
        
        # Maximum zenith angle (half the FOV)
        self.max_zenith_angle = np.radians(self.fov_degrees / 2)
        
        # Focal length parameter (depends on projection model)
        # For equidistant: r = f * theta, so f = r_max / max_zenith
        if self.projection == FisheyeProjection.EQUIDISTANT:
            self.f = self.r_max / self.max_zenith_angle
        elif self.projection == FisheyeProjection.EQUISOLID:
            self.f = self.r_max / (2 * np.sin(self.max_zenith_angle / 2))
        elif self.projection == FisheyeProjection.STEREOGRAPHIC:
            self.f = self.r_max / (2 * np.tan(self.max_zenith_angle / 2))
        elif self.projection == FisheyeProjection.ORTHOGRAPHIC:
            self.f = self.r_max / np.sin(self.max_zenith_angle)
    
    def pixel_to_angles(
        self,
        pixel_coords: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert pixel coordinates to (azimuth, elevation) angles.
        
        Args:
            pixel_coords: Nx2 array of (x, y) pixel coordinates
            
        Returns:
            azimuth: N array in radians (0 to 2π, 0=North, increasing clockwise)
            elevation: N array in radians (0=horizon, π/2=zenith)
            
        Notes:
            - Points outside the FOV circle will have elevation < 0
            - Azimuth convention: 0=North, 90°=East (clockwise when looking up)
        """
        pixel_coords = np.atleast_2d(pixel_coords)
        
        # Calculate offset from principal point
        dx = pixel_coords[:, 0] - self.cx
        dy = pixel_coords[:, 1] - self.cy
        
        # Radial distance from center
        r = np.sqrt(dx**2 + dy**2)
        
        # Convert radius to zenith angle based on projection model
        zenith_angle = self._radius_to_zenith(r)
        
        # Elevation = 90° - zenith angle
        elevation = np.pi / 2 - zenith_angle
        
        # Azimuth from pixel position
        # Note: Image coordinate system typically has y increasing downward
        # For a camera looking up at the sky:
        #   - "up" in image often corresponds to North (or needs calibration)
        #   - We use atan2(dx, -dy) to get angle from "up" direction
        if self.azimuth_clockwise:
            raw_azimuth = np.arctan2(dx, -dy)  # CW from up
        else:
            raw_azimuth = np.arctan2(-dx, -dy)  # CCW from up
        
        # Apply rotation offset and normalize to [0, 2π)
        azimuth = (raw_azimuth + np.radians(self.azimuth_offset)) % (2 * np.pi)
        
        return azimuth, elevation
    
    def angles_to_pixel(
        self,
        azimuth: np.ndarray,
        elevation: np.ndarray
    ) -> np.ndarray:
        """
        Convert (azimuth, elevation) angles to pixel coordinates.
        
        Args:
            azimuth: N array in radians (0 to 2π)
            elevation: N array in radians (0=horizon, π/2=zenith)
            
        Returns:
            pixel_coords: Nx2 array of (x, y) pixel coordinates
        """
        azimuth = np.atleast_1d(azimuth)
        elevation = np.atleast_1d(elevation)
        
        # Zenith angle from elevation
        zenith_angle = np.pi / 2 - elevation
        
        # Convert zenith angle to radius
        r = self._zenith_to_radius(zenith_angle)
        
        # Convert azimuth to image angle
        image_angle = azimuth - np.radians(self.azimuth_offset)
        
        # Convert to pixel offsets
        if self.azimuth_clockwise:
            dx = r * np.sin(image_angle)
            dy = -r * np.cos(image_angle)
        else:
            dx = -r * np.sin(image_angle)
            dy = -r * np.cos(image_angle)
        
        # Add principal point offset
        x = dx + self.cx
        y = dy + self.cy
        
        return np.column_stack([x, y])
    
    def pixel_to_ray_direction(
        self,
        pixel_coords: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Convert pixel coordinates to 3D ray directions in local ENU frame.
        
        Args:
            pixel_coords: Nx2 array of (x, y) pixel coordinates
            normalize: If True, return unit vectors
            
        Returns:
            rays: Nx3 array of ray direction vectors in ENU coordinates
                  (East, North, Up)
                  
        Notes:
            The camera is assumed to be pointing at zenith with:
            - East = +X
            - North = +Y  
            - Up = +Z (zenith)
        """
        azimuth, elevation = self.pixel_to_angles(pixel_coords)
        return self.angles_to_ray_direction(azimuth, elevation, normalize)
    
    def angles_to_ray_direction(
        self,
        azimuth: np.ndarray,
        elevation: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Convert (azimuth, elevation) to 3D ray direction in ENU coordinates.
        
        Args:
            azimuth: N array in radians (0=North, π/2=East)
            elevation: N array in radians (0=horizon, π/2=zenith)
            normalize: If True, return unit vectors
            
        Returns:
            rays: Nx3 array [East, North, Up] direction vectors
            
        Notes:
            Azimuth convention: 0 = North, 90° = East (clockwise from above)
            This matches standard geographic convention.
        """
        azimuth = np.atleast_1d(azimuth)
        elevation = np.atleast_1d(elevation)
        
        cos_el = np.cos(elevation)
        sin_el = np.sin(elevation)
        cos_az = np.cos(azimuth)
        sin_az = np.sin(azimuth)
        
        # Standard spherical to Cartesian (ENU convention)
        # Az=0 (North) -> Y+, Az=90° (East) -> X+
        east = cos_el * sin_az    # X
        north = cos_el * cos_az   # Y
        up = sin_el               # Z
        
        rays = np.column_stack([east, north, up])
        
        if normalize:
            norms = np.linalg.norm(rays, axis=1, keepdims=True)
            rays = rays / np.maximum(norms, 1e-10)
        
        return rays
    
    def _radius_to_zenith(self, r: np.ndarray) -> np.ndarray:
        """Convert radial distance to zenith angle based on projection model."""
        # Clip to valid range
        r = np.clip(r, 0, self.r_max)
        
        if self.projection == FisheyeProjection.EQUIDISTANT:
            # r = f * theta => theta = r / f
            theta = r / self.f
        elif self.projection == FisheyeProjection.EQUISOLID:
            # r = 2f * sin(theta/2) => theta = 2 * arcsin(r / 2f)
            theta = 2 * np.arcsin(np.clip(r / (2 * self.f), -1, 1))
        elif self.projection == FisheyeProjection.STEREOGRAPHIC:
            # r = 2f * tan(theta/2) => theta = 2 * arctan(r / 2f)
            theta = 2 * np.arctan(r / (2 * self.f))
        elif self.projection == FisheyeProjection.ORTHOGRAPHIC:
            # r = f * sin(theta) => theta = arcsin(r / f)
            theta = np.arcsin(np.clip(r / self.f, -1, 1))
        else:
            raise ValueError(f"Unknown projection: {self.projection}")
        
        return theta
    
    def _zenith_to_radius(self, theta: np.ndarray) -> np.ndarray:
        """Convert zenith angle to radial distance based on projection model."""
        theta = np.clip(theta, 0, self.max_zenith_angle)
        
        if self.projection == FisheyeProjection.EQUIDISTANT:
            r = self.f * theta
        elif self.projection == FisheyeProjection.EQUISOLID:
            r = 2 * self.f * np.sin(theta / 2)
        elif self.projection == FisheyeProjection.STEREOGRAPHIC:
            r = 2 * self.f * np.tan(theta / 2)
        elif self.projection == FisheyeProjection.ORTHOGRAPHIC:
            r = self.f * np.sin(theta)
        else:
            raise ValueError(f"Unknown projection: {self.projection}")
        
        return r
    
    def get_elevation_at_radius(self, r: float) -> float:
        """Get elevation angle (radians) at given radial distance from center."""
        zenith = self._radius_to_zenith(np.array([r]))[0]
        return np.pi / 2 - zenith
    
    def get_radius_at_elevation(self, elevation: float) -> float:
        """Get radial distance (pixels) for given elevation angle (radians)."""
        zenith = np.pi / 2 - elevation
        return self._zenith_to_radius(np.array([zenith]))[0]
    
    def is_in_fov(self, pixel_coords: np.ndarray) -> np.ndarray:
        """Check if pixel coordinates are within the field of view."""
        pixel_coords = np.atleast_2d(pixel_coords)
        dx = pixel_coords[:, 0] - self.cx
        dy = pixel_coords[:, 1] - self.cy
        r = np.sqrt(dx**2 + dy**2)
        return r <= self.r_max
    
    def create_elevation_grid(self) -> np.ndarray:
        """Create a grid of elevation angles for each pixel."""
        y, x = np.mgrid[0:self.image_height, 0:self.image_width]
        coords = np.column_stack([x.ravel(), y.ravel()])
        _, elevation = self.pixel_to_angles(coords)
        return elevation.reshape(self.image_height, self.image_width)
    
    def create_azimuth_grid(self) -> np.ndarray:
        """Create a grid of azimuth angles for each pixel."""
        y, x = np.mgrid[0:self.image_height, 0:self.image_width]
        coords = np.column_stack([x.ravel(), y.ravel()])
        azimuth, _ = self.pixel_to_angles(coords)
        return azimuth.reshape(self.image_height, self.image_width)
    
    def get_angular_resolution(self, elevation: float = np.pi/4) -> float:
        """
        Get approximate angular resolution in radians per pixel at given elevation.
        
        This is useful for estimating pointing accuracy.
        
        Args:
            elevation: Elevation angle in radians (default: 45°)
            
        Returns:
            Angular resolution in radians per pixel
        """
        # For equidistant projection, this is constant
        if self.projection == FisheyeProjection.EQUIDISTANT:
            return self.max_zenith_angle / self.r_max
        
        # For other projections, compute numerically
        r = self.get_radius_at_elevation(elevation)
        r_plus = min(r + 1, self.r_max)
        r_minus = max(r - 1, 0)
        
        el_plus = self.get_elevation_at_radius(r_plus)
        el_minus = self.get_elevation_at_radius(r_minus)
        
        return (el_minus - el_plus) / (r_plus - r_minus)


@dataclass
class CameraCalibration:
    """
    Complete calibration parameters for a fisheye sky camera.
    
    This stores both the camera model and its geographic position/orientation.
    """
    # Camera position
    latitude: float  # degrees
    longitude: float  # degrees  
    height: float  # meters above sea level
    
    # Image parameters
    image_width: int
    image_height: int
    
    # Fisheye parameters
    fov_degrees: float = 180.0
    projection: FisheyeProjection = FisheyeProjection.EQUIDISTANT
    cx: Optional[float] = None
    cy: Optional[float] = None
    
    # Orientation (needs to be calibrated, e.g., from sun position)
    azimuth_offset: float = 0.0  # Degrees: angle from image "up" to geographic North
    azimuth_clockwise: bool = True
    
    # Optional metadata
    name: str = "camera"
    
    _camera_model: Optional[FisheyeCameraModel] = field(default=None, repr=False)
    
    @property
    def camera_model(self) -> FisheyeCameraModel:
        """Get the FisheyeCameraModel for this calibration."""
        if self._camera_model is None:
            self._camera_model = FisheyeCameraModel(
                image_width=self.image_width,
                image_height=self.image_height,
                fov_degrees=self.fov_degrees,
                projection=self.projection,
                cx=self.cx,
                cy=self.cy,
                azimuth_offset=self.azimuth_offset,
                azimuth_clockwise=self.azimuth_clockwise
            )
        return self._camera_model
    
    def to_dict(self) -> dict:
        """Convert calibration to dictionary."""
        return {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'height': self.height,
            'image_width': self.image_width,
            'image_height': self.image_height,
            'fov_degrees': self.fov_degrees,
            'projection': self.projection.value,
            'cx': self.cx,
            'cy': self.cy,
            'azimuth_offset': self.azimuth_offset,
            'azimuth_clockwise': self.azimuth_clockwise,
            'name': self.name
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CameraCalibration':
        """Create calibration from dictionary."""
        data = data.copy()
        if 'projection' in data and isinstance(data['projection'], str):
            data['projection'] = FisheyeProjection(data['projection'])
        return cls(**data)


# Pre-defined calibrations for known cameras
SIRTA_IPSL_CALIBRATION = CameraCalibration(
    name="SIRTA-IPSL",
    latitude=48.7132,
    longitude=2.207,
    height=177.5,
    image_width=1024,  # Will be updated from actual image
    image_height=1024,
    fov_degrees=180.0,
    projection=FisheyeProjection.EQUIDISTANT,
    azimuth_offset=0.0,  # Needs calibration
    azimuth_clockwise=True
)

ECTL_CALIBRATION = CameraCalibration(
    name="ECTL",
    latitude=48.600518087374105,
    longitude=2.3467954996250784,
    height=90.0,
    image_width=1024,  # Will be updated from actual image
    image_height=1024,
    fov_degrees=180.0,
    projection=FisheyeProjection.EQUIDISTANT,
    azimuth_offset=0.0,  # Needs calibration
    azimuth_clockwise=True
)
