#!/usr/bin/env python3
"""Test plane projection at 11km altitude using azimuth/zenith calibration.

Coordinate Convention:
- Image indexing: image[row, col] where row=vertical (y), col=horizontal (x)
- Standard CV convention used here: x=horizontal, y=vertical
- The converter module handles the translation to cheick_code's inverted convention
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
import math
from geographiclib import geodesic

# Add paths
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "cheick_code"))
sys.path.insert(0, os.path.join(REPO_ROOT, "azimuth_zenith_calibration"))

from azimuth_zenith_calibration.converter import AzimuthZenithMapper


# Camera location constants (SIRTA)
SIRTA_LONGITUDE = 2.208
SIRTA_LATITUDE = 48.713
SIRTA_HEIGHT = 160  # meters above sea level


def calculate_latitude_longitude(azimuth_deg, zenith_deg, target_altitude_m, 
                                 camera_lon=SIRTA_LONGITUDE, camera_lat=SIRTA_LATITUDE, 
                                 camera_height=SIRTA_HEIGHT):
    """
    Calculate latitude and longitude for a given azimuth and zenith angle at a target altitude.
    
    Parameters
    ----------
    azimuth_deg : float
        Azimuth angle in degrees (0=North, 90=East, -90=West, ±180=South)
    zenith_deg : float
        Zenith angle in degrees (0=straight up, 90=horizon)
    target_altitude_m : float
        Target altitude in meters above sea level
    camera_lon, camera_lat : float
        Camera location (longitude, latitude)
    camera_height : float
        Camera height in meters above sea level
    
    Returns
    -------
    lat, lon : float
        Target latitude and longitude in degrees
    """
    wgs84 = geodesic.Geodesic.WGS84
    
    # Convert zenith to elevation angle
    elevation_angle_deg = 90 - zenith_deg
    elevation_angle_rad = math.radians(elevation_angle_deg)
    
    # Altitude difference
    delta_altitude_m = target_altitude_m - camera_height
    
    # Distance on surface: distance = altitude / tan(elevation)
    distance_on_surface = delta_altitude_m / math.tan(elevation_angle_rad)
    
    # Solve forward geodesic problem
    direct_result = wgs84.Direct(camera_lat, camera_lon, azimuth_deg, distance_on_surface)
    
    return direct_result['lat2'], direct_result['lon2']


def filter_zenith_circular_mask(zenith_map, center_x, center_y, radius):
    """
    Filter zenith map to only include pixels within a circular region.
    
    Parameters
    ----------
    zenith_map : np.ndarray
        Original zenith angle map in radians
    center_x : float
        Horizontal center coordinate (pixels)
    center_y : float
        Vertical center coordinate (pixels)
    radius : float
        Radius of the circular region (pixels)
    
    Returns
    -------
    filtered_map : np.ndarray
        Zenith map with values outside the circle set to 90 degrees (pi/2 radians)
    mask : np.ndarray (bool)
        Boolean mask indicating valid pixels (True inside circle, False outside)
    """
    # Get image dimensions
    height, width = zenith_map.shape
    
    # Create coordinate grids
    x_coords = np.arange(width)
    y_coords = np.arange(height)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Calculate distance from center for each pixel
    distances = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # Create circular mask
    mask = distances <= radius
    
    # Create filtered map
    filtered_map = zenith_map.copy()
    filtered_map[~mask] = np.pi / 2  # Set outside pixels to 90 degrees
    
    return filtered_map, mask


def create_projection_grid(square_size_km=75, resolution=1024, cloud_height_km=11.0):
    """
    Create interpolation grid for projection (similar to init_grid_interpolation).
    
    Parameters
    ----------
    square_size_km : float
        Size of the square in km
    resolution : int
        Number of pixels in the output grid
    cloud_height_km : float
        Altitude of the projection plane in km
    
    Returns
    -------
    interpolation_grid_xy : tuple of np.ndarray
        X and Y coordinate grids in meters
    interpolation_azimuth : np.ndarray
        Azimuth angles for each grid point (radians)
    interpolation_zenith : np.ndarray
        Zenith angles for each grid point (radians)
    """
    # Convert to meters
    square_size_m = square_size_km * 1000
    cloud_height_m = cloud_height_km * 1000
    
    half_size = square_size_m / 2
    step = square_size_m / (resolution - 1)
    
    # Create coordinate arrays
    x = np.arange(-half_size, half_size + step, step)
    y = np.arange(-half_size, half_size + step, step)
    interpolation_grid_xy = np.meshgrid(x, y)
    
    # Calculate zenith and azimuth
    r = np.sqrt(interpolation_grid_xy[0]**2 + interpolation_grid_xy[1]**2)
    interpolation_zenith = np.arctan(r / cloud_height_m)
    interpolation_azimuth = np.arctan2(interpolation_grid_xy[1], interpolation_grid_xy[0])
    
    # Adjust azimuth to match convention (0=North, 90=East)
    interpolation_azimuth = (interpolation_azimuth - 3*np.pi/2) % (2*np.pi) - np.pi
    
    return interpolation_grid_xy, interpolation_azimuth, interpolation_zenith


def project_image_with_interpolation(img_array, azimuth_array, zenith_array, 
                                     square_size_km=75, resolution=1024, 
                                     cloud_height_km=11.0):
    """
    Project image using grid interpolation method (similar to project_image).
    
    Parameters
    ----------
    img_array : np.ndarray
        Input image (H, W, C)
    azimuth_array : np.ndarray
        Azimuth angle for each pixel (radians)
    zenith_array : np.ndarray
        Zenith angle for each pixel (radians)
    square_size_km : float
        Size of output grid in km
    resolution : int
        Output resolution
    cloud_height_km : float
        Projection altitude in km
    
    Returns
    -------
    projected_image : np.ndarray
        Projected image
    grid_xy : tuple
        X and Y coordinate grids in km
    """
    # Create projection grid
    grid_xy, interp_azimuth, interp_zenith = create_projection_grid(
        square_size_km, resolution, cloud_height_km
    )
    
    # Stack azimuth and zenith into grid
    azimuth_zenith_grid = np.stack([interp_azimuth, interp_zenith], axis=-1)
    
    # Flatten azimuth and zenith arrays, filtering out NaNs
    flat_azimuth = azimuth_array.flatten()
    flat_zenith = zenith_array.flatten()
    
    mask_nan = np.isnan(flat_azimuth) | np.isnan(flat_zenith)
    filtered_azimuth = flat_azimuth[~mask_nan]
    filtered_zenith = flat_zenith[~mask_nan]
    azimuth_zenith = np.stack([filtered_azimuth, filtered_zenith], axis=-1)
    
    # Create coordinate grid for raw image
    x_r = np.arange(azimuth_array.shape[0])
    y_r = np.arange(azimuth_array.shape[1])
    grid = np.meshgrid(x_r, y_r)
    coordinates = np.stack([grid[0].T.flatten()[~mask_nan], grid[1].T.flatten()[~mask_nan]], axis=-1)
    
    # Create interpolator from (azimuth, zenith) to raw pixel coordinates
    azimuth_zenith_to_pixel_raw = LinearNDInterpolator(azimuth_zenith, coordinates)
    
    # Map projection grid to raw pixel coordinates
    projected_grid = azimuth_zenith_to_pixel_raw(azimuth_zenith_grid)
    
    # Interpolate RGB values
    pixel_to_rgb = RegularGridInterpolator((x_r, y_r), img_array, bounds_error=False, fill_value=0)
    projected_image = pixel_to_rgb(projected_grid).astype(np.uint8)
    
    # Return projected image and grid in km
    grid_xy_km = (grid_xy[0] / 1000, grid_xy[1] / 1000)
    
    return projected_image, grid_xy_km


def project_to_plane(azimuth, zenith, altitude_km=11.0, camera_altitude_km=0.16, max_zenith_deg=85.0):
    """
    Project pixels to a horizontal plane at given altitude.
    
    Parameters
    ----------
    azimuth : np.ndarray
        Azimuth angles in radians
    zenith : np.ndarray
        Zenith angles in radians (0 = straight up, π/2 = horizon)
    altitude_km : float
        Altitude of the projection plane in km
    camera_altitude_km : float
        Camera altitude above sea level in km (SIRTA ≈ 160m)
    max_zenith_deg : float
        Maximum zenith angle in degrees to include (crop beyond this)
    
    Returns
    -------
    X, Y : np.ndarray
        Coordinates in km (North, East) relative to camera
    valid : np.ndarray (bool)
        Mask indicating valid projections
    """
    # Direction vectors (unit sphere)
    # Using Simon's convention from the calibration
    X_dir = -np.cos(azimuth) * np.sin(zenith)
    Y_dir = np.sin(azimuth) * np.sin(zenith)
    Z_dir = np.cos(zenith)
    
    # Ray equation: P = camera_pos + t * direction
    # We want Z = altitude_km
    # camera_altitude_km + t * Z_dir = altitude_km
    
    target_height = altitude_km - camera_altitude_km
    
    # Avoid division by zero and crop at max zenith angle
    max_zenith_rad = np.radians(max_zenith_deg)
    valid = (Z_dir > 1e-6) & (zenith < max_zenith_rad)
    
    # Calculate parameter t
    t = np.full_like(zenith, np.nan)
    t[valid] = target_height / Z_dir[valid]
    
    # Check if ray goes upward
    valid &= (t > 0)
    
    # Calculate intersection
    X = t * X_dir  # North (km)
    Y = t * Y_dir  # East (km)
    
    return X, Y, valid


def project_image_to_plane(image, azimuth_map, zenith_map, altitude_km=11.0,
                          grid_resolution_km=0.1, grid_extent_km=50, max_zenith_deg=80.0):
    """
    Project entire image to a horizontal plane at given altitude.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (H, W, C)
    azimuth_map : np.ndarray
        Azimuth angle for each pixel
    zenith_map : np.ndarray
        Zenith angle for each pixel
    altitude_km : float
        Altitude of projection plane
    grid_resolution_km : float
        Resolution of output grid in km
    grid_extent_km : float
        Half-width of output grid in km
    max_zenith_deg : float
        Maximum zenith angle in degrees to include
    
    Returns
    -------
    projected : np.ndarray
        Projected image on the plane
    grid_X, grid_Y : np.ndarray
        Coordinate arrays for the grid
    """
    # Project all pixels to plane
    X, Y, valid = project_to_plane(azimuth_map, zenith_map, altitude_km, max_zenith_deg=max_zenith_deg)
    
    # Create output grid
    n_points = int(2 * grid_extent_km / grid_resolution_km)
    grid_X = np.linspace(-grid_extent_km, grid_extent_km, n_points)
    grid_Y = np.linspace(-grid_extent_km, grid_extent_km, n_points)
    
    # Initialize output
    if len(image.shape) == 3:
        projected = np.zeros((n_points, n_points, image.shape[2]))
    else:
        projected = np.zeros((n_points, n_points))
    count = np.zeros((n_points, n_points))
    
    # Map each valid pixel to grid
    valid_mask = valid & ~np.isnan(X) & ~np.isnan(Y)
    valid_mask &= (np.abs(X) < grid_extent_km) & (np.abs(Y) < grid_extent_km)
    
    if np.any(valid_mask):
        # Find grid indices
        i_indices = np.searchsorted(grid_X, X[valid_mask])
        j_indices = np.searchsorted(grid_Y, Y[valid_mask])
        
        # Clip to valid range
        i_indices = np.clip(i_indices, 0, n_points - 1)
        j_indices = np.clip(j_indices, 0, n_points - 1)
        
        # Get pixel values
        y_coords, x_coords = np.where(valid_mask)
        
        # Accumulate values
        for idx in range(len(y_coords)):
            y_px, x_px = y_coords[idx], x_coords[idx]
            i_grid, j_grid = i_indices[idx], j_indices[idx]
            
            if len(image.shape) == 3:
                projected[j_grid, i_grid] += image[y_px, x_px]
            else:
                projected[j_grid, i_grid] += image[y_px, x_px]
            count[j_grid, i_grid] += 1
    
    # Average pixels in same grid cell
    mask = count > 0
    if len(image.shape) == 3:
        for c in range(projected.shape[2]):
            projected[mask, c] /= count[mask]
    else:
        projected[mask] /= count[mask]
    
    return projected, grid_X, grid_Y


def main():
    # Load image
    image_path = "/data/common/STEREOSTUDYIPSL/Datasets/gQg5IUvV/srf02_0a_skyimgLz2_v01_20250406_044600_853/20250406/20250406044600_01.jpg"
    print(f"Loading image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    image = np.array(Image.open(image_path))
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")
    
    # Initialize mapper
    print("\nInitializing azimuth/zenith mapper for SIRTA_W...")
    image_shape = image.shape[:2]
    mapper = AzimuthZenithMapper(site="SIRTA_W", image_shape=image_shape)
    
    # Generate mapping (use step=2 for faster processing)
    print("Generating azimuth/zenith mapping (this may take a moment)...")
    mapping = mapper.generate_mapping(step=2)
    
    azimuth_map = mapping['azimuth']
    zenith_map = mapping['zenith']
    x_grid = mapping['x']
    y_grid = mapping['y']
    
    print(f"Mapping shape: {azimuth_map.shape}")
    print(f"Azimuth range: [{np.nanmin(azimuth_map):.3f}, {np.nanmax(azimuth_map):.3f}] rad")
    print(f"Zenith range (before filtering): [{np.nanmin(zenith_map):.3f}, {np.nanmax(zenith_map):.3f}] rad")
    
    # Apply circular filter to zenith map
    print("\nApplying circular filter to zenith map...")
    center_x = 512 / 2  # Adjusted for step=2 downsampling
    center_y = 384 / 2  # Adjusted for step=2 downsampling
    radius = 380 / 2    # Adjusted for step=2 downsampling
    
    zenith_map, circle_mask = filter_zenith_circular_mask(zenith_map, center_x, center_y, radius)
    print(f"Center: ({center_x}, {center_y}) pixels (in downsampled coordinates)")
    print(f"Radius: {radius} pixels")
    print(f"Pixels inside circle: {np.sum(circle_mask)} ({100*np.sum(circle_mask)/circle_mask.size:.1f}%)")
    print(f"Zenith range (after filtering): [{np.degrees(np.nanmin(zenith_map[circle_mask])):.1f}, {np.degrees(np.nanmax(zenith_map[circle_mask])):.1f}] deg")
    
    # Downsample image to match mapping
    image_downsampled = image[::2, ::2]
    
    # Project using grid interpolation method
    print("\nProjecting image using grid interpolation method...")
    square_size_km = 75  # 75km x 75km grid
    output_resolution = 1024  # output image size
    altitude_km = 11.0
    
    projected, grid_xy_km = project_image_with_interpolation(
        image_downsampled,
        azimuth_map,
        zenith_map,
        square_size_km=square_size_km,
        resolution=output_resolution,
        cloud_height_km=altitude_km
    )
    
    # Extract grid coordinates for plotting
    grid_X_km = grid_xy_km[0][0, :] / 1  # X coordinates (East-West)
    grid_Y_km = grid_xy_km[1][:, 0] / 1  # Y coordinates (North-South)
    
    print(f"Projected image shape: {projected.shape}")
    print(f"Grid extent: {grid_X_km[0]:.1f} to {grid_X_km[-1]:.1f} km (East-West)")
    print(f"             {grid_Y_km[0]:.1f} to {grid_Y_km[-1]:.1f} km (North-South)")
    
    # Calculate lat/lon for a sample point (center of image)
    center_azimuth_deg = 0  # North
    center_zenith_deg = 45  # 45 degrees from vertical
    sample_lat, sample_lon = calculate_latitude_longitude(
        center_azimuth_deg, center_zenith_deg, altitude_km * 1000
    )
    print(f"\nSample lat/lon calculation:")
    print(f"  Azimuth: {center_azimuth_deg}° (North), Zenith: {center_zenith_deg}°")
    print(f"  At {altitude_km}km altitude: ({sample_lat:.6f}°, {sample_lon:.6f}°)")
    
    # Plot side by side
    print("\nCreating visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Fisheye Image', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('X (pixels)')
    # Projected image
    if len(projected.shape) == 3:
        projected_display = projected.astype(np.uint8)
    else:
        projected_display = projected
    
    im = axes[1].imshow(
        projected,
        extent=[grid_X_km[0], grid_X_km[-1], grid_Y_km[0], grid_Y_km[-1]],
        origin='lower',
        aspect='equal'
    )
    axes[1].set_title('Projected to Horizontal Plane at 11km Altitude\n(Using Grid Interpolation)', 
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel('East (km)', fontsize=12)
    axes[1].set_ylabel('North (km)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add text annotations
    axes[1].text(0.02, 0.98, 'Camera location', 
                transform=axes[1].transAxes,
                fontsize=10, color='red',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_path = "/data/common/STEREOSTUDYIPSL/plane_projection_11km.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    
    # Show plot
    plt.show()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
