#!/usr/bin/env python3
"""
Project flight positions onto camera images for both SIRTA (IPSL) and Orsay (ECTL) cameras.

This script:
1. Loads filtered flight data (from ADS-B)
2. Converts lat/lon/altitude to azimuth/zenith relative to each camera
3. Converts azimuth/zenith to pixel coordinates using the Fripon calibration model
4. Creates visualizations showing flight positions on camera images

Author: Copilot
Date: 2025
"""

import os
import sys
import math
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Any
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    plt = None

try:
    from geographiclib import geodesic
except ImportError:
    geodesic = None
    print("Warning: geographiclib not installed. Install with: pip install geographiclib")

# Add cheick_code to path for calibration functions
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_CURRENT_DIR)
_CHEICK_CODE = os.path.join(_REPO_ROOT, "cheick_code")
if _CHEICK_CODE not in sys.path:
    sys.path.insert(0, _CHEICK_CODE)

# Import calibration functions
try:
    from calibration.calibrationFripon import (
        readCalParams, 
        invModel,
        Cartesian2Spherical
    )
    from calibration.baseCalibration import readCalParams as readCalParamsBase
    CALIBRATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import calibration functions: {e}")
    CALIBRATION_AVAILABLE = False


@dataclass
class CameraConfig:
    """Configuration for a ground-based camera."""
    name: str
    site_code: str  # Code used in calibration params (SIRTA_W, Orsay)
    latitude: float  # Camera latitude in degrees
    longitude: float  # Camera longitude in degrees
    height: float  # Camera height above ground in meters
    image_shape: Tuple[int, int]  # (height, width) of images
    
    
# Camera configurations
CAMERAS = {
    'SIRTA': CameraConfig(
        name='SIRTA (IPSL)',
        site_code='SIRTA_W',
        latitude=48.713,
        longitude=2.208,
        height=156,  # SIRTA altitude
        image_shape=(768, 1024)  # Standard SIRTA image size
    ),
    'Orsay': CameraConfig(
        name='Orsay (ECTL)',
        site_code='Orsay',
        latitude=48.706433,
        longitude=2.179331,
        height=90,  # Approximate height
        image_shape=(1280, 960)  # Standard Orsay image size
    )
}


def calculate_azimuth_zenith(
    camera_lat: float, 
    camera_lon: float, 
    camera_height: float,
    target_lat: float, 
    target_lon: float, 
    target_altitude_m: float
) -> Tuple[float, float]:
    """
    Calculate azimuth and zenith angles from camera to a target position.
    
    Parameters
    ----------
    camera_lat : float
        Camera latitude in degrees
    camera_lon : float
        Camera longitude in degrees  
    camera_height : float
        Camera height above ground in meters
    target_lat : float
        Target latitude in degrees
    target_lon : float
        Target longitude in degrees
    target_altitude_m : float
        Target altitude in meters
        
    Returns
    -------
    Tuple[float, float]
        (azimuth, zenith) in degrees
        Azimuth: 0° = North, 90° = East, 180° = South, 270° = West
        Zenith: 0° = overhead, 90° = horizon
    """
    if geodesic is None:
        raise ImportError("geographiclib is required for azimuth/zenith calculation")
    
    wgs84 = geodesic.Geodesic.WGS84
    
    # Compute the geodesic inverse problem between the two points
    inverse_coords = wgs84.Inverse(camera_lat, camera_lon, target_lat, target_lon)
    
    # Azimuth from camera to target
    azimuth = inverse_coords['azi1']
    
    # Distance between points on the surface (meters)
    distance_on_surface = inverse_coords['s12']
    
    # Difference in altitude
    delta_altitude_m = target_altitude_m - camera_height
    
    if delta_altitude_m <= 0:
        # Target is below or at camera level - not visible in sky
        return azimuth, 90.0
    
    # Calculate the straight-line distance using Pythagorean theorem
    straight_distance = math.sqrt(distance_on_surface**2 + delta_altitude_m**2)
    
    # Elevation angle
    elevation_angle = math.degrees(math.asin(delta_altitude_m / straight_distance))
    
    # Zenith angle is 90 degrees minus the elevation angle
    zenith = 90 - elevation_angle
    
    return azimuth, zenith


def azimuth_zenith_to_direction_vector(azimuth_deg: float, zenith_deg: float) -> np.ndarray:
    """
    Convert azimuth and zenith angles to a 3D unit direction vector.
    
    The coordinate system used by cheick_code calibration:
    - X: points North
    - Y: points East  
    - Z: points Up (zenith)
    
    Parameters
    ----------
    azimuth_deg : float
        Azimuth angle in degrees (0° = North, 90° = East)
    zenith_deg : float
        Zenith angle in degrees (0° = overhead, 90° = horizon)
        
    Returns
    -------
    np.ndarray
        3D unit vector [X, Y, Z]
    """
    az_rad = math.radians(azimuth_deg)
    zen_rad = math.radians(zenith_deg)
    
    # Convert to Cartesian coordinates
    # X = sin(zenith) * cos(azimuth) - North component
    # Y = sin(zenith) * sin(azimuth) - East component
    # Z = cos(zenith) - Up component
    x = math.sin(zen_rad) * math.cos(az_rad)
    y = math.sin(zen_rad) * math.sin(az_rad)
    z = math.cos(zen_rad)
    
    return np.array([x, y, z])


def world_to_image_fripon(
    direction_vector: np.ndarray,
    site: str,
    image_shape: Tuple[int, int]
) -> Tuple[Optional[float], Optional[float]]:
    """
    Convert a 3D direction vector to image pixel coordinates using Fripon calibration.
    
    Parameters
    ----------
    direction_vector : np.ndarray
        3D unit direction vector [X, Y, Z]
    site : str
        Camera site code ('SIRTA_W' or 'Orsay')
    image_shape : Tuple[int, int]
        Image shape (height, width)
        
    Returns
    -------
    Tuple[Optional[float], Optional[float]]
        (x, y) pixel coordinates or (None, None) if conversion fails
    """
    if not CALIBRATION_AVAILABLE:
        return None, None
    
    try:
        # Read calibration parameters
        params = readCalParams(site=site)
        if params is None or len(params) != 6:
            print(f"Warning: Invalid calibration params for {site}")
            return None, None
            
        b, x0, y0, theta, K1, phi = params
        
        # Prepare the position vector (shape: 3 x 1)
        Px = direction_vector.reshape(3, 1)
        
        # Call the inverse model to get pixel coordinates
        x, y = invModel(Px, b, x0, y0, theta, K1, phi, site=site.replace('_W', ''))
        
        # Extract scalar values
        x_val = float(x[0]) if hasattr(x, '__len__') else float(x)
        y_val = float(y[0]) if hasattr(y, '__len__') else float(y)
        
        # Check if coordinates are within image bounds
        height, width = image_shape
        if 0 <= x_val < height and 0 <= y_val < width:
            return x_val, y_val
        else:
            return None, None
            
    except Exception as e:
        print(f"Error in world_to_image: {e}")
        return None, None


def project_flight_to_camera(
    flight_lat: float,
    flight_lon: float,
    flight_alt_m: float,
    camera: CameraConfig
) -> Dict[str, Any]:
    """
    Project a single flight position onto a camera image.
    
    Parameters
    ----------
    flight_lat : float
        Flight latitude in degrees
    flight_lon : float
        Flight longitude in degrees
    flight_alt_m : float
        Flight altitude in meters
    camera : CameraConfig
        Camera configuration
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing projection results:
        - azimuth: Azimuth angle in degrees
        - zenith: Zenith angle in degrees
        - pixel_x: X pixel coordinate (or None)
        - pixel_y: Y pixel coordinate (or None)
        - in_fov: Whether the point is in the camera's field of view
    """
    # Calculate azimuth and zenith
    azimuth, zenith = calculate_azimuth_zenith(
        camera.latitude, camera.longitude, camera.height,
        flight_lat, flight_lon, flight_alt_m
    )
    
    result = {
        'azimuth': azimuth,
        'zenith': zenith,
        'pixel_x': None,
        'pixel_y': None,
        'in_fov': False
    }
    
    # Only project if zenith angle is reasonable (< 85 degrees)
    if zenith >= 85:
        return result
    
    # Convert to direction vector
    direction = azimuth_zenith_to_direction_vector(azimuth, zenith)
    
    # Project to image coordinates
    px, py = world_to_image_fripon(direction, camera.site_code, camera.image_shape)
    
    if px is not None and py is not None:
        result['pixel_x'] = px
        result['pixel_y'] = py
        result['in_fov'] = True
    
    return result


def project_flights_dataframe(
    df: pd.DataFrame,
    camera: CameraConfig
) -> pd.DataFrame:
    """
    Project all flights in a DataFrame onto a camera image.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: latitude, longitude, altitude (in feet)
    camera : CameraConfig
        Camera configuration
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with added columns for projection results
    """
    print(f"\nProjecting {len(df)} flight positions onto {camera.name}...")
    
    # Convert altitude from feet to meters if needed
    if 'altitude' in df.columns and df['altitude'].max() > 20000:
        # Likely in feet
        alt_m = df['altitude'] * 0.3048
    else:
        alt_m = df['altitude']
    
    # Project each point
    results = []
    for idx, row in df.iterrows():
        proj = project_flight_to_camera(
            row['latitude'],
            row['longitude'],
            alt_m.loc[idx],
            camera
        )
        results.append(proj)
    
    # Add results to DataFrame
    proj_df = pd.DataFrame(results)
    prefix = camera.site_code.replace('_W', '').lower()
    proj_df.columns = [f'{prefix}_{col}' for col in proj_df.columns]
    
    result_df = pd.concat([df.reset_index(drop=True), proj_df], axis=1)
    
    # Count visible points
    in_fov_col = f'{prefix}_in_fov'
    visible = result_df[in_fov_col].sum()
    print(f"  - {visible} / {len(df)} points are visible in {camera.name}")
    
    return result_df


def visualize_flight_projections(
    df: pd.DataFrame,
    camera: CameraConfig,
    title: str = None,
    output_path: str = None,
    background_image: np.ndarray = None
):
    """
    Create a visualization of flight positions projected onto a camera image.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with projection columns
    camera : CameraConfig
        Camera configuration
    title : str, optional
        Plot title
    output_path : str, optional
        Path to save the plot
    background_image : np.ndarray, optional
        Background image to display
    """
    if plt is None:
        print("Warning: matplotlib not available for visualization")
        return
    
    prefix = camera.site_code.replace('_W', '').lower()
    px_col = f'{prefix}_pixel_x'
    py_col = f'{prefix}_pixel_y'
    in_fov_col = f'{prefix}_in_fov'
    
    # Filter to visible points
    visible_df = df[df[in_fov_col] == True].copy()
    
    if len(visible_df) == 0:
        print(f"No visible flight positions for {camera.name}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Show background image if provided
    if background_image is not None:
        ax.imshow(background_image)
    else:
        # Create a blank image
        height, width = camera.image_shape
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # Flip Y axis for image coordinates
        ax.set_facecolor('lightgray')
    
    # Color points by zenith angle
    zenith_col = f'{prefix}_zenith'
    scatter = ax.scatter(
        visible_df[py_col],  # Note: y coordinate is horizontal in image
        visible_df[px_col],  # Note: x coordinate is vertical in image
        c=visible_df[zenith_col],
        cmap='viridis_r',
        s=20,
        alpha=0.6,
        edgecolors='white',
        linewidths=0.5
    )
    
    plt.colorbar(scatter, ax=ax, label='Zenith Angle (degrees)')
    
    # Add title
    if title is None:
        title = f'Flight Positions Projected onto {camera.name}'
    ax.set_title(title)
    ax.set_xlabel('Image X (pixels)')
    ax.set_ylabel('Image Y (pixels)')
    
    # Add statistics
    stats_text = (
        f"Visible: {len(visible_df)} / {len(df)} positions\n"
        f"Zenith range: {visible_df[zenith_col].min():.1f}° - {visible_df[zenith_col].max():.1f}°"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    
    plt.close()


def create_stereo_comparison(
    df: pd.DataFrame,
    output_path: str = None
):
    """
    Create a side-by-side comparison of flight projections on both cameras.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with projection columns for both cameras
    output_path : str, optional
        Path to save the plot
    """
    if plt is None:
        print("Warning: matplotlib not available for visualization")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for idx, (camera_key, camera) in enumerate(CAMERAS.items()):
        ax = axes[idx]
        prefix = camera.site_code.replace('_W', '').lower()
        
        px_col = f'{prefix}_pixel_x'
        py_col = f'{prefix}_pixel_y'
        in_fov_col = f'{prefix}_in_fov'
        zenith_col = f'{prefix}_zenith'
        
        # Filter to visible points
        if in_fov_col in df.columns:
            visible_df = df[df[in_fov_col] == True].copy()
        else:
            visible_df = pd.DataFrame()
        
        height, width = camera.image_shape
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_facecolor('lightgray')
        
        if len(visible_df) > 0:
            scatter = ax.scatter(
                visible_df[py_col],
                visible_df[px_col],
                c=visible_df[zenith_col],
                cmap='viridis_r',
                s=30,
                alpha=0.7,
                edgecolors='white',
                linewidths=0.5
            )
            plt.colorbar(scatter, ax=ax, label='Zenith (°)')
            
            stats = f"Visible: {len(visible_df)}\nZenith: {visible_df[zenith_col].min():.1f}°-{visible_df[zenith_col].max():.1f}°"
        else:
            stats = "No visible positions"
        
        ax.set_title(f'{camera.name}')
        ax.set_xlabel('Image X (pixels)')
        ax.set_ylabel('Image Y (pixels)')
        ax.text(0.02, 0.98, stats, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Flight Positions Projected onto Both Cameras', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved stereo comparison to: {output_path}")
    
    plt.close()


def main():
    """Main function to project flights onto camera images."""
    # Paths
    flights_dir = Path('/data/common/STEREOSTUDYIPSL/Flights')
    filtered_file = flights_dir / '2025-04-06_filtered.parquet'
    
    if not filtered_file.exists():
        print(f"Error: Filtered flight file not found: {filtered_file}")
        print("Please run process_flights.py first to generate the filtered data.")
        return
    
    # Load filtered flight data
    print(f"Loading filtered flight data from: {filtered_file}")
    df = pd.read_parquet(filtered_file)
    print(f"Loaded {len(df)} flight positions")
    
    # Print data info
    print(f"\nData columns: {list(df.columns)}")
    print(f"Altitude range: {df['altitude'].min():.0f} - {df['altitude'].max():.0f} feet")
    print(f"Lat range: {df['latitude'].min():.4f} - {df['latitude'].max():.4f}")
    print(f"Lon range: {df['longitude'].min():.4f} - {df['longitude'].max():.4f}")
    
    # Project onto both cameras
    result_df = df.copy()
    for camera_key, camera in CAMERAS.items():
        result_df = project_flights_dataframe(result_df, camera)
    
    # Save projected data
    output_file = flights_dir / '2025-04-06_projected.parquet'
    result_df.to_parquet(output_file, index=False)
    print(f"\nSaved projected data to: {output_file}")
    
    # Create summary
    print("\n=== Projection Summary ===")
    for camera_key, camera in CAMERAS.items():
        prefix = camera.site_code.replace('_W', '').lower()
        in_fov_col = f'{prefix}_in_fov'
        if in_fov_col in result_df.columns:
            visible = result_df[in_fov_col].sum()
            print(f"{camera.name}: {visible} / {len(result_df)} positions visible ({100*visible/len(result_df):.1f}%)")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    for camera_key, camera in CAMERAS.items():
        output_path = flights_dir / f'flight_projections_{camera_key.lower()}.png'
        visualize_flight_projections(
            result_df,
            camera,
            output_path=str(output_path)
        )
    
    # Create stereo comparison
    stereo_output = flights_dir / 'flight_projections_stereo.png'
    create_stereo_comparison(result_df, output_path=str(stereo_output))
    
    print("\nDone!")
    
    return result_df


if __name__ == '__main__':
    main()
