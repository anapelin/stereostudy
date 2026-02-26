#!/usr/bin/env python3
"""
Reproject fisheye camera images to a common horizontal plane.

This script demonstrates the reprojection of SIRTA and Orsay camera images
to a horizontal plane at 10 km altitude, enabling direct comparison.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from scipy.interpolate import griddata
from typing import Tuple, Dict, Optional
import argparse


class PlaneReprojector:
    """
    Reprojects fisheye camera images to a common horizontal plane.
    """
    
    def __init__(self, 
                 camera_config_path: str,
                 plane_altitude_m: float = 10000,
                 plane_resolution_m: float = 50):
        """
        Initialize reprojector with camera configuration.
        
        Parameters:
        -----------
        camera_config_path : str
            Path to camera JSON config file
        plane_altitude_m : float
            Altitude of projection plane in meters (default: 10km)
        plane_resolution_m : float
            Spatial resolution of plane grid in meters (default: 50m)
        """
        self.config_path = Path(camera_config_path)
        self.plane_altitude_m = plane_altitude_m
        self.plane_resolution_m = plane_resolution_m
        
        # Load configuration
        with open(self.config_path) as f:
            self.config = json.load(f)
        
        self.camera_name = self.config['name']
        self.camera_elevation_m = self.config['location']['elevation_m']
        self.image_width = self.config['image']['width']
        self.image_height = self.config['image']['height']
        
        # Load azimuth/zenith map
        self.azimuth, self.zenith = self.load_azimuth_zenith_map()
        
        # Compute projection coordinates
        self.compute_projection_coordinates()
        
        print(f"✓ Initialized {self.camera_name}")
        print(f"  Plane altitude: {self.plane_altitude_m/1000:.1f} km")
        print(f"  Camera elevation: {self.camera_elevation_m} m")
        print(f"  Resolution: {self.plane_resolution_m} m/pixel")
    
    def load_azimuth_zenith_map(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load pre-computed azimuth and zenith arrays.
        
        Returns:
        --------
        azimuth : np.ndarray
            (H, W) array of azimuth angles in degrees
        zenith : np.ndarray
            (H, W) array of zenith angles in degrees
        """
        az_zen_path = self.config_path.parent.parent / self.config['calibration']['azimuth_zenith_map']
        
        if not az_zen_path.exists():
            raise FileNotFoundError(f"Azimuth/zenith map not found: {az_zen_path}")
        
        data = np.load(az_zen_path)
        azimuth = data['azimuth']
        zenith = data['zenith']
        
        print(f"  Loaded {az_zen_path.name}")
        print(f"    Azimuth range: [{azimuth.min():.1f}°, {azimuth.max():.1f}°]")
        print(f"    Zenith range: [{zenith.min():.1f}°, {zenith.max():.1f}°]")
        
        return azimuth, zenith
    
    def compute_projection_coordinates(self):
        """
        Compute East-North coordinates for each camera pixel on the projection plane.
        """
        # Height difference from camera to plane
        delta_h = self.plane_altitude_m - self.camera_elevation_m
        
        # Convert angles to radians
        zen_rad = np.radians(self.zenith)
        az_rad = np.radians(self.azimuth)
        
        # Distance along ray to plane (d = delta_h / cos(zenith))
        # Avoid division by zero for horizon pixels (zenith = 90°)
        cos_zen = np.cos(zen_rad)
        
        # Mask for valid pixels (not looking below horizon)
        valid_mask = cos_zen > 0.01  # zenith < ~89.4°
        
        # Distance to plane
        d = np.full_like(zen_rad, np.nan)
        d[valid_mask] = delta_h / cos_zen[valid_mask]
        
        # Horizontal distance from camera
        r = d * np.sin(zen_rad)
        
        # East-North components (note: azimuth 0° = North, 90° = East)
        self.east = r * np.sin(az_rad)
        self.north = r * np.cos(az_rad)
        self.valid_mask = valid_mask
        
        # Compute extent
        valid_east = self.east[valid_mask]
        valid_north = self.north[valid_mask]
        
        if len(valid_east) > 0:
            self.extent = {
                'east_min': np.nanmin(valid_east),
                'east_max': np.nanmax(valid_east),
                'north_min': np.nanmin(valid_north),
                'north_max': np.nanmax(valid_north)
            }
            print(f"  Projection extent:")
            print(f"    East: [{self.extent['east_min']/1000:.1f}, {self.extent['east_max']/1000:.1f}] km")
            print(f"    North: [{self.extent['north_min']/1000:.1f}, {self.extent['north_max']/1000:.1f}] km")
    
    def create_plane_grid(self, extent_km: float = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create uniform grid on projection plane.
        
        Parameters:
        -----------
        extent_km : float
            Size of grid in each direction from camera (km)
            
        Returns:
        --------
        grid_east : np.ndarray
            2D array of east coordinates (meters)
        grid_north : np.ndarray
            2D array of north coordinates (meters)
        """
        extent_m = extent_km * 1000
        
        # Create grid
        east_coords = np.arange(-extent_m, extent_m, self.plane_resolution_m)
        north_coords = np.arange(-extent_m, extent_m, self.plane_resolution_m)
        
        grid_east, grid_north = np.meshgrid(east_coords, north_coords)
        
        return grid_east, grid_north
    
    def reproject_image(self, 
                       image: np.ndarray,
                       grid_east: np.ndarray,
                       grid_north: np.ndarray,
                       method: str = 'linear') -> np.ndarray:
        """
        Reproject camera image to plane grid.
        
        Parameters:
        -----------
        image : np.ndarray
            Input camera image (H, W, 3) or (H, W)
        grid_east : np.ndarray
            2D array of east coordinates for output grid
        grid_north : np.ndarray
            2D array of north coordinates for output grid
        method : str
            Interpolation method: 'linear', 'nearest', 'cubic'
            
        Returns:
        --------
        projected_image : np.ndarray
            Reprojected image on plane grid
        """
        # Get valid source coordinates
        valid_pixels = self.valid_mask
        source_east = self.east[valid_pixels]
        source_north = self.north[valid_pixels]
        
        # Handle RGB or grayscale
        is_color = len(image.shape) == 3
        
        if is_color:
            # Process each channel separately
            h, w, c = image.shape
            projected = np.zeros((grid_east.shape[0], grid_east.shape[1], c))
            
            for ch in range(c):
                source_values = image[:, :, ch][valid_pixels]
                
                # Interpolate
                projected[:, :, ch] = griddata(
                    points=(source_east, source_north),
                    values=source_values,
                    xi=(grid_east, grid_north),
                    method=method,
                    fill_value=0
                )
            
            projected = np.clip(projected, 0, 255).astype(np.uint8)
        else:
            # Grayscale
            source_values = image[valid_pixels]
            
            projected = griddata(
                points=(source_east, source_north),
                values=source_values,
                xi=(grid_east, grid_north),
                method=method,
                fill_value=0
            )
            
            projected = np.clip(projected, 0, 255).astype(np.uint8)
        
        return projected


def visualize_reprojection(original_image: np.ndarray,
                          reprojected_image: np.ndarray,
                          camera_name: str,
                          grid_extent_km: float,
                          plane_altitude_km: float):
    """
    Visualize original and reprojected images side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original fisheye image
    axes[0].imshow(original_image)
    axes[0].set_title(f'{camera_name}\nOriginal Fisheye Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Reprojected image
    extent_m = grid_extent_km * 1000
    axes[1].imshow(reprojected_image, 
                   extent=[-extent_m/1000, extent_m/1000, -extent_m/1000, extent_m/1000],
                   origin='lower')
    axes[1].set_title(f'{camera_name}\nReprojected at {plane_altitude_km:.0f} km Altitude', 
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel('East (km)', fontsize=12)
    axes[1].set_ylabel('North (km)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    return fig


def process_sample_images(camera_name: str, 
                         config_path: Path,
                         num_samples: int = 3,
                         extent_km: float = 30,
                         plane_altitude_m: float = 10000):
    """
    Process and display sample reprojected images for a camera.
    """
    print(f"\n{'='*70}")
    print(f"Processing {camera_name}")
    print('='*70)
    
    # Initialize reprojector
    reprojector = PlaneReprojector(
        camera_config_path=config_path,
        plane_altitude_m=plane_altitude_m,
        plane_resolution_m=50
    )
    
    # Get image directory
    # Navigate from config/cameras/ up to Codebase, then to workspace root
    workspace_root = config_path.parent.parent.parent
    projected_dir_rel = reprojector.config['data']['projected_dir']
    image_dir = workspace_root / projected_dir_rel
    
    # Try to resolve the path
    if not image_dir.exists():
        # Alternative: try from workspace root directly
        image_dir = Path('/data/common/STEREOSTUDYIPSL') / projected_dir_rel.lstrip('../')
    
    if not image_dir.exists():
        print(f"✗ Image directory not found: {image_dir}")
        return
    
    # Find sample images
    image_files = sorted(list(image_dir.glob('*.jpg')))[:num_samples]
    
    if not image_files:
        print(f"✗ No images found in {image_dir}")
        return
    
    print(f"\n✓ Found {len(image_files)} sample images")
    
    # Create plane grid
    grid_east, grid_north = reprojector.create_plane_grid(extent_km=extent_km)
    print(f"✓ Created plane grid: {grid_east.shape}")
    
    # Process each sample
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing {image_path.name}...")
        
        # Load image
        original_loaded = Image.open(image_path).convert('RGB')
        
        # Resize to match calibration map dimensions if needed
        if original_loaded.size != (reprojector.image_width, reprojector.image_height):
            print(f"  Resizing from {original_loaded.size} to ({reprojector.image_width}, {reprojector.image_height})")
            original = np.array(original_loaded.resize(
                (reprojector.image_width, reprojector.image_height),
                Image.Resampling.LANCZOS
            ))
        else:
            original = np.array(original_loaded)
        
        print(f"  Image shape: {original.shape}")
        
        # Reproject
        print(f"  Reprojecting to {plane_altitude_m/1000:.0f} km plane...")
        reprojected = reprojector.reproject_image(original, grid_east, grid_north)
        print(f"  Reprojected shape: {reprojected.shape}")
        
        # Visualize
        fig = visualize_reprojection(
            original, 
            reprojected, 
            camera_name,
            extent_km,
            plane_altitude_m / 1000
        )
        plt.show()
        
        print(f"  ✓ Done")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='Reproject camera images to horizontal plane')
    parser.add_argument('--camera', choices=['sirta', 'orsay', 'both'], default='both',
                       help='Camera to process')
    parser.add_argument('--samples', type=int, default=3,
                       help='Number of sample images to process')
    parser.add_argument('--extent', type=float, default=30,
                       help='Grid extent in km from camera')
    parser.add_argument('--altitude', type=float, default=10,
                       help='Projection plane altitude in km')
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    config_dir = script_dir.parent / 'config' / 'cameras'
    
    sirta_config = config_dir / 'sirta_camera.json'
    orsay_config = config_dir / 'orsay_camera.json'
    
    print("\n" + "="*70)
    print("CAMERA IMAGE REPROJECTION TO HORIZONTAL PLANE")
    print("="*70)
    print(f"Plane altitude: {args.altitude:.1f} km")
    print(f"Grid extent: ±{args.extent:.1f} km")
    print(f"Samples per camera: {args.samples}")
    
    # Process cameras
    if args.camera in ['sirta', 'both']:
        if sirta_config.exists():
            process_sample_images(
                'SIRTA (IPSL)',
                sirta_config,
                num_samples=args.samples,
                extent_km=args.extent,
                plane_altitude_m=args.altitude * 1000
            )
        else:
            print(f"\n✗ SIRTA config not found: {sirta_config}")
    
    if args.camera in ['orsay', 'both']:
        if orsay_config.exists():
            process_sample_images(
                'Orsay (ECTL)',
                orsay_config,
                num_samples=args.samples,
                extent_km=args.extent,
                plane_altitude_m=args.altitude * 1000
            )
        else:
            print(f"\n✗ Orsay config not found: {orsay_config}")
    
    print("\n" + "="*70)
    print("✓ Processing complete")
    print("="*70)


if __name__ == "__main__":
    main()
