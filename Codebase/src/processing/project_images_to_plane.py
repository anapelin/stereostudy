#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project all-sky camera images onto a horizontal plane at 10km altitude using cheick_code calibration.

This script:
1. Loads all-sky camera images from the gQg5IUvV dataset
2. Uses the Fripon/Original calibration model from cheick_code
3. Projects each image onto a horizontal plane at 10km altitude
4. Saves the projected images

Author: Generated for STEREOSTUDYIPSL project
Date: December 2025
"""

import os
import sys
import glob
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Optional
import matplotlib.pyplot as plt

# Add cheick_code to path
CHEICK_CODE_PATH = '/data/common/STEREOSTUDYIPSL/cheick_code'
if CHEICK_CODE_PATH not in sys.path:
    sys.path.insert(0, CHEICK_CODE_PATH)

# Import calibration functions
from calibration.baseCalibration import readCalParams, Spherical2Cartesian, Cartesian2Spherical, initR
from calibration.calibrationFripon import world2image
from calibration.useCalibration import worldToImage
# Import directly using importlib to bypass __init__.py
import importlib.util
spec = importlib.util.spec_from_file_location("position", os.path.join(CHEICK_CODE_PATH, "setup_variable", "position.py"))
position_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(position_module)
readLatLon = position_module.readLatLon
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator, griddata

# Configuration
INPUT_DIR = '/data/common/STEREOSTUDYIPSL/Datasets/gQg5IUvV/RAW/20250406'
OUTPUT_DIR = '/data/common/STEREOSTUDYIPSL/Datasets/gQg5IUvV_projected_10km'
SITE = 'SIRTA'
PLANE_ALTITUDE_KM = 10.0
EARTH_RADIUS_KM = 6370.0

# Projection plane settings
PROJECTION_SIZE = 1024  # Output image size (pixels)
PROJECTION_EXTENT_KM = 75  # Size of the projection area (km x km)


class PlaneProjector:
    """Projects all-sky camera images onto a horizontal plane at specified altitude."""
    
    def __init__(self, site: str = 'SIRTA', altitude_km: float = 10.0, 
                 projection_size: int = 1024, projection_extent_km: float = 75):
        """
        Initialize the plane projector.
        
        Args:
            site: Camera site name ('SIRTA' or 'Orsay')
            altitude_km: Altitude of projection plane in km
            projection_size: Output image size in pixels
            projection_extent_km: Spatial extent of projection plane in km
        """
        self.site = site
        self.altitude_km = altitude_km
        self.projection_size = projection_size
        self.projection_extent_km = projection_extent_km
        
        # Get site location
        self.lat_site, self.lon_site = readLatLon(site, path=os.path.join(CHEICK_CODE_PATH, 'params.csv'))
        print(f"Camera site: {site} at ({self.lat_site:.6f}°N, {self.lon_site:.6f}°E)")
        
        # Create projection grid
        self._create_projection_grid()
        
    def _create_projection_grid(self):
        """Create the projection plane grid in local coordinates."""
        # Create grid from -extent/2 to +extent/2 km
        half_extent = self.projection_extent_km / 2.0
        x_coords = np.linspace(-half_extent, half_extent, self.projection_size)
        y_coords = np.linspace(-half_extent, half_extent, self.projection_size)
        
        # Create meshgrid (X = East, Y = North in local coordinates)
        self.X_grid, self.Y_grid = np.meshgrid(x_coords, y_coords)
        
        # Z is constant at the plane altitude
        self.Z_grid = np.full_like(self.X_grid, self.altitude_km)
        
        print(f"Created projection grid: {self.projection_size}x{self.projection_size} pixels")
        print(f"Spatial extent: {self.projection_extent_km}km x {self.projection_extent_km}km")
        print(f"Plane altitude: {self.altitude_km}km")
        
    def _local_to_normalized_position(self, x_km: float, y_km: float, z_km: float) -> np.ndarray:
        """
        Convert local cartesian coordinates to normalized position vector for calibration.
        
        Args:
            x_km: East coordinate in km (relative to site)
            y_km: North coordinate in km (relative to site)
            z_km: Altitude in km
            
        Returns:
            Normalized position vector [X, Y, Z] where:
            - X corresponds to -North direction
            - Y corresponds to East direction  
            - Z corresponds to Up direction
            - Vector is normalized to unit length
        """
        # Normalize the position vector
        norm = np.sqrt(x_km**2 + y_km**2 + z_km**2)
        
        # Apply the coordinate transformation used in cheick_code
        # Note: Y (North) gets negative sign in first component
        position = np.array([
            -y_km / norm,   # X component (negative North)
             x_km / norm,   # Y component (East)
             z_km / norm    # Z component (Up)
        ])
        
        return position
    
    def project_image(self, image: np.ndarray, image_shape: Optional[Tuple] = None) -> np.ndarray:
        """
        Project an all-sky image onto the horizontal plane.
        
        Args:
            image: Input all-sky camera image (H, W, C)
            image_shape: Shape tuple for calibration (default: image.shape)
            
        Returns:
            Projected image on the horizontal plane (projection_size, projection_size, C)
        """
        if image_shape is None:
            image_shape = image.shape
            
        # Initialize output image
        if len(image.shape) == 3:
            projected = np.zeros((self.projection_size, self.projection_size, image.shape[2]), dtype=image.dtype)
        else:
            projected = np.zeros((self.projection_size, self.projection_size), dtype=image.dtype)
        
        print(f"  Projecting {self.projection_size}x{self.projection_size} grid points...")
        
        # Process each pixel in the projection plane
        for i in tqdm(range(self.projection_size), desc="  Rows", leave=False):
            for j in range(self.projection_size):
                # Get local coordinates for this grid point
                x_km = self.X_grid[i, j]
                y_km = self.Y_grid[i, j]
                z_km = self.Z_grid[i, j]
                

                
                # Convert to normalized position vector
                position = self._local_to_normalized_position(x_km, y_km, z_km)


                
                # Project to image coordinates using calibration
                try:
                    px, py = worldToImage(
                        XPosition=position,
                        imageShape=np.array(image_shape),
                        zoom=False,
                        methodRead="csv",
                        site=self.site
                    )
                    
                    # Check if projection is within image bounds
                    if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                        # Bilinear interpolation
                        x0, y0 = int(px), int(py)
                        x1, y1 = min(x0 + 1, image.shape[1] - 1), min(y0 + 1, image.shape[0] - 1)
                        
                        wx = px - x0
                        wy = py - y0
                        
                        if len(image.shape) == 3:
                            # Color image
                            projected[i, j] = (
                                (1 - wx) * (1 - wy) * image[y0, x0] +
                                wx * (1 - wy) * image[y0, x1] +
                                (1 - wx) * wy * image[y1, x0] +
                                wx * wy * image[y1, x1]
                            )
                        else:
                            # Grayscale image
                            projected[i, j] = (
                                (1 - wx) * (1 - wy) * image[y0, x0] +
                                wx * (1 - wy) * image[y0, x1] +
                                (1 - wx) * wy * image[y1, x0] +
                                wx * wy * image[y1, x1]
                            )
                            
                except Exception as e:
                    # Skip points that can't be projected
                    continue
                # self.pixel_to_rgb = RegularGridInterpolator((x_r, y_r), img_array, bounds_error=False, fill_value=0)
        
        return projected
    
    def visualize_projection_grid(self, output_path: Optional[str] = None):
        """
        Visualize the projection grid for debugging.
        
        Args:
            output_path: Path to save the visualization (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot spatial extent
        axes[0].scatter(self.X_grid.flatten()[::100], self.Y_grid.flatten()[::100], 
                       c='blue', s=1, alpha=0.5)
        axes[0].set_xlabel('East (km)')
        axes[0].set_ylabel('North (km)')
        axes[0].set_title(f'Projection Plane at {self.altitude_km}km altitude')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_aspect('equal')
        
        # Plot radial distance from camera
        distances = np.sqrt(self.X_grid**2 + self.Y_grid**2 + self.Z_grid**2)
        im = axes[1].imshow(distances, cmap='viridis', origin='lower')
        axes[1].set_title('Distance from Camera (km)')
        axes[1].set_xlabel('Pixel X')
        axes[1].set_ylabel('Pixel Y')
        plt.colorbar(im, ax=axes[1], label='Distance (km)')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Grid visualization saved to: {output_path}")
        plt.show()


def process_all_images(input_dir: str, output_dir: str, pattern: str = '*_01.jpg',
                       site: str = 'SIRTA', altitude_km: float = 10.0,
                       max_images: Optional[int] = None):
    """
    Process all images in the input directory and save projected versions.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save projected images
        pattern: Glob pattern to match input images
        site: Camera site name
        altitude_km: Projection plane altitude in km
        max_images: Maximum number of images to process (None for all)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all images
    image_paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    print(f"\n{'='*70}")
    print(f"PLANE PROJECTION PIPELINE")
    print(f"{'='*70}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Pattern: {pattern}")
    print(f"Found {len(image_paths)} images to process")
    print(f"{'='*70}\n")
    
    if len(image_paths) == 0:
        print("No images found! Check the input directory and pattern.")
        return
    
    # Initialize projector
    projector = PlaneProjector(site=site, altitude_km=altitude_km)
    
    # Save grid visualization
    grid_viz_path = os.path.join(output_dir, 'projection_grid_visualization.png')
    projector.visualize_projection_grid(output_path=grid_viz_path)
    
    # Process each image
    print(f"\nProcessing images...")
    for idx, image_path in enumerate(tqdm(image_paths, desc="Overall progress")):
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"  Warning: Could not load {image_path}")
                continue
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get filename
            filename = os.path.basename(image_path)
            output_filename = filename.replace('.jpg', f'_projected_{int(altitude_km)}km.jpg')
            output_path = os.path.join(output_dir, output_filename)
            
            # Skip if already processed
            if os.path.exists(output_path):
                continue
            
            print(f"\n[{idx+1}/{len(image_paths)}] Processing: {filename}")
            
            # Project image
            projected = projector.project_image(image_rgb, image_shape=image.shape)
            
            # Convert back to BGR for saving
            projected_bgr = cv2.cvtColor(projected, cv2.COLOR_RGB2BGR)
            
            # Save projected image
            cv2.imwrite(output_path, projected_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
        except Exception as e:
            print(f"  Error processing {image_path}: {e}")
            continue
    
    print(f"\n{'='*70}")
    print(f"Processing complete!")
    print(f"Projected images saved to: {output_dir}")
    print(f"{'='*70}\n")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Project all-sky camera images onto a horizontal plane using calibration.'
    )
    parser.add_argument('--input-dir', type=str, default=INPUT_DIR,
                       help='Input directory containing images')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                       help='Output directory for projected images')
    parser.add_argument('--pattern', type=str, default='*_01.jpg',
                       help='Glob pattern to match input images')
    parser.add_argument('--site', type=str, default=SITE,
                       help='Camera site (SIRTA or Orsay)')
    parser.add_argument('--altitude', type=float, default=PLANE_ALTITUDE_KM,
                       help='Projection plane altitude in km')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images to process (for testing)')
    parser.add_argument('--projection-size', type=int, default=1024,
                       help='Output image size in pixels')
    parser.add_argument('--projection-extent', type=float, default=75,
                       help='Spatial extent of projection in km')
    
    args = parser.parse_args()
    
    # Update global settings
    global PROJECTION_SIZE, PROJECTION_EXTENT_KM
    PROJECTION_SIZE = 1024
    PROJECTION_EXTENT_KM = 75
    
    # Process images
    process_all_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        site=args.site,
        altitude_km=args.altitude,
        max_images=args.max_images
    )


if __name__ == '__main__':
    main()
