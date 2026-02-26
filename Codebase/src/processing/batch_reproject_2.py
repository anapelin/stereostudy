#!/usr/bin/env python3
"""Batch reproject SIRTA images using azimuth-zenith map and ECTL-style projection."""

import os
import sys
import glob
import numpy as np
from PIL import Image
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from pathlib import Path

# Add paths - go up to workspace root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, WORKSPACE_ROOT)


def load_sirta_calibration():
    """Load SIRTA azimuth/zenith calibration map."""
    calib_path = os.path.join(WORKSPACE_ROOT, "Codebase", "data", "azimuth_zenith_map_full_corrected.npz")
    data = np.load(calib_path)
    
    # Load and convert to radians
    azimuth_deg = data['azimuth']  # Shape: (768, 1024)
    zenith_deg = data['zenith']
    
    azimuth_rad = np.radians(azimuth_deg)
    zenith_rad = np.radians(zenith_deg)
    
    print(f"Loaded SIRTA calibration: {azimuth_rad.shape}")
    print(f"  Azimuth range: [{np.degrees(np.nanmin(azimuth_rad)):.1f}, {np.degrees(np.nanmax(azimuth_rad)):.1f}] deg")
    print(f"  Zenith range: [{np.degrees(np.nanmin(zenith_rad)):.1f}, {np.degrees(np.nanmax(zenith_rad)):.1f}] deg")
    
    return azimuth_rad, zenith_rad


def create_projection_grid(square_size_km=75, resolution=1024, cloud_height_km=10.0):
    """Create interpolation grid for projection (same as ECTL method)."""
    square_size_m = square_size_km * 1000
    cloud_height_m = cloud_height_km * 1000
    
    half_size = square_size_m / 2
    step = square_size_m / (resolution - 1)
    
    x = np.arange(-half_size, half_size + step, step)
    y = np.arange(-half_size, half_size + step, step)
    interpolation_grid_xy = np.meshgrid(x, y)
    
    r = np.sqrt(interpolation_grid_xy[0]**2 + interpolation_grid_xy[1]**2)
    interpolation_zenith = np.arctan(r / cloud_height_m)
    interpolation_azimuth = np.arctan2(interpolation_grid_xy[1], interpolation_grid_xy[0])
    interpolation_azimuth = (interpolation_azimuth - 3*np.pi/2) % (2*np.pi) - np.pi
    
    return interpolation_grid_xy, interpolation_azimuth, interpolation_zenith


def init_interpolators(azimuth_array, zenith_array, resolution=1024, max_zenith_deg=80):
    """Initialize interpolators using ECTL-style method."""
    image_size = azimuth_array.shape  # (768, 1024)
    
    # Create restriction array to filter out large zenith angles
    max_zenith_rad = np.radians(max_zenith_deg)
    restriction_array = np.where(zenith_array > max_zenith_rad, np.nan, 1)
    
    # Flatten and apply restriction
    flattened_azimuth = (restriction_array * azimuth_array).flatten()
    flattened_zenith = (restriction_array * zenith_array).flatten()
    
    # Identify and filter NaN values
    mask_nan_azimuth = np.isnan(flattened_azimuth)
    mask_nan_zenith = np.isnan(flattened_zenith)
    mask_combined_nan = mask_nan_azimuth | mask_nan_zenith
    
    filtered_azimuth = flattened_azimuth[~mask_combined_nan]
    filtered_zenith = flattened_zenith[~mask_combined_nan]
    
    # Combine filtered azimuth and zenith angles
    azimuth_zenith = np.stack([filtered_azimuth, filtered_zenith], axis=-1)
    
    # Create coordinate grid
    x_r = np.arange(image_size[0])
    y_r = np.arange(image_size[1])
    grid = np.meshgrid(x_r, y_r)
    coordinates = np.stack([grid[0].T.flatten()[~mask_combined_nan], 
                           grid[1].T.flatten()[~mask_combined_nan]], axis=-1)
    
    # Create interpolator from azimuth-zenith to pixel coordinates
    azimuth_zenith_to_pixel_raw = LinearNDInterpolator(azimuth_zenith, coordinates)
    
    print(f"Initialized interpolators:")
    print(f"  Valid pixels: {len(coordinates)} / {image_size[0] * image_size[1]}")
    print(f"  Image size: {image_size}")
    
    return azimuth_zenith_to_pixel_raw, image_size


def project_image_sirta(img_array, azimuth_array, zenith_array, 
                        square_size_km=75, resolution=1024, 
                        cloud_height_km=10.0, max_zenith_deg=80):
    """Project SIRTA image using ECTL-style method."""
    
    # Initialize interpolators
    azimuth_zenith_to_pixel_raw, image_size = init_interpolators(
        azimuth_array, zenith_array, resolution, max_zenith_deg
    )
    
    # Create projection grid
    grid_xy, interp_azimuth, interp_zenith = create_projection_grid(
        square_size_km, resolution, cloud_height_km
    )
    
    # Combine azimuth and zenith into grid
    azimuth_zenith_grid = np.stack([interp_azimuth, interp_zenith], axis=-1)
    
    # Map azimuth-zenith grid to pixel coordinates
    projected_grid = azimuth_zenith_to_pixel_raw(azimuth_zenith_grid)
    
    # Create RGB interpolator
    x_r = np.arange(img_array.shape[0])
    y_r = np.arange(img_array.shape[1])
    pixel_to_rgb = RegularGridInterpolator((x_r, y_r), img_array, 
                                           bounds_error=False, fill_value=0)
    
    # Interpolate RGB values
    projected_image = pixel_to_rgb(projected_grid)
    
    # Replace NaN with 0 and convert to uint8
    projected_image = np.nan_to_num(projected_image, nan=0.0)
    projected_image = np.clip(projected_image, 0, 255).astype(np.uint8)
    
    return projected_image


def process_single_image(input_path, output_path, azimuth_map, zenith_map):
    """Process a single image and save the reprojection."""
    try:
        # Load image
        image = np.array(Image.open(input_path))
        
        # Check if image needs resizing to match calibration
        if image.shape[:2] != azimuth_map.shape:
            # Resize to match calibration map (768, 1024)
            pil_img = Image.fromarray(image)
            image = np.array(pil_img.resize(
                (azimuth_map.shape[1], azimuth_map.shape[0]),  # (width, height)
                Image.Resampling.LANCZOS
            ))
        
        # Project image
        projected = project_image_sirta(
            image,
            azimuth_map,
            zenith_map,
            square_size_km=75,
            resolution=1024,
            cloud_height_km=10.0,
            max_zenith_deg=80
        )
        
        # Save projected image
        Image.fromarray(projected).save(output_path, quality=95)
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # Paths
    input_dir = os.path.join(WORKSPACE_ROOT, "Datasets", "gQg5IUvV", "RAW", "20250406")
    output_dir = os.path.join(WORKSPACE_ROOT, "Datasets", "gQg5IUvV", "reprojected_ectl_style_10km")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Get list of input images (only those ending in _01.jpg)
    input_images = sorted(glob.glob(os.path.join(input_dir, "*_01.jpg")))
    print(f"Found {len(input_images)} images to process (files ending in _01.jpg)")
    
    if len(input_images) == 0:
        print("No images found!")
        return
    
    # Load SIRTA calibration
    print("\nLoading SIRTA calibration...")
    azimuth_map, zenith_map = load_sirta_calibration()
    
    print(f"\nCalibration loaded:")
    print(f"  Azimuth shape: {azimuth_map.shape}")
    print(f"  Zenith shape: {zenith_map.shape}")
    
    # Process each image
    print(f"\nProcessing images...")
    success_count = 0
    
    for i, input_path in enumerate(input_images, 1):
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, filename)
        
        print(f"[{i}/{len(input_images)}] Processing {filename}...", end=" ")
        
        if process_single_image(input_path, output_path, azimuth_map, zenith_map):
            print("✓")
            success_count += 1
        else:
            print("✗")
    
    print(f"\nCompleted: {success_count}/{len(input_images)} images processed successfully")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
