#!/usr/bin/env python3
"""Batch reproject all images from the dataset to horizontal plane at 11km altitude."""

import os
import sys
import glob
import numpy as np
import cv2
from PIL import Image
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from pathlib import Path
from typing import Tuple

# Add paths - go up to workspace root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, WORKSPACE_ROOT)

from azimuth_zenith_calibration.converter import AzimuthZenithMapper


def increase_brightness(image: np.ndarray, alpha: float = 1.0, beta: int = 50) -> np.ndarray:
    """Augments the brightness of an image by adjusting its contrast and brightness."""
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta) 
    return adjusted_image


def increase_contrast(image: np.ndarray, clip_limit: float = 3.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Increases the contrast of an image using the CLAHE algorithm."""
    # Convert the input image from BGR to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    
    # Split the LAB image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    # Create a CLAHE object with specified clip limit and tile grid size
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Apply CLAHE on the L-channel
    enhanced_l_channel = clahe.apply(l_channel)
    
    # Merge the enhanced L-channel with the original A and B channels
    merged_channels = cv2.merge([enhanced_l_channel, a_channel, b_channel])
    
    # Convert the merged channels back from LAB to BGR color space
    contrast_enhanced_image = cv2.cvtColor(merged_channels, cv2.COLOR_Lab2BGR)
    return contrast_enhanced_image


def decrease_warmth(image: np.ndarray, adjustment_value: int = 50) -> np.ndarray:
    """Decreases the warmth of an image by adjusting the blue and red channels."""
    # Split the input image into its Blue, Green, and Red channels
    blue_channel, green_channel, red_channel = cv2.split(image)

    # Decrease the blue channel's intensity to reduce warmth
    blue_channel = cv2.subtract(blue_channel, adjustment_value)  
    
    # Increase the red channel's intensity to compensate and maintain color balance
    red_channel = cv2.add(red_channel, adjustment_value) 
    
    # Clip the blue channel values to ensure they remain within [0, 255]
    blue_channel = np.clip(blue_channel, 0, 255)
    
    # Clip the red channel values to ensure they remain within [0, 255]
    red_channel = np.clip(red_channel, 0, 255)
    
    # Merge the adjusted channels back into a single image
    cooler_image = cv2.merge([blue_channel, green_channel, red_channel])
    return cooler_image


def enhance_image(image: np.ndarray) -> np.ndarray:
    """Enhances an image by increasing brightness, contrast, and decreasing warmth."""
    # Increase the brightness of the image
    brighter_image = increase_brightness(image)
    
    # Increase the contrast of the brighter image
    more_contrasted_image = increase_contrast(brighter_image)
    
    # Decrease the warmth of the more contrasted image to finalize the enhancement
    enhanced_image = decrease_warmth(more_contrasted_image)
    return enhanced_image


def filter_zenith_circular_mask(zenith_map, center_x, center_y, radius):
    """Filter zenith map to only include pixels within a circular region."""
    height, width = zenith_map.shape
    x_coords = np.arange(width)
    y_coords = np.arange(height)
    X, Y = np.meshgrid(x_coords, y_coords)
    distances = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    mask = distances <= radius
    filtered_map = zenith_map.copy()
    filtered_map[~mask] = np.pi / 2  # Set outside pixels to 90 degrees
    return filtered_map, mask


def create_projection_grid(square_size_km=75, resolution=1024, cloud_height_km=10.0):
    """Create interpolation grid for projection."""
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


def project_image_with_interpolation(img_array, azimuth_array, zenith_array, 
                                     square_size_km=75, resolution=1024, 
                                     cloud_height_km=10.0):
    """Project image using grid interpolation method."""
    # Create projection grid
    grid_xy, interp_azimuth, interp_zenith = create_projection_grid(
        square_size_km, resolution, cloud_height_km
    )
    
    azimuth_zenith_grid = np.stack([interp_azimuth, interp_zenith], axis=-1)
    
    # Flatten and filter NaNs
    flat_azimuth = azimuth_array.flatten()
    flat_zenith = zenith_array.flatten()
    mask_nan = np.isnan(flat_azimuth) | np.isnan(flat_zenith)
    
    filtered_azimuth = flat_azimuth[~mask_nan]
    filtered_zenith = flat_zenith[~mask_nan]
    azimuth_zenith = np.stack([filtered_azimuth, filtered_zenith], axis=-1)
    
    # Create coordinate grid
    x_r = np.arange(azimuth_array.shape[0])
    y_r = np.arange(azimuth_array.shape[1])
    grid = np.meshgrid(x_r, y_r)
    coordinates = np.stack([grid[0].T.flatten()[~mask_nan], grid[1].T.flatten()[~mask_nan]], axis=-1)
    
    # Create interpolator
    azimuth_zenith_to_pixel_raw = LinearNDInterpolator(azimuth_zenith, coordinates)
    projected_grid = azimuth_zenith_to_pixel_raw(azimuth_zenith_grid)
    
    # Interpolate RGB values
    pixel_to_rgb = RegularGridInterpolator((x_r, y_r), img_array, bounds_error=False, fill_value=0)
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
        
        # Convert RGB to BGR for OpenCV processing
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Apply image enhancement
        enhanced_image = enhance_image(image_bgr)
        
        # Convert back to RGB
        enhanced_image_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
        
        # Downsample enhanced image to match mapping (step=2)
        image_downsampled = enhanced_image_rgb[::2, ::2]
        
        # Project image
        projected = project_image_with_interpolation(
            image_downsampled,
            azimuth_map,
            zenith_map,
            square_size_km=75,
            resolution=1024,  # Output resolution
            cloud_height_km=10.0
        )
        
        # Save projected image
        Image.fromarray(projected).save(output_path, quality=95)
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def main():
    # Paths
    input_dir = "/data/common/STEREOSTUDYIPSL/Datasets/gQg5IUvV/RAW/20250406"
    output_dir = "/data/common/STEREOSTUDYIPSL/Datasets/gQg5IUvV/reprojected_10km"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Get list of input images (only those ending in _01.jpg)
    input_images = sorted(glob.glob(os.path.join(input_dir, "*_01.jpg")))
    print(f"Found {len(input_images)} images to process (files ending in _01.jpg)")
    
    if len(input_images) == 0:
        print("No images found!")
        return
    
    # Initialize mapper and generate calibration once
    print("\nInitializing azimuth/zenith mapper...")
    image_shape = (768, 1024)  # Original image shape
    mapper = AzimuthZenithMapper(site="SIRTA_W", image_shape=image_shape)
    
    print("Generating azimuth/zenith mapping...")
    mapping = mapper.generate_mapping(step=2)  # Downsample by factor of 2
    
    azimuth_map = mapping['azimuth']
    zenith_map = mapping['zenith']
    
    # Apply circular filter
    print("Applying circular filter to zenith map...")
    center_x = 512 / 2  # Adjusted for step=2
    center_y = 384 / 2
    radius = 380 / 2
    
    zenith_map, _ = filter_zenith_circular_mask(zenith_map, center_x, center_y, radius)
    print(f"Zenith map filtered with circle at ({center_x}, {center_y}), radius={radius}px")
    
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
