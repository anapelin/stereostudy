#!/usr/bin/env python3
"""
Reproject IPSL camera images using ground_camera_projector.

This script uses the GroundCameraProjector class with IPSL calibration data
to reproject raw IPSL images to a top-down view.
"""

import os
import sys
import glob
import numpy as np
from PIL import Image
from pathlib import Path

# Add parent directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "Codebase", "src", "processing"))

from ground_camera_projector import GroundCameraProjector


# IPSL/SIRTA camera parameters
IPSL_LATITUDE = 48.7133
IPSL_LONGITUDE = 2.2081
IPSL_HEIGHT = 162  # meters above sea level

# Projection parameters (matching ECTL style)
CLOUD_HEIGHT = 10000  # 10km
SQUARE_SIZE = 75000   # 75km x 75km
RESOLUTION = 1024     # 1024x1024 pixels
MAX_ZENITH_ANGLE = 80 # degrees


def setup_ipsl_projector():
    """
    Initialize the GroundCameraProjector for IPSL camera.
    
    Returns:
        GroundCameraProjector instance configured for IPSL
    """
    # Path to IPSL calibration files
    calibration_dir = os.path.join(WORKSPACE_ROOT, "Codebase", "config", "IPSL")
    
    # Verify calibration files exist
    azimuth_file = os.path.join(calibration_dir, "calibration", "azimuth_visible.jp2")
    zenith_file = os.path.join(calibration_dir, "calibration", "zenith_visible.jp2")
    
    # Create calibration subdirectory if needed
    calib_subdir = os.path.join(calibration_dir, "calibration")
    os.makedirs(calib_subdir, exist_ok=True)
    
    # Check if files need to be moved
    src_azimuth = os.path.join(calibration_dir, "azimuth_visible.jp2")
    src_zenith = os.path.join(calibration_dir, "zenith_visible.jp2")
    
    if os.path.exists(src_azimuth) and not os.path.exists(azimuth_file):
        import shutil
        shutil.copy(src_azimuth, azimuth_file)
        print(f"  Copied azimuth calibration to: {azimuth_file}")
    
    if os.path.exists(src_zenith) and not os.path.exists(zenith_file):
        import shutil
        shutil.copy(src_zenith, zenith_file)
        print(f"  Copied zenith calibration to: {zenith_file}")
    
    if not os.path.exists(azimuth_file) or not os.path.exists(zenith_file):
        raise FileNotFoundError(
            f"Calibration files not found in {calib_subdir}\n"
            f"Expected: azimuth_visible.jp2 and zenith_visible.jp2\n"
            f"Run visualize_azimuth_zenith.py first to generate calibration files."
        )
    
    print("Initializing IPSL projector...")
    print(f"  Calibration directory: {calibration_dir}")
    print(f"  Camera location: {IPSL_LATITUDE}°N, {IPSL_LONGITUDE}°E @ {IPSL_HEIGHT}m")
    print(f"  Projection: {SQUARE_SIZE/1000:.0f}km × {SQUARE_SIZE/1000:.0f}km @ {CLOUD_HEIGHT/1000:.0f}km altitude")
    print(f"  Resolution: {RESOLUTION}×{RESOLUTION} pixels")
    
    projector = GroundCameraProjector(
        raw_dir=calibration_dir,
        category="visible",
        resolution=RESOLUTION,
        longitude=IPSL_LONGITUDE,
        latitude=IPSL_LATITUDE,
        height_above_ground=IPSL_HEIGHT,
        cloud_height=CLOUD_HEIGHT,
        square_size=SQUARE_SIZE,
        max_zenith_angle=MAX_ZENITH_ANGLE
    )
    
    print("✓ Projector initialized successfully")
    return projector


def process_single_image(input_path, output_path, projector):
    """
    Process a single image through the projector.
    
    Args:
        input_path: Path to input image
        output_path: Path to save projected image
        projector: GroundCameraProjector instance
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load image
        image = np.array(Image.open(input_path))
        
        # Check if image needs resizing to match calibration (768x1024)
        expected_shape = (768, 1024, 3)
        if image.shape[:2] != expected_shape[:2]:
            print(f"    Resizing from {image.shape} to {expected_shape[:2]}")
            pil_img = Image.fromarray(image)
            image = np.array(pil_img.resize(
                (expected_shape[1], expected_shape[0]),  # (width, height)
                Image.Resampling.LANCZOS
            ))
        
        # Project image
        projected = projector.project_image(image, uint8=True)
        
        # Save projected image
        Image.fromarray(projected).save(output_path, quality=95)
        return True
        
    except Exception as e:
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main execution function."""
    print("="*80)
    print("IPSL IMAGE REPROJECTION")
    print("="*80)
    
    # Paths
    input_dir = os.path.join(WORKSPACE_ROOT, "Datasets", "gQg5IUvV", "RAW", "20250406")
    output_dir = os.path.join(WORKSPACE_ROOT, "Datasets", "gQg5IUvV", "PROJECTED_GROUND")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Get list of input images (all JPG files, or filter by pattern)
    input_pattern = os.path.join(input_dir, "*.jpg")
    input_images = sorted(glob.glob(input_pattern))
    
    print(f"\nFound {len(input_images)} images to process")
    
    if len(input_images) == 0:
        print("No images found!")
        return
    
    # Show first few filenames
    print("\nFirst 3 images:")
    for img_path in input_images[:3]:
        print(f"  {os.path.basename(img_path)}")
    
    # Initialize projector
    print("\n" + "-"*80)
    try:
        projector = setup_ipsl_projector()
    except Exception as e:
        print(f"\n✗ Failed to initialize projector: {e}")
        return
    
    # Process each image
    print("\n" + "-"*80)
    print("Processing images...")
    print("-"*80)
    
    success_count = 0
    
    for i, input_path in enumerate(input_images, 1):
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, filename)
        
        print(f"[{i}/{len(input_images)}] {filename}...", end=" ")
        
        if process_single_image(input_path, output_path, projector):
            print("✓")
            success_count += 1
        else:
            print("✗")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total images: {len(input_images)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(input_images) - success_count}")
    print(f"Output directory: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
