#%%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify plane projection on a single image.

This script projects one all-sky image onto a plane at 11km to verify the setup works.
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Add cheick_code to path
CHEICK_CODE_PATH = '/data/common/STEREOSTUDYIPSL/cheick_code'
if CHEICK_CODE_PATH not in sys.path:
    sys.path.insert(0, CHEICK_CODE_PATH)

from calibration.useCalibration import worldToImage
# Import readLatLon directly to avoid AVION dependency
import pandas as pd

def readLatLon(site, path='/data/common/STEREOSTUDYIPSL/cheick_code/params.csv'):
    """Extract latitude and longitude for a site."""
    data = pd.read_csv(path, index_col=0)
    line = data.loc[site]
    return line['lat'], line['lon']

# Configuration
TEST_IMAGE = '/data/common/STEREOSTUDYIPSL/Datasets/gQg5IUvV/srf02_0a_skyimgLz2_v01_20250406_044600_853/20250406/20250406050000_01.jpg'
OUTPUT_DIR = '/data/common/STEREOSTUDYIPSL/test_projection_output'
SITE = 'SIRTA'
ALTITUDE_KM = 11.0

#%%

def test_single_point_projection():
    """Test projecting a single point at 11km altitude."""
    print("\n" + "="*70)
    print("Testing single point projection")
    print("="*70)
    
    # Get site coordinates
    lat_site, lon_site = readLatLon(SITE, path=os.path.join(CHEICK_CODE_PATH, 'params.csv'))
    print(f"Site location: {lat_site:.6f}°N, {lon_site:.6f}°E")
    
    # Test point: 10km North, 5km East, 11km altitude
    x_km = 5.0   # East
    y_km = 10.0  # North
    z_km = 11.0  # Altitude
    
    # Normalize
    norm = np.sqrt(x_km**2 + y_km**2 + z_km**2)
    position = np.array([-y_km/norm, x_km/norm, z_km/norm])
    
    print(f"\nTest point: ({x_km}km E, {y_km}km N, {z_km}km alt)")
    print(f"Normalized position vector: {position}")
    
    # Project to image
    try:
        px, py = worldToImage(
            XPosition=position,
            imageShape=np.array([768, 1024, 3]),
            zoom=False,
            methodRead="csv",
            site=SITE
        )
        print(f"✓ Projected to pixel coordinates: ({px:.2f}, {py:.2f})")
        return True
    except Exception as e:
        print(f"✗ Projection failed: {e}")
        return False

def test_image_projection():
    """Test projecting a complete image."""
    print("\n" + "="*70)
    print("Testing full image projection")
    print("="*70)
    
    # Check if test image exists
    if not os.path.exists(TEST_IMAGE):
        print(f"✗ Test image not found: {TEST_IMAGE}")
        return False
    
    # Load image
    print(f"Loading image: {TEST_IMAGE}")
    image = cv2.imread(TEST_IMAGE)
    if image is None:
        print("✗ Failed to load image")
        return False
    
    print(f"✓ Image loaded: {image.shape}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get site coordinates
    lat_site, lon_site = readLatLon(SITE, path=os.path.join(CHEICK_CODE_PATH, 'params.csv'))
    
    # Output configuration: 768x768 center crop
    projection_size = 768
    extent_km = 40  # 40km x 40km area at 11km altitude
    
    print(f"\nCreating projection grid: {projection_size}x{projection_size} pixels")
    print(f"Spatial extent: {extent_km}km x {extent_km}km at {ALTITUDE_KM}km altitude")
    
    # Create grid in ground coordinates (East-North)
    half_extent = extent_km / 2.0
    x_coords = np.linspace(-half_extent, half_extent, projection_size)  # East-West
    y_coords = np.linspace(half_extent, -half_extent, projection_size)  # North-South (top to bottom)
    
    # Initialize output
    projected = np.zeros((projection_size, projection_size, 3), dtype=np.uint8)
    
    print("Projecting image...")
    
    # Project each pixel in the output grid
    valid_pixels = 0
    for row in range(projection_size):
        if row % 64 == 0:  # Progress indicator
            print(f"  Progress: {100*row//projection_size}%")
        
        for col in range(projection_size):
            # Ground coordinates for this output pixel
            x_km = x_coords[col]  # East
            y_km = y_coords[row]  # North
            z_km = ALTITUDE_KM    # Altitude
            
            # Normalize position vector
            norm = np.sqrt(x_km**2 + y_km**2 + z_km**2)
            if norm == 0:
                continue
            
            position = np.array([-y_km/norm, x_km/norm, z_km/norm])
            
            try:
                # Project to original image coordinates
                px, py = worldToImage(
                    XPosition=position,
                    imageShape=np.array(image.shape),
                    zoom=False,
                    methodRead="csv",
                    site=SITE
                )
                
                # Check bounds and sample
                px_int, py_int = int(round(px)), int(round(py))
                if 0 <= px_int < image.shape[1] and 0 <= py_int < image.shape[0]:
                    projected[row, col] = image[py_int, px_int]
                    valid_pixels += 1
                    
            except Exception:
                # Skip invalid projections
                continue
    
    print(f"✓ Projection complete: {valid_pixels}/{projection_size**2} valid pixels ({100*valid_pixels/projection_size**2:.1f}%)")
    
    # Save results
    output_path = os.path.join(OUTPUT_DIR, 'test_projected_image.jpg')
    cv2.imwrite(output_path, projected)
    print(f"✓ Saved projected image to: {output_path}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Original image
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original All-Sky Image')
    axes[0].axis('off')
    
    # Projected image
    axes[1].imshow(cv2.cvtColor(projected, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Projected at {ALTITUDE_KM}km altitude\n({extent_km}km x {extent_km}km)')
    axes[1].set_xlabel('East-West')
    axes[1].set_ylabel('North-South')
    
    viz_path = os.path.join(OUTPUT_DIR, 'test_projection_comparison.png')
    plt.tight_layout()
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {viz_path}")
    plt.close()
    
    return True
#%%
def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PLANE PROJECTION TEST SUITE")
    print("="*70)
    print(f"Site: {SITE}")
    print(f"Altitude: {ALTITUDE_KM} km")
    print(f"Calibration path: {CHEICK_CODE_PATH}")
    print("="*70)
    
    # Test 1: Single point projection
    test1_ok = test_single_point_projection()
    
    # Test 2: Full image projection
    test2_ok = test_image_projection()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Single point projection: {'✓ PASS' if test1_ok else '✗ FAIL'}")
    print(f"Full image projection:   {'✓ PASS' if test2_ok else '✗ FAIL'}")
    print("="*70)
    
    if test1_ok and test2_ok:
        print("\n✓ All tests passed! Ready to process full dataset.")
        print(f"\nTo process all images, run:")
        print(f"  python project_images_to_plane.py")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")

if __name__ == '__main__':
    main()

# %%
