#!/usr/bin/env python3
"""Quick test of the calibration system"""

import sys
import numpy as np

# Add cheick_code to path
sys.path.insert(0, '/data/common/STEREOSTUDYIPSL/cheick_code')

from calibration.useCalibration import worldToImage

# Test coordinates
print("Testing worldToImage function...")
print("="*60)

# Simple test: point at 10km East, 5km North, 11km altitude
x_km, y_km, z_km = 10.0, 5.0, 11.0
norm = np.sqrt(x_km**2 + y_km**2 + z_km**2)
position = np.array([-y_km/norm, x_km/norm, z_km/norm])

print(f"Input position (East, North, Up): ({x_km}, {y_km}, {z_km}) km")
print(f"Normalized vector: {position}")

try:
    image_shape = np.array([768, 1024, 3])
    x_pixel, y_pixel = worldToImage(
        XPosition=position,
        imageShape=image_shape,
        methodRead="csv",
        site='SIRTA',
        zoom=False
    )
    print(f"✓ Success! Projected to pixel: ({x_pixel:.2f}, {y_pixel:.2f})")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
