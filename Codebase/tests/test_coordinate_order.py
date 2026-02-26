#!/usr/bin/env python3
"""Test to verify coordinate ordering issue."""

import numpy as np
import matplotlib.pyplot as plt

# Simulate what converter.py does
height, width = 768, 1024
xs = np.arange(0, width, 4)
ys = np.arange(0, height, 4)

print(f"Image shape: ({height}, {width})")
print(f"xs range: 0 to {xs[-1]} (length {len(xs)})")
print(f"ys range: 0 to {ys[-1]} (length {len(ys)})")

# Create meshgrid
x_grid, y_grid = np.meshgrid(xs, ys)

print(f"\nAfter meshgrid:")
print(f"x_grid.shape: {x_grid.shape}")
print(f"y_grid.shape: {y_grid.shape}")
print(f"x_grid[0, 0] = {x_grid[0, 0]}, x_grid[0, -1] = {x_grid[0, -1]}")
print(f"y_grid[0, 0] = {y_grid[0, 0]}, y_grid[-1, 0] = {y_grid[-1, 0]}")

# Create a test pattern: gradient in X direction
test_pattern = x_grid.copy()

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: x_grid values
im1 = axes[0].imshow(x_grid, cmap='viridis')
axes[0].set_title('X Grid Values\n(should increase left to right)')
axes[0].set_xlabel('Columns')
axes[0].set_ylabel('Rows')
plt.colorbar(im1, ax=axes[0])

# Plot 2: y_grid values
im2 = axes[1].imshow(y_grid, cmap='viridis')
axes[1].set_title('Y Grid Values\n(should increase top to bottom)')
axes[1].set_xlabel('Columns')
axes[1].set_ylabel('Rows')
plt.colorbar(im2, ax=axes[1])

# Plot 3: Mark calibration center
center_x, center_y = 384.72, 518.53
axes[2].imshow(x_grid, cmap='gray')
axes[2].plot(center_x/4, center_y/4, 'r+', markersize=20, markeredgewidth=3, label=f'Calib center ({center_x:.0f}, {center_y:.0f})')
axes[2].plot(width/2/4, height/2/4, 'gx', markersize=20, markeredgewidth=3, label=f'Geom center ({width/2:.0f}, {height/2:.0f})')
axes[2].set_title('Calibration vs Geometric Center')
axes[2].set_xlabel('Columns (X/4)')
axes[2].set_ylabel('Rows (Y/4)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/data/common/STEREOSTUDYIPSL/coordinate_order_test.png', dpi=150)
print(f"\nSaved test visualization to: coordinate_order_test.png")

# The issue: 
print("\n" + "="*60)
print("THE ISSUE:")
print("="*60)
print(f"Calibration center: x0={center_x:.1f}, y0={center_y:.1f}")
print(f"Geometric center: x={width/2:.1f}, y={height/2:.1f}")
print(f"\nIn the image coordinates (row, col) used by imshow:")
print(f"  - Row index = Y coordinate")
print(f"  - Col index = X coordinate")
print(f"\nSo the calibration center should appear at:")
print(f"  - Column = {center_x:.1f} (X coordinate)")
print(f"  - Row = {center_y:.1f} (Y coordinate)")
print(f"\nIn the downsampled (step=4) visualization:")
print(f"  - Column = {center_x/4:.1f}")
print(f"  - Row = {center_y/4:.1f}")
print("="*60)
