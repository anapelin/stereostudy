#!/usr/bin/env python3
"""Filter zenith map to only include pixels within a circular region."""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add paths
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "cheick_code"))
sys.path.insert(0, os.path.join(REPO_ROOT, "azimuth_zenith_calibration"))

from azimuth_zenith_calibration.converter import AzimuthZenithMapper


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


def main():
    # Initialize mapper
    print("Initializing azimuth/zenith mapper for SIRTA_W...")
    image_shape = (768, 1024)  # SIRTA camera dimensions
    mapper = AzimuthZenithMapper(site="SIRTA_W", image_shape=image_shape)
    
    # Generate mapping
    print("Generating azimuth/zenith mapping...")
    mapping = mapper.generate_mapping(step=1)
    
    zenith_map = mapping['zenith']
    azimuth_map = mapping['azimuth']
    
    print(f"Zenith map shape: {zenith_map.shape}")
    print(f"Zenith range (before filtering): [{np.degrees(np.nanmin(zenith_map)):.1f}, {np.degrees(np.nanmax(zenith_map)):.1f}] deg")
    
    # Apply circular filter
    center_x = 512  # horizontal center
    center_y = 384  # vertical center
    radius = 380    # pixels
    
    print(f"\nApplying circular mask:")
    print(f"  Center: ({center_x}, {center_y}) pixels")
    print(f"  Radius: {radius} pixels")
    
    filtered_zenith, mask = filter_zenith_circular_mask(zenith_map, center_x, center_y, radius)
    
    print(f"\nPixels inside circle: {np.sum(mask)} ({100*np.sum(mask)/mask.size:.1f}%)")
    print(f"Pixels outside circle: {np.sum(~mask)} ({100*np.sum(~mask)/mask.size:.1f}%)")
    print(f"Zenith range (after filtering): [{np.degrees(np.nanmin(filtered_zenith[mask])):.1f}, {np.degrees(np.nanmax(filtered_zenith[mask])):.1f}] deg")
    
    # Create visualization
    print("\nCreating visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    extent = [0, image_shape[1], image_shape[0], 0]  # [left, right, bottom, top]
    
    # Original zenith map
    im1 = axes[0].imshow(np.degrees(zenith_map), cmap='viridis', vmin=0, vmax=90,
                         extent=extent, origin='upper')
    axes[0].set_title('Original Zenith Map', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('X - Horizontal (pixels)', fontsize=12)
    axes[0].set_ylabel('Y - Vertical (pixels)', fontsize=12)
    
    # Draw circle on original
    circle1 = plt.Circle((center_x, center_y), radius, color='red', fill=False, 
                         linewidth=2, linestyle='--', label=f'Radius={radius}px')
    axes[0].add_patch(circle1)
    axes[0].plot(center_x, center_y, 'r+', markersize=15, markeredgewidth=2, 
                label=f'Center ({center_x}, {center_y})')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Zenith Angle (degrees)', rotation=270, labelpad=20)
    
    # Filtered zenith map
    im2 = axes[1].imshow(np.degrees(filtered_zenith), cmap='viridis', vmin=0, vmax=90,
                         extent=extent, origin='upper')
    axes[1].set_title('Filtered Zenith Map\n(Outside circle set to 90°)', 
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel('X - Horizontal (pixels)', fontsize=12)
    axes[1].set_ylabel('Y - Vertical (pixels)', fontsize=12)
    
    # Draw circle on filtered
    circle2 = plt.Circle((center_x, center_y), radius, color='red', fill=False, 
                         linewidth=2, linestyle='--', label=f'Radius={radius}px')
    axes[1].add_patch(circle2)
    axes[1].plot(center_x, center_y, 'r+', markersize=15, markeredgewidth=2,
                label=f'Center ({center_x}, {center_y})')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Zenith Angle (degrees)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    # Save figure
    output_path = "/data/common/STEREOSTUDYIPSL/zenith_circular_filter.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    
    plt.show()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
