#!/usr/bin/env python3
"""Visualize the azimuth and zenith maps from camera calibration.

Coordinate Convention:
- Displays use standard image convention: x=horizontal (columns), y=vertical (rows)
- The converter module internally handles cheick_code's convention (x=vertical, y=horizontal)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from PIL import Image as PILImage

# Add paths - navigate to workspace root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "cheick_code"))
sys.path.insert(0, WORKSPACE_ROOT)  # Add workspace root so we can import azimuth_zenith_calibration

from azimuth_zenith_calibration.converter import AzimuthZenithMapper


def create_angle_visualization(azimuth, zenith, title=""):
    """
    Create a color-coded visualization of azimuth and zenith.
    
    Uses HSV color space:
    - Hue = azimuth (direction)
    - Saturation = sin(zenith) (distance from center)
    - Value = constant (brightness)
    """
    # Normalize azimuth to 0-1 (for hue)
    azimuth_norm = (azimuth + np.pi) / (2 * np.pi)
    
    # Use sin(zenith) for saturation (0 at zenith, 1 at horizon)
    saturation = np.sin(zenith)
    
    # Constant value
    value = np.ones_like(azimuth)
    
    # Stack into HSV
    hsv = np.stack([azimuth_norm, saturation, value], axis=-1)
    
    # Convert to RGB
    rgb = hsv_to_rgb(hsv)
    
    return rgb


def save_calibration_as_jp2(azimuth_map, zenith_map, output_dir):
    """
    Save azimuth and zenith maps as JP2 files for skycam compatibility.
    
    Args:
        azimuth_map: Azimuth array in radians
        zenith_map: Zenith array in radians
        output_dir: Directory to save JP2 files
    """
    import os
    from PIL import Image as PILImage
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nSaving calibration maps as JP2 files...")
    print(f"Output directory: {output_dir}")
    
    # Convert float32 radians to uint16 for JP2 storage
    # Azimuth: -π to π -> scale to 0-65535
    # Zenith: 0 to π/2 -> scale to 0-65535
    
    # Save azimuth map
    azimuth_path = os.path.join(output_dir, "azimuth_visible.jp2")
    # Scale from [-π, π] to [0, 65535]
    azimuth_scaled = ((azimuth_map + np.pi) / (2 * np.pi) * 65535).astype(np.uint16)
    # Handle NaN values
    azimuth_scaled = np.nan_to_num(azimuth_scaled, nan=0)
    PILImage.fromarray(azimuth_scaled, mode='I;16').save(azimuth_path)
    print(f"  ✓ Saved azimuth map: {azimuth_path}")
    print(f"    Encoding: uint16, range [0, 65535] represents [-π, π] radians")
    
    # Save zenith map
    zenith_path = os.path.join(output_dir, "zenith_visible.jp2")
    # Scale from [0, π/2] to [0, 65535]
    zenith_scaled = (zenith_map / (np.pi / 2) * 65535).astype(np.uint16)
    # Handle NaN values
    zenith_scaled = np.nan_to_num(zenith_scaled, nan=0)
    PILImage.fromarray(zenith_scaled, mode='I;16').save(zenith_path)
    print(f"  ✓ Saved zenith map: {zenith_path}")
    print(f"    Encoding: uint16, range [0, 65535] represents [0, π/2] radians")
    
    # Also save as NPZ for exact numerical preservation
    npz_path = os.path.join(output_dir, "azimuth_zenith_visible_calibration.npz")
    np.savez_compressed(npz_path, azimuth=azimuth_map, zenith=zenith_map)
    print(f"  ✓ Saved NPZ archive: {npz_path}")
    
    # Save conversion info
    info_path = os.path.join(output_dir, "calibration_info.txt")
    with open(info_path, 'w') as f:
        f.write("IPSL Camera Calibration Maps\n")
        f.write("=" * 50 + "\n\n")
        f.write("Files:\n")
        f.write("  - azimuth_visible.jp2: Azimuth angles\n")
        f.write("  - zenith_visible.jp2: Zenith angles\n")
        f.write("  - azimuth_zenith_visible_calibration.npz: Raw float data\n\n")
        f.write("JP2 Encoding:\n")
        f.write("  Format: uint16 (0-65535)\n\n")
        f.write("  Azimuth conversion:\n")
        f.write("    radians = (uint16_value / 65535) * 2π - π\n")
        f.write("    range: [-π, π] radians or [-180°, 180°]\n\n")
        f.write("  Zenith conversion:\n")
        f.write("    radians = (uint16_value / 65535) * π/2\n")
        f.write("    range: [0, π/2] radians or [0°, 90°]\n\n")
        f.write(f"Image shape: {azimuth_map.shape}\n")
        f.write(f"Camera: SIRTA IPSL (768x1024)\n")
    print(f"  ✓ Saved conversion info: {info_path}")
    
    # Verify file sizes
    azimuth_size = os.path.getsize(azimuth_path) / 1024  # KB
    zenith_size = os.path.getsize(zenith_path) / 1024
    npz_size = os.path.getsize(npz_path) / 1024
    print(f"\nFile sizes:")
    print(f"  azimuth_visible.jp2: {azimuth_size:.1f} KB")
    print(f"  zenith_visible.jp2: {zenith_size:.1f} KB")
    print(f"  azimuth_zenith_visible_calibration.npz: {npz_size:.1f} KB")
    
    return azimuth_path, zenith_path


def main():
    # Initialize mapper
    print("Initializing azimuth/zenith mapper for SIRTA_W...")
    image_shape = (768, 1024)  # SIRTA camera dimensions
    mapper = AzimuthZenithMapper(site="SIRTA_W", image_shape=image_shape)
    
    # Generate mapping at different resolutions for visualization
    print("\nGenerating full-resolution mapping (this may take a minute)...")
    mapping_full = mapper.generate_mapping(step=1)  # Full resolution
    
    azimuth_map = mapping_full['azimuth']
    zenith_map = mapping_full['zenith']
    x_grid = mapping_full['x']
    y_grid = mapping_full['y']
    
    print(f"Mapping shape: {azimuth_map.shape}")
    print(f"Azimuth range: [{np.nanmin(azimuth_map):.3f}, {np.nanmax(azimuth_map):.3f}] rad")
    print(f"           or: [{np.degrees(np.nanmin(azimuth_map)):.1f}, {np.degrees(np.nanmax(azimuth_map)):.1f}] deg")
    print(f"Zenith range:  [{np.nanmin(zenith_map):.3f}, {np.nanmax(zenith_map):.3f}] rad")
    print(f"           or: [{np.degrees(np.nanmin(zenith_map)):.1f}, {np.degrees(np.nanmax(zenith_map)):.1f}] deg")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Azimuth map (color-coded by direction)
    ax1 = plt.subplot(2, 3, 1)
    extent = [0, image_shape[1], image_shape[0], 0]  # [left, right, bottom, top]
    im1 = ax1.imshow(np.degrees(azimuth_map), cmap='hsv', vmin=-180, vmax=180, 
                     extent=extent, origin='upper')
    ax1.set_title('Azimuth Map (Direction)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X - Horizontal (pixels)')
    ax1.set_ylabel('Y - Vertical (pixels)')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Azimuth (degrees)', rotation=270, labelpad=20)
    ax1.grid(True, alpha=0.3)
    
    # 2. Zenith map (color-coded by angle from vertical)
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(np.degrees(zenith_map), cmap='viridis', vmin=0, vmax=90,
                     extent=extent, origin='upper')
    ax2.set_title('Zenith Map (Angle from Vertical)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X - Horizontal (pixels)')
    ax2.set_ylabel('Y - Vertical (pixels)')
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Zenith (degrees)', rotation=270, labelpad=20)
    ax2.grid(True, alpha=0.3)
    
    # 3. Combined HSV visualization
    ax3 = plt.subplot(2, 3, 3)
    rgb_vis = create_angle_visualization(azimuth_map, zenith_map)
    ax3.imshow(rgb_vis, extent=extent, origin='upper')
    ax3.set_title('Combined: Direction & Distance', fontsize=14, fontweight='bold')
    ax3.set_xlabel('X - Horizontal (pixels)')
    ax3.set_ylabel('Y - Vertical (pixels)')
    ax3.text(0.02, 0.98, 'Hue = Direction\nBrightness = Distance from center',
            transform=ax3.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Azimuth histogram
    ax4 = plt.subplot(2, 3, 4)
    valid_azimuth = azimuth_map[~np.isnan(azimuth_map)]
    ax4.hist(np.degrees(valid_azimuth), bins=72, color='steelblue', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Azimuth (degrees)', fontsize=11)
    ax4.set_ylabel('Pixel Count', fontsize=11)
    ax4.set_title('Azimuth Distribution', fontsize=12, fontweight='bold')
    ax4.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='North')
    ax4.axvline(90, color='green', linestyle='--', linewidth=1, alpha=0.5, label='East')
    ax4.axvline(-90, color='green', linestyle='--', linewidth=1, alpha=0.5, label='West')
    ax4.axvline(180, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax4.axvline(-180, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='South')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. Zenith histogram
    ax5 = plt.subplot(2, 3, 5)
    valid_zenith = zenith_map[~np.isnan(zenith_map)]
    ax5.hist(np.degrees(valid_zenith), bins=45, color='darkgreen', alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Zenith (degrees)', fontsize=11)
    ax5.set_ylabel('Pixel Count', fontsize=11)
    ax5.set_title('Zenith Distribution', fontsize=12, fontweight='bold')
    ax5.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Straight up')
    ax5.axvline(60, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='60° (typical limit)')
    ax5.axvline(90, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='Horizon')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # 6. Polar plot showing field of view
    ax6 = plt.subplot(2, 3, 6, projection='polar')
    
    # Sample points for visualization
    sample_step = 8
    az_sample = azimuth_map[::sample_step, ::sample_step].ravel()
    zen_sample = zenith_map[::sample_step, ::sample_step].ravel()
    
    # Remove NaNs
    valid = ~np.isnan(az_sample) & ~np.isnan(zen_sample)
    az_sample = az_sample[valid]
    zen_sample = zen_sample[valid]
    
    # Plot in polar coordinates (azimuth = angle, zenith = radius)
    scatter = ax6.scatter(az_sample, np.degrees(zen_sample), 
                         c=np.degrees(zen_sample), cmap='viridis',
                         s=2, alpha=0.5)
    ax6.set_theta_zero_location('N')
    ax6.set_theta_direction(-1)
    ax6.set_ylim(0, 90)
    ax6.set_title('Field of View (Polar)', fontsize=14, fontweight='bold', pad=20)
    ax6.set_ylabel('Zenith angle (degrees)', labelpad=30)
    
    # Add reference circles
    ax6.plot(np.linspace(0, 2*np.pi, 100), np.full(100, 30), 'r--', alpha=0.3, linewidth=1)
    ax6.plot(np.linspace(0, 2*np.pi, 100), np.full(100, 60), 'r--', alpha=0.5, linewidth=1)
    ax6.text(0, 30, '30°', ha='center', fontsize=8, color='red')
    ax6.text(0, 60, '60°', ha='center', fontsize=8, color='red')
    
    plt.tight_layout()
    
    # Save figure
    output_path = "/data/common/STEREOSTUDYIPSL/testing.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    
    # Create a second figure focusing on the camera center region
    print("\nCreating detailed center view...")
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get center region (middle 50%)
    h, w = azimuth_map.shape
    h_start, h_end = h//4, 3*h//4
    w_start, w_end = w//4, 3*w//4
    
    az_center = azimuth_map[h_start:h_end, w_start:w_end]
    zen_center = zenith_map[h_start:h_end, w_start:w_end]
    
    # Azimuth detail
    im1 = axes2[0].imshow(np.degrees(az_center), cmap='twilight', interpolation='bilinear')
    axes2[0].set_title('Azimuth - Center Region Detail', fontsize=13, fontweight='bold')
    axes2[0].set_xlabel('X (pixels)')
    axes2[0].set_ylabel('Y (pixels)')
    cbar1 = plt.colorbar(im1, ax=axes2[0])
    cbar1.set_label('Azimuth (degrees)', rotation=270, labelpad=15)
    
    # Save calibration maps as JP2 files for skycam
    ipsl_config_dir = os.path.join(WORKSPACE_ROOT, "Codebase", "config", "IPSL")
    save_calibration_as_jp2(azimuth_map, zenith_map, ipsl_config_dir)
    
    # Zenith detail
    im2 = axes2[1].imshow(np.degrees(zen_center), cmap='plasma', interpolation='bilinear')
    axes2[1].set_title('Zenith - Center Region Detail', fontsize=13, fontweight='bold')
    axes2[1].set_xlabel('X (pixels)')
    axes2[1].set_ylabel('Y (pixels)')
    cbar2 = plt.colorbar(im2, ax=axes2[1])
    cbar2.set_label('Zenith (degrees)', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    print()
    output_path2 = "/data/common/STEREOSTUDYIPSL/testing_detail.png"
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Saved detail view to: {output_path2}")

    # Print statistics
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    print(f"Total pixels in map: {azimuth_map.size}")
    print(f"Valid pixels: {np.sum(~np.isnan(azimuth_map))}")
    print(f"\nAzimuth (direction):")
    print(f"  Mean: {np.degrees(np.nanmean(azimuth_map)):.2f}°")
    print(f"  Std:  {np.degrees(np.nanstd(azimuth_map)):.2f}°")
    print(f"\nZenith (angle from vertical):")
    print(f"  Mean: {np.degrees(np.nanmean(zenith_map)):.2f}°")
    print(f"  Std:  {np.degrees(np.nanstd(zenith_map)):.2f}°")
    print(f"  Min:  {np.degrees(np.nanmin(zenith_map)):.2f}°")
    print(f"  Max:  {np.degrees(np.nanmax(zenith_map)):.2f}°")
    print(f"\nField of view coverage:")
    zenith_60 = np.sum(zenith_map < np.radians(60))
    zenith_total = np.sum(~np.isnan(zenith_map))
    print(f"  Pixels within 60° zenith: {zenith_60} ({100*zenith_60/zenith_total:.1f}%)")
    
    print("\n" + "="*60)
    print("Done! Check the output images.")
    print("="*60)


if __name__ == "__main__":
    main()
