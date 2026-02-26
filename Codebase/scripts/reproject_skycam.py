#!/usr/bin/env python3
"""
Reproject camera images using proper calibrations:
- IPSL (SIRTA site): Uses cheick_code Fripon calibration  
- ECTL (Bretigny site): Uses skycam JP2 calibration

Both are reprojected to a common 10 km altitude plane.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from typing import Tuple
import argparse
from datetime import datetime

# Add paths to workspace modules
workspace_root = Path(__file__).parent.parent.parent
cheick_code_path = workspace_root / "cheick_code"
skycam_path = workspace_root / "Codebase" / "config" / "utils" / "skycam" / "src"

if str(cheick_code_path) not in sys.path:
    sys.path.insert(0, str(cheick_code_path))
if str(skycam_path) not in sys.path:
    sys.path.insert(0, str(skycam_path))

# Import skycam for ECTL
from skycam.adapters.calibration import JP2CalibrationLoader
from skycam.domain.models import ProjectionSettings, Position
from skycam.domain.projection import ProjectionService

# Import SIRTA calibration
from scipy.interpolate import griddata


class ECTLReprojector:
    """Reprojector for ECTL (Bretigny) using skycam library."""
    
    def __init__(self, plane_altitude_m: float = 10000):
        """Initialize ECTL reprojector.
        
        Args:
            plane_altitude_m: Altitude of projection plane in meters
        """
        self.plane_altitude_m = plane_altitude_m
        self.site_name = "ECTL (Bretigny)"
        
        # Load ECTL calibration
        calibration_dir = workspace_root / "Codebase" / "config" / "ECTL"
        print(f"  Loading ECTL calibration from {calibration_dir}")
        
        loader = JP2CalibrationLoader(calibration_dir)
        self.calibration = loader.load("visible")
        
        # Configure projection settings
        self.settings = ProjectionSettings(
            resolution=1024,
            cloud_height=plane_altitude_m,
            square_size=75000.0,  # 75 km square
            max_zenith_angle=80.0
        )
        
        # Create projection service
        self.projector = ProjectionService(
            calibration=self.calibration,
            settings=self.settings,
            calibration_path=calibration_dir
        )
        
        print(f"✓ Initialized ECTL (Bretigny) reprojector")
        print(f"  Plane altitude: {plane_altitude_m/1000:.1f} km")
        print(f"  Image size: {self.calibration.image_size}")
        print(f"  Output resolution: {self.settings.resolution}")
        print(f"  Grid size: {self.settings.square_size/1000:.1f} km")
    
    def reproject_image(self, image: np.ndarray) -> np.ndarray:
        """Reproject ECTL image to horizontal plane.
        
        Args:
            image: Input image (H, W, 3) RGB array
            
        Returns:
            Reprojected image on plane grid
        """
        # Ensure image matches calibration size
        h, w = self.calibration.image_size
        if image.shape[:2] != (h, w):
            pil_img = Image.fromarray(image)
            image = np.array(pil_img.resize((w, h), Image.Resampling.LANCZOS))
        
        # Skycam expects (H, W, C) format
        projected = self.projector.project(image)
        
        return projected
    
    def get_grid_extent(self) -> Tuple[float, float]:
        """Get grid extent in kilometers.
        
        Returns:
            (extent_km, extent_km) for visualization
        """
        extent_km = self.settings.square_size / 2000  # half-size in km
        return extent_km, extent_km


class SIRTAReprojector:
    """Reprojector for IPSL (SIRTA) using cheick_code calibration."""
    
    def __init__(self, plane_altitude_m: float = 10000):
        """Initialize SIRTA reprojector.
        
        Args:
            plane_altitude_m: Altitude of projection plane in meters
        """
        self.plane_altitude_m = plane_altitude_m
        self.site_name = "IPSL (SIRTA)"
        self.camera_elevation_m = 156
        
        # Load SIRTA azimuth/zenith map
        az_zen_map_path = workspace_root / "Codebase" / "data" / "azimuth_zenith_map_full_corrected.npz"
        print(f"  Loading SIRTA calibration from {az_zen_map_path.name}")
        
        data = np.load(az_zen_map_path)
        self.azimuth = data['azimuth']
        self.zenith = data['zenith']
        self.image_height, self.image_width = self.azimuth.shape
        
        print(f"✓ Initialized IPSL (SIRTA) reprojector")
        print(f"  Plane altitude: {plane_altitude_m/1000:.1f} km")
        print(f"  Camera elevation: {self.camera_elevation_m} m")
        print(f"  Image size: {self.image_width} × {self.image_height}")
        
        # Compute projection coordinates
        self._compute_projection_coords()
    
    def _compute_projection_coords(self):
        """Compute East-North coordinates for projection."""
        delta_h = self.plane_altitude_m - self.camera_elevation_m
        
        # Convert to radians
        zen_rad = np.radians(self.zenith)
        az_rad = np.radians(self.azimuth)
        
        # Valid pixels (not near horizon)
        cos_zen = np.cos(zen_rad)
        self.valid_mask = cos_zen > 0.01
        
        # Distance to plane
        d = np.full_like(zen_rad, np.nan)
        d[self.valid_mask] = delta_h / cos_zen[self.valid_mask]
        
        # Horizontal distance
        r = d * np.sin(zen_rad)
        
        # East-North components
        self.east = r * np.sin(az_rad)
        self.north = r * np.cos(az_rad)
        
        # Grid extent (same as ECTL for comparison)
        max_extent = 37500  # meters (half of 75km)
        self.grid_extent_m = max_extent
    
    def reproject_image(self, image: np.ndarray) -> np.ndarray:
        """Reproject SIRTA image to horizontal plane.
        
        Args:
            image: Input image (H, W, 3) RGB array
            
        Returns:
            Reprojected image on plane grid
        """
        # Resize if needed
        if image.shape[:2] != (self.image_height, self.image_width):
            pil_img = Image.fromarray(image)
            image = np.array(pil_img.resize(
                (self.image_width, self.image_height),
                Image.Resampling.LANCZOS
            ))
        
        # Create grid matching ECTL resolution
        resolution = 1024
        extent = self.grid_extent_m
        coords = np.linspace(-extent, extent, resolution)
        grid_east, grid_north = np.meshgrid(coords, coords)
        
        # Get valid source coordinates
        valid_pixels = self.valid_mask
        source_east = self.east[valid_pixels]
        source_north = self.north[valid_pixels]
        
        # Reproject each channel
        projected = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        
        for ch in range(3):
            source_values = image[:, :, ch][valid_pixels]
            
            projected[:, :, ch] = griddata(
                points=(source_east, source_north),
                values=source_values,
                xi=(grid_east, grid_north),
                method='linear',
                fill_value=0
            )
        
        return np.clip(projected, 0, 255).astype(np.uint8)
    
    def get_grid_extent(self) -> Tuple[float, float]:
        """Get grid extent in kilometers."""
        extent_km = self.grid_extent_m / 1000
        return extent_km, extent_km


def visualize_reprojection(original: np.ndarray,
                          reprojected: np.ndarray,
                          site_name: str,
                          extent_km: float,
                          altitude_km: float,
                          output_dir: Path,
                          timestamp: str,
                          img_name: str):
    """Visualize original and reprojected images."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    
    # Original
    axes[0].imshow(original)
    axes[0].set_title(f'{site_name}\nOriginal Fisheye Image', 
                     fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Reprojected
    axes[1].imshow(reprojected,
                   extent=[-extent_km, extent_km, -extent_km, extent_km],
                   origin='lower')
    axes[1].set_title(f'{site_name}\nReprojected at {altitude_km:.0f} km',
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel('East (km)', fontsize=12)
    axes[1].set_ylabel('North (km)', fontsize=12)
    axes[1].grid(True, alpha=0.3, color='white', linewidth=0.5)
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    
    plt.tight_layout()
    
    # Save with timestamp and original image name
    safe_site_name = site_name.replace(' ', '_').replace('(', '').replace(')', '')
    output_filename = f"{timestamp}_{safe_site_name}_{img_name}.png"
    output_path = output_dir / output_filename
    plt.savefig(output_path, dpi=150)
    print(f"  Saved: {output_path}")
    
    plt.show()


def process_site(site: str, num_samples: int, plane_altitude_m: float, output_dir: Path):
    """Process images from a specific site."""
    print(f"\n{'='*70}")
    print(f"Processing {site}")
    print('='*70)
    
    if site == "SIRTA":
        reprojector = SIRTAReprojector(plane_altitude_m=plane_altitude_m)
        image_dir = workspace_root / "Datasets" / "gQg5IUvV" / "RAW" / "20250406"
    elif site == "ECTL":
        reprojector = ECTLReprojector(plane_altitude_m=plane_altitude_m)
        image_dir = workspace_root / "Datasets" / "OdnkTZQ8" / "RAW" / "RAW"
    else:
        print(f"✗ Unknown site: {site}")
        return
    
    if not image_dir.exists():
        print(f"✗ Image directory not found: {image_dir}")
        return
    
    # Find sample images (JP2 for ECTL, JPG for both)
    if site == "ECTL":
        jpg_files = list(image_dir.glob('*.jpg'))
        jp2_files = list(image_dir.glob('*.jp2'))
        image_files = sorted(jpg_files + jp2_files)[:num_samples]
    else:
        image_files = sorted(list(image_dir.glob('*.jpg')))[:num_samples]
    
    if not image_files:
        print(f"✗ No images found in {image_dir}")
        return
    
    print(f"\n✓ Found {len(image_files)} sample images")
    
    # Get grid extent
    extent_km, _ = reprojector.get_grid_extent()
    
    # Generate timestamp for this batch
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process each sample
    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing {img_path.name}...")
        
        try:
            # Load image
            original = np.array(Image.open(img_path).convert('RGB'))
            print(f"  Loaded image: {original.shape}")
            
            # Reproject
            print(f"  Reprojecting to {plane_altitude_m/1000:.0f} km plane...")
            reprojected = reprojector.reproject_image(original)
            print(f"  Reprojected shape: {reprojected.shape}")
            
            # Visualize
            img_name_stem = img_path.stem  # filename without extension
            visualize_reprojection(
                original,
                reprojected,
                reprojector.site_name,
                extent_km,
                plane_altitude_m / 1000,
                output_dir,
                timestamp,
                img_name_stem
            )
            
            print(f"  ✓ Done")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Reproject camera images using proper calibrations'
    )
    parser.add_argument('--site', choices=['SIRTA', 'ECTL', 'both'], default='both',
                       help='Site to process (SIRTA=IPSL, ECTL=Bretigny)')
    parser.add_argument('--samples', type=int, default=2,
                       help='Number of sample images to process')
    parser.add_argument('--altitude', type=float, default=10,
                       help='Projection plane altitude in km')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for saved images (default: Codebase/outputs)')
    
    args = parser.parse_args()
    
    plane_altitude_m = args.altitude * 1000
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = workspace_root / "Codebase" / "outputs"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("CAMERA IMAGE REPROJECTION")
    print("="*70)
    print(f"IPSL = SIRTA site (using cheick_code Fripon calibration)")
    print(f"ECTL = Bretigny site (using skycam JP2 calibration)")
    print(f"Plane altitude: {args.altitude:.1f} km")
    print(f"Samples per site: {args.samples}")
    print(f"Output directory: {output_dir}")
    
    # Process sites
    if args.site in ['SIRTA', 'both']:
        process_site('SIRTA', args.samples, plane_altitude_m, output_dir)
    
    if args.site in ['ECTL', 'both']:
        process_site('ECTL', args.samples, plane_altitude_m, output_dir)
    
    print("\n" + "="*70)
    print("✓ Processing complete")
    print(f"Images saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()