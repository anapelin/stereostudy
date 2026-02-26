#!/usr/bin/env python3
"""
Verification script for camera configurations and calibration data.
"""

import json
import numpy as np
from pathlib import Path

def verify_camera_config(config_path: Path):
    """Verify a camera configuration file."""
    print(f"\n{'='*60}")
    print(f"Verifying: {config_path.name}")
    print('='*60)
    
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"✓ Camera: {config['name']}")
    print(f"✓ Site Code: {config['site_code']}")
    print(f"✓ Location: {config['location']['latitude']}°N, {config['location']['longitude']}°E")
    print(f"✓ Elevation: {config['location']['elevation_m']} m")
    print(f"✓ Image Size: {config['image']['width']} × {config['image']['height']}")
    
    # Verify azimuth/zenith map
    az_zen_map_path = Path(config_path.parent.parent) / config['calibration']['azimuth_zenith_map']
    if az_zen_map_path.exists():
        data = np.load(az_zen_map_path)
        print(f"✓ Azimuth/Zenith Map: {az_zen_map_path.name}")
        print(f"  - Size: {az_zen_map_path.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"  - Shape: {data['azimuth'].shape}")
        print(f"  - Site: {data['site']}")
        print(f"  - Azimuth range: [{data['azimuth'].min():.1f}°, {data['azimuth'].max():.1f}°]")
        print(f"  - Zenith range: [{data['zenith'].min():.1f}°, {data['zenith'].max():.1f}°]")
    else:
        print(f"✗ Azimuth/Zenith Map NOT FOUND: {az_zen_map_path}")
        return False
    
    return True

def main():
    """Main verification routine."""
    print("\n" + "="*60)
    print("CAMERA CONFIGURATION VERIFICATION")
    print("="*60)
    
    config_dir = Path(__file__).parent.parent / "config" / "cameras"
    
    if not config_dir.exists():
        print(f"\n✗ Configuration directory not found: {config_dir}")
        return
    
    configs = list(config_dir.glob("*.json"))
    
    if not configs:
        print(f"\n✗ No configuration files found in {config_dir}")
        return
    
    print(f"\nFound {len(configs)} camera configuration(s)")
    
    success_count = 0
    for config_path in sorted(configs):
        if verify_camera_config(config_path):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {success_count}/{len(configs)} configurations verified")
    print('='*60)
    
    if success_count == len(configs):
        print("\n✓ All camera configurations are valid!")
    else:
        print(f"\n✗ {len(configs) - success_count} configuration(s) failed verification")

if __name__ == "__main__":
    main()
