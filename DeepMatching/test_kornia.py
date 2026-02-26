#!/usr/bin/env python3
"""Test script to verify Kornia LoFTR works."""

print("Testing Kornia LoFTR...")

try:
    from kornia.feature import LoFTR
    print("✓ LoFTR import successful")
except ImportError as e:
    print(f"✗ LoFTR import failed: {e}")
    exit(1)

try:
    loftr = LoFTR(pretrained="outdoor")
    print("✓ LoFTR model loaded (outdoor weights)")
    print(f"  Model type: {type(loftr)}")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    exit(1)

print("\nAll tests passed! Kornia LoFTR is working correctly.")
