"""
Data loading and matching utilities.
"""
import datetime
import re
from pathlib import Path
from typing import List, Dict
import numpy as np
from PIL import Image


def extract_timestamp_from_ipsl(filename: str) -> datetime.datetime:
    """
    Extract timestamp from IPSL filename format: YYYYMMDDHHMMSS_XX.jpg
    
    Args:
        filename: IPSL filename
        
    Returns:
        datetime object or None
    """
    match = re.search(r'(\d{14})', filename)
    if match:
        timestamp_str = match.group(1)
        return datetime.datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
    return None


def extract_timestamp_from_ectl(filename: str) -> datetime.datetime:
    """
    Extract timestamp from ECTL filename format: image_YYYYMMDDHHMMSS.jpg
    
    Args:
        filename: ECTL filename
        
    Returns:
        datetime object or None
    """
    match = re.search(r'image_(\d{14})', filename)
    if match:
        timestamp_str = match.group(1)
        return datetime.datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
    return None


def find_matching_image_pairs(dataset1_images: List[Path], 
                              dataset2_images: List[Path]) -> List[Dict]:
    """
    Find image pairs with matching timestamps.
    
    Args:
        dataset1_images: List of IPSL image paths
        dataset2_images: List of ECTL image paths
        
    Returns:
        List of dictionaries with matched pairs
    """
    # Parse timestamps for both datasets
    dataset1_parsed = []
    for img_path in dataset1_images:
        timestamp = extract_timestamp_from_ipsl(img_path.name)
        if timestamp:
            dataset1_parsed.append({
                'path': img_path,
                'timestamp': timestamp,
                'filename': img_path.name
            })
    
    dataset2_parsed = []
    for img_path in dataset2_images:
        timestamp = extract_timestamp_from_ectl(img_path.name)
        if timestamp:
            dataset2_parsed.append({
                'path': img_path,
                'timestamp': timestamp,
                'filename': img_path.name
            })
    
    # Find exact matches
    matched_pairs = []
    for item1 in dataset1_parsed:
        for item2 in dataset2_parsed:
            if item1['timestamp'] == item2['timestamp']:
                matched_pairs.append({
                    'ipsl_path': item1['path'],
                    'ipsl_filename': item1['filename'],
                    'ectl_path': item2['path'],
                    'ectl_filename': item2['filename'],
                    'timestamp': item1['timestamp']
                })
                break
    
    return matched_pairs


def filter_pairs_by_time(matched_pairs: List[Dict], 
                        start_hour: int, 
                        end_hour: int) -> List[Dict]:
    """
    Filter matched pairs to specific time window.
    
    Args:
        matched_pairs: List of matched image pairs
        start_hour: Start hour (inclusive)
        end_hour: End hour (exclusive)
        
    Returns:
        Filtered list of pairs
    """
    filtered = []
    for pair in matched_pairs:
        hour = pair['timestamp'].hour
        if start_hour <= hour < end_hour:
            filtered.append(pair)
    return filtered


def load_image_pair(pair: Dict, flip_ectl: bool = True) -> tuple:
    """
    Load a pair of images.
    
    Args:
        pair: Dictionary with image paths
        flip_ectl: Whether to vertically flip ECTL image
        
    Returns:
        tuple: (ipsl_image, ectl_image) as numpy arrays
    """
    ipsl_image = np.array(Image.open(pair['ipsl_path']).convert('RGB'))
    ectl_image = np.array(Image.open(pair['ectl_path']).convert('RGB'))
    
    if flip_ectl:
        ectl_image = np.flipud(ectl_image)
    
    return ipsl_image, ectl_image


def load_datasets(dataset1_dir: Path, dataset2_dir: Path) -> tuple:
    """
    Load all images from both datasets.
    
    Args:
        dataset1_dir: IPSL dataset directory
        dataset2_dir: ECTL dataset directory
        
    Returns:
        tuple: (dataset1_images, dataset2_images) as lists of Paths
    """
    dataset1_images = sorted(list(dataset1_dir.glob('*.jpg')))
    dataset2_images = sorted(list(dataset2_dir.glob('*.jpg')))
    
    print("="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Dataset 1 (IPSL): {len(dataset1_images)} images")
    if len(dataset1_images) > 0:
        print(f"  First: {dataset1_images[0].name}")
        print(f"  Last:  {dataset1_images[-1].name}")
    
    print(f"\nDataset 2 (ECTL): {len(dataset2_images)} images")
    if len(dataset2_images) > 0:
        print(f"  First: {dataset2_images[0].name}")
        print(f"  Last:  {dataset2_images[-1].name}")
    print("="*60)
    
    return dataset1_images, dataset2_images
