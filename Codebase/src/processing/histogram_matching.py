# Histogram Matching Implementation

from skimage import exposure
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from PIL import Image
import os
import cv2

def compute_color_histogram(image_paths: List[str], num_samples: int = 50) -> np.ndarray:
    if len(image_paths) == 0:
        print("Error: No images provided")
        return None
    
    samples = min(num_samples, len(image_paths))
    sampled_paths = np.random.choice(image_paths, samples, replace=False)
    
    all_images = []
    print(f"Computing color histogram from {samples} sample images...")
    
    for path in tqdm(sampled_paths, desc="Loading samples"):
        try:
            img = Image.open(path)
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np.array(img)
            
            # Ensure image has valid shape
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                all_images.append(img_array)
            else:
                print(f"Skipping {os.path.basename(path)}: invalid shape {img_array.shape}")
        except Exception as e:
            print(f"Error loading {os.path.basename(path)}: {e}")
            continue
    
    if not all_images:
        print("Error: No valid images loaded")
        return None
    
    print(f"Successfully loaded {len(all_images)} images")
    
    # Find common dimensions (use the most common size)
    shapes = [img.shape for img in all_images]
    from collections import Counter
    most_common_shape = Counter(shapes).most_common(1)[0][0]
    print(f"Using shape: {most_common_shape}")
    
    # Filter images to match common shape
    filtered_images = [img for img in all_images if img.shape == most_common_shape]
    
    if not filtered_images:
        print("Error: No images with matching dimensions")
        return None
    
    # Compute mean image as reference
    mean_image = np.mean(filtered_images, axis=0).astype(np.uint8)
    return mean_image


def match_histogram(img1, img2) -> np.ndarray:
    """
    Match the histogram of source image to reference image.
    
    Args:
        source_image: Image to transform (RGB)
        reference_image: Reference image with target histogram (RGB)
        
    Returns:
        Histogram-matched image
    """
    # Load images if they are file paths
    def load_img(img):
        if isinstance(img, str):
            # It's a file path
            loaded = Image.open(img)
            if loaded.mode != 'RGB':
                loaded = loaded.convert('RGB')
            return np.array(loaded)
        else:
            # Already a numpy array
            return img
    
    img1_array = load_img(img1)
    img2_array = load_img(img2)
    
    # Apply histogram matching per channel
    matched = exposure.match_histograms(img1_array, img2_array, channel_axis=2)
    return matched.astype(np.uint8)


def visualize_histograms(img1, img2, img2_matched=None):
    """
    Visualize color histograms for comparison.
    
    Args:
        img1: Dataset 1 image (file path or numpy array)
        img2: Dataset 2 original image (file path or numpy array)
        img2_matched: Dataset 2 image after histogram matching (file path or numpy array, optional)
    """
    # Load images if they are file paths
    def load_img(img):
        if isinstance(img, str):
            # It's a file path
            loaded = Image.open(img)
            if loaded.mode != 'RGB':
                loaded = loaded.convert('RGB')
            return np.array(loaded)
        else:
            # Already a numpy array
            return img
    
    img1_array = load_img(img1)
    img2_array = load_img(img2)
    img2_matched_array = load_img(img2_matched) if img2_matched is not None else None
    
    colors = ['red', 'green', 'blue']
    labels = ['Red', 'Green', 'Blue']
    
    num_plots = 3 if img2_matched_array is not None else 2
    fig, axes = plt.subplots(2, num_plots, figsize=(6*num_plots, 8))
    
    # Show images
    axes[0, 0].imshow(img1_array)
    axes[0, 0].set_title('Dataset 1 (Reference)', fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.flip(img2_array, 0))
    axes[0, 1].set_title('Dataset 2 (Original)', fontweight='bold')
    axes[0, 1].axis('off')
    
    if img2_matched_array is not None:
        axes[0, 2].imshow(img2_matched_array)
        axes[0, 2].set_title('Dataset 2 (Matched)', fontweight='bold')
        axes[0, 2].axis('off')
    
    # Plot histograms
    for i, (color, label) in enumerate(zip(colors, labels)):
        # Dataset 1 histogram
        hist1, bins1 = np.histogram(img1_array[:,:,i].flatten(), bins=256, range=(0, 256))
        axes[1, 0].plot(bins1[:-1], hist1, color=color, alpha=0.7, label=label)
        
        # Dataset 2 original histogram
        hist2, bins2 = np.histogram(img2_array[:,:,i].flatten(), bins=256, range=(0, 256))
        axes[1, 1].plot(bins2[:-1], hist2, color=color, alpha=0.7, label=label)
        
        # Dataset 2 matched histogram
        if img2_matched_array is not None:
            hist3, bins3 = np.histogram(img2_matched_array[:,:,i].flatten(), bins=256, range=(0, 256))
            axes[1, 2].plot(bins3[:-1], hist3, color=color, alpha=0.7, label=label)
    
    axes[1, 0].set_title('Dataset 1 Histogram')
    axes[1, 0].set_xlabel('Pixel Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].set_title('Dataset 2 Original Histogram')
    axes[1, 1].set_xlabel('Pixel Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    if img2_matched_array is not None:
        axes[1, 2].set_title('Dataset 2 Matched Histogram')
        axes[1, 2].set_xlabel('Pixel Value')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].legend()
        axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

print("✓ Histogram matching functions defined")