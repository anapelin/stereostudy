"""
Image preprocessing utilities.
"""
import numpy as np
from skimage import exposure
from scipy.ndimage import median_filter, binary_dilation


def match_histogram(source_image: np.ndarray, 
                   reference_image: np.ndarray) -> np.ndarray:
    """
    Match the histogram of source image to reference image.
    
    Args:
        source_image: Image to transform (RGB numpy array)
        reference_image: Reference image with target histogram (RGB numpy array)
        
    Returns:
        Histogram-matched image
    """
    matched = exposure.match_histograms(source_image, reference_image, channel_axis=2)
    return matched.astype(np.uint8)


def remove_sun_pixels(image: np.ndarray, 
                     brightness_threshold: int = 240) -> np.ndarray:
    """
    Simple sun removal: replace very bright pixels with surrounding sky color.
    
    Args:
        image: RGB image as numpy array
        brightness_threshold: Minimum brightness (0-255) to consider as sun
        
    Returns:
        Image with sun pixels replaced
    """
    # Convert to grayscale to find brightest pixels
    gray = np.mean(image, axis=2)
    
    # Find sun pixels (very bright areas)
    sun_mask = gray > brightness_threshold
    
    # Only proceed if we found sun pixels
    if not np.any(sun_mask):
        print("    No bright sun pixels found")
        return image.copy()
    
    # Dilate the mask slightly to include sun glare around edges
    sun_mask_dilated = binary_dilation(sun_mask, iterations=2)
    
    # Find replacement color: median of non-sun pixels in upper part of image
    h, w = image.shape[:2]
    sky_region = image[:h//2, :][~sun_mask_dilated[:h//2, :]]
    
    if len(sky_region) > 100:
        # Use median color from sky region
        replacement_color = np.median(sky_region, axis=0).astype(np.uint8)
    else:
        # Fallback: use median of all non-sun pixels
        non_sun_pixels = image[~sun_mask_dilated]
        if len(non_sun_pixels) > 0:
            replacement_color = np.median(non_sun_pixels, axis=0).astype(np.uint8)
        else:
            replacement_color = np.array([135, 206, 235], dtype=np.uint8)  # Sky blue
    
    # Create output image
    output_image = image.copy()
    output_image[sun_mask_dilated] = replacement_color
    
    # Apply a small median filter to smooth transitions
    for i in range(3):
        channel = output_image[:, :, i].astype(np.float32)
        mask_float = sun_mask_dilated.astype(np.float32)
        channel_blurred = median_filter(channel, size=3)
        output_image[:, :, i] = (
            channel * (1 - mask_float) + channel_blurred * mask_float
        ).astype(np.uint8)
    
    num_pixels_removed = np.sum(sun_mask_dilated)
    percent_removed = 100 * num_pixels_removed / sun_mask_dilated.size
    print(f"    Removed {num_pixels_removed} sun pixels ({percent_removed:.2f}%)")
    
    return output_image


def preprocess_image(image: np.ndarray,
                    reference_image: np.ndarray = None,
                    apply_histogram_matching: bool = False,
                    remove_sun: bool = False) -> np.ndarray:
    """
    Apply preprocessing steps to an image.
    
    Args:
        image: Input image
        reference_image: Reference for histogram matching
        apply_histogram_matching: Whether to apply histogram matching
        remove_sun: Whether to remove sun pixels
        
    Returns:
        Preprocessed image
    """
    processed = image.copy()
    
    if remove_sun:
        processed = remove_sun_pixels(processed)
    
    if apply_histogram_matching and reference_image is not None:
        processed = match_histogram(processed, reference_image)
    
    return processed
