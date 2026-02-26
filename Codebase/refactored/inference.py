"""
Segmentation inference utilities.
"""
import numpy as np
import torch
from config import TASK, ID2LABEL


def run_inference(image: np.ndarray, 
                 model, 
                 processor, 
                 device,
                 threshold: float = 0.5,
                 mask_threshold: float = 0.5,
                 overlap_threshold: float = 0.8):
    """
    Run segmentation inference on an image.
    
    Args:
        image: RGB image as numpy array
        model: Segmentation model
        processor: Image processor
        device: Device to run on
        threshold: Classification threshold
        mask_threshold: Mask threshold
        overlap_threshold: Overlap area threshold
        
    Returns:
        Segmentation results
    """
    # Prepare inputs
    inputs = processor([image], return_tensors="pt").to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process
    target_sizes = [image.shape[:2]]
    
    if TASK == "panoptic":
        kwargs = {
            "threshold": threshold,
            "mask_threshold": mask_threshold,
            "overlap_mask_area_threshold": overlap_threshold,
        }
        segmentation = processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=target_sizes,
            **kwargs
        )[0]
    else:
        kwargs = {
            "threshold": threshold,
            "mask_threshold": mask_threshold,
            "overlap_mask_area_threshold": overlap_threshold,
            "return_binary_maps": True,
        }
        segmentation = processor.post_process_instance_segmentation(
            outputs,
            target_sizes=target_sizes,
            **kwargs
        )[0]
    
    return segmentation


def extract_statistics(segmentation, label_name: str = 'contrail'):
    """
    Extract statistics from segmentation results.
    
    Args:
        segmentation: Segmentation results
        label_name: Label to extract stats for
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_segments': 0,
        'contrail_segments': 0,
        'total_pixels': 0,
        'contrail_pixels': 0,
    }
    
    if 'segments_info' not in segmentation:
        return stats
    
    segments_info = segmentation['segments_info']
    stats['total_segments'] = len(segments_info)
    
    # Count contrail segments and pixels
    if 'segmentation' in segmentation:
        seg_mask = segmentation['segmentation'].cpu().numpy()
        stats['total_pixels'] = seg_mask.size
        
        for seg_info in segments_info:
            label_id = seg_info['label_id']
            if ID2LABEL.get(label_id, '') == label_name:
                stats['contrail_segments'] += 1
                # Count pixels for this segment
                segment_id = seg_info['id']
                stats['contrail_pixels'] += np.sum(seg_mask == segment_id)
    
    return stats


def process_image_pair(ipsl_image, ectl_image, model, processor, device,
                      apply_histogram_matching=False):
    """
    Process a pair of images through the inference pipeline.
    
    Args:
        ipsl_image: IPSL image
        ectl_image: ECTL image
        model: Segmentation model
        processor: Image processor
        device: Device to run on
        apply_histogram_matching: Whether to apply histogram matching
        
    Returns:
        dict: Results including segmentations and statistics
    """
    from image_processing import preprocess_image
    
    # Preprocess IPSL image
    ipsl_processed = preprocess_image(
        ipsl_image,
        reference_image=ectl_image if apply_histogram_matching else None,
        apply_histogram_matching=apply_histogram_matching
    )
    
    # Run inference
    print("  Running inference on IPSL image...")
    ipsl_seg = run_inference(ipsl_processed, model, processor, device)
    
    print("  Running inference on ECTL image...")
    ectl_seg = run_inference(ectl_image, model, processor, device)
    
    # Extract statistics
    ipsl_stats = extract_statistics(ipsl_seg)
    ectl_stats = extract_statistics(ectl_seg)
    
    print(f"\n  IPSL: {ipsl_stats['total_segments']} segments "
          f"({ipsl_stats['contrail_segments']} contrails, "
          f"{ipsl_stats['contrail_pixels']} pixels)")
    print(f"  ECTL: {ectl_stats['total_segments']} segments "
          f"({ectl_stats['contrail_segments']} contrails, "
          f"{ectl_stats['contrail_pixels']} pixels)")
    
    return {
        'ipsl_image': ipsl_processed,
        'ectl_image': ectl_image,
        'ipsl_seg': ipsl_seg,
        'ectl_seg': ectl_seg,
        'ipsl_stats': ipsl_stats,
        'ectl_stats': ectl_stats,
    }
