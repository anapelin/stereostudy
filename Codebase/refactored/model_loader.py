"""
Model loading utilities for segmentation.
"""
import torch
from transformers import (
    Mask2FormerImageProcessor,
    Mask2FormerForUniversalSegmentation,
)
from config import BASE_MODEL, CHECKPOINT_DIR, ID2LABEL


def load_processor():
    """
    Load and configure the image processor.
    
    Returns:
        Mask2FormerImageProcessor: Configured processor
    """
    processor = Mask2FormerImageProcessor.from_pretrained(
        BASE_MODEL,
        do_resize=False,
        do_rescale=False,
        do_normalize=True,
        do_reduce_labels=True,
        ignore_index=255,
    )
    return processor


def load_model(device="cuda"):
    """
    Load the segmentation model.
    
    Args:
        device: Device to load model on ('cuda' or 'cpu')
        
    Returns:
        model: Loaded model
        device: Device model is on
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        CHECKPOINT_DIR,
        id2label=ID2LABEL,
        ignore_mismatched_sizes=True,
    )
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    print(f"  Model classes: {model.config.num_labels}")
    
    return model, device


def setup_model_and_processor():
    """
    Complete setup for model and processor.
    
    Returns:
        tuple: (model, processor, device)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = load_processor()
    model, device = load_model(device)
    
    return model, processor, device
