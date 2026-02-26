"""
Utility functions for stereo matchers.
"""

import logging
from pathlib import Path
from typing import Union, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def get_device(device: Optional[str] = None) -> str:
    """
    Get the appropriate device for computation.
    
    Args:
        device: Explicit device string, or None for auto-detection
        
    Returns:
        Device string ('cuda' or 'cpu')
    """
    if device is not None:
        return device
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_image(
    path: Union[str, Path],
    as_tensor: bool = False,
    grayscale: bool = False,
    resize: Optional[Tuple[int, int]] = None
) -> Union[np.ndarray, torch.Tensor]:
    """
    Load an image from disk.
    
    Args:
        path: Path to the image file
        as_tensor: If True, return as torch tensor
        grayscale: If True, convert to grayscale
        resize: Optional (width, height) to resize to
        
    Returns:
        Image as numpy array (H, W, C) or torch tensor (C, H, W)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    # Load with PIL
    img = Image.open(path)
    
    if grayscale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    
    if resize is not None:
        img = img.resize(resize, Image.LANCZOS)
    
    # Convert to numpy
    arr = np.array(img)
    
    if as_tensor:
        # Convert to tensor
        if arr.ndim == 2:
            # Grayscale: (H, W) -> (1, H, W)
            tensor = torch.from_numpy(arr).float().unsqueeze(0) / 255.0
        else:
            # RGB: (H, W, C) -> (C, H, W)
            tensor = torch.from_numpy(arr).float().permute(2, 0, 1) / 255.0
        return tensor
    
    return arr


def ensure_tensor(
    image: Union[np.ndarray, torch.Tensor],
    device: str = 'cpu',
    normalize: bool = True
) -> torch.Tensor:
    """
    Ensure input is a torch tensor.
    
    Args:
        image: Input image (numpy or tensor)
        device: Device to move tensor to
        normalize: If True, normalize to 0-1 range
        
    Returns:
        Torch tensor
    """
    if isinstance(image, np.ndarray):
        # Convert numpy to tensor
        tensor = torch.from_numpy(image.copy())
        
        # Handle dimensions
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)  # Add channel dim
        elif tensor.ndim == 3 and tensor.shape[-1] in [1, 3, 4]:
            tensor = tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    else:
        tensor = image.clone()
    
    # Ensure float
    tensor = tensor.float()
    
    # Normalize if needed
    if normalize and tensor.max() > 1.0:
        tensor = tensor / 255.0
    
    return tensor.to(device)


def ensure_numpy(
    image: Union[np.ndarray, torch.Tensor],
    denormalize: bool = True
) -> np.ndarray:
    """
    Ensure input is a numpy array.
    
    Args:
        image: Input image (numpy or tensor)
        denormalize: If True and values are 0-1, multiply by 255
        
    Returns:
        Numpy array
    """
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
    else:
        arr = image.copy()
    
    # Handle tensor dimensions (C, H, W) -> (H, W, C)
    if arr.ndim == 3 and arr.shape[0] in [1, 3, 4]:
        arr = arr.transpose(1, 2, 0)
    
    # Squeeze single channel
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    
    # Denormalize if needed
    if denormalize and arr.max() <= 1.0:
        arr = (arr * 255).astype(np.uint8)
    
    return arr


def resize_image(
    image: Union[np.ndarray, torch.Tensor],
    size: Tuple[int, int],
    keep_aspect: bool = True
) -> Union[np.ndarray, torch.Tensor]:
    """
    Resize an image to the target size.
    
    Args:
        image: Input image
        size: Target (width, height)
        keep_aspect: If True, resize while maintaining aspect ratio
        
    Returns:
        Resized image (same type as input)
    """
    import torch.nn.functional as F
    
    is_tensor = isinstance(image, torch.Tensor)
    
    if not is_tensor:
        # Convert to tensor for resizing
        tensor = ensure_tensor(image, normalize=False)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
    else:
        tensor = image
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
    
    _, _, h, w = tensor.shape
    target_w, target_h = size
    
    if keep_aspect:
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
    else:
        new_w, new_h = target_w, target_h
    
    # Resize
    resized = F.interpolate(tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
    
    # Squeeze batch dimension
    resized = resized.squeeze(0)
    
    if not is_tensor:
        return ensure_numpy(resized, denormalize=True)
    
    return resized


def pad_to_size(
    image: Union[np.ndarray, torch.Tensor],
    size: Tuple[int, int],
    pad_value: float = 0
) -> Union[np.ndarray, torch.Tensor]:
    """
    Pad an image to the target size.
    
    Args:
        image: Input image
        size: Target (width, height)
        pad_value: Value to use for padding
        
    Returns:
        Padded image (same type as input)
    """
    import torch.nn.functional as F
    
    is_tensor = isinstance(image, torch.Tensor)
    
    if not is_tensor:
        tensor = ensure_tensor(image, normalize=False)
    else:
        tensor = image
    
    if tensor.ndim == 3:
        c, h, w = tensor.shape
    else:
        h, w = tensor.shape
        c = 1
        tensor = tensor.unsqueeze(0)
    
    target_w, target_h = size
    
    pad_w = max(0, target_w - w)
    pad_h = max(0, target_h - h)
    
    # Pad (left, right, top, bottom)
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), value=pad_value)
    
    if not is_tensor:
        return ensure_numpy(padded, denormalize=True)
    
    return padded


def make_divisible(
    image: Union[np.ndarray, torch.Tensor],
    divisor: int = 8,
    mode: str = 'pad'
) -> Union[np.ndarray, torch.Tensor]:
    """
    Make image dimensions divisible by a factor.
    
    Args:
        image: Input image
        divisor: Factor to make dimensions divisible by
        mode: 'pad' or 'crop'
        
    Returns:
        Modified image
    """
    is_tensor = isinstance(image, torch.Tensor)
    
    if is_tensor:
        if image.ndim == 3:
            _, h, w = image.shape
        else:
            h, w = image.shape
    else:
        if image.ndim == 3:
            h, w, _ = image.shape
        else:
            h, w = image.shape
    
    new_h = ((h + divisor - 1) // divisor) * divisor
    new_w = ((w + divisor - 1) // divisor) * divisor
    
    if mode == 'pad':
        return pad_to_size(image, (new_w, new_h))
    else:
        # Crop
        if is_tensor:
            if image.ndim == 3:
                return image[:, :new_h, :new_w]
            return image[:new_h, :new_w]
        else:
            if image.ndim == 3:
                return image[:new_h, :new_w, :]
            return image[:new_h, :new_w]


def scale_keypoints(
    keypoints: np.ndarray,
    original_size: Tuple[int, int],
    new_size: Tuple[int, int]
) -> np.ndarray:
    """
    Scale keypoints from one image size to another.
    
    Args:
        keypoints: Nx2 array of (x, y) coordinates
        original_size: (width, height) of original image
        new_size: (width, height) of new image
        
    Returns:
        Scaled keypoints
    """
    scale_x = new_size[0] / original_size[0]
    scale_y = new_size[1] / original_size[1]
    
    scaled = keypoints.copy()
    scaled[:, 0] *= scale_x
    scaled[:, 1] *= scale_y
    
    return scaled


def compute_epipolar_error(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    F: np.ndarray
) -> np.ndarray:
    """
    Compute epipolar error for matched keypoints.
    
    Args:
        kpts0: Nx2 keypoints in image 0
        kpts1: Nx2 keypoints in image 1
        F: 3x3 fundamental matrix
        
    Returns:
        N-length array of epipolar errors
    """
    # Convert to homogeneous coordinates
    ones = np.ones((len(kpts0), 1))
    pts0 = np.hstack([kpts0, ones])  # Nx3
    pts1 = np.hstack([kpts1, ones])  # Nx3
    
    # Compute epipolar lines in image 1
    lines1 = (F @ pts0.T).T  # Nx3
    
    # Normalize lines
    norm = np.sqrt(lines1[:, 0]**2 + lines1[:, 1]**2) + 1e-8
    lines1 = lines1 / norm[:, None]
    
    # Compute distance from points to lines
    errors = np.abs(np.sum(pts1 * lines1, axis=1))
    
    return errors


def estimate_fundamental_matrix(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    method: str = 'ransac'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate fundamental matrix from matched keypoints.
    
    Args:
        kpts0: Nx2 keypoints in image 0
        kpts1: Nx2 keypoints in image 1
        method: 'ransac' or '8point'
        
    Returns:
        Tuple of (fundamental_matrix, inlier_mask)
    """
    import cv2
    
    if len(kpts0) < 8:
        return None, np.zeros(len(kpts0), dtype=bool)
    
    if method == 'ransac':
        F, mask = cv2.findFundamentalMat(
            kpts0, kpts1,
            cv2.FM_RANSAC,
            ransacReprojThreshold=3.0,
            confidence=0.99
        )
    else:
        F, mask = cv2.findFundamentalMat(
            kpts0, kpts1,
            cv2.FM_8POINT
        )
    
    if mask is None:
        mask = np.zeros(len(kpts0), dtype=bool)
    else:
        mask = mask.ravel().astype(bool)
    
    return F, mask
