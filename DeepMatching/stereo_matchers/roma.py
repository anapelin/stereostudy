"""
RoMa (Robust Dense Feature Matching) matcher implementation.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch

from .base import BaseMatcher

logger = logging.getLogger(__name__)


class RoMaMatcher(BaseMatcher):
    """
    RoMa matcher for robust dense feature matching.
    
    RoMa is a robust dense feature matching method that produces
    high-quality dense correspondences between image pairs.
    
    Args:
        device: Device to run on ('cuda', 'cpu', or None for auto)
        model_type: Type of model ('outdoor', 'indoor', or 'dinov2')
        resolution: Resolution for matching (224, 560, or 896)
        
    Example:
        >>> matcher = RoMaMatcher(device='cuda', model_type='outdoor')
        >>> result = matcher.match(img1, img2)
        >>> print(f"Found {result.num_matches} matches")
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        model_type: str = 'outdoor',
        resolution: int = 560,
        upsample_res: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize RoMa matcher.
        
        Args:
            device: Device to run on
            model_type: Model variant ('outdoor', 'indoor', 'dinov2')
            resolution: Processing resolution
            upsample_res: Upsampling resolution for fine matches
        """
        # Set attributes before super().__init__() since name property needs them
        self.model_type = model_type
        self.resolution = resolution
        self.upsample_res = upsample_res or resolution * 2
        self.requires_grayscale = False  # RoMa uses RGB
        super().__init__(device=device, **kwargs)
    
    @property
    def name(self) -> str:
        return f"RoMa-{self.model_type}"
    
    def _load_model(self) -> None:
        """Load the RoMa model."""
        try:
            from romatch import roma_outdoor, roma_indoor
        except ImportError:
            try:
                # Alternative import
                import sys
                sys.path.append('/data/common/STEREOSTUDYIPSL/DeepMatching/RoMa')
                from romatch import roma_outdoor, roma_indoor
            except ImportError:
                raise ImportError(
                    "romatch is required for RoMa. Install with:\n"
                    "pip install romatch\n"
                    "or clone from: https://github.com/Parskatt/RoMa"
                )
        
        logger.info(f"Loading RoMa {self.model_type} model...")
        
        # use_custom_corr=False uses pure PyTorch local correlation
        # (slower but doesn't require local_corr CUDA extension)
        if self.model_type == 'indoor':
            self.model = roma_indoor(
                device=self.device,
                use_custom_corr=False
            )
        else:
            self.model = roma_outdoor(
                device=self.device,
                use_custom_corr=False
            )
        
        self.model.eval()
        logger.info(f"RoMa model loaded on {self.device}")
    
    def _match_impl(
        self,
        image0: torch.Tensor,
        image1: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run RoMa matching.
        
        Args:
            image0: First image tensor (1, 3, H, W) in 0-1 range
            image1: Second image tensor (1, 3, H, W) in 0-1 range
            
        Returns:
            Tuple of (keypoints0, keypoints1, confidence)
        """
        from PIL import Image as PILImage
        
        # RoMa expects PIL Images, not tensors
        # Convert tensors to PIL Images
        def tensor_to_pil(tensor):
            """Convert (1, C, H, W) tensor in 0-1 range to PIL RGB Image."""
            # Handle grayscale
            if tensor.shape[1] == 1:
                tensor = tensor.repeat(1, 3, 1, 1)
            # (1, 3, H, W) -> (H, W, 3)
            arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            # Convert to uint8
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
            return PILImage.fromarray(arr, mode='RGB')
        
        pil_img0 = tensor_to_pil(image0)
        pil_img1 = tensor_to_pil(image1)
        
        # Get original sizes
        w0, h0 = pil_img0.size  # PIL uses (width, height)
        w1, h1 = pil_img1.size
        
        # Get warp and certainty - RoMa expects PIL Images
        warp, certainty = self.model.match(
            pil_img0,
            pil_img1,
            device=self.device
        )
        
        # Sample matches from the dense correspondence
        matches, certainty_vals = self.model.sample(
            warp, 
            certainty,
            num=min(10000, w0 * h0 // 4)  # Sample up to 10k matches
        )
        
        # Convert normalized coordinates to pixel coordinates
        kpts0, kpts1 = self.model.to_pixel_coordinates(
            matches, 
            h0, w0, h1, w1
        )
        
        # Convert to numpy
        keypoints0 = kpts0.cpu().numpy()
        keypoints1 = kpts1.cpu().numpy()
        confidence = certainty_vals.cpu().numpy()
        
        # Flatten if needed
        if confidence.ndim > 1:
            confidence = confidence.flatten()
        
        return keypoints0, keypoints1, confidence


class RoMaTinyMatcher(BaseMatcher):
    """
    RoMa Tiny matcher - faster but less accurate version.
    
    Uses the tiny variant of RoMa for faster inference with
    slightly reduced accuracy.
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        **kwargs
    ):
        # Set attributes before super().__init__()
        self.requires_grayscale = False
        super().__init__(device=device, **kwargs)
    
    @property
    def name(self) -> str:
        return "RoMa-Tiny"
    
    def _load_model(self) -> None:
        """Load RoMa Tiny model."""
        try:
            from romatch import tiny_roma_v1_outdoor
        except ImportError:
            raise ImportError(
                "romatch is required. Install with: pip install romatch"
            )
        
        logger.info("Loading RoMa Tiny model...")
        self.model = tiny_roma_v1_outdoor(device=self.device)
        self.model.eval()
        logger.info(f"RoMa Tiny loaded on {self.device}")
    
    def _match_impl(
        self,
        image0: torch.Tensor,
        image1: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run RoMa Tiny matching."""
        if image0.shape[1] == 1:
            image0 = image0.repeat(1, 3, 1, 1)
        if image1.shape[1] == 1:
            image1 = image1.repeat(1, 3, 1, 1)
        
        h0, w0 = image0.shape[2:]
        h1, w1 = image1.shape[2:]
        
        warp, certainty = self.model.match(
            image0.squeeze(0),
            image1.squeeze(0),
            device=self.device
        )
        
        matches, certainty_vals = self.model.sample(
            warp, certainty,
            num=min(10000, w0 * h0 // 4)
        )
        
        kpts0, kpts1 = self.model.to_pixel_coordinates(
            matches, h0, w0, h1, w1
        )
        
        keypoints0 = kpts0.cpu().numpy()
        keypoints1 = kpts1.cpu().numpy()
        confidence = certainty_vals.cpu().numpy().flatten()
        
        return keypoints0, keypoints1, confidence
