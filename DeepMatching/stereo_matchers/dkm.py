"""
DKM (Dense Kernelized Feature Matching) matcher implementation.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch

from .base import BaseMatcher

logger = logging.getLogger(__name__)


class DKMMatcher(BaseMatcher):
    """
    DKM matcher for dense kernelized feature matching.
    
    DKM uses dense feature matching with a kernelized approach
    for robust correspondence estimation.
    
    Args:
        device: Device to run on ('cuda', 'cpu', or None for auto)
        model_type: Type of model ('outdoor', 'indoor', 'mega_synthetic')
        
    Example:
        >>> matcher = DKMMatcher(device='cuda', model_type='outdoor')
        >>> result = matcher.match(img1, img2)
        >>> print(f"Found {result.num_matches} matches")
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        model_type: str = 'outdoor',
        **kwargs
    ):
        """
        Initialize DKM matcher.
        
        Args:
            device: Device to run on
            model_type: Model variant ('outdoor', 'indoor', 'mega_synthetic')
        """
        # Set attributes before super().__init__() since name property needs them
        self.model_type = model_type
        self.requires_grayscale = False  # DKM uses RGB
        super().__init__(device=device, **kwargs)
    
    @property
    def name(self) -> str:
        return f"DKM-{self.model_type}"
    
    def _load_model(self) -> None:
        """Load the DKM model."""
        try:
            from dkm import DKMv3
        except ImportError:
            try:
                # Try alternative import path
                import sys
                sys.path.append('/data/common/STEREOSTUDYIPSL/DeepMatching/DKM')
                from dkm import DKMv3
            except ImportError:
                raise ImportError(
                    "dkm is required for DKM matching. Install from:\n"
                    "https://github.com/Parskatt/DKM\n"
                    "pip install dkm"
                )
        
        logger.info(f"Loading DKM {self.model_type} model...")
        
        # Load appropriate model
        if self.model_type == 'indoor':
            self.model = DKMv3(weights='indoor', device=self.device)
        elif self.model_type == 'mega_synthetic':
            self.model = DKMv3(weights='mega_synthetic', device=self.device)
        else:
            self.model = DKMv3(weights='outdoor', device=self.device)
        
        self.model.eval()
        logger.info(f"DKM model loaded on {self.device}")
    
    def _match_impl(
        self,
        image0: torch.Tensor,
        image1: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run DKM matching.
        
        Args:
            image0: First image tensor (1, 3, H, W)
            image1: Second image tensor (1, 3, H, W)
            
        Returns:
            Tuple of (keypoints0, keypoints1, confidence)
        """
        # DKM expects RGB images
        if image0.shape[1] == 1:
            image0 = image0.repeat(1, 3, 1, 1)
        if image1.shape[1] == 1:
            image1 = image1.repeat(1, 3, 1, 1)
        
        # Get original sizes
        h0, w0 = image0.shape[2:]
        h1, w1 = image1.shape[2:]
        
        # Get dense warp and certainty
        dense_matches, dense_certainty = self.model.match(
            image0.squeeze(0),
            image1.squeeze(0)
        )
        
        # Sample sparse matches from dense prediction
        sparse_matches, sparse_certainty = self.model.sample(
            dense_matches,
            dense_certainty,
            num=min(10000, w0 * h0 // 4)
        )
        
        # Convert to pixel coordinates
        kpts0, kpts1 = self.model.to_pixel_coordinates(
            sparse_matches,
            h0, w0, h1, w1
        )
        
        keypoints0 = kpts0.cpu().numpy()
        keypoints1 = kpts1.cpu().numpy()
        confidence = sparse_certainty.cpu().numpy().flatten()
        
        return keypoints0, keypoints1, confidence


class DKMLiteMatcher(BaseMatcher):
    """
    DKM Lite matcher - faster version with reduced accuracy.
    
    Uses a lightweight version of DKM for faster inference.
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
        return "DKM-Lite"
    
    def _load_model(self) -> None:
        """Load DKM Lite model."""
        try:
            from dkm import DKMv3
        except ImportError:
            raise ImportError(
                "dkm is required. Clone from: https://github.com/Parskatt/DKM"
            )
        
        logger.info("Loading DKM Lite model...")
        # Use the lightweight version if available
        self.model = DKMv3(weights='outdoor', device=self.device, h=384, w=512)
        self.model.eval()
        logger.info(f"DKM Lite loaded on {self.device}")
    
    def _match_impl(
        self,
        image0: torch.Tensor,
        image1: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run DKM Lite matching."""
        if image0.shape[1] == 1:
            image0 = image0.repeat(1, 3, 1, 1)
        if image1.shape[1] == 1:
            image1 = image1.repeat(1, 3, 1, 1)
        
        h0, w0 = image0.shape[2:]
        h1, w1 = image1.shape[2:]
        
        dense_matches, dense_certainty = self.model.match(
            image0.squeeze(0),
            image1.squeeze(0)
        )
        
        sparse_matches, sparse_certainty = self.model.sample(
            dense_matches,
            dense_certainty,
            num=min(5000, w0 * h0 // 8)
        )
        
        kpts0, kpts1 = self.model.to_pixel_coordinates(
            sparse_matches, h0, w0, h1, w1
        )
        
        keypoints0 = kpts0.cpu().numpy()
        keypoints1 = kpts1.cpu().numpy()
        confidence = sparse_certainty.cpu().numpy().flatten()
        
        return keypoints0, keypoints1, confidence
