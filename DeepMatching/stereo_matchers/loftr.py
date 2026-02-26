"""
LoFTR (Local Feature TRansformer) matcher implementation using Kornia.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch

from .base import BaseMatcher

logger = logging.getLogger(__name__)


class LoFTRMatcher(BaseMatcher):
    """
    LoFTR matcher using Kornia's implementation.
    
    LoFTR is a detector-free local feature matching method that uses
    a coarse-to-fine matching strategy with transformers.
    
    Args:
        device: Device to run on ('cuda', 'cpu', or None for auto)
        pretrained: Pretrained weights ('outdoor' or 'indoor')
        
    Example:
        >>> matcher = LoFTRMatcher(device='cuda', pretrained='outdoor')
        >>> result = matcher.match(img1, img2)
        >>> print(f"Found {result.num_matches} matches")
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        pretrained: str = 'outdoor',
        **kwargs
    ):
        """
        Initialize LoFTR matcher.
        
        Args:
            device: Device to run on
            pretrained: Which pretrained weights to use ('outdoor' or 'indoor')
        """
        # Set attributes before super().__init__() since name property needs them
        self.pretrained = pretrained
        self.requires_grayscale = True
        super().__init__(device=device, **kwargs)
    
    @property
    def name(self) -> str:
        return f"LoFTR-{self.pretrained}"
    
    def _load_model(self) -> None:
        """Load the LoFTR model from Kornia."""
        try:
            from kornia.feature import LoFTR as KorniaLoFTR
        except ImportError:
            raise ImportError(
                "kornia is required for LoFTR. Install with: pip install kornia"
            )
        
        logger.info(f"Loading LoFTR with {self.pretrained} weights...")
        
        self.model = KorniaLoFTR(pretrained=self.pretrained)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"LoFTR model loaded on {self.device}")
    
    def _match_impl(
        self,
        image0: torch.Tensor,
        image1: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run LoFTR matching.
        
        Args:
            image0: First image tensor (1, 1, H, W) - grayscale
            image1: Second image tensor (1, 1, H, W) - grayscale
            
        Returns:
            Tuple of (keypoints0, keypoints1, confidence)
        """
        # Ensure grayscale (1 channel)
        if image0.shape[1] == 3:
            image0 = 0.299 * image0[:, 0:1] + 0.587 * image0[:, 1:2] + 0.114 * image0[:, 2:3]
        if image1.shape[1] == 3:
            image1 = 0.299 * image1[:, 0:1] + 0.587 * image1[:, 1:2] + 0.114 * image1[:, 2:3]
        
        # Prepare input dict
        input_dict = {
            "image0": image0,
            "image1": image1
        }
        
        # Run matching
        output = self.model(input_dict)
        
        # Extract results
        keypoints0 = output['keypoints0'].cpu().numpy()
        keypoints1 = output['keypoints1'].cpu().numpy()
        confidence = output['confidence'].cpu().numpy()
        
        return keypoints0, keypoints1, confidence


class LoFTRMatcherLightGlue(BaseMatcher):
    """
    LoFTR matcher using LightGlue's implementation (alternative).
    
    This uses the LightGlue library if available, which may have
    different performance characteristics than Kornia's version.
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        pretrained: str = 'outdoor',
        **kwargs
    ):
        # Set attributes before super().__init__() since name property needs them
        self.pretrained = pretrained
        self.requires_grayscale = True
        super().__init__(device=device, **kwargs)
    
    @property
    def name(self) -> str:
        return f"LoFTR-LG-{self.pretrained}"
    
    def _load_model(self) -> None:
        """Load LoFTR via LightGlue."""
        try:
            from lightglue import LoFTR
        except ImportError:
            raise ImportError(
                "LightGlue is required. Install from: "
                "https://github.com/cvg/LightGlue"
            )
        
        self.model = LoFTR(pretrained=self.pretrained).to(self.device)
        self.model.eval()
    
    def _match_impl(
        self,
        image0: torch.Tensor,
        image1: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run LightGlue LoFTR matching."""
        # Ensure grayscale
        if image0.shape[1] == 3:
            image0 = 0.299 * image0[:, 0:1] + 0.587 * image0[:, 1:2] + 0.114 * image0[:, 2:3]
        if image1.shape[1] == 3:
            image1 = 0.299 * image1[:, 0:1] + 0.587 * image1[:, 1:2] + 0.114 * image1[:, 2:3]
        
        # Run matching
        output = self.model({'image0': image0, 'image1': image1})
        
        keypoints0 = output['keypoints0'].cpu().numpy()
        keypoints1 = output['keypoints1'].cpu().numpy()
        confidence = output.get('confidence', torch.ones(len(keypoints0))).cpu().numpy()
        
        return keypoints0, keypoints1, confidence
