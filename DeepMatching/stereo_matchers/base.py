"""
Base classes and interfaces for stereo matchers.
"""

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, Dict, Any, List

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """
    Standardized output format for all stereo matchers.
    
    Attributes:
        keypoints0: Nx2 array of keypoint coordinates in image 1
        keypoints1: Nx2 array of keypoint coordinates in image 2
        confidence: N-length array of confidence scores (0-1)
        num_matches: Total number of matches found
        inference_time: Time taken for inference in seconds
        model_name: Name of the model used
        extra: Additional model-specific outputs
    """
    keypoints0: np.ndarray
    keypoints1: np.ndarray
    confidence: np.ndarray
    num_matches: int
    inference_time: float
    model_name: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the result structure."""
        assert len(self.keypoints0) == len(self.keypoints1) == len(self.confidence), \
            "Keypoints and confidence arrays must have the same length"
        assert self.keypoints0.shape[1] == 2, "Keypoints must be Nx2"
        assert self.keypoints1.shape[1] == 2, "Keypoints must be Nx2"
    
    def filter_by_confidence(self, threshold: float = 0.5) -> 'MatchResult':
        """
        Filter matches by confidence threshold.
        
        Args:
            threshold: Minimum confidence to keep
            
        Returns:
            New MatchResult with filtered matches
        """
        mask = self.confidence >= threshold
        return MatchResult(
            keypoints0=self.keypoints0[mask],
            keypoints1=self.keypoints1[mask],
            confidence=self.confidence[mask],
            num_matches=int(mask.sum()),
            inference_time=self.inference_time,
            model_name=self.model_name,
            extra=self.extra
        )
    
    def top_k(self, k: int) -> 'MatchResult':
        """
        Get top-k matches by confidence.
        
        Args:
            k: Number of top matches to return
            
        Returns:
            New MatchResult with top-k matches
        """
        if k >= self.num_matches:
            return self
        
        indices = np.argsort(self.confidence)[::-1][:k]
        return MatchResult(
            keypoints0=self.keypoints0[indices],
            keypoints1=self.keypoints1[indices],
            confidence=self.confidence[indices],
            num_matches=k,
            inference_time=self.inference_time,
            model_name=self.model_name,
            extra=self.extra
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'keypoints0': self.keypoints0,
            'keypoints1': self.keypoints1,
            'confidence': self.confidence,
            'num_matches': self.num_matches,
            'inference_time': self.inference_time,
            'model_name': self.model_name,
            'extra': self.extra
        }
    
    @property
    def confidence_stats(self) -> Dict[str, float]:
        """Get confidence statistics."""
        if self.num_matches == 0:
            return {'mean': 0.0, 'median': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        return {
            'mean': float(np.mean(self.confidence)),
            'median': float(np.median(self.confidence)),
            'std': float(np.std(self.confidence)),
            'min': float(np.min(self.confidence)),
            'max': float(np.max(self.confidence))
        }


class BaseMatcher(ABC):
    """
    Abstract base class for all stereo matchers.
    
    All matcher implementations should inherit from this class
    and implement the `_match_impl` method.
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the matcher.
        
        Args:
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
            **kwargs: Additional model-specific parameters
        """
        self.device = self._get_device(device)
        self.model = None
        self.model_loaded = False
        self._kwargs = kwargs
        
        logger.info(f"Initializing {self.name} on device: {self.device}")
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the matcher."""
        pass
    
    @abstractmethod
    def _load_model(self) -> None:
        """Load the model. Called lazily on first match."""
        pass
    
    @abstractmethod
    def _match_impl(
        self,
        image0: torch.Tensor,
        image1: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Internal matching implementation.
        
        Args:
            image0: First image as tensor (C, H, W)
            image1: Second image as tensor (C, H, W)
            
        Returns:
            Tuple of (keypoints0, keypoints1, confidence)
        """
        pass
    
    def _get_device(self, device: Optional[str]) -> str:
        """Determine the device to use."""
        if device is not None:
            return device
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def _preprocess_image(
        self,
        image: Union[np.ndarray, torch.Tensor],
        grayscale: bool = False
    ) -> torch.Tensor:
        """
        Preprocess image to tensor format.
        
        Args:
            image: Input image (numpy array or torch tensor)
            grayscale: Whether to convert to grayscale
            
        Returns:
            Preprocessed tensor (1, C, H, W)
        """
        # Convert numpy to tensor
        if isinstance(image, np.ndarray):
            # Ensure float and 0-1 range
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            elif image.max() > 1.0:
                image = image.astype(np.float32) / 255.0
            else:
                image = image.astype(np.float32)
            
            # Handle dimensions
            if image.ndim == 2:
                # Grayscale: (H, W) -> (1, 1, H, W)
                image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
            elif image.ndim == 3:
                # RGB/BGR: (H, W, C) -> (1, C, H, W)
                image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")
        
        elif isinstance(image, torch.Tensor):
            # Ensure 4D tensor
            if image.ndim == 2:
                image = image.unsqueeze(0).unsqueeze(0)
            elif image.ndim == 3:
                image = image.unsqueeze(0)
            
            # Ensure float
            if image.dtype != torch.float32:
                image = image.float()
            
            # Ensure 0-1 range
            if image.max() > 1.0:
                image = image / 255.0
        
        else:
            raise TypeError(f"Expected numpy array or torch tensor, got {type(image)}")
        
        # Convert to grayscale if needed
        if grayscale and image.shape[1] == 3:
            # RGB to grayscale using luminance formula
            image = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        
        # Move to device
        image = image.to(self.device)
        
        return image
    
    def match(
        self,
        image0: Union[np.ndarray, torch.Tensor],
        image1: Union[np.ndarray, torch.Tensor],
        **kwargs
    ) -> MatchResult:
        """
        Match features between two images.
        
        Args:
            image0: First image (numpy array or torch tensor)
            image1: Second image (numpy array or torch tensor)
            **kwargs: Additional matching parameters
            
        Returns:
            MatchResult with standardized output
        """
        # Load model if not loaded
        if not self.model_loaded:
            logger.info(f"Loading {self.name} model...")
            self._load_model()
            self.model_loaded = True
        
        # Preprocess images
        grayscale = kwargs.get('grayscale', getattr(self, 'requires_grayscale', False))
        img0 = self._preprocess_image(image0, grayscale=grayscale)
        img1 = self._preprocess_image(image1, grayscale=grayscale)
        
        # Time the inference
        start_time = time.time()
        
        try:
            with torch.no_grad():
                kpts0, kpts1, conf = self._match_impl(img0, img1)
        except Exception as e:
            logger.error(f"Error during matching with {self.name}: {e}")
            # Return empty result on error
            return MatchResult(
                keypoints0=np.zeros((0, 2)),
                keypoints1=np.zeros((0, 2)),
                confidence=np.zeros(0),
                num_matches=0,
                inference_time=time.time() - start_time,
                model_name=self.name,
                extra={'error': str(e)}
            )
        
        inference_time = time.time() - start_time
        
        # Ensure numpy arrays
        if isinstance(kpts0, torch.Tensor):
            kpts0 = kpts0.cpu().numpy()
        if isinstance(kpts1, torch.Tensor):
            kpts1 = kpts1.cpu().numpy()
        if isinstance(conf, torch.Tensor):
            conf = conf.cpu().numpy()
        
        # Create result
        result = MatchResult(
            keypoints0=kpts0,
            keypoints1=kpts1,
            confidence=conf,
            num_matches=len(kpts0),
            inference_time=inference_time,
            model_name=self.name
        )
        
        logger.info(f"{self.name}: Found {result.num_matches} matches in {inference_time:.3f}s")
        
        return result
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device}, loaded={self.model_loaded})"
