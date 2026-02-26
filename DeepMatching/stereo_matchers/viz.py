"""
Visualization utilities for stereo matching results.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch

from .base import MatchResult

logger = logging.getLogger(__name__)


def visualize_matches(
    image0: np.ndarray,
    image1: np.ndarray,
    result: Union[MatchResult, Dict],
    top_k: Optional[int] = None,
    confidence_threshold: float = 0.0,
    figsize: Tuple[int, int] = (16, 8),
    line_width: float = 0.5,
    point_size: float = 4,
    alpha: float = 0.7,
    color_by_confidence: bool = True,
    cmap: str = 'viridis',
    title: Optional[str] = None,
    show_stats: bool = True
) -> Figure:
    """
    Visualize feature matches between two images.
    
    Args:
        image0: First image (H, W, 3) or (H, W)
        image1: Second image (H, W, 3) or (H, W)
        result: MatchResult or dict with keypoints and confidence
        top_k: Number of top matches to show (by confidence)
        confidence_threshold: Minimum confidence to show
        figsize: Figure size
        line_width: Width of match lines
        point_size: Size of keypoint markers
        alpha: Transparency of lines
        color_by_confidence: Color matches by confidence
        cmap: Colormap to use
        title: Optional title
        show_stats: Whether to show statistics overlay
        
    Returns:
        matplotlib Figure
    """
    # Convert dict to MatchResult if needed
    if isinstance(result, dict):
        result = MatchResult(
            keypoints0=result['keypoints0'],
            keypoints1=result['keypoints1'],
            confidence=result.get('confidence', np.ones(len(result['keypoints0']))),
            num_matches=len(result['keypoints0']),
            inference_time=result.get('inference_time', 0),
            model_name=result.get('model_name', '')
        )
    
    # Filter by confidence
    if confidence_threshold > 0:
        result = result.filter_by_confidence(confidence_threshold)
    
    # Get top-k
    if top_k is not None and top_k < result.num_matches:
        result = result.top_k(top_k)
    
    kpts0 = result.keypoints0
    kpts1 = result.keypoints1
    conf = result.confidence
    
    # Ensure images are 3-channel for display
    if image0.ndim == 2:
        image0 = np.stack([image0] * 3, axis=-1)
    if image1.ndim == 2:
        image1 = np.stack([image1] * 3, axis=-1)
    
    # Normalize images to 0-255 uint8
    if image0.max() <= 1.0:
        image0 = (image0 * 255).astype(np.uint8)
    if image1.max() <= 1.0:
        image1 = (image1 * 255).astype(np.uint8)
    
    # Create figure with two subplots
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)
    
    ax0.imshow(image0)
    ax0.set_title('Image 0', fontsize=12)
    ax0.axis('off')
    
    ax1.imshow(image1)
    ax1.set_title('Image 1', fontsize=12)
    ax1.axis('off')
    
    # Color mapping
    if color_by_confidence and len(conf) > 0:
        # Normalize confidence to 0-1
        if conf.max() > conf.min():
            norm_conf = (conf - conf.min()) / (conf.max() - conf.min())
        else:
            norm_conf = np.ones_like(conf)
        colors = cm.get_cmap(cmap)(norm_conf)
    else:
        colors = ['lime'] * len(kpts0)
    
    # Draw matches
    for i in range(len(kpts0)):
        # Draw points
        ax0.scatter(kpts0[i, 0], kpts0[i, 1], c=[colors[i]], s=point_size, zorder=5)
        ax1.scatter(kpts1[i, 0], kpts1[i, 1], c=[colors[i]], s=point_size, zorder=5)
        
        # Draw connecting line
        con = ConnectionPatch(
            xyA=(kpts0[i, 0], kpts0[i, 1]),
            xyB=(kpts1[i, 0], kpts1[i, 1]),
            coordsA="data", coordsB="data",
            axesA=ax0, axesB=ax1,
            color=colors[i],
            linewidth=line_width,
            alpha=alpha
        )
        ax1.add_artist(con)
    
    # Add title
    if title is None:
        title = f"{result.model_name}" if result.model_name else "Feature Matches"
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Add statistics
    if show_stats and result.num_matches > 0:
        stats_text = (
            f"Matches: {result.num_matches}\n"
            f"Time: {result.inference_time:.3f}s\n"
            f"Conf: {conf.mean():.3f}±{conf.std():.3f}"
        )
        fig.text(0.02, 0.02, stats_text, fontsize=10, 
                family='monospace', verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add colorbar if using confidence colors
    # if color_by_confidence and len(conf) > 0:
    #     sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(conf.min(), conf.max()))
    #     sm.set_array([])
    #     cbar = fig.colorbar(sm, ax=[ax0, ax1], shrink=0.5, aspect=30, pad=0.02)
    #     cbar.set_label('Confidence', fontsize=10)

    #     # put it in the middle of the figure
    #     cbar.ax.yaxis.set_label_position('left')
    #     cbar.ax.yaxis.set_ticks_position('left')
    
    plt.tight_layout()
    
    return fig


def visualize_matches_stacked(
    image0: np.ndarray,
    image1: np.ndarray,
    result: Union[MatchResult, Dict],
    top_k: Optional[int] = 100,
    figsize: Tuple[int, int] = (12, 12),
    **kwargs
) -> Figure:
    """
    Visualize matches with images stacked vertically.
    
    Args:
        image0: First image
        image1: Second image
        result: MatchResult or dict
        top_k: Number of top matches to show
        figsize: Figure size
        **kwargs: Additional arguments for line styling
        
    Returns:
        matplotlib Figure
    """
    if isinstance(result, dict):
        result = MatchResult(
            keypoints0=result['keypoints0'],
            keypoints1=result['keypoints1'],
            confidence=result.get('confidence', np.ones(len(result['keypoints0']))),
            num_matches=len(result['keypoints0']),
            inference_time=result.get('inference_time', 0),
            model_name=result.get('model_name', '')
        )
    
    if top_k is not None and top_k < result.num_matches:
        result = result.top_k(top_k)
    
    # Ensure 3-channel
    if image0.ndim == 2:
        image0 = np.stack([image0] * 3, axis=-1)
    if image1.ndim == 2:
        image1 = np.stack([image1] * 3, axis=-1)
    
    # Normalize
    if image0.max() <= 1.0:
        image0 = (image0 * 255).astype(np.uint8)
    if image1.max() <= 1.0:
        image1 = (image1 * 255).astype(np.uint8)
    
    # Resize images to same width
    h0, w0 = image0.shape[:2]
    h1, w1 = image1.shape[:2]
    
    # Stack images vertically
    if w0 != w1:
        # Resize image1 to match image0 width
        scale = w0 / w1
        new_h1 = int(h1 * scale)
        from PIL import Image
        img1_pil = Image.fromarray(image1).resize((w0, new_h1), Image.LANCZOS)
        image1_resized = np.array(img1_pil)
        # Scale keypoints
        kpts1_scaled = result.keypoints1.copy()
        kpts1_scaled[:, 0] *= scale
        kpts1_scaled[:, 1] *= scale
    else:
        image1_resized = image1
        kpts1_scaled = result.keypoints1
        new_h1 = h1
    
    # Stack images
    stacked = np.vstack([image0, image1_resized])
    
    # Adjust keypoints for stacked image
    kpts1_offset = kpts1_scaled.copy()
    kpts1_offset[:, 1] += h0  # Offset by height of first image
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(stacked)
    ax.axis('off')
    
    # Color by confidence
    conf = result.confidence
    if len(conf) > 0 and conf.max() > conf.min():
        norm_conf = (conf - conf.min()) / (conf.max() - conf.min())
    else:
        norm_conf = np.ones(len(conf))
    colors = cm.get_cmap('viridis')(norm_conf)
    
    # Draw matches
    line_width = kwargs.get('line_width', 0.5)
    alpha = kwargs.get('alpha', 0.7)
    
    for i in range(len(result.keypoints0)):
        ax.plot(
            [result.keypoints0[i, 0], kpts1_offset[i, 0]],
            [result.keypoints0[i, 1], kpts1_offset[i, 1]],
            color=colors[i],
            linewidth=line_width,
            alpha=alpha
        )
    
    # Add title and stats
    title = f"{result.model_name}: {result.num_matches} matches" if result.model_name else f"{result.num_matches} matches"
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    return fig


def compare_models(
    image0: np.ndarray,
    image1: np.ndarray,
    results: Dict[str, Union[MatchResult, Dict]],
    top_k: Optional[int] = 100,
    figsize: Tuple[int, int] = (20, 10),
    show_original: bool = True
) -> Figure:
    """
    Compare multiple models side-by-side.
    
    Args:
        image0: First image
        image1: Second image
        results: Dict of model_name -> MatchResult
        top_k: Number of top matches to show per model
        figsize: Figure size
        show_original: Whether to show original image pair
        
    Returns:
        matplotlib Figure
    """
    n_models = len(results)
    n_cols = n_models + (1 if show_original else 0)
    
    fig, axes = plt.subplots(2, n_cols, figsize=figsize)
    
    # Ensure images are displayable
    if image0.ndim == 2:
        image0 = np.stack([image0] * 3, axis=-1)
    if image1.ndim == 2:
        image1 = np.stack([image1] * 3, axis=-1)
    if image0.max() <= 1.0:
        image0 = (image0 * 255).astype(np.uint8)
    if image1.max() <= 1.0:
        image1 = (image1 * 255).astype(np.uint8)
    
    col_idx = 0
    
    # Show original images
    if show_original:
        axes[0, col_idx].imshow(image0)
        axes[0, col_idx].set_title('Image 0 (Original)', fontsize=10)
        axes[0, col_idx].axis('off')
        
        axes[1, col_idx].imshow(image1)
        axes[1, col_idx].set_title('Image 1 (Original)', fontsize=10)
        axes[1, col_idx].axis('off')
        
        col_idx += 1
    
    # Show each model's results
    for model_name, result in results.items():
        if isinstance(result, dict):
            result = MatchResult(
                keypoints0=result['keypoints0'],
                keypoints1=result['keypoints1'],
                confidence=result.get('confidence', np.ones(len(result['keypoints0']))),
                num_matches=len(result['keypoints0']),
                inference_time=result.get('inference_time', 0),
                model_name=model_name
            )
        
        if top_k is not None and top_k < result.num_matches:
            result = result.top_k(top_k)
        
        # Draw matches on copies of images
        img0_copy = image0.copy()
        img1_copy = image1.copy()
        
        # Draw keypoints
        conf = result.confidence
        if len(conf) > 0 and conf.max() > conf.min():
            norm_conf = (conf - conf.min()) / (conf.max() - conf.min())
        else:
            norm_conf = np.ones(len(conf)) if len(conf) > 0 else np.array([])
        
        colors = cm.get_cmap('viridis')(norm_conf) * 255
        
        for i in range(len(result.keypoints0)):
            x0, y0 = int(result.keypoints0[i, 0]), int(result.keypoints0[i, 1])
            x1, y1 = int(result.keypoints1[i, 0]), int(result.keypoints1[i, 1])
            color = colors[i][:3].astype(np.uint8).tolist()
            
            # Draw circles
            import cv2
            cv2.circle(img0_copy, (x0, y0), 3, color, -1)
            cv2.circle(img1_copy, (x1, y1), 3, color, -1)
        
        # Display
        axes[0, col_idx].imshow(img0_copy)
        axes[0, col_idx].set_title(f'{model_name}\n{result.num_matches} matches', fontsize=10)
        axes[0, col_idx].axis('off')
        
        axes[1, col_idx].imshow(img1_copy)
        stats = f'{result.inference_time:.3f}s | conf: {conf.mean():.2f}' if len(conf) > 0 else f'{result.inference_time:.3f}s'
        axes[1, col_idx].set_title(stats, fontsize=9)
        axes[1, col_idx].axis('off')
        
        col_idx += 1
    
    plt.tight_layout()
    
    return fig


def plot_confidence_histogram(
    results: Dict[str, Union[MatchResult, Dict]],
    figsize: Tuple[int, int] = (12, 5),
    bins: int = 50
) -> Figure:
    """
    Plot confidence histogram for multiple models.
    
    Args:
        results: Dict of model_name -> MatchResult
        figsize: Figure size
        bins: Number of histogram bins
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    for model_name, result in results.items():
        if isinstance(result, dict):
            conf = result.get('confidence', [])
        else:
            conf = result.confidence
        
        if len(conf) > 0:
            ax.hist(conf, bins=bins, alpha=0.5, label=f'{model_name} (n={len(conf)})')
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Match Confidence Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def plot_match_comparison(
    results: Dict[str, Union[MatchResult, Dict]],
    figsize: Tuple[int, int] = (14, 6)
) -> Figure:
    """
    Plot comparison bars for multiple models.
    
    Args:
        results: Dict of model_name -> MatchResult
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    models = list(results.keys())
    n_matches = []
    times = []
    mean_conf = []
    
    for name, result in results.items():
        if isinstance(result, dict):
            n_matches.append(result.get('num_matches', len(result.get('keypoints0', []))))
            times.append(result.get('inference_time', 0))
            conf = result.get('confidence', [])
            mean_conf.append(np.mean(conf) if len(conf) > 0 else 0)
        else:
            n_matches.append(result.num_matches)
            times.append(result.inference_time)
            mean_conf.append(np.mean(result.confidence) if result.num_matches > 0 else 0)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    # Number of matches
    bars = axes[0].bar(models, n_matches, color=colors)
    axes[0].set_ylabel('Number of Matches')
    axes[0].set_title('Matches Found')
    axes[0].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, n_matches):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val}', ha='center', va='bottom', fontsize=10)
    
    # Inference time
    bars = axes[1].bar(models, times, color=colors)
    axes[1].set_ylabel('Time (seconds)')
    axes[1].set_title('Inference Time')
    axes[1].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, times):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Mean confidence
    bars = axes[2].bar(models, mean_conf, color=colors)
    axes[2].set_ylabel('Mean Confidence')
    axes[2].set_title('Average Match Confidence')
    axes[2].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, mean_conf):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    return fig


def create_match_video_frame(
    image0: np.ndarray,
    image1: np.ndarray,
    result: MatchResult,
    frame_size: Tuple[int, int] = (1920, 1080),
    top_k: int = 100
) -> np.ndarray:
    """
    Create a single video frame showing matches.
    
    Args:
        image0: First image
        image1: Second image
        result: MatchResult
        frame_size: Output frame size (width, height)
        top_k: Number of matches to show
        
    Returns:
        Frame as numpy array
    """
    import cv2
    
    if top_k < result.num_matches:
        result = result.top_k(top_k)
    
    # Ensure 3-channel
    if image0.ndim == 2:
        image0 = cv2.cvtColor(image0, cv2.COLOR_GRAY2BGR)
    if image1.ndim == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    
    # Resize to fit frame
    h0, w0 = image0.shape[:2]
    h1, w1 = image1.shape[:2]
    
    target_w = frame_size[0] // 2
    target_h = frame_size[1]
    
    scale0 = min(target_w / w0, target_h / h0)
    scale1 = min(target_w / w1, target_h / h1)
    
    new_w0, new_h0 = int(w0 * scale0), int(h0 * scale0)
    new_w1, new_h1 = int(w1 * scale1), int(h1 * scale1)
    
    img0_resized = cv2.resize(image0, (new_w0, new_h0))
    img1_resized = cv2.resize(image1, (new_w1, new_h1))
    
    # Create canvas
    canvas = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
    
    # Place images
    y0 = (target_h - new_h0) // 2
    canvas[y0:y0+new_h0, 0:new_w0] = img0_resized
    
    y1 = (target_h - new_h1) // 2
    x1_offset = target_w
    canvas[y1:y1+new_h1, x1_offset:x1_offset+new_w1] = img1_resized
    
    # Draw matches
    for i in range(len(result.keypoints0)):
        pt0 = (int(result.keypoints0[i, 0] * scale0), 
               int(result.keypoints0[i, 1] * scale0 + y0))
        pt1 = (int(result.keypoints1[i, 0] * scale1 + x1_offset),
               int(result.keypoints1[i, 1] * scale1 + y1))
        
        # Color by confidence
        conf = result.confidence[i]
        color = cm.viridis(conf)[:3]
        color = tuple(int(c * 255) for c in color[::-1])  # BGR
        
        cv2.circle(canvas, pt0, 3, color, -1)
        cv2.circle(canvas, pt1, 3, color, -1)
        cv2.line(canvas, pt0, pt1, color, 1, cv2.LINE_AA)
    
    # Add text
    text = f"{result.model_name}: {result.num_matches} matches ({result.inference_time:.3f}s)"
    cv2.putText(canvas, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2, cv2.LINE_AA)
    
    return canvas
