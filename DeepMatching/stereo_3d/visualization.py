"""
Visualization tools for altitude estimation results.

This module provides specialized visualizations for:
- Fisheye images with altitude-colored matches
- 3D point cloud views
- Altitude histograms and profiles
- Quality diagnostic plots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from typing import Optional, Tuple, List
import warnings


def visualize_matches_with_altitude(
    img1: np.ndarray,
    img2: np.ndarray,
    altitude_result,
    top_k: Optional[int] = None,
    min_altitude: Optional[float] = None,
    max_altitude: Optional[float] = None,
    cmap: str = 'viridis',
    figsize: Tuple[int, int] = (16, 8),
    point_size: int = 30,
    line_alpha: float = 0.3,
    show_colorbar: bool = True,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Visualize stereo matches colored by estimated altitude.
    
    Creates a side-by-side view of both images with matched points
    connected by lines, colored by altitude.
    
    Args:
        img1: First image (camera 1)
        img2: Second image (camera 2)
        altitude_result: AltitudeResult with triangulated points
        top_k: Only show top K highest-confidence matches
        min_altitude: Minimum altitude for colormap (auto if None)
        max_altitude: Maximum altitude for colormap (auto if None)
        cmap: Matplotlib colormap name
        figsize: Figure size (width, height)
        point_size: Size of match points
        line_alpha: Transparency of connecting lines
        show_colorbar: Whether to show altitude colorbar
        title: Optional figure title
        
    Returns:
        matplotlib Figure
    """
    n_matches = altitude_result.num_points
    if n_matches == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No matches to display", ha='center', va='center')
        return fig
    
    # Filter to top K if specified
    if top_k is not None and top_k < n_matches:
        if altitude_result.confidence is not None:
            # Sort by confidence
            idx = np.argsort(altitude_result.confidence)[::-1][:top_k]
        else:
            # Random sample
            idx = np.random.choice(n_matches, size=top_k, replace=False)
        
        kpts0 = altitude_result.keypoints0[idx]
        kpts1 = altitude_result.keypoints1[idx]
        altitudes = altitude_result.altitudes[idx]
    else:
        kpts0 = altitude_result.keypoints0
        kpts1 = altitude_result.keypoints1
        altitudes = altitude_result.altitudes
    
    # Setup colormap
    if min_altitude is None:
        min_altitude = np.nanpercentile(altitudes, 5)
    if max_altitude is None:
        max_altitude = np.nanpercentile(altitudes, 95)
    
    norm = Normalize(vmin=min_altitude, vmax=max_altitude)
    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(norm(altitudes))
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Show images
    axes[0].imshow(img1)
    axes[0].set_title('Camera 1 (IPSL)', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(img2)
    axes[1].set_title('Camera 2 (ECTL)', fontsize=12)
    axes[1].axis('off')
    
    # Plot points
    for i in range(len(kpts0)):
        color = colors[i]
        
        # Points on each image
        axes[0].scatter(kpts0[i, 0], kpts0[i, 1], c=[color], s=point_size, 
                       edgecolors='white', linewidths=0.5, zorder=10)
        axes[1].scatter(kpts1[i, 0], kpts1[i, 1], c=[color], s=point_size,
                       edgecolors='white', linewidths=0.5, zorder=10)
    
    # Add colorbar
    if show_colorbar:
        sm = ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.02)
        cbar.set_label('Altitude (m ASL)', fontsize=10)
    
    # Title
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_3d_pointcloud_view(
    altitude_result,
    camera_positions: Optional[List[Tuple[float, float, float]]] = None,
    view_elev: float = 30,
    view_azim: float = -60,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'viridis',
    point_size: int = 10,
    show_cameras: bool = True,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Create 3D visualization of triangulated points.
    
    Shows the point cloud in ENU coordinates with optional
    camera positions.
    
    Args:
        altitude_result: AltitudeResult with triangulated points
        camera_positions: Optional list of (E, N, U) camera positions
        view_elev: Elevation angle for 3D view
        view_azim: Azimuth angle for 3D view
        figsize: Figure size
        cmap: Colormap for altitude coloring
        point_size: Size of points
        show_cameras: Whether to show camera positions
        title: Optional figure title
        
    Returns:
        matplotlib Figure
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    n_points = altitude_result.num_points
    if n_points == 0:
        ax.text2D(0.5, 0.5, "No points to display", ha='center', va='center',
                  transform=ax.transAxes)
        return fig
    
    # Extract ENU coordinates
    e = altitude_result.points_3d_enu[:, 0] / 1000  # Convert to km
    n = altitude_result.points_3d_enu[:, 1] / 1000
    u = altitude_result.points_3d_enu[:, 2] / 1000
    
    # Color by altitude
    altitudes_km = altitude_result.altitudes / 1000
    
    # Plot points
    scatter = ax.scatter(e, n, u, c=altitudes_km, cmap=cmap, s=point_size,
                        alpha=0.7, edgecolors='none')
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Altitude (km ASL)', fontsize=10)
    
    # Show camera positions
    if show_cameras and camera_positions is not None:
        for i, (ce, cn, cu) in enumerate(camera_positions):
            ax.scatter([ce/1000], [cn/1000], [cu/1000], c='red', s=200, 
                      marker='^', edgecolors='black', linewidths=2,
                      label=f'Camera {i+1}' if i == 0 else None, zorder=100)
    
    # Labels
    ax.set_xlabel('East (km)')
    ax.set_ylabel('North (km)')
    ax.set_zlabel('Up (km)')
    
    # Set view angle
    ax.view_init(elev=view_elev, azim=view_azim)
    
    # Title
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Equal aspect ratio (approximately)
    max_range = max(e.max() - e.min(), n.max() - n.min(), u.max() - u.min()) / 2
    mid_e = (e.max() + e.min()) / 2
    mid_n = (n.max() + n.min()) / 2
    mid_u = (u.max() + u.min()) / 2
    ax.set_xlim(mid_e - max_range, mid_e + max_range)
    ax.set_ylim(mid_n - max_range, mid_n + max_range)
    ax.set_zlim(mid_u - max_range, mid_u + max_range)
    
    plt.tight_layout()
    return fig


def plot_altitude_histogram(
    altitude_result,
    bins: int = 50,
    range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[int, int] = (10, 6),
    show_stats: bool = True,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot histogram of altitude estimates.
    
    Args:
        altitude_result: AltitudeResult with altitudes
        bins: Number of histogram bins
        range: Optional (min, max) range for histogram
        figsize: Figure size
        show_stats: Whether to show statistics annotation
        title: Optional figure title
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_points = altitude_result.num_points
    if n_points == 0:
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
        return fig
    
    altitudes_km = altitude_result.altitudes / 1000
    
    # Determine range
    if range is None:
        range = (np.percentile(altitudes_km, 1), np.percentile(altitudes_km, 99))
    else:
        range = (range[0] / 1000, range[1] / 1000)
    
    # Plot histogram
    ax.hist(altitudes_km, bins=bins, range=range, color='steelblue', 
            edgecolor='white', alpha=0.8)
    
    ax.set_xlabel('Altitude (km ASL)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    else:
        ax.set_title('Altitude Distribution', fontsize=12)
    
    # Add statistics
    if show_stats:
        stats = altitude_result.altitude_stats
        stats_text = (
            f"n = {stats['count']}\n"
            f"Mean: {stats['mean']/1000:.2f} km\n"
            f"Median: {stats['median']/1000:.2f} km\n"
            f"Std: {stats['std']/1000:.2f} km"
        )
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                va='top', ha='right', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig


def plot_altitude_cross_section(
    altitude_result,
    azimuth: Optional[float] = None,
    azimuth_tolerance: float = 10.0,
    figsize: Tuple[int, int] = (12, 5),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot altitude profile along a specific azimuth direction.
    
    Args:
        altitude_result: AltitudeResult with triangulated points
        azimuth: Azimuth direction for cross-section (degrees, 0=North)
                 If None, uses the azimuth with most points
        azimuth_tolerance: Points within this angle are included (degrees)
        figsize: Figure size
        title: Optional figure title
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_points = altitude_result.num_points
    if n_points == 0:
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
        return fig
    
    # Compute azimuth of each point from reference
    e = altitude_result.points_3d_enu[:, 0]
    n = altitude_result.points_3d_enu[:, 1]
    point_azimuths = np.degrees(np.arctan2(e, n)) % 360
    
    # Determine azimuth to use
    if azimuth is None:
        # Use histogram to find most common azimuth
        hist, edges = np.histogram(point_azimuths, bins=36)
        best_bin = np.argmax(hist)
        azimuth = (edges[best_bin] + edges[best_bin + 1]) / 2
    
    # Filter points near the azimuth
    az_diff = np.abs(point_azimuths - azimuth)
    az_diff = np.minimum(az_diff, 360 - az_diff)  # Handle wraparound
    mask = az_diff <= azimuth_tolerance
    
    if mask.sum() == 0:
        ax.text(0.5, 0.5, f"No points near azimuth {azimuth:.0f}°", 
                ha='center', va='center')
        return fig
    
    # Horizontal distance and altitude
    horiz_dist = np.sqrt(e[mask]**2 + n[mask]**2) / 1000  # km
    altitudes = altitude_result.altitudes[mask] / 1000  # km
    uncertainties = altitude_result.uncertainties[mask] / 1000  # km
    
    # Sort by distance
    sort_idx = np.argsort(horiz_dist)
    horiz_dist = horiz_dist[sort_idx]
    altitudes = altitudes[sort_idx]
    uncertainties = uncertainties[sort_idx]
    
    # Plot
    ax.errorbar(horiz_dist, altitudes, yerr=uncertainties, 
                fmt='o', markersize=4, alpha=0.6, 
                capsize=2, elinewidth=0.5)
    
    ax.set_xlabel('Horizontal Distance (km)', fontsize=11)
    ax.set_ylabel('Altitude (km ASL)', fontsize=11)
    
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    else:
        ax.set_title(f'Altitude Cross-Section (Azimuth: {azimuth:.0f}° ± {azimuth_tolerance:.0f}°)',
                     fontsize=12)
    
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    return fig


def plot_quality_diagnostics(
    altitude_result,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Create diagnostic plots for quality assessment.
    
    Shows distributions of:
    - Triangulation angles
    - Ray miss distances
    - Uncertainties
    - Elevation angles
    
    Args:
        altitude_result: AltitudeResult to analyze
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    n_points = altitude_result.num_points
    if n_points == 0:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
        return fig
    
    # 1. Triangulation angle distribution
    ax = axes[0, 0]
    ax.hist(altitude_result.triangulation_angles, bins=50, 
            color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(20, color='green', linestyle='--', label='Good range')
    ax.axvline(50, color='green', linestyle='--')
    ax.set_xlabel('Triangulation Angle (°)')
    ax.set_ylabel('Count')
    ax.set_title('Triangulation Angle Distribution')
    ax.legend()
    
    # 2. Ray miss distance distribution  
    ax = axes[0, 1]
    miss_clipped = np.clip(altitude_result.ray_miss_distances, 0, 
                           np.percentile(altitude_result.ray_miss_distances, 99))
    ax.hist(miss_clipped, bins=50, color='darkorange', 
            edgecolor='white', alpha=0.8)
    ax.set_xlabel('Ray Miss Distance (m)')
    ax.set_ylabel('Count')
    ax.set_title('Ray Miss Distance Distribution')
    ax.axvline(500, color='red', linestyle='--', label='Threshold')
    ax.legend()
    
    # 3. Uncertainty distribution
    ax = axes[1, 0]
    unc_clipped = np.clip(altitude_result.uncertainties, 0,
                          np.percentile(altitude_result.uncertainties, 99))
    ax.hist(unc_clipped / 1000, bins=50, color='green',
            edgecolor='white', alpha=0.8)
    ax.set_xlabel('Altitude Uncertainty (km)')
    ax.set_ylabel('Count')
    ax.set_title('Uncertainty Distribution')
    
    # 4. Altitude vs Triangulation Angle scatter
    ax = axes[1, 1]
    scatter = ax.scatter(altitude_result.triangulation_angles, 
                        altitude_result.altitudes / 1000,
                        c=altitude_result.uncertainties / 1000,
                        cmap='plasma', alpha=0.5, s=10)
    ax.set_xlabel('Triangulation Angle (°)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Altitude vs Triangulation Angle')
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Uncertainty (km)')
    
    plt.tight_layout()
    return fig


def plot_fisheye_overlay(
    img: np.ndarray,
    altitude_result,
    camera: int = 1,
    cmap: str = 'viridis',
    alpha: float = 0.7,
    figsize: Tuple[int, int] = (10, 10),
    show_grid: bool = True,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Overlay altitude-colored points on fisheye image with optional grid.
    
    Args:
        img: Fisheye image
        altitude_result: AltitudeResult with matches
        camera: Which camera (1 or 2) for keypoints
        cmap: Colormap for altitude
        alpha: Transparency
        figsize: Figure size
        show_grid: Whether to show elevation/azimuth grid
        title: Optional title
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.imshow(img)
    
    n_points = altitude_result.num_points
    if n_points > 0:
        if camera == 1:
            kpts = altitude_result.keypoints0
            elevations = altitude_result.elevations_cam1
        else:
            kpts = altitude_result.keypoints1
            elevations = altitude_result.elevations_cam2
        
        # Color by altitude
        norm = Normalize(vmin=np.nanpercentile(altitude_result.altitudes, 5),
                        vmax=np.nanpercentile(altitude_result.altitudes, 95))
        colors = plt.get_cmap(cmap)(norm(altitude_result.altitudes))
        
        scatter = ax.scatter(kpts[:, 0], kpts[:, 1], c=colors, s=20,
                           alpha=alpha, edgecolors='white', linewidths=0.3)
        
        # Add colorbar
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Altitude (m ASL)')
    
    # Add elevation grid circles
    if show_grid:
        h, w = img.shape[:2]
        cx, cy = w / 2, h / 2
        r_max = min(w, h) / 2
        
        # Draw elevation circles (every 15°)
        for el in [15, 30, 45, 60, 75]:
            r = r_max * (1 - el / 90)
            circle = plt.Circle((cx, cy), r, fill=False, 
                               color='white', linestyle='--', 
                               linewidth=0.5, alpha=0.5)
            ax.add_patch(circle)
            ax.text(cx + r * 0.7, cy - r * 0.7, f'{el}°', 
                   color='white', fontsize=8, alpha=0.7)
        
        # Draw azimuth lines (every 45°)
        for az in range(0, 360, 45):
            az_rad = np.radians(az)
            ax.plot([cx, cx + r_max * np.sin(az_rad)],
                   [cy, cy - r_max * np.cos(az_rad)],
                   'w--', linewidth=0.5, alpha=0.5)
    
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_summary_figure(
    img1: np.ndarray,
    img2: np.ndarray,
    altitude_result,
    filtered_result,
    figsize: Tuple[int, int] = (18, 12),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive summary figure with multiple panels.
    
    Args:
        img1, img2: Input images
        altitude_result: Raw AltitudeResult
        filtered_result: Filtered AltitudeResult
        figsize: Figure size
        title: Optional super title
        
    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Create grid
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Top row: Images with matches
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.imshow(img1)
    if filtered_result.num_points > 0 and filtered_result.keypoints0 is not None:
        ax1.scatter(filtered_result.keypoints0[:, 0], filtered_result.keypoints0[:, 1],
                   c=filtered_result.altitudes, cmap='viridis', s=20, alpha=0.7)
    ax1.set_title(f'Camera 1 - {filtered_result.num_points} filtered points')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.imshow(img2)
    if filtered_result.num_points > 0 and filtered_result.keypoints1 is not None:
        scatter = ax2.scatter(filtered_result.keypoints1[:, 0], filtered_result.keypoints1[:, 1],
                             c=filtered_result.altitudes, cmap='viridis', s=20, alpha=0.7)
    ax2.set_title(f'Camera 2 - {filtered_result.num_points} filtered points')
    ax2.axis('off')
    
    # Middle row: Altitude histogram and 2D spatial view
    ax3 = fig.add_subplot(gs[1, :2])
    if filtered_result.num_points > 0:
        ax3.hist(filtered_result.altitudes / 1000, bins=40, 
                color='steelblue', edgecolor='white', alpha=0.8)
    ax3.set_xlabel('Altitude (km)')
    ax3.set_ylabel('Count')
    ax3.set_title('Altitude Distribution (Filtered)')
    
    ax4 = fig.add_subplot(gs[1, 2:])
    if filtered_result.num_points > 0:
        scatter = ax4.scatter(filtered_result.points_3d_enu[:, 0] / 1000,
                             filtered_result.points_3d_enu[:, 1] / 1000,
                             c=filtered_result.altitudes / 1000,
                             cmap='viridis', s=20, alpha=0.7)
        cbar = fig.colorbar(scatter, ax=ax4)
        cbar.set_label('Altitude (km)')
    ax4.set_xlabel('East (km)')
    ax4.set_ylabel('North (km)')
    ax4.set_title('Spatial Distribution (Top View)')
    ax4.set_aspect('equal')
    
    # Bottom row: Quality metrics
    ax5 = fig.add_subplot(gs[2, 0])
    if altitude_result.num_points > 0:
        ax5.hist(altitude_result.triangulation_angles, bins=40,
                color='steelblue', alpha=0.5, label='All')
    if filtered_result.num_points > 0:
        ax5.hist(filtered_result.triangulation_angles, bins=40,
                color='green', alpha=0.7, label='Filtered')
    ax5.set_xlabel('Triangulation Angle (°)')
    ax5.set_ylabel('Count')
    ax5.set_title('Triangulation Angles')
    ax5.legend()
    
    ax6 = fig.add_subplot(gs[2, 1])
    if altitude_result.num_points > 0:
        miss_clip = np.clip(altitude_result.ray_miss_distances, 0,
                           np.percentile(altitude_result.ray_miss_distances, 98))
        ax6.hist(miss_clip, bins=40, color='steelblue', alpha=0.5, label='All')
    if filtered_result.num_points > 0:
        ax6.hist(filtered_result.ray_miss_distances, bins=40,
                color='green', alpha=0.7, label='Filtered')
    ax6.set_xlabel('Miss Distance (m)')
    ax6.set_ylabel('Count')
    ax6.set_title('Ray Miss Distances')
    ax6.legend()
    
    ax7 = fig.add_subplot(gs[2, 2])
    if altitude_result.num_points > 0:
        ax7.scatter(altitude_result.triangulation_angles,
                   altitude_result.altitudes / 1000,
                   c='steelblue', alpha=0.3, s=5, label='All')
    if filtered_result.num_points > 0:
        ax7.scatter(filtered_result.triangulation_angles,
                   filtered_result.altitudes / 1000,
                   c='green', alpha=0.5, s=15, label='Filtered')
    ax7.set_xlabel('Triangulation Angle (°)')
    ax7.set_ylabel('Altitude (km)')
    ax7.set_title('Altitude vs Angle')
    ax7.legend()
    
    # Text summary
    ax8 = fig.add_subplot(gs[2, 3])
    ax8.axis('off')
    
    stats = filtered_result.altitude_stats
    summary_text = (
        f"Summary Statistics\n"
        f"{'='*25}\n"
        f"Total matches: {altitude_result.num_points}\n"
        f"Filtered: {filtered_result.num_points} ({100*filtered_result.num_points/max(1,altitude_result.num_points):.1f}%)\n\n"
        f"Altitude (filtered):\n"
        f"  Mean: {stats['mean']/1000:.2f} km\n"
        f"  Std: {stats['std']/1000:.2f} km\n"
        f"  Median: {stats['median']/1000:.2f} km\n"
        f"  Range: {stats['min']/1000:.2f} - {stats['max']/1000:.2f} km\n\n"
    )
    
    if filtered_result.num_points > 0:
        qstats = filtered_result.quality_stats
        summary_text += (
            f"Quality:\n"
            f"  Tri. angle: {qstats['median_tri_angle']:.1f}°\n"
            f"  Miss dist: {qstats['median_miss_distance']:.0f} m\n"
            f"  Uncertainty: {qstats['median_uncertainty']:.0f} m"
        )
    
    ax8.text(0.1, 0.95, summary_text, transform=ax8.transAxes,
            fontsize=10, fontfamily='monospace', va='top')
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    return fig
