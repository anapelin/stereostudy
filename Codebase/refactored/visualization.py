"""
Visualization utilities for stereo camera comparison.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def visualize_inference_pair(ipsl_image, ectl_image, ipsl_seg, ectl_seg, 
                            ipsl_filename, ectl_filename, timestamp,
                            histogram_matched=False):
    """
    Visualize inference results for a matched pair.
    
    Args:
        ipsl_image: IPSL image (numpy array)
        ectl_image: ECTL image (numpy array)
        ipsl_seg: IPSL segmentation results
        ectl_seg: ECTL segmentation results
        ipsl_filename: IPSL filename
        ectl_filename: ECTL filename
        timestamp: Image timestamp
        histogram_matched: Whether histogram matching was applied
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    
    # Row 1: Original images
    axes[0, 0].imshow(ipsl_image)
    title1 = f"IPSL\n{ipsl_filename}"
    if histogram_matched:
        title1 += "\n[Histogram Matched]"
    axes[0, 0].set_title(title1, fontsize=10)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(ectl_image)
    axes[0, 1].set_title(f"ECTL\n{ectl_filename}", fontsize=10)
    axes[0, 1].axis('off')
    
    # Row 2: Segmentation masks
    if 'segmentation' in ipsl_seg:
        ipsl_mask = ipsl_seg['segmentation'].cpu().numpy()
        axes[1, 0].imshow(ipsl_mask, cmap='tab20')
        axes[1, 0].set_title('IPSL Segmentation Mask', fontsize=10)
        axes[1, 0].axis('off')
    
    if 'segmentation' in ectl_seg:
        ectl_mask = ectl_seg['segmentation'].cpu().numpy()
        axes[1, 1].imshow(ectl_mask, cmap='tab20')
        axes[1, 1].set_title('ECTL Segmentation Mask', fontsize=10)
        axes[1, 1].axis('off')
    
    # Row 3: Overlays
    if 'segmentation' in ipsl_seg:
        axes[2, 0].imshow(ipsl_image)
        axes[2, 0].imshow(ipsl_mask, alpha=0.5, cmap='tab20')
        axes[2, 0].set_title('IPSL Overlay', fontsize=10)
        axes[2, 0].axis('off')
    
    if 'segmentation' in ectl_seg:
        axes[2, 1].imshow(ectl_image)
        axes[2, 1].imshow(ectl_mask, alpha=0.5, cmap='tab20')
        axes[2, 1].set_title('ECTL Overlay', fontsize=10)
        axes[2, 1].axis('off')
    
    plt.suptitle(f"Timestamp: {timestamp}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


def visualize_flights_on_image(image, flights_df, title="Flights on Camera Image"):
    """
    Visualize flight trajectories overlaid on camera image.
    
    Args:
        image: Camera image (numpy array)
        flights_df: DataFrame with pixel_x, pixel_y, visible columns
        title: Plot title
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    ax.imshow(image)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Filter visible flights
    visible_flights = flights_df[flights_df['visible']].copy()
    
    if len(visible_flights) == 0:
        print("⚠ No visible flights in this image")
        plt.tight_layout()
        plt.show()
        return
    
    # Plot flight trajectories
    if 'flight_id' in visible_flights.columns:
        colors = plt.cm.rainbow(np.linspace(0, 1, 
                                           len(visible_flights['flight_id'].unique())))
        
        for i, flight_id in enumerate(visible_flights['flight_id'].unique()):
            flight_points = visible_flights[visible_flights['flight_id'] == flight_id]
            
            if len(flight_points) > 1:
                # Plot trajectory line
                ax.plot(
                    flight_points['pixel_x'], 
                    flight_points['pixel_y'],
                    'o-', 
                    color=colors[i],
                    linewidth=2, 
                    markersize=6, 
                    alpha=0.8,
                    label=f'{flight_id}'
                )
            else:
                # Single point
                ax.plot(
                    flight_points['pixel_x'].values[0],
                    flight_points['pixel_y'].values[0],
                    'o',
                    color=colors[i],
                    markersize=8,
                    alpha=0.8,
                    label=f'{flight_id}'
                )
        
        # Add legend if not too many flights
        if len(visible_flights['flight_id'].unique()) <= 15:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    else:
        # Plot all points without flight IDs
        ax.scatter(visible_flights['pixel_x'], visible_flights['pixel_y'], 
                  c='red', s=50, alpha=0.7, label='Flight positions')
        ax.legend(loc='upper right')
    
    print(f"✓ Plotted {len(visible_flights)} visible flight points")
    
    plt.tight_layout()
    plt.show()


def create_comparison_table(results):
    """
    Create comparison table from results.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        DataFrame with comparison statistics
    """
    if len(results) == 0:
        print("No results to display")
        return None
    
    comparison_data = []
    
    for i, result in enumerate(results):
        comparison_data.append({
            'Pair': i + 1,
            'Timestamp': result['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'IPSL_Segments': result['ipsl_stats']['total_segments'],
            'IPSL_Contrails': result['ipsl_stats']['contrail_segments'],
            'IPSL_Pixels': result['ipsl_stats']['contrail_pixels'],
            'ECTL_Segments': result['ectl_stats']['total_segments'],
            'ECTL_Contrails': result['ectl_stats']['contrail_segments'],
            'ECTL_Pixels': result['ectl_stats']['contrail_pixels'],
            'Pixel_Diff': (result['ipsl_stats']['contrail_pixels'] - 
                          result['ectl_stats']['contrail_pixels']),
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    print("\n" + "="*100)
    print("COMPARISON TABLE")
    print("="*100)
    print(df_comparison.to_string(index=False))
    print("="*100)
    
    # Summary statistics
    print("\nSUMMARY STATISTICS:")
    print(f"  Total pairs processed: {len(results)}")
    print(f"\n  IPSL:")
    print(f"    Total contrail segments: {df_comparison['IPSL_Contrails'].sum()}")
    print(f"    Total contrail pixels: {df_comparison['IPSL_Pixels'].sum()}")
    print(f"    Avg contrails per image: {df_comparison['IPSL_Contrails'].mean():.2f}")
    print(f"\n  ECTL:")
    print(f"    Total contrail segments: {df_comparison['ECTL_Contrails'].sum()}")
    print(f"    Total contrail pixels: {df_comparison['ECTL_Pixels'].sum()}")
    print(f"    Avg contrails per image: {df_comparison['ECTL_Contrails'].mean():.2f}")
    
    return df_comparison


def visualize_calibration_maps(azimuth_map, zenith_map, title_prefix=""):
    """
    Visualize azimuth and zenith calibration maps.
    
    Args:
        azimuth_map: Azimuth map (radians or degrees)
        zenith_map: Zenith map (radians or degrees)
        title_prefix: Prefix for plot titles
    """
    # Convert to degrees if in radians
    if np.nanmax(azimuth_map) < 10:
        azimuth_map = np.degrees(azimuth_map)
    if np.nanmax(zenith_map) < 10:
        zenith_map = np.degrees(zenith_map)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(azimuth_map, cmap='hsv')
    axes[0].set_title(f'{title_prefix}Azimuth Map (degrees)', fontsize=12)
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], label='Azimuth (deg)')
    
    im2 = axes[1].imshow(zenith_map, cmap='viridis')
    axes[1].set_title(f'{title_prefix}Zenith Map (degrees)', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], label='Zenith (deg)')
    
    plt.tight_layout()
    plt.show()
