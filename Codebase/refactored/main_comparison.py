#!/usr/bin/env python3
"""
Main script for stereo camera inference comparison.

This script orchestrates the complete workflow:
1. Load datasets and find matched image pairs
2. Load segmentation model
3. Run inference on matched pairs
4. Generate visualizations and statistics
5. Optionally project and visualize flight trajectories
"""
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import (
    DATASET1_DIR, DATASET2_DIR, USE_HISTOGRAM_MATCHING,
    TIME_FILTER_START_HOUR, TIME_FILTER_END_HOUR,
    INFERENCE_THRESHOLD, INFERENCE_MASK_THRESHOLD, INFERENCE_OVERLAP_THRESHOLD
)
from data_utils import (
    load_datasets, find_matching_image_pairs, filter_pairs_by_time, 
    load_image_pair
)
from model_loader import setup_model_and_processor
from inference import process_image_pair
from visualization import (
    visualize_inference_pair, create_comparison_table,
    visualize_flights_on_image
)
from flight_projection import (
    setup_skycam, load_flight_data, filter_flights_by_timestamp,
    project_flights_to_pixels
)


def main(num_pairs_to_process=10, include_flights=False):
    """
    Main execution function.
    
    Args:
        num_pairs_to_process: Number of image pairs to process
        include_flights: Whether to include flight projection
    """
    print("="*80)
    print("STEREO CAMERA INFERENCE COMPARISON")
    print("="*80)
    
    # Step 1: Load datasets
    print("\n[1/6] Loading datasets...")
    dataset1_images, dataset2_images = load_datasets(DATASET1_DIR, DATASET2_DIR)
    
    # Step 2: Find matched pairs
    print("\n[2/6] Finding matched image pairs...")
    matched_pairs = find_matching_image_pairs(dataset1_images, dataset2_images)
    print(f"✓ Found {len(matched_pairs)} matched image pairs")
    
    # Filter by time window
    print(f"\nFiltering for images between {TIME_FILTER_START_HOUR}:00 and "
          f"{TIME_FILTER_END_HOUR}:00...")
    matched_pairs = filter_pairs_by_time(
        matched_pairs, 
        TIME_FILTER_START_HOUR, 
        TIME_FILTER_END_HOUR
    )
    print(f"✓ {len(matched_pairs)} pairs in time window")
    
    if len(matched_pairs) == 0:
        print("⚠ No matched pairs found. Exiting.")
        return
    
    # Limit number of pairs
    num_pairs_to_process = min(num_pairs_to_process, len(matched_pairs))
    matched_pairs = matched_pairs[:num_pairs_to_process]
    
    # Step 3: Load model
    print("\n[3/6] Loading segmentation model...")
    model, processor, device = setup_model_and_processor()
    
    # Step 4: Process pairs
    print(f"\n[4/6] Processing {num_pairs_to_process} matched pairs...")
    print(f"Histogram matching: {'ENABLED' if USE_HISTOGRAM_MATCHING else 'DISABLED'}\n")
    
    results = []
    
    for i, pair in enumerate(matched_pairs):
        print(f"\n{'='*80}")
        print(f"Processing pair {i+1}/{num_pairs_to_process}")
        print(f"Timestamp: {pair['timestamp']}")
        print(f"{'='*80}")
        
        try:
            # Load images
            ipsl_image, ectl_image = load_image_pair(pair, flip_ectl=True)
            
            print(f"  IPSL image shape: {ipsl_image.shape}")
            print(f"  ECTL image shape: {ectl_image.shape} [vertically flipped]")
            
            # Process through inference pipeline
            result = process_image_pair(
                ipsl_image, ectl_image, model, processor, device,
                apply_histogram_matching=USE_HISTOGRAM_MATCHING
            )
            
            # Add metadata
            result['timestamp'] = pair['timestamp']
            result['ipsl_filename'] = pair['ipsl_filename']
            result['ectl_filename'] = pair['ectl_filename']
            
            results.append(result)
            
            # Visualize
            visualize_inference_pair(
                result['ipsl_image'], result['ectl_image'],
                result['ipsl_seg'], result['ectl_seg'],
                pair['ipsl_filename'], pair['ectl_filename'],
                pair['timestamp'],
                histogram_matched=USE_HISTOGRAM_MATCHING
            )
            
        except Exception as e:
            print(f"  ✗ Error processing pair: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Step 5: Generate comparison table
    print(f"\n[5/6] Generating comparison statistics...")
    df_comparison = create_comparison_table(results)
    
    # Step 6: Flight projection (optional)
    if include_flights and len(results) > 0:
        print(f"\n[6/6] Processing flight trajectories...")
        try:
            # Setup skycam
            aircraft_projector, projection_settings = setup_skycam()
            
            # Load flight data
            flights_df = load_flight_data()
            
            if len(flights_df) > 0:
                # Process first result as example
                test_result = results[0]
                
                # Filter flights
                flights_filtered = filter_flights_by_timestamp(
                    flights_df,
                    test_result['timestamp']
                )
                
                if len(flights_filtered) > 0:
                    # Project to pixels
                    flights_projected = project_flights_to_pixels(
                        flights_filtered,
                        aircraft_projector,
                        projection_settings
                    )
                    
                    # Visualize
                    visualize_flights_on_image(
                        test_result['ectl_image'],
                        flights_projected,
                        title=f"Flights on ECTL Image\n"
                              f"{test_result['ectl_filename']}\n"
                              f"{test_result['timestamp']}"
                    )
                else:
                    print("  No flights found in time window")
            else:
                print("  No flight data available")
                
        except Exception as e:
            print(f"  ✗ Error processing flights: {e}")
            import traceback
            traceback.print_exc()
    elif not include_flights:
        print(f"\n[6/6] Flight projection skipped (include_flights=False)")
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"✓ Processed {len(results)} image pairs successfully")
    print(f"✓ Results saved in memory (results list)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run stereo camera inference comparison"
    )
    parser.add_argument(
        '--num-pairs', 
        type=int, 
        default=10,
        help='Number of image pairs to process (default: 10)'
    )
    parser.add_argument(
        '--include-flights',
        action='store_true',
        help='Include flight trajectory projection'
    )
    
    args = parser.parse_args()
    
    main(
        num_pairs_to_process=args.num_pairs,
        include_flights=args.include_flights
    )
