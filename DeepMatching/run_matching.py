#!/usr/bin/env python3
"""
Command-line interface for running stereo matching benchmarks.

Usage:
    python run_matching.py --folder_a /path/to/folder_a --folder_b /path/to/folder_b
    python run_matching.py --folder_a /path/to/folder_a --folder_b /path/to/folder_b --models loftr roma
    python run_matching.py --folder_a /path/to/folder_a --folder_b /path/to/folder_b --pairs img001,img002
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run stereo matching benchmark on image pairs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all models on all matching image pairs
    python run_matching.py --folder_a images/camera1 --folder_b images/camera2 --models all

    # Run specific models
    python run_matching.py --folder_a images/camera1 --folder_b images/camera2 --models loftr roma

    # Run on specific pairs
    python run_matching.py --folder_a images/camera1 --folder_b images/camera2 --pairs img001,img002

    # Save outputs to custom directory
    python run_matching.py --folder_a images/camera1 --folder_b images/camera2 --output results/
        """
    )
    
    parser.add_argument(
        '--folder_a', '-a',
        type=str,
        required=True,
        help='Path to first image folder'
    )
    
    parser.add_argument(
        '--folder_b', '-b',
        type=str,
        required=True,
        help='Path to second image folder'
    )
    
    parser.add_argument(
        '--models', '-m',
        type=str,
        nargs='+',
        default=['all'],
        help='Models to use (loftr, roma, dkm, or all)'
    )
    
    parser.add_argument(
        '--pairs', '-p',
        type=str,
        nargs='*',
        help='Specific image pairs to process (comma-separated base names)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--max_pairs',
        type=int,
        default=None,
        help='Maximum number of pairs to process'
    )
    
    parser.add_argument(
        '--top_k',
        type=int,
        default=100,
        help='Number of top matches to visualize'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda or cpu)'
    )
    
    parser.add_argument(
        '--save_vis',
        action='store_true',
        help='Save visualization images'
    )
    
    parser.add_argument(
        '--no_display',
        action='store_true',
        help='Do not display figures (useful for headless mode)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


def find_image_pairs(
    folder_a: Path,
    folder_b: Path,
    specific_pairs: Optional[List[str]] = None
) -> List[Tuple[Path, Path]]:
    """
    Find matching image pairs between two folders.
    
    Args:
        folder_a: First folder
        folder_b: Second folder
        specific_pairs: Optional list of specific base names to match
        
    Returns:
        List of (path_a, path_b) tuples
    """
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Get all images in folder_a
    images_a = {}
    for ext in extensions:
        for path in folder_a.glob(f'*{ext}'):
            base = path.stem
            images_a[base] = path
        for path in folder_a.glob(f'*{ext.upper()}'):
            base = path.stem
            images_a[base] = path
    
    # Get all images in folder_b
    images_b = {}
    for ext in extensions:
        for path in folder_b.glob(f'*{ext}'):
            base = path.stem
            images_b[base] = path
        for path in folder_b.glob(f'*{ext.upper()}'):
            base = path.stem
            images_b[base] = path
    
    # Find matching pairs
    pairs = []
    
    if specific_pairs:
        # Use specific pairs
        for pair_str in specific_pairs:
            names = pair_str.split(',')
            if len(names) == 1:
                # Same name in both folders
                name = names[0].strip()
                if name in images_a and name in images_b:
                    pairs.append((images_a[name], images_b[name]))
                else:
                    logger.warning(f"Could not find pair for: {name}")
            elif len(names) == 2:
                # Different names
                name_a, name_b = names[0].strip(), names[1].strip()
                if name_a in images_a and name_b in images_b:
                    pairs.append((images_a[name_a], images_b[name_b]))
                else:
                    logger.warning(f"Could not find pair: {name_a}, {name_b}")
    else:
        # Find all matching base names
        common_names = set(images_a.keys()) & set(images_b.keys())
        for name in sorted(common_names):
            pairs.append((images_a[name], images_b[name]))
    
    return pairs


def load_image_pair(path_a: Path, path_b: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load an image pair."""
    img_a = np.array(Image.open(path_a).convert('RGB'))
    img_b = np.array(Image.open(path_b).convert('RGB'))
    return img_a, img_b


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate folders
    folder_a = Path(args.folder_a)
    folder_b = Path(args.folder_b)
    
    if not folder_a.exists():
        logger.error(f"Folder A does not exist: {folder_a}")
        sys.exit(1)
    if not folder_b.exists():
        logger.error(f"Folder B does not exist: {folder_b}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find image pairs
    logger.info(f"Finding image pairs between {folder_a} and {folder_b}...")
    pairs = find_image_pairs(folder_a, folder_b, args.pairs)
    
    if not pairs:
        logger.error("No matching image pairs found!")
        sys.exit(1)
    
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    
    logger.info(f"Found {len(pairs)} image pairs to process")
    
    # Import stereo matchers
    try:
        from stereo_matchers import ModelBenchmark
        from stereo_matchers.viz import (
            visualize_matches, compare_models, 
            plot_confidence_histogram, plot_match_comparison
        )
    except ImportError as e:
        logger.error(f"Could not import stereo_matchers: {e}")
        logger.info("Make sure you're running from the correct directory or add to PYTHONPATH")
        sys.exit(1)
    
    # Determine models
    if 'all' in args.models:
        models = ['loftr', 'roma', 'dkm']
    else:
        models = args.models
    
    logger.info(f"Using models: {models}")
    logger.info(f"Device: {args.device or 'auto'}")
    
    # Load image pairs
    logger.info("Loading image pairs...")
    image_pairs = []
    pair_names = []
    
    for path_a, path_b in pairs:
        try:
            img_a, img_b = load_image_pair(path_a, path_b)
            image_pairs.append((img_a, img_b))
            pair_names.append(f"{path_a.stem}")
            logger.debug(f"Loaded: {path_a.name} <-> {path_b.name}")
        except Exception as e:
            logger.error(f"Error loading {path_a} / {path_b}: {e}")
    
    if not image_pairs:
        logger.error("No images could be loaded!")
        sys.exit(1)
    
    logger.info(f"Loaded {len(image_pairs)} image pairs")
    
    # Run benchmark
    logger.info("Starting benchmark...")
    start_time = time.time()
    
    benchmark = ModelBenchmark(models=models, device=args.device)
    results = benchmark.run(image_pairs, verbose=True)
    
    total_time = time.time() - start_time
    logger.info(f"\nTotal benchmark time: {total_time:.2f}s")
    
    # Generate comparison table
    df = benchmark.compare()
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # Save results
    csv_path = output_dir / 'benchmark_results.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")
    
    # Generate report
    report = benchmark.generate_report(output_dir / 'benchmark_report.txt')
    print(report)
    
    # Generate visualizations
    if args.save_vis or not args.no_display:
        import matplotlib
        if args.no_display:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Comparison plot
        fig = benchmark.plot_comparison(figsize=(16, 12))
        if args.save_vis:
            fig.savefig(output_dir / 'comparison_plot.png', dpi=150, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {output_dir / 'comparison_plot.png'}")
        if not args.no_display:
            plt.show()
        plt.close(fig)
        
        # Visualize matches for each pair
        if args.save_vis:
            vis_dir = output_dir / 'visualizations'
            vis_dir.mkdir(exist_ok=True)
            
            for i, ((img_a, img_b), pair_name) in enumerate(zip(image_pairs, pair_names)):
                # Get results for this pair
                pair_results = {
                    name: benchmark_result.results[i]
                    for name, benchmark_result in results.items()
                }
                
                # Compare models
                fig = compare_models(img_a, img_b, pair_results, top_k=args.top_k)
                fig.savefig(vis_dir / f'{pair_name}_comparison.png', dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                # Individual model visualizations
                for model_name, result in pair_results.items():
                    fig = visualize_matches(
                        img_a, img_b, result,
                        top_k=args.top_k,
                        title=f"{model_name}: {pair_name}"
                    )
                    fig.savefig(vis_dir / f'{pair_name}_{model_name}.png', dpi=150, bbox_inches='tight')
                    plt.close(fig)
            
            logger.info(f"Visualizations saved to {vis_dir}")
        
        # Confidence histogram
        if len(image_pairs) == 1:
            first_pair_results = {
                name: result.results[0]
                for name, result in results.items()
            }
            fig = plot_confidence_histogram(first_pair_results)
            if args.save_vis:
                fig.savefig(output_dir / 'confidence_histogram.png', dpi=150, bbox_inches='tight')
            if not args.no_display:
                plt.show()
            plt.close(fig)
    
    logger.info("\nBenchmark complete!")
    logger.info(f"Results saved to: {output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
