"""
Benchmarking utilities for comparing stereo matchers.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import torch

from .base import BaseMatcher, MatchResult

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """
    Container for benchmark results.
    
    Attributes:
        model_name: Name of the model
        results: List of MatchResult objects
        avg_inference_time: Average inference time
        avg_num_matches: Average number of matches
        total_time: Total benchmarking time
    """
    model_name: str
    results: List[MatchResult]
    avg_inference_time: float = 0.0
    avg_num_matches: float = 0.0
    total_time: float = 0.0
    
    def __post_init__(self):
        if self.results:
            self.avg_inference_time = np.mean([r.inference_time for r in self.results])
            self.avg_num_matches = np.mean([r.num_matches for r in self.results])
    
    @property
    def confidence_stats(self) -> Dict[str, float]:
        """Aggregate confidence statistics across all results."""
        all_conf = np.concatenate([r.confidence for r in self.results if r.num_matches > 0])
        if len(all_conf) == 0:
            return {'mean': 0.0, 'median': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        return {
            'mean': float(np.mean(all_conf)),
            'median': float(np.median(all_conf)),
            'std': float(np.std(all_conf)),
            'min': float(np.min(all_conf)),
            'max': float(np.max(all_conf))
        }


class ModelBenchmark:
    """
    Benchmark utility for comparing multiple stereo matchers.
    
    Args:
        models: List of model names to benchmark ('loftr', 'roma', 'dkm')
        device: Device to run on
        
    Example:
        >>> benchmark = ModelBenchmark(models=['loftr', 'roma', 'dkm'])
        >>> results = benchmark.run(image_pairs)
        >>> df = benchmark.compare()
        >>> benchmark.plot_comparison()
    """
    
    MODEL_REGISTRY = {
        'loftr': ('stereo_matchers.loftr', 'LoFTRMatcher'),
        'loftr_indoor': ('stereo_matchers.loftr', 'LoFTRMatcher', {'pretrained': 'indoor'}),
        'loftr_outdoor': ('stereo_matchers.loftr', 'LoFTRMatcher', {'pretrained': 'outdoor'}),
        'roma': ('stereo_matchers.roma', 'RoMaMatcher'),
        'roma_indoor': ('stereo_matchers.roma', 'RoMaMatcher', {'model_type': 'indoor'}),
        'roma_outdoor': ('stereo_matchers.roma', 'RoMaMatcher', {'model_type': 'outdoor'}),
        'roma_tiny': ('stereo_matchers.roma', 'RoMaTinyMatcher'),
        'dkm': ('stereo_matchers.dkm', 'DKMMatcher'),
        'dkm_indoor': ('stereo_matchers.dkm', 'DKMMatcher', {'model_type': 'indoor'}),
        'dkm_outdoor': ('stereo_matchers.dkm', 'DKMMatcher', {'model_type': 'outdoor'}),
        'dkm_lite': ('stereo_matchers.dkm', 'DKMLiteMatcher'),
    }
    
    def __init__(
        self,
        models: List[str] = None,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize benchmark.
        
        Args:
            models: List of model names (use 'all' for all available)
            device: Device to run on
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._kwargs = kwargs
        
        if models is None or models == ['all']:
            models = ['loftr', 'roma', 'dkm']
        
        self.model_names = models
        self.matchers: Dict[str, BaseMatcher] = {}
        self.benchmark_results: Dict[str, BenchmarkResult] = {}
        
        logger.info(f"Benchmark initialized with models: {models}")
    
    def _load_matcher(self, model_name: str) -> Optional[BaseMatcher]:
        """Load a matcher by name."""
        if model_name not in self.MODEL_REGISTRY:
            logger.warning(f"Unknown model: {model_name}")
            return None
        
        entry = self.MODEL_REGISTRY[model_name]
        module_path = entry[0]
        class_name = entry[1]
        init_kwargs = entry[2] if len(entry) > 2 else {}
        
        try:
            # Import the module
            import importlib
            module = importlib.import_module(module_path)
            matcher_class = getattr(module, class_name)
            
            # Merge kwargs
            all_kwargs = {**init_kwargs, **self._kwargs}
            
            # Create matcher
            matcher = matcher_class(device=self.device, **all_kwargs)
            return matcher
            
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            return None
    
    def load_models(self) -> None:
        """Load all specified models."""
        for name in self.model_names:
            if name not in self.matchers:
                matcher = self._load_matcher(name)
                if matcher is not None:
                    self.matchers[name] = matcher
                    logger.info(f"Loaded {name}")
    
    def run(
        self,
        image_pairs: List[Tuple[np.ndarray, np.ndarray]],
        verbose: bool = True
    ) -> Dict[str, BenchmarkResult]:
        """
        Run benchmark on image pairs.
        
        Args:
            image_pairs: List of (image0, image1) tuples
            verbose: Whether to print progress
            
        Returns:
            Dictionary of model_name -> BenchmarkResult
        """
        # Load models if not loaded
        self.load_models()
        
        n_pairs = len(image_pairs)
        
        for model_name, matcher in self.matchers.items():
            if verbose:
                print(f"\n{'='*60}")
                print(f"Running {model_name} on {n_pairs} image pairs...")
                print(f"{'='*60}")
            
            results = []
            total_start = time.time()
            
            for i, (img0, img1) in enumerate(image_pairs):
                try:
                    result = matcher.match(img0, img1)
                    results.append(result)
                    
                    if verbose:
                        print(f"  Pair {i+1}/{n_pairs}: {result.num_matches} matches "
                              f"in {result.inference_time:.3f}s")
                        
                except Exception as e:
                    logger.error(f"Error matching pair {i} with {model_name}: {e}")
                    # Create empty result
                    results.append(MatchResult(
                        keypoints0=np.zeros((0, 2)),
                        keypoints1=np.zeros((0, 2)),
                        confidence=np.zeros(0),
                        num_matches=0,
                        inference_time=0.0,
                        model_name=model_name,
                        extra={'error': str(e)}
                    ))
            
            total_time = time.time() - total_start
            
            benchmark_result = BenchmarkResult(
                model_name=model_name,
                results=results,
                total_time=total_time
            )
            
            self.benchmark_results[model_name] = benchmark_result
            
            if verbose:
                print(f"\n{model_name} Summary:")
                print(f"  Total time: {total_time:.2f}s")
                print(f"  Avg inference: {benchmark_result.avg_inference_time:.3f}s")
                print(f"  Avg matches: {benchmark_result.avg_num_matches:.1f}")
        
        return self.benchmark_results
    
    def run_single(
        self,
        image0: np.ndarray,
        image1: np.ndarray
    ) -> Dict[str, MatchResult]:
        """
        Run all models on a single image pair.
        
        Args:
            image0: First image
            image1: Second image
            
        Returns:
            Dictionary of model_name -> MatchResult
        """
        # Load models if not loaded
        self.load_models()
        
        results = {}
        for model_name, matcher in self.matchers.items():
            try:
                results[model_name] = matcher.match(image0, image1)
            except Exception as e:
                logger.error(f"Error with {model_name}: {e}")
                results[model_name] = MatchResult(
                    keypoints0=np.zeros((0, 2)),
                    keypoints1=np.zeros((0, 2)),
                    confidence=np.zeros(0),
                    num_matches=0,
                    inference_time=0.0,
                    model_name=model_name,
                    extra={'error': str(e)}
                )
        
        return results
    
    def compare(self) -> pd.DataFrame:
        """
        Generate comparison DataFrame.
        
        Returns:
            DataFrame with comparison metrics
        """
        if not self.benchmark_results:
            raise ValueError("No benchmark results. Run benchmark first.")
        
        data = []
        for name, result in self.benchmark_results.items():
            conf_stats = result.confidence_stats
            data.append({
                'Model': name,
                'Avg_Time_s': result.avg_inference_time,
                'Avg_Matches': result.avg_num_matches,
                'Total_Time_s': result.total_time,
                'Conf_Mean': conf_stats['mean'],
                'Conf_Median': conf_stats['median'],
                'Conf_Std': conf_stats['std'],
                'Conf_Min': conf_stats['min'],
                'Conf_Max': conf_stats['max'],
                'Num_Pairs': len(result.results),
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Avg_Matches', ascending=False)
        
        return df
    
    def plot_comparison(
        self,
        figsize: Tuple[int, int] = (16, 12),
        save_path: Optional[str] = None
    ):
        """
        Plot comparison charts.
        
        Args:
            figsize: Figure size
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt
        
        if not self.benchmark_results:
            raise ValueError("No benchmark results. Run benchmark first.")
        
        df = self.compare()
        models = df['Model'].tolist()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Average matches
        ax = axes[0, 0]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
        bars = ax.bar(models, df['Avg_Matches'], color=colors)
        ax.set_ylabel('Average Matches')
        ax.set_title('Average Matches per Image Pair')
        ax.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars, df['Avg_Matches']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.0f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Inference time
        ax = axes[0, 1]
        bars = ax.bar(models, df['Avg_Time_s'], color=colors)
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Average Inference Time')
        ax.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars, df['Avg_Time_s']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.3f}s', ha='center', va='bottom', fontsize=10)
        
        # 3. Confidence distribution
        ax = axes[1, 0]
        x = np.arange(len(models))
        width = 0.35
        ax.bar(x - width/2, df['Conf_Mean'], width, label='Mean', color='steelblue')
        ax.bar(x + width/2, df['Conf_Median'], width, label='Median', color='coral')
        ax.errorbar(x - width/2, df['Conf_Mean'], yerr=df['Conf_Std'], 
                   fmt='none', color='black', capsize=3)
        ax.set_ylabel('Confidence')
        ax.set_title('Match Confidence Statistics')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        
        # 4. Matches per second (efficiency)
        ax = axes[1, 1]
        efficiency = df['Avg_Matches'] / df['Avg_Time_s']
        bars = ax.bar(models, efficiency, color=colors)
        ax.set_ylabel('Matches / Second')
        ax.set_title('Matching Efficiency')
        ax.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars, efficiency):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.0f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        return fig
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a text report of the benchmark results.
        
        Args:
            output_path: Optional path to save the report
            
        Returns:
            Report as string
        """
        if not self.benchmark_results:
            return "No benchmark results. Run benchmark first."
        
        df = self.compare()
        
        lines = [
            "=" * 80,
            "STEREO MATCHER BENCHMARK REPORT",
            "=" * 80,
            "",
            f"Models tested: {', '.join(self.model_names)}",
            f"Device: {self.device}",
            "",
            "-" * 80,
            "SUMMARY TABLE",
            "-" * 80,
            "",
            df.to_string(index=False),
            "",
            "-" * 80,
            "KEY FINDINGS",
            "-" * 80,
            ""
        ]
        
        # Best performers
        best_matches = df.loc[df['Avg_Matches'].idxmax()]
        best_speed = df.loc[df['Avg_Time_s'].idxmin()]
        best_confidence = df.loc[df['Conf_Mean'].idxmax()]
        
        lines.extend([
            f"✓ Most matches: {best_matches['Model']} ({best_matches['Avg_Matches']:.0f} avg)",
            f"✓ Fastest: {best_speed['Model']} ({best_speed['Avg_Time_s']:.3f}s avg)",
            f"✓ Best confidence: {best_confidence['Model']} ({best_confidence['Conf_Mean']:.3f} mean)",
            "",
            "=" * 80,
        ])
        
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
        
        return report


def quick_benchmark(
    image0: np.ndarray,
    image1: np.ndarray,
    models: List[str] = None,
    device: Optional[str] = None
) -> pd.DataFrame:
    """
    Quick benchmark on a single image pair.
    
    Args:
        image0: First image
        image1: Second image
        models: List of models to benchmark
        device: Device to use
        
    Returns:
        Comparison DataFrame
    """
    if models is None:
        models = ['loftr', 'roma', 'dkm']
    
    benchmark = ModelBenchmark(models=models, device=device)
    benchmark.run([(image0, image1)], verbose=False)
    
    return benchmark.compare()
