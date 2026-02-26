# Stereo Matchers

A unified Python interface for multiple stereo feature matching models, designed for easy integration with existing image processing pipelines.

## Features

- **Unified Interface**: All matchers use the same API for easy comparison
- **Multiple Models**: LoFTR, RoMa, DKM, and more
- **Benchmarking**: Built-in tools for comparing model performance
- **Visualization**: Publication-ready match visualizations
- **GPU Support**: Automatic CUDA detection and optimization

## Supported Models

| Model | Type | Speed | Accuracy | Notes |
|-------|------|-------|----------|-------|
| LoFTR | Dense | Fast | Good | Via Kornia, easy to install |
| LoFTR-Indoor | Dense | Fast | Good | Indoor-trained variant |
| RoMa | Dense | Medium | Excellent | State-of-the-art accuracy |
| RoMa-Tiny | Dense | Fast | Good | Faster, slightly less accurate |
| DKM | Dense | Medium | Excellent | Dense kernelized matching |

## Installation

### Basic Installation

```bash
# Clone or copy to your project
cd /path/to/DeepMatching

# Install dependencies
pip install -r requirements.txt
```

### Model-Specific Setup

#### LoFTR (via Kornia - Recommended for beginners)
```bash
pip install kornia kornia-moons
```

#### RoMa
```bash
pip install romatch
# OR clone manually:
git clone https://github.com/Parskatt/RoMa.git
cd RoMa && pip install -e .
```

#### DKM
```bash
git clone https://github.com/Parskatt/DKM.git
cd DKM && pip install -e .
```

## Quick Start

### From Python (Notebook Integration)

```python
from stereo_matchers import LoFTRMatcher, RoMaMatcher, DKMMatcher

# Initialize a matcher
matcher = LoFTRMatcher(device='cuda')

# Match two images (numpy arrays or torch tensors)
result = matcher.match(img1, img2)

# Access results
print(f"Found {result.num_matches} matches in {result.inference_time:.3f}s")
print(f"Keypoints shape: {result.keypoints0.shape}")
print(f"Confidence mean: {result.confidence.mean():.3f}")
```

### Benchmarking Multiple Models

```python
from stereo_matchers import ModelBenchmark

# Create benchmark with multiple models
benchmark = ModelBenchmark(models=['loftr', 'roma', 'dkm'])

# Run on image pairs
image_pairs = [(img1, img2), (img3, img4), ...]
results = benchmark.run(image_pairs)

# Get comparison DataFrame
df = benchmark.compare()
print(df)

# Plot comparison
fig = benchmark.plot_comparison()
```

### Visualization

```python
from stereo_matchers.viz import visualize_matches, compare_models

# Visualize single model results
fig = visualize_matches(img1, img2, result, top_k=100)

# Compare multiple models
results_dict = {
    'LoFTR': loftr_result,
    'RoMa': roma_result,
    'DKM': dkm_result
}
fig = compare_models(img1, img2, results_dict)
```

### Command Line Interface

```bash
# Run all models on matching image pairs
python run_matching.py --folder_a /path/to/camera1 --folder_b /path/to/camera2

# Run specific models
python run_matching.py -a images/left -b images/right --models loftr roma

# Save visualizations
python run_matching.py -a images/left -b images/right --save_vis --output results/

# Limit number of pairs
python run_matching.py -a images/left -b images/right --max_pairs 5

# Use specific device
python run_matching.py -a images/left -b images/right --device cuda:1
```

## API Reference

### MatchResult

All matchers return a `MatchResult` object:

```python
@dataclass
class MatchResult:
    keypoints0: np.ndarray   # Nx2 - keypoints in image 1
    keypoints1: np.ndarray   # Nx2 - keypoints in image 2
    confidence: np.ndarray   # N   - confidence scores (0-1)
    num_matches: int         # Total number of matches
    inference_time: float    # Time in seconds
    model_name: str          # Name of the model used
    extra: Dict              # Model-specific outputs
```

#### Methods

```python
# Filter by confidence threshold
filtered = result.filter_by_confidence(threshold=0.5)

# Get top-k matches by confidence
top_matches = result.top_k(k=100)

# Get confidence statistics
stats = result.confidence_stats  # {'mean', 'median', 'std', 'min', 'max'}

# Convert to dictionary
d = result.to_dict()
```

### Base Matcher Interface

All matchers inherit from `BaseMatcher`:

```python
class MyMatcher(BaseMatcher):
    def __init__(self, device=None, **kwargs):
        super().__init__(device=device, **kwargs)
    
    @property
    def name(self) -> str:
        return "MyMatcher"
    
    def _load_model(self):
        # Load model weights
        pass
    
    def _match_impl(self, image0, image1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Return keypoints0, keypoints1, confidence
        pass
```

### Input Format

Matchers accept:
- **NumPy arrays**: (H, W, 3) RGB or (H, W) grayscale, uint8 or float32
- **Torch tensors**: (C, H, W) or (1, C, H, W), float32 in 0-1 range

Images are automatically preprocessed:
- Converted to correct format (grayscale for LoFTR, RGB for RoMa/DKM)
- Normalized to 0-1 range
- Moved to appropriate device

## Output Files

When using the CLI with `--save_vis`, the following are generated:

```
outputs/
├── benchmark_results.csv    # Metrics table
├── benchmark_report.txt     # Text summary
├── comparison_plot.png      # Comparison charts
└── visualizations/
    ├── pair001_comparison.png     # All models side-by-side
    ├── pair001_loftr.png          # LoFTR matches
    ├── pair001_roma.png           # RoMa matches
    └── ...
```

## Interpreting Results

### Key Metrics

- **num_matches**: Total correspondences found. Higher isn't always better - many low-confidence matches may indicate noise.
- **confidence**: Model's certainty about each match (0-1). Higher mean confidence suggests more reliable matches.
- **inference_time**: Processing speed. Important for real-time applications.

### Model Selection Guidelines

| Use Case | Recommended Model | Reason |
|----------|-------------------|--------|
| Quick prototyping | LoFTR | Easiest to install, fast |
| Best accuracy | RoMa | State-of-the-art matching |
| Real-time applications | LoFTR or RoMa-Tiny | Fastest inference |
| Indoor scenes | LoFTR-Indoor | Trained on indoor data |
| Large viewpoint changes | RoMa or DKM | Better at wide baselines |

## Troubleshooting

### Common Issues

**1. CUDA out of memory**
```python
# Use smaller images
matcher.match(img1[::2, ::2], img2[::2, ::2])

# Or use CPU
matcher = LoFTRMatcher(device='cpu')
```

**2. Model not found / Import error**
```bash
# Check if model is installed
python -c "import kornia"  # For LoFTR
python -c "import romatch"  # For RoMa
```

**3. Slow performance**
```python
# Ensure GPU is being used
import torch
print(torch.cuda.is_available())  # Should be True

# Check which device the model is on
print(matcher.device)
```

**4. Few matches found**
- Try lowering confidence threshold
- Ensure images have overlapping content
- Check image quality (blur, exposure)

### Performance Tips

1. **Batch processing**: Process multiple pairs to amortize model loading time
2. **Image resolution**: Resize very large images (>2000px) for faster inference
3. **GPU memory**: Use `torch.cuda.empty_cache()` between large batches
4. **Warmup**: First inference is slower due to CUDA kernel compilation

## Integration Example

Here's how to integrate with the existing stereo inference notebook:

```python
# In your notebook
import sys
sys.path.append('/data/common/STEREOSTUDYIPSL/DeepMatching')

from stereo_matchers import LoFTRMatcher, RoMaMatcher
from stereo_matchers.viz import visualize_matches

# After loading and preprocessing your image pairs...
matcher = LoFTRMatcher(device='cuda')

for pair in matched_pairs:
    # Your existing preprocessing
    ipsl_image = load_and_preprocess(pair['ipsl_path'])
    ectl_image = load_and_preprocess(pair['ectl_path'])
    
    # Run matching
    result = matcher.match(ipsl_image, ectl_image)
    
    # Visualize
    fig = visualize_matches(ipsl_image, ectl_image, result, top_k=100)
    plt.show()
```

## License

MIT License - feel free to use and modify.

## References

- **LoFTR**: Sun et al., "LoFTR: Detector-Free Local Feature Matching with Transformers", CVPR 2021
- **RoMa**: Edstedt et al., "RoMa: Robust Dense Feature Matching", CVPR 2024
- **DKM**: Edstedt et al., "Deep Kernelized Dense Geometric Matching", CVPR 2023
