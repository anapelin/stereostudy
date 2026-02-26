# Stereo Camera Inference Comparison - Refactored

This directory contains a refactored version of the stereo camera inference comparison notebook, organized into modular Python scripts for easier maintenance and execution.

## Project Structure

```
refactored/
├── config.py                  # Configuration and constants
├── model_loader.py           # Model and processor loading
├── data_utils.py             # Dataset loading and matching
├── image_processing.py       # Image preprocessing utilities
├── inference.py              # Segmentation inference
├── flight_projection.py      # Flight trajectory projection
├── visualization.py          # Plotting and visualization
├── main_comparison.py        # Main orchestration script
└── README.md                 # This file
```

## Module Overview

### config.py
Contains all configuration parameters, paths, and constants:
- Dataset paths (IPSL and ECTL)
- Model configuration
- Camera locations and calibration paths
- Projection settings
- Processing options

### model_loader.py
Functions for loading the segmentation model:
- `load_processor()` - Load image processor
- `load_model()` - Load segmentation model
- `setup_model_and_processor()` - Complete setup

### data_utils.py
Dataset operations and image pair matching:
- `extract_timestamp_from_ipsl()` - Parse IPSL filenames
- `extract_timestamp_from_ectl()` - Parse ECTL filenames
- `find_matching_image_pairs()` - Match images by timestamp
- `filter_pairs_by_time()` - Filter by time window
- `load_image_pair()` - Load matched image pairs
- `load_datasets()` - Load all images from both datasets

### image_processing.py
Image preprocessing functions:
- `match_histogram()` - Histogram matching
- `remove_sun_pixels()` - Sun removal
- `preprocess_image()` - Combined preprocessing pipeline

### inference.py
Segmentation inference operations:
- `run_inference()` - Run model inference
- `extract_statistics()` - Extract segmentation statistics
- `process_image_pair()` - Complete inference pipeline for image pairs

### flight_projection.py
Flight trajectory projection using skycam:
- `setup_skycam()` - Initialize aircraft projector
- `load_flight_data()` - Load and standardize flight data
- `filter_flights_by_timestamp()` - Filter flights by time and altitude
- `project_flights_to_pixels()` - Project GPS coordinates to pixels

### visualization.py
Plotting and visualization functions:
- `visualize_inference_pair()` - Side-by-side inference visualization
- `visualize_flights_on_image()` - Flight trajectory overlay
- `create_comparison_table()` - Statistics comparison table
- `visualize_calibration_maps()` - Calibration map visualization

### main_comparison.py
Main orchestration script that runs the complete workflow.

## Usage

### Basic Usage

Run the comparison on 10 image pairs:

```bash
python main_comparison.py
```

### Advanced Options

Process more pairs:
```bash
python main_comparison.py --num-pairs 20
```

Include flight trajectory projection:
```bash
python main_comparison.py --include-flights
```

Combine options:
```bash
python main_comparison.py --num-pairs 20 --include-flights
```

### Using as a Library

You can also import and use the modules in your own scripts:

```python
from config import DATASET1_DIR, DATASET2_DIR
from data_utils import load_datasets, find_matching_image_pairs
from model_loader import setup_model_and_processor
from inference import process_image_pair

# Load datasets
dataset1_images, dataset2_images = load_datasets(DATASET1_DIR, DATASET2_DIR)

# Find matched pairs
matched_pairs = find_matching_image_pairs(dataset1_images, dataset2_images)

# Setup model
model, processor, device = setup_model_and_processor()

# Process a pair
# ... (see main_comparison.py for full example)
```

## Configuration

Edit `config.py` to customize:

- **Dataset paths**: Change `DATASET1_NAME` and `DATASET2_NAME`
- **Time filtering**: Modify `TIME_FILTER_START_HOUR` and `TIME_FILTER_END_HOUR`
- **Histogram matching**: Set `USE_HISTOGRAM_MATCHING = True/False`
- **Model settings**: Update model paths and parameters
- **Projection settings**: Adjust altitude, coverage area, resolution

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- transformers (Hugging Face)
- scikit-image
- pandas
- numpy
- matplotlib
- PIL
- scipy
- pytz

Install dependencies:
```bash
pip install torch transformers scikit-image pandas numpy matplotlib pillow scipy pytz
```

## Output

The script produces:

1. **Console output**: Progress and statistics for each processed pair
2. **Visualizations**: 
   - Side-by-side image comparisons with segmentation overlays
   - Flight trajectory projections (if enabled)
3. **Comparison table**: Statistics table with contrail detection metrics
4. **In-memory results**: `results` list containing all processed pairs

## Notes

- The script automatically uses CUDA if available, otherwise falls back to CPU
- ECTL images are automatically flipped vertically during loading
- Flight projection requires the skycam library from the Betatesting folder
- Processing time depends on the number of pairs and hardware capabilities

## Troubleshooting

**Model loading errors**: Check that model checkpoint exists at `CHECKPOINT_DIR`

**Dataset not found**: Verify paths in `config.py` match your directory structure

**CUDA out of memory**: Reduce `num_pairs_to_process` or use smaller batches

**Flight projection errors**: Ensure skycam library is accessible and flight data file exists
