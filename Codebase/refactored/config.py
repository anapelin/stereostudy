"""
Configuration settings for stereo camera inference comparison.
"""
from pathlib import Path
import os

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Workspace and dataset paths
WORKSPACE_DIR = Path("/data/common/STEREOSTUDYIPSL")
DATASETS_DIR = WORKSPACE_DIR / "Datasets"

DATASET1_NAME = "gQg5IUvV"  # IPSL dataset
DATASET2_NAME = "OdnkTZQ8"  # ECTL dataset

DATASET1_DIR = DATASETS_DIR / DATASET1_NAME / "PROJECTED"
DATASET2_DIR = DATASETS_DIR / DATASET2_NAME / "PROJECTED"

# Model configuration
TASK = "panoptic"
MODEL_SIZE = "base"
RUN_ID = "polygon"

TRAILVISION_DIR = Path("/data/common/TRAILVISION")
SEGMENTATION_DIR = TRAILVISION_DIR / "segmentation"
MODELS_DIR = SEGMENTATION_DIR / "models"
CHECKPOINT_DIR = MODELS_DIR / TASK / RUN_ID

# Base model path
BASE_MODEL = "/data/common/STEREOSTUDYIPSL/Codebase/FineTuning/contrail_segmentation"

# Calibration paths
SIRTA_CALIB_PATH = WORKSPACE_DIR / "Codebase" / "data" / "azimuth_zenith_map_full_corrected.npz"
ECTL_CALIB_DIR = WORKSPACE_DIR / "Codebase" / "config" / "ECTL"

# Flight data
FLIGHTS_DIR = WORKSPACE_DIR / "Flights"
FLIGHT_CSV = FLIGHTS_DIR / "2025-04-06_summary.csv"

# Camera locations
SIRTA_LAT = 48.7133  # degrees
SIRTA_LON = 2.2081   # degrees
SIRTA_ALT = 162.0    # meters above sea level

ECTL_LAT = 48.7133   # degrees
ECTL_LON = 2.2081    # degrees
ECTL_ALT = 162.0     # meters above sea level

# Projection settings
PROJECTION_CLOUD_HEIGHT = 10000.0  # meters (10km)
PROJECTION_SQUARE_SIZE = 75000.0   # meters (75km)
PROJECTION_RESOLUTION = 1024       # pixels

# Processing options
USE_HISTOGRAM_MATCHING = False
TIME_FILTER_START_HOUR = 7
TIME_FILTER_END_HOUR = 8

# Inference parameters
INFERENCE_THRESHOLD = 0.5
INFERENCE_MASK_THRESHOLD = 0.5
INFERENCE_OVERLAP_THRESHOLD = 0.8

# Segmentation categories
CATEGORIES = [
    {"id": 1, "name": "contrail", "isthing": 1, "color": [255, 0, 0]},
]

if TASK == "panoptic":
    BACKGROUND_CATEGORY_ID = len(CATEGORIES) + 1
    CATEGORIES.append({
        "id": BACKGROUND_CATEGORY_ID,
        "name": "sky",
        "isthing": 0,
        "color": [135, 206, 235],
    })

# Create id2label mapping
ID2LABEL = {id: label["name"] for id, label in enumerate(CATEGORIES)}
