# STEREOSTUDY

> **Stereo Cloud Height Estimation and Contrail Detection**  
> EUROCONTROL / IPSL Collaborative Research Project

Automated altitude estimation of clouds and aircraft contrails using stereoscopic computer vision with two ground-based all-sky cameras separated by ~16 km in the Île-de-France region, south of Paris.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Scientific Objectives](#scientific-objectives)
- [System Architecture](#system-architecture)
- [Camera Systems](#camera-systems)
- [Repository Structure](#repository-structure)
- [Module Reference](#module-reference)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Technology Stack](#technology-stack)
- [Data Access](#data-access)
- [Contributing](#contributing)

---

## 🔬 Overview

STEREOSTUDY estimates the **3D altitude of atmospheric features** (clouds and aircraft contrails) by observing them simultaneously from two geographically separated ground-based cameras using **stereo triangulation**. The system combines:

- **Fisheye camera calibration** — Pixel ↔ (azimuth, zenith) conversion using Fripon polynomial models
- **Image reprojection** — Transforming hemispherical images onto common horizontal planes at target altitudes
- **Deep feature matching** — LoFTR, RoMa, DKM neural networks for finding corresponding points
- **Stereo triangulation** — 3D altitude estimation from matched feature pairs with uncertainty quantification
- **Contrail segmentation** — Mask2Former-based panoptic segmentation for contrail detection
- **Flight correlation** — ADS-B trajectory projection and comparison with detected features
- **Contrail advection** — Wind-driven contrail displacement modeling using ERA5 meteorological data

**Camera Locations:**
- **SIRTA Observatory** (IPSL, Palaiseau): 48.7133°N, 2.2081°E, 156-177m ASL
- **ECTL** (EUROCONTROL, Brétigny-sur-Orge): 48.6005°N, 2.3468°E, 90m ASL
- **Baseline distance:** ~15 km (optimal for 8-12 km altitude measurement)

---

## 🎯 Scientific Objectives

1. **Cloud Base Height Estimation** — Ground-based alternative to ceilometers and lidar
2. **Contrail Altitude Retrieval** — Validate ADS-B altitude data for aviation environmental studies
3. **Contrail Persistence Analysis** — Track contrail evolution and windborne advection
4. **Validation Data** — Provide ground-truth altitude measurements for satellite/model validation
5. **Atmospheric Research** — Study cloud formation and contrail-cirrus interactions

**Key Research Questions:**
- Can stereo vision from all-sky cameras reliably estimate cloud/contrail altitude?
- How do measurement uncertainties scale with baseline distance and altitude?
- Can we track individual contrails across time using wind field advection models?
- How do different deep learning matchers (LoFTR, RoMa, DKM) perform for atmospheric features?

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  STEREOSTUDY Processing Pipeline                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📷 SIRTA Camera      📷 ECTL Camera      ✈️ ADS-B Data        │
│  (1024×768 fisheye)   (1280×960 fisheye)  (OpenSky Network)    │
│         │                    │                    │             │
│         ├────────────────────┤                    │             │
│         │                                          │             │
│   ┌─────▼──────────────────┐              ┌──────▼─────────┐   │
│   │  Camera Calibration    │              │ Flight Filter  │   │
│   │  Fripon Model / JP2    │              │ & Projection   │   │
│   │  pixel ↔ (az, zen)     │              └────────┬───────┘   │
│   └─────┬──────────────────┘                       │           │
│         │                                           │           │
│   ┌─────▼──────────────────┐                       │           │
│   │ Image Reprojection     │                       │           │
│   │ Fisheye → 10km Plane   │                       │           │
│   │ 75×75 km @ 1024² px    │                       │           │
│   └─────┬──────────────────┘                       │           │
│         │                                           │           │
│   ┌─────▼────────┐   ┌──────────────┐             │           │
│   │  Contrail    │   │    Stereo    │             │           │
│   │ Segmentation │   │   Feature    │             │           │
│   │(Mask2Former) │   │   Matching   │             │           │
│   └──────────────┘   │ LoFTR/RoMa   │             │           │
│                      │    /DKM      │             │           │
│                      └─────┬────────┘             │           │
│                            │                      │           │
│                      ┌─────▼────────┐             │           │
│                      │      3D      │◄────────────┘           │
│                      │Triangulation │                         │
│                      │+Uncertainty  │                         │
│                      └─────┬────────┘                         │
│                            │                                  │
│                      ┌─────▼────────┐                         │
│                      │Visualization │                         │
│                      │& Validation  │                         │
│                      └──────────────┘                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📷 Camera Systems

### SIRTA (IPSL) Camera

| Parameter | Value |
|-----------|-------|
| **Location** | SIRTA Observatory, Palaiseau, France |
| **Coordinates** | 48.7133°N, 2.2081°E |
| **Elevation** | 156–177.5 m ASL |
| **Image Size** | 1024 × 768 pixels (RGB) |
| **Field of View** | 180° hemispherical |
| **Calibration** | Fripon 6-parameter polynomial model |
| **Site Code** | `SIRTA_W` |
| **Dataset ID** | `gQg5IUvV` |

### ECTL (Orsay) Camera

| Parameter | Value |
|-----------|-------|
| **Location** | EUROCONTROL Experimental Centre, Brétigny-sur-Orge |
| **Coordinates** | 48.6005°N, 2.3468°E |
| **Elevation** | 90 m ASL |
| **Image Size** | 1280 × 960 pixels (RGB) |
| **Field of View** | 180° hemispherical |
| **Calibration** | JP2 calibration maps + Fripon model |
| **Site Code** | `Orsay` |
| **Dataset ID** | `OdnkTZQ8` |

### Stereo Configuration

| Parameter | Value |
|-----------|-------|
| **Baseline Distance** | ~15 km |
| **Baseline Orientation** | Approximately East-West |
| **Target Altitude** | 8–12 km (contrail cruise altitude) |
| **Grid Coverage** | 75 km × 75 km at target altitude |
| **Projection Resolution** | 1024 × 1024 pixels |
| **Pixel Size** | ~73 m/pixel at 10 km altitude |

---

## 📁 Repository Structure

```
stereostudy/
│
├── 📄 README.md             ← This file
├── 📄 .gitignore            ← Git exclusions
│
├── 📦 cheick_code/          ← Camera Calibration (Fripon/M1 models)
├── 📦 DeepMatching/         ← Stereo Feature Matching & 3D Triangulation
├── 📦 azimuth_zenith_calibration/  ← Az/Zen Mapper Package
├── 📦 Flights/              ← ADS-B Flight Processing & Advection
└── 📦 Codebase/             ← Main Processing Pipeline
```

---

## 📚 Module Reference

### 1. `cheick_code/` — Camera Calibration

Converts between pixel coordinates and angular sky coordinates (azimuth, zenith) using Fripon polynomial model.

**Key Files:**
- `calibration/baseCalibration.py` — Core math, rotation matrices, parameter I/O
- `calibration/calibrationFripon.py` — Fripon model: `model()`, `invModel()`
- `calibration/FriponModel.py` — OOP interface
- `calibration/OriginalModel.py` — 23-parameter original model
- `setup_variable/position.py` — Camera lat/lon reader
- `params.csv` — Calibration parameters for all sites

**Usage:**
```python
from cheick_code.calibration.FriponModel import FriponModel

model = FriponModel("params.csv", "SIRTA_W")
azimuth, zenith, distance = model.pixel_to_world(512, 384)
ix, iy = model.world_to_pixel(180.0, 45.0)
```

---

### 2. `DeepMatching/` — Stereo Feature Matching & Triangulation

Unified interface for deep learning stereo matchers (LoFTR, RoMa, DKM) with 3D triangulation.

**`stereo_matchers/`** — Feature matching:
- `base.py` — Abstract `BaseMatcher` + `MatchResult` dataclass
- `loftr.py` — LoFTR matcher (Kornia)
- `roma.py` — RoMa matcher
- `dkm.py` — DKM matcher
- `benchmark.py` — Performance comparison

**`stereo_3d/`** — 3D reconstruction:
- `triangulation.py` — `AtmosphericTriangulator` for altitude estimation
- `coordinates.py` — WGS84/ECEF/ENU transforms
- `fisheye_model.py` — Generic fisheye camera model
- `uncertainty.py` — Error propagation
- `quality.py` — Match quality assessment

**Usage:**
```python
from stereo_matchers import LoFTRMatcher
from stereo_3d import AtmosphericTriangulator

# Match features
matcher = LoFTRMatcher(device='cuda')
matches = matcher.match(img1, img2)

# Triangulate
triangulator = AtmosphericTriangulator(cam1, cam2)
results = triangulator.triangulate_matches(matches)

# Analyze
altitudes = [r.altitude / 1000 for r in results]
print(f"Mean altitude: {np.mean(altitudes):.2f} ± {np.std(altitudes):.2f} km")
```

---

### 3. `azimuth_zenith_calibration/` — Az/Zen Mapper

Batch pixel → (azimuth, zenith) conversion using pre-computed calibration maps (NPZ files).

**Files:**
- `converter.py` — `AzimuthZenithMapper` class
- `cli.py` — Command-line interface

**Usage:**
```python
from azimuth_zenith_calibration import AzimuthZenithMapper

mapper = AzimuthZenithMapper("azimuth_map.npz", "zenith_map.npz")
azimuths, zeniths = mapper.pixels_to_angles(ix_array, iy_array)
```

---

### 4. `Flights/` — ADS-B Flight Processing

Filter and project ADS-B flight data from OpenSky Network.

**Scripts:**
- `process_flights.py` — Filter flights visible from camera FOV
- `project_flights_to_cameras.py` — Project 3D tracks to camera pixels
- `Advection/advection.py` — Contrail advection with ERA5 winds

**Usage:**
```bash
# Filter flights
python process_flights.py --date 2025-04-06 --output filtered.csv

# Project to pixels
python project_flights_to_cameras.py --flights filtered.csv --output projected.csv

# Advect contrail
python Advection/advection.py --start-position 48.65,2.25,10000 --duration 60min
```

---

### 5. `Codebase/` — Main Processing Pipeline

Comprehensive pipeline for reprojection, segmentation, and stereo matching.

**`src/processing/`** — Image reprojection:
- `batch_reproject_dataset.py` — Batch fisheye → plane reprojection
- `project_images_to_plane.py` — Per-pixel projection logic
- `ground_camera_projector.py` — `GroundCameraProjector` class
- `histogram_matching.py` — Cross-camera color matching

**`src/visualization/`** — Visualization tools:
- `visualize_azimuth_zenith.py` — Az/zen map visualization

**`src/utils/`** — Utilities:
- `filter_zenith_circle.py` — Circular zenith mask
- `explain_optical_center.py` — Optical center diagnostics

**`refactored/`** — Modular inference pipeline:
- `main_comparison.py` — CLI orchestrator
- `model_loader.py` — Mask2Former loader
- `inference.py` — Segmentation inference
- `flight_projection.py` — Flight overlay

**`FineTuning/`** — Model training:
- `finetune_mask2former.ipynb` — Training notebook
- `contrail_segmentation/` — Model checkpoint (config.json, preprocessor_config.json)

**`notebooks/`** — Analysis notebooks:
- `stereo_miniprojector_inference.ipynb` — Full pipeline (41MB)
- `dataset_analysis.ipynb` — Dataset EDA
- `stereo_inference_comparison.ipynb` — Model comparison
- `adsb_extraction.ipynb` — ADS-B extraction

**`tests/`** — Unit tests:
- `test_plane_projection.py` — Plane projection tests
- `test_coordinate_order.py` — Coordinate system validation

**`docs/`** — Technical documentation:
- `CALIBRATION_ANALYSIS_AND_PLAN.md`
- `PLANE_PROJECTION_TEST_RESULTS.md`
- `SKYCAM_REPROJECTION_PLAN.md`
- `SKYCAM_RESEARCH_SUMMARY.md`

---

## 🔧 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for deep learning matchers)
- 16+ GB RAM (for large image processing)

### Install Dependencies

```bash
# Clone repository
git clone https://github.com/anapelin/stereostudy.git
cd stereostudy

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies
pip install -r Codebase/requirements.txt

# Install DeepMatching module
cd DeepMatching
pip install -e .
cd ..

# Install cheick_code calibration
cd cheick_code
pip install -e .
cd ..
```

### Install Optional Dependencies

```bash
# For Mask2Former segmentation
pip install transformers[torch] accelerate

# For contrail advection
pip install pycontrails

# For notebook visualization
pip install jupyter matplotlib seaborn
```

---

## 🚀 Quick Start

### 1. Camera Calibration

```python
from cheick_code.calibration.FriponModel import FriponModel

# Load SIRTA camera model
sirta_model = FriponModel("cheick_code/params.csv", "SIRTA_W")

# Convert pixel to sky coordinates
azimuth, zenith, dist = sirta_model.pixel_to_world(512, 384)
print(f"Azimuth: {azimuth:.2f}°, Zenith: {zenith:.2f}°")

# Convert sky coordinates to pixel
ix, iy = sirta_model.world_to_pixel(180.0, 45.0)
print(f"Pixel: ({ix:.1f}, {iy:.1f})")
```

### 2. Image Reprojection

```bash
# Batch reproject dataset to 10 km horizontal plane
python Codebase/src/processing/batch_reproject_dataset.py \
    --input-dir /path/to/gQg5IUvV/ \
    --output-dir /path/to/gQg5IUvV_projected_10km/ \
    --altitude 10000 \
    --resolution 1024 \
    --extent-km 75
```

### 3. Stereo Feature Matching

```python
from stereo_matchers import LoFTRMatcher
from PIL import Image
import numpy as np

# Load images
img1 = np.array(Image.open("sirta_reprojected.jpg"))
img2 = np.array(Image.open("ectl_reprojected.jpg"))

# Match features
matcher = LoFTRMatcher(device='cuda')
matches = matcher.match(img1, img2)

print(f"Found {len(matches.keypoints0)} matches")
print(f"Mean confidence: {matches.confidence.mean():.3f}")
```

### 4. 3D Triangulation

```python
from stereo_3d import AtmosphericTriangulator, FisheyeCameraModel

# Define cameras
cam1 = FisheyeCameraModel(lat=48.7133, lon=2.2081, alt=177.5)
cam2 = FisheyeCameraModel(lat=48.6005, lon=2.3468, alt=90.0)

# Triangulate
triangulator = AtmosphericTriangulator(cam1, cam2)
results = triangulator.triangulate_matches(matches, uncertainties=True)

# Filter high-quality matches
high_quality = [r for r in results if r.quality_score > 0.8]
altitudes = [r.altitude / 1000 for r in high_quality]  # km

print(f"High-quality matches: {len(high_quality)}")
print(f"Mean altitude: {np.mean(altitudes):.2f} ± {np.std(altitudes):.2f} km")
```

### 5. Run Full Pipeline

```bash
# Run modular inference pipeline with flight overlay
python Codebase/refactored/main_comparison.py \
    --model-path Codebase/FineTuning/contrail_segmentation/ \
    --dataset-dir /path/to/gQg5IUvV_projected_10km/ \
    --flights Flights/2025-04-06_final.csv \
    --output-dir outputs/
```

---

## 🔄 Pipeline Walkthrough

### Step 1: Camera Calibration

Load calibration parameters for each camera and verify accuracy:

```bash
python Codebase/scripts/verify_camera_configs.py \
    --config Codebase/config/cameras/sirta_camera.json
```

### Step 2: Image Reprojection

Transform fisheye images to common horizontal plane:

```bash
# SIRTA camera
python Codebase/src/processing/batch_reproject_dataset.py \
    --camera SIRTA_W \
    --altitude 10000

# ECTL camera
python Codebase/src/processing/batch_reproject_dataset.py \
    --camera Orsay \
    --altitude 10000
```

### Step 3: Contrail Segmentation (Optional)

Segment contrails using fine-tuned Mask2Former:

```python
from Codebase.refactored.model_loader import load_segmentation_model
from Codebase.refactored.inference import run_inference

model = load_segmentation_model("Codebase/FineTuning/contrail_segmentation/")
masks = run_inference(model, reprojected_image)
```

### Step 4: Stereo Matching

Find corresponding features between camera pairs:

```bash
cd DeepMatching
python run_matching.py \
    --image1 /path/to/sirta_reprojected.jpg \
    --image2 /path/to/ectl_reprojected.jpg \
    --matcher loftr \
    --output matches.npz
```

### Step 5: 3D Triangulation

Estimate altitudes from matched features:

```python
from DeepMatching.stereo_3d import AtmosphericTriangulator

results = triangulator.triangulate_matches(
    matches,
    azimuth_zenith_func1=sirta_model.pixel_to_world,
    azimuth_zenith_func2=ectl_model.pixel_to_world
)
```

### Step 6: Validation

Compare estimated altitudes with ADS-B flight data:

```bash
python Flights/project_flights_to_cameras.py \
    --flights Flights/2025-04-06_filtered.csv \
    --camera-model cheick_code/params.csv \
    --output projected_flights.csv
```

---

## 🛠️ Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Calibration** | NumPy, SciPy (Newton-Raphson), OpenCV |
| **Projection** | SciPy interpolation, Numba JIT compilation |
| **Feature Matching** | PyTorch, Kornia (LoFTR), RoMa, DKM |
| **Segmentation** | HuggingFace Transformers (Mask2Former), CUDA |
| **Triangulation** | NumPy (vectorized ECEF/ENU transforms) |
| **Geodesy** | geographiclib (WGS84), pyproj |
| **Flight Data** | pandas, pycontrails, ERA5 (ECMWF CDS) |
| **Visualization** | matplotlib, OpenCV, PIL/Pillow |

---

## 💾 Data Access

### Image Datasets

**SIRTA Camera (`gQg5IUvV`):**
- Raw fisheye images: 1024×768 RGB
- Reprojected images: 1024×1024 RGB @ 10 km altitude
- Temporal coverage: 2025-04-06 (sample day)

**ECTL Camera (`OdnkTZQ8`):**
- Raw fisheye images: 1280×960 RGB
- Reprojected images: 1024×1024 RGB @ 10 km altitude
- Temporal coverage: 2025-04-06 (sample day)

### ADS-B Flight Data

**Source:** OpenSky Network (https://opensky-network.org/)
- Coverage: Île-de-France region
- Fields: timestamp, callsign, lat, lon, altitude, velocity
- Processed files: `Flights/2025-04-06_*.csv`

### Calibration Data

**Format:** CSV (Fripon parameters), NPZ (azimuth/zenith maps), JP2 (calibration maps)
- SIRTA: 6-parameter Fripon model
- ECTL: JP2 calibration maps 
---

## 🤝 Contributing

This is an internal EUROCONTROL/IPSL research project. For questions or collaboration:

- **Primary Contact:** Ana-Maria Pelin
- **GitHub:** https://github.com/anapelin/stereostudy

---

## 📄 License

Internal EUROCONTROL/IPSL research project.  
`Betatesting/skycam/` library: EUPL-1.2 (European Union Public License)

---

## 🙏 Acknowledgments

- **SIRTA Observatory** (IPSL) — Camera access and data
- **EUROCONTROL** — Project funding and ECTL camera
- **OpenSky Network** — ADS-B data
- **ECMWF** — ERA5 meteorological reanalysis
- **Kornia**, **RoMa**, **DKM** — Deep learning stereo matchers
- **HuggingFace** — Mask2Former model hub

---

**Last Updated:** February 2026  
**Repository:** https://github.com/anapelin/stereostudy
