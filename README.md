# stereostudy

Stereo cloud-height estimation and contrail detection using a pair of ground-based all-sky (fisheye) cameras at SIRTA/IPSL (Palaiseau) and ECTL (Bretigny), ~15 km apart.

## Repository Structure

```
stereostudy/
├── cheick_code/              # Camera calibration (Fripon/M1 models)
│   ├── calibration/          #   Core calibration math & models
│   ├── setup_variable/       #   Site parameters & position reader
│   ├── image/                #   Image loading utilities
│   └── params.csv            #   Calibration parameters for all sites
│
├── DeepMatching/             # Stereo feature matching + 3D triangulation
│   ├── stereo_matchers/      #   Unified API for LoFTR, RoMa, DKM
│   ├── stereo_3d/            #   Triangulation, coordinates, uncertainty
│   ├── run_matching.py       #   CLI entry point
│   └── examples/             #   Usage notebooks
│
└── .gitignore
```

## Modules

### `cheick_code/` — Camera Calibration

Original calibration code for the Fripon fisheye camera model. Provides pixel ↔ (azimuth, zenith) ↔ world coordinate transforms needed by all downstream processing.

**Key files:**
- `calibration/baseCalibration.py` — Core math (coordinate conversions, rotation matrices, parameter I/O)
- `calibration/calibrationFripon.py` — Fripon model: `model()` (pixel→world), `invModel()` (world→pixel)
- `calibration/useCalibration.py` — Dispatcher that selects Fripon or M1 model based on parameter count
- `setup_variable/position.py` — Camera lat/lon reader from `params.csv`

### `DeepMatching/` — Stereo Matching & 3D Triangulation

Unified interface for deep-learning stereo feature matching (LoFTR, RoMa, DKM) with 3D atmospheric triangulation.

**`stereo_matchers/`** — Feature matching:
- `base.py` — Abstract `BaseMatcher` + `MatchResult` dataclass
- `loftr.py`, `roma.py`, `dkm.py` — Model implementations
- `benchmark.py` — Performance comparison across models
- `viz.py` — Match visualization

**`stereo_3d/`** — 3D reconstruction:
- `coordinates.py` — WGS84, ECEF, ENU coordinate transforms
- `fisheye_model.py` — Generic fisheye camera model (equidistant/equisolid/stereographic)
- `triangulation.py` — `AtmosphericTriangulator` for stereo altitude estimation
- `uncertainty.py` — Geometric error propagation
- `quality.py` — Match quality assessment and filtering