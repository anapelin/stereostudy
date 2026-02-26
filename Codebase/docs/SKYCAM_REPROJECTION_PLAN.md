# Skycam Reprojection Implementation Plan

## Executive Summary

This document outlines a plan to use the `skycam` package for reprojecting images from both SIRTA (IPSL) and Orsay (ECTL) cameras to a common plane projection. The implementation will leverage existing azimuth/zenith calibration maps and camera configuration data.

---

## 1. Current State Analysis

### 1.1 Skycam Package Status

The `skycam` package (v0.0.1.dev0) is currently a minimal stub with no implemented functionality:
- Only contains version information
- No modules or classes exposed
- Appears to be a placeholder/development package

**Finding**: The skycam package is not yet usable for reprojection. We need to either:
1. Implement our own reprojection system based on existing calibration data
2. Wait for skycam development (not recommended)
3. Use existing calibration tools from `cheick_code` (recommended)

### 1.2 Available Resources

#### Camera Calibration Data

**SIRTA (IPSL) Camera:**
- Site code: `SIRTA_W`
- Location: 48.713°N, 2.208°E, 156m elevation
- Image size: 768 × 1024 pixels
- Calibration model: Fripon (6 parameters)
- Azimuth/Zenith map: Available in `Codebase/data/azimuth_zenith_map_full_corrected.npz`

**Orsay (ECTL) Camera:**
- Site code: `Orsay`
- Location: 48.706433°N, 2.179331°E, 90m elevation
- Image size: 1280 × 960 pixels
- Calibration model: Fripon (6 parameters)
- Azimuth/Zenith map: **NOT YET GENERATED**

#### Calibration Parameters (from `cheick_code/params.csv`)

```csv
SIRTA_W: b=[0.00445, 2.56e-09, 4.14e-14, -4.55e-19, 5.46e-25], 
         x0=384.72, y0=518.53, w=[-0.0428, 0.0061, 0.0035], 
         K1=0.000624, phi=0.2791

Orsay:   b=[0.339, -0.0158, 0.0043, -0.0033, 0.00044], 
         x0=610.33, y0=483.07, w=[2.476, 2.898, 1.893], 
         K1=0.0018, phi=5.030
```

---

## 2. Recommended Approach: Custom Reprojection System

Since `skycam` is not functional, we'll build our own reprojection system using existing tools.

### 2.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Reprojection Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Generate Azimuth/Zenith Maps                            │
│     ├─ SIRTA: ✓ Already exists                              │
│     └─ Orsay: ⚠ Need to generate                            │
│                                                               │
│  2. Define Common Projection Plane                           │
│     ├─ Altitude: 10 km (typical contrail altitude)          │
│     ├─ Coordinate system: East-North grid                    │
│     └─ Resolution: 50m per pixel                             │
│                                                               │
│  3. For Each Camera:                                         │
│     ├─ Map camera pixels → (azimuth, zenith)                │
│     ├─ Compute ray direction from camera                     │
│     ├─ Intersect ray with projection plane                   │
│     └─ Determine (East, North) coordinates on plane          │
│                                                               │
│  4. Resampling                                               │
│     ├─ Create common plane grid                              │
│     ├─ For each plane pixel, find source camera pixel        │
│     └─ Interpolate pixel values                              │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Implementation Components

#### Component 1: Azimuth/Zenith Map Generator

**Purpose**: Generate pixel-wise (azimuth, zenith) mapping for both cameras

**Tool**: `azimuth_zenith_calibration` package (already available)

**Command for SIRTA** (already done):
```bash
python -m azimuth_zenith_calibration.cli \
    --site SIRTA_W \
    --shape 768,1024,3 \
    --step 1 \
    -o Codebase/data/azimuth_zenith_map_SIRTA.npz
```

**Command for Orsay** (TO DO):
```bash
python -m azimuth_zenith_calibration.cli \
    --site Orsay \
    --shape 1280,960,3 \
    --step 1 \
    -o Codebase/data/azimuth_zenith_map_Orsay.npz
```

**Output format** (`.npz` file):
- `azimuth`: (H, W) array of azimuth angles in degrees
- `zenith`: (H, W) array of zenith angles in degrees
- `x`, `y`: Pixel coordinate grids
- `site`, `image_shape`, `step`: Metadata

#### Component 2: Camera Configuration Files

**Purpose**: Store camera-specific parameters for reprojection

**Format**: JSON configuration files

**Location**: `Codebase/config/cameras/`

**SIRTA Configuration** (`sirta_camera.json`):
```json
{
    "name": "SIRTA (IPSL)",
    "site_code": "SIRTA_W",
    "location": {
        "latitude": 48.713,
        "longitude": 2.208,
        "elevation_m": 156
    },
    "image": {
        "width": 1024,
        "height": 768,
        "format": "RGB"
    },
    "calibration": {
        "model": "Fripon",
        "params_file": "../../cheick_code/params.csv",
        "azimuth_zenith_map": "../data/azimuth_zenith_map_full_corrected.npz"
    },
    "projection": {
        "type": "fisheye",
        "fov_degrees": 180
    }
}
```

**Orsay Configuration** (`orsay_camera.json`):
```json
{
    "name": "Orsay (ECTL)",
    "site_code": "Orsay",
    "location": {
        "latitude": 48.706433,
        "longitude": 2.179331,
        "elevation_m": 90
    },
    "image": {
        "width": 960,
        "height": 1280,
        "format": "RGB"
    },
    "calibration": {
        "model": "Fripon",
        "params_file": "../../cheick_code/params.csv",
        "azimuth_zenith_map": "../data/azimuth_zenith_map_Orsay.npz"
    },
    "projection": {
        "type": "fisheye",
        "fov_degrees": 180
    }
}
```

#### Component 3: Reprojection Class

**Purpose**: Core reprojection logic

**File**: `Codebase/src/projection/plane_reprojector.py`

**Key Methods**:

```python
class PlaneReprojector:
    """
    Reprojects fisheye camera images to a common horizontal plane.
    """
    
    def __init__(self, 
                 camera_config_path: str,
                 plane_altitude_m: float = 10000,
                 plane_resolution_m: float = 50):
        """
        Initialize reprojector with camera configuration.
        
        Parameters:
        -----------
        camera_config_path : str
            Path to camera JSON config file
        plane_altitude_m : float
            Altitude of projection plane in meters (default: 10km)
        plane_resolution_m : float
            Spatial resolution of plane grid in meters (default: 50m)
        """
        pass
    
    def load_azimuth_zenith_map(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load pre-computed azimuth and zenith arrays.
        
        Returns:
        --------
        azimuth : np.ndarray
            (H, W) array of azimuth angles in degrees
        zenith : np.ndarray
            (H, W) array of zenith angles in degrees
        """
        pass
    
    def compute_ray_plane_intersection(self,
                                      azimuth: np.ndarray,
                                      zenith: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute intersection of camera rays with horizontal plane.
        
        Parameters:
        -----------
        azimuth : np.ndarray
            Azimuth angles in degrees (0° = North)
        zenith : np.ndarray
            Zenith angles in degrees (0° = overhead, 90° = horizon)
            
        Returns:
        --------
        east : np.ndarray
            East coordinates on plane (meters from camera)
        north : np.ndarray
            North coordinates on plane (meters from camera)
        """
        # Height difference from camera to plane
        delta_h = plane_altitude_m - camera_elevation_m
        
        # For each pixel:
        # 1. Convert (az, zen) to unit direction vector
        # 2. Scale by distance to reach plane: d = delta_h / cos(zen)
        # 3. Project to horizontal components
        
        # Vectorized computation:
        zen_rad = np.radians(zenith)
        az_rad = np.radians(azimuth)
        
        # Distance along ray to plane
        d = delta_h / np.cos(zen_rad)
        
        # Horizontal distance from camera
        r = d * np.sin(zen_rad)
        
        # East-North components
        east = r * np.sin(az_rad)
        north = r * np.cos(az_rad)
        
        return east, north
    
    def create_plane_grid(self, 
                         extent_km: float = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create uniform grid on projection plane.
        
        Parameters:
        -----------
        extent_km : float
            Size of grid in each direction from camera (km)
            
        Returns:
        --------
        grid_east : np.ndarray
            2D array of east coordinates
        grid_north : np.ndarray
            2D array of north coordinates
        """
        pass
    
    def reproject_image(self, 
                       image: np.ndarray,
                       plane_grid: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Reproject camera image to plane grid.
        
        Parameters:
        -----------
        image : np.ndarray
            Input camera image (H, W, 3)
        plane_grid : Tuple[np.ndarray, np.ndarray]
            (grid_east, grid_north) defining output plane
            
        Returns:
        --------
        projected_image : np.ndarray
            Reprojected image on plane grid
        """
        # For each output pixel:
        # 1. Get (east, north) coordinates
        # 2. Find corresponding (azimuth, zenith) from inverse lookup
        # 3. Map (azimuth, zenith) to camera (x, y)
        # 4. Interpolate pixel value from camera image
        pass
```

#### Component 4: Stereo Matching & Alignment

**Purpose**: Align reprojected images from both cameras

**File**: `Codebase/src/projection/stereo_alignment.py`

**Approach**:
1. Reproject both images to same plane at same altitude
2. Images are now in common coordinate system (East-North)
3. Can directly compare/overlay/difference images
4. Use cross-correlation or feature matching for fine alignment

---

## 3. Implementation Steps

### Phase 1: Setup & Configuration (Immediate)

- [x] Install conda environment with dependencies
- [x] Analyze existing calibration system
- [ ] **Create Orsay azimuth/zenith map**
- [ ] Create camera configuration JSON files
- [ ] Create directory structure: `Codebase/config/cameras/`

### Phase 2: Core Reprojection (Week 1)

- [ ] Implement `PlaneReprojector` class
- [ ] Test with SIRTA images
- [ ] Validate projection geometry
- [ ] Add visualization tools

### Phase 3: Stereo Processing (Week 2)

- [ ] Implement `StereoAlignment` class
- [ ] Process matched image pairs
- [ ] Generate comparison visualizations
- [ ] Compute alignment metrics

### Phase 4: Integration & Testing (Week 3)

- [ ] Integrate with existing notebooks
- [ ] Batch process all matched pairs
- [ ] Performance optimization
- [ ] Documentation

---

## 4. Mathematical Foundation

### 4.1 Coordinate Systems

**World Coordinate System**:
- Origin: Camera location
- X-axis: Points North
- Y-axis: Points East
- Z-axis: Points Up (zenith)

**Spherical Coordinates**:
- Azimuth (θ): Angle from North, measured clockwise (0-360°)
- Zenith (φ): Angle from vertical (0° = overhead, 90° = horizon)

**Conversion to Cartesian**:
```
X = sin(φ) * cos(θ)  [North component]
Y = sin(φ) * sin(θ)  [East component]
Z = cos(φ)           [Up component]
```

### 4.2 Ray-Plane Intersection

For a horizontal plane at altitude `h_plane`:

1. Camera at (0, 0, h_camera)
2. Ray direction: unit vector from (θ, φ)
3. Plane equation: z = h_plane

**Intersection point**:
```
t = (h_plane - h_camera) / Z
P_x = X * t  [North distance]
P_y = Y * t  [East distance]
```

### 4.3 Pixel Value Interpolation

Use bilinear interpolation for resampling:

```python
def interpolate_pixel(image, x, y):
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = x0 + 1, y0 + 1
    
    wx = x - x0
    wy = y - y0
    
    value = (1-wx)*(1-wy)*image[y0, x0] + \
            wx*(1-wy)*image[y0, x1] + \
            (1-wx)*wy*image[y1, x0] + \
            wx*wy*image[y1, x1]
    
    return value
```

---

## 5. Expected Outputs

### 5.1 Configuration Files

```
Codebase/config/
└── cameras/
    ├── sirta_camera.json
    └── orsay_camera.json
```

### 5.2 Azimuth/Zenith Maps

```
Codebase/data/
├── azimuth_zenith_map_SIRTA.npz     (already exists)
└── azimuth_zenith_map_Orsay.npz     (to be generated)
```

### 5.3 Reprojection Code

```
Codebase/src/projection/
├── __init__.py
├── plane_reprojector.py
├── stereo_alignment.py
└── utils.py
```

### 5.4 Notebooks

```
Codebase/notebooks/
├── test_plane_reprojection.ipynb    (testing/validation)
└── stereo_comparison_reprojected.ipynb  (final comparison)
```

---

## 6. Validation Strategy

### 6.1 Geometric Validation

1. **Known landmarks**: Verify that known ground features (buildings, etc.) align correctly
2. **Horizon check**: Verify that zenith ≈ 90° pixels project to infinity
3. **Symmetry**: Check that symmetric features in camera remain symmetric on plane

### 6.2 Photometric Validation

1. **Brightness preservation**: Check that pixel intensities are preserved
2. **Edge sharpness**: Verify that sharp edges remain sharp after reprojection
3. **No artifacts**: Check for interpolation artifacts or discontinuities

### 6.3 Stereo Validation

1. **Feature correspondence**: Same real-world features should align between cameras
2. **Temporal consistency**: Same aircraft track should appear in both reprojected images
3. **Geometric consistency**: Distances/sizes should be consistent between cameras

---

## 7. Next Actions

### Immediate (Today):

1. **Generate Orsay azimuth/zenith map**:
   ```bash
   cd /data/common/STEREOSTUDYIPSL
   python -m azimuth_zenith_calibration.cli \
       --site Orsay \
       --shape 1280,960,3 \
       --step 1 \
       -o Codebase/data/azimuth_zenith_map_Orsay.npz
   ```

2. **Create camera configuration files** as specified above

3. **Set up directory structure**:
   ```bash
   mkdir -p Codebase/config/cameras
   mkdir -p Codebase/src/projection
   ```

### This Week:

1. Implement `PlaneReprojector` class
2. Test with single SIRTA image
3. Validate projection accuracy
4. Create visualization tools

---

## 8. References

- **Calibration System**: `cheick_code/calibration/`
- **Existing Maps**: `Codebase/data/azimuth_zenith_map_full_corrected.npz`
- **Flight Projection**: `Flights/project_flights_to_cameras.py`
- **Calibration Analysis**: `Codebase/docs/CALIBRATION_ANALYSIS_AND_PLAN.md`

---

## Conclusion

While the `skycam` package is not currently functional, we have all the necessary tools and data to implement a robust reprojection system:

1. **Calibration data**: Available for both cameras
2. **Azimuth/zenith maps**: One exists, one can be generated
3. **Mathematical framework**: Well-documented in existing code
4. **Implementation path**: Clear steps forward

The recommended approach leverages existing, proven calibration tools rather than waiting for external package development. This gives us full control over the reprojection process and ensures compatibility with the existing workflow.
