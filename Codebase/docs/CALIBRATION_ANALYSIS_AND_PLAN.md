# Calibration Analysis and Implementation Plan

## Executive Summary

This document analyzes the calibration system in `cheick_code/calibration/` and provides a plan to obtain pixel->(azimuth, zenith) mapping and plane projection at desired altitude **without relying on the deco and cloud libraries**.

---

## 1. Calibration Architecture Analysis

### 1.1 Models Available

The codebase implements **two camera models**:

1. **Fripon Model** (`FriponModel.py`, `calibrationFripon.py`)
   - Simplified model with 12 parameters
   - Used for SIRTA and Orsay sites
   - Parameters: `[b[5], x0, y0, w[3], K1, phi]`

2. **Original Model** (`OriginalModel.py`, `calibrationM1.py`)
   - Complex model with 23 parameters
   - Parameters: `[alphaX, alphaY, s, xo[2], w[3], t[3], k[8], p[4]]`

### 1.2 Calibration Parameters (Fripon Model for SIRTA)

From `params.csv`, the SIRTA calibration parameters are:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `a1` (b[0]) | 224.53 | Polynomial coeff (1st order) |
| `a2` (b[1]) | -6.52 | Polynomial coeff (3rd order) |
| `a3` (b[2]) | -4.75 | Polynomial coeff (5th order) |
| `a4` (b[3]) | 4.16 | Polynomial coeff (7th order) |
| `a5` (b[4]) | -0.96 | Polynomial coeff (9th order) |
| `xo` | 384.72 | Image center X coordinate |
| `yo` | 518.53 | Image center Y coordinate |
| `wx` (w[0]) | -0.0428 | Rotation angle X (radians) |
| `wy` (w[1]) | 0.0061 | Rotation angle Y (radians) |
| `wz` (w[2]) | 0.0035 | Rotation angle Z (radians) |
| `K1` | 0.000624 | Phase distortion weight |
| `phi` | 0.279 | Phase shift (radians) |
| `lat` | 48.713 | Site latitude |
| `lon` | 2.208 | Site longitude |

---

## 2. Forward Model: World → Image (Current Implementation)

The **forward projection** (3D world coordinates → 2D image pixels) follows this pipeline:

### Pipeline Steps:

```
3D Position (X, Y, Z) 
    ↓
[1] Rotation (using rotation matrix R from w[3])
    ↓
(X', Y', Z') rotated coordinates
    ↓
[2] Cartesian to Spherical conversion
    ↓
(zenith_angle, azimuth_angle)
    ↓
[3] Radial distortion correction (9th degree polynomial)
    ↓
r_distorted
    ↓
[4] Phase distortion correction
    ↓
R (final radius in pixels)
    ↓
[5] Polar to Cartesian
    ↓
(x_pixel, y_pixel)
```

### Mathematical Details:

#### Step 1: Rotation Matrix (Cardan angles)
```python
# Rotation: Rz(w[0]) * Ry(w[1]) * Rx(w[2])
R[0,0] = cos(w0)*cos(w1)
R[1,0] = sin(w0)*cos(w1)
R[2,0] = -sin(w1)

R[0,1] = -sin(w0)*cos(w2) + cos(w0)*sin(w1)*sin(w2)
R[1,1] = cos(w0)*cos(w2) + sin(w0)*sin(w1)*sin(w2)
R[2,1] = cos(w1)*sin(w2)

R[0,2] = sin(w0)*sin(w2) + cos(w0)*sin(w1)*cos(w2)
R[1,2] = -cos(w0)*sin(w2) + sin(w0)*sin(w1)*cos(w2)
R[2,2] = cos(w1)*cos(w2)
```

For SIRTA, method='Nico' applies: `P' = R^(-1) * P`

#### Step 2: Cartesian to Spherical
```python
zenith = arctan2(sqrt(X'^2 + Y'^2), Z')
azimuth = arctan2(Y', X')
```

#### Step 3: Radial Distortion (9th order polynomial)
```python
r_distorted = b[0]*zenith + b[1]*zenith^3 + b[2]*zenith^5 + 
              b[3]*zenith^7 + b[4]*zenith^9
```

#### Step 4: Phase Distortion
```python
R = r_distorted / (1 + K1 * sin(azimuth + phi))
```

#### Step 5: Polar to Cartesian
```python
x_pixel = R * cos(azimuth) + x0
y_pixel = R * sin(azimuth) + y0
```

---

## 3. Required: Inverse Model (Image → World)

To obtain **pixel → (azimuth, zenith)** mapping, we need to **reverse** the forward model.

### 3.1 Simplified Inverse Pipeline

```
(x_pixel, y_pixel)
    ↓
[5'] Cartesian to Polar
    ↓
(R, azimuth)
    ↓
[4'] Inverse phase distortion
    ↓
r_distorted
    ↓
[3'] Inverse radial distortion (solve 9th degree polynomial)
    ↓
zenith_angle
    ↓
[2'] Spherical to Cartesian (at unit radius)
    ↓
(X', Y', Z') direction vector
    ↓
[1'] Inverse rotation
    ↓
(X, Y, Z) world direction vector
```

### 3.2 Mathematical Implementation

#### Step 5': Cartesian to Polar
```python
# Given pixel coordinates
dx = x_pixel - x0
dy = y_pixel - y0
R = sqrt(dx^2 + dy^2)
azimuth = arctan2(dy, dx)
```

#### Step 4': Inverse Phase Distortion
```python
# Solve for r_distorted
r_distorted = R * (1 + K1 * sin(azimuth + phi))
```

#### Step 3': Inverse Radial Distortion
This is the **most challenging step** - solving:
```
r_distorted = b[0]*z + b[1]*z^3 + b[2]*z^5 + b[3]*z^7 + b[4]*z^9
```

**Solution approaches:**
1. **Newton-Raphson iteration** (recommended)
2. **Lookup table** with interpolation
3. **Numerical root finding** (scipy.optimize.fsolve)

Newton-Raphson pseudocode:
```python
def inverse_poly9(r_distorted, b, initial_guess=r_distorted/b[0]):
    z = initial_guess
    for iteration in range(max_iterations):
        f = b[0]*z + b[1]*z**3 + b[2]*z**5 + b[3]*z**7 + b[4]*z**9 - r_distorted
        df = b[0] + 3*b[1]*z**2 + 5*b[2]*z**4 + 7*b[3]*z**6 + 9*b[4]*z**8
        z_new = z - f/df
        if abs(z_new - z) < tolerance:
            return z_new
        z = z_new
    return z
```

#### Step 2': Spherical to Cartesian (unit sphere)
```python
X' = cos(azimuth) * sin(zenith)
Y' = sin(azimuth) * sin(zenith)
Z' = cos(zenith)
```

#### Step 1': Inverse Rotation
```python
# Apply rotation matrix R (not R^(-1))
[X, Y, Z] = R * [X', Y', Z']
```

---

## 4. Implementation Plan: Pixel → (Azimuth, Zenith) Mapping

### Phase 1: Create Standalone Calibration Module (No Dependencies)

**File:** `standalone_calibration.py`

```python
import numpy as np

class CameraCalibration:
    """
    Standalone camera calibration for all-sky fisheye cameras.
    No dependency on deco or cloud libraries.
    """
    
    def __init__(self, params_dict):
        """
        Initialize with calibration parameters.
        
        Parameters:
        -----------
        params_dict : dict
            Dictionary containing:
            - 'b': [b0, b1, b2, b3, b4] (radial polynomial)
            - 'x0', 'y0': center coordinates
            - 'w': [wx, wy, wz] (rotation angles)
            - 'K1': phase distortion weight
            - 'phi': phase shift
            - 'lat', 'lon': site coordinates
        """
        self.b = np.array(params_dict['b'])
        self.x0 = params_dict['x0']
        self.y0 = params_dict['y0']
        self.w = np.array(params_dict['w'])
        self.K1 = params_dict['K1']
        self.phi = params_dict['phi']
        self.lat = params_dict['lat']
        self.lon = params_dict['lon']
        self.R = self._build_rotation_matrix()
        
    def _build_rotation_matrix(self):
        """Build rotation matrix from Cardan angles."""
        # Implementation of rotation matrix
        pass
    
    def pixel_to_angles(self, x_pixel, y_pixel):
        """
        Convert pixel coordinates to (azimuth, zenith) angles.
        
        Returns:
        --------
        azimuth : float (radians, 0 to 2π)
        zenith : float (radians, 0 to π/2)
        """
        # Step 5': Cartesian to Polar
        dx = x_pixel - self.x0
        dy = y_pixel - self.y0
        R = np.sqrt(dx**2 + dy**2)
        azimuth = np.arctan2(dy, dx)
        
        # Step 4': Inverse phase distortion
        r_distorted = R * (1 + self.K1 * np.sin(azimuth + self.phi))
        
        # Step 3': Inverse polynomial (Newton-Raphson)
        zenith = self._inverse_poly9(r_distorted)
        
        return azimuth, zenith
    
    def _inverse_poly9(self, r_distorted, max_iter=20, tol=1e-8):
        """Solve 9th degree polynomial using Newton-Raphson."""
        # Initial guess
        z = r_distorted / self.b[0] if self.b[0] != 0 else r_distorted
        
        for _ in range(max_iter):
            # Polynomial: f(z) = b0*z + b1*z^3 + ... - r_distorted
            f = (self.b[0]*z + self.b[1]*z**3 + self.b[2]*z**5 + 
                 self.b[3]*z**7 + self.b[4]*z**9 - r_distorted)
            
            # Derivative
            df = (self.b[0] + 3*self.b[1]*z**2 + 5*self.b[2]*z**4 + 
                  7*self.b[3]*z**6 + 9*self.b[4]*z**8)
            
            if abs(df) < 1e-12:
                break
                
            z_new = z - f / df
            
            if abs(z_new - z) < tol:
                return z_new
            z = z_new
        
        return z
    
    def pixel_to_direction(self, x_pixel, y_pixel):
        """
        Convert pixel to 3D direction vector (world frame).
        
        Returns:
        --------
        direction : np.array([X, Y, Z]) - unit vector
        """
        azimuth, zenith = self.pixel_to_angles(x_pixel, y_pixel)
        
        # Spherical to Cartesian (camera frame)
        X_cam = np.cos(azimuth) * np.sin(zenith)
        Y_cam = np.sin(azimuth) * np.sin(zenith)
        Z_cam = np.cos(zenith)
        
        # Rotate to world frame
        direction_cam = np.array([X_cam, Y_cam, Z_cam])
        direction_world = self.R @ direction_cam
        
        return direction_world
```

### Phase 2: Create Pixel Grid Mapping

**File:** `generate_angle_map.py`

```python
import numpy as np
from standalone_calibration import CameraCalibration

def create_angle_maps(calib, image_shape=(768, 1024)):
    """
    Pre-compute azimuth and zenith for every pixel.
    
    Returns:
    --------
    azimuth_map : np.array(image_shape) - azimuth for each pixel
    zenith_map : np.array(image_shape) - zenith for each pixel
    """
    height, width = image_shape
    azimuth_map = np.zeros((height, width))
    zenith_map = np.zeros((height, width))
    
    for y in range(height):
        for x in range(width):
            az, zen = calib.pixel_to_angles(x, y)
            azimuth_map[y, x] = az
            zenith_map[y, x] = zen
    
    return azimuth_map, zenith_map

def save_angle_maps(azimuth_map, zenith_map, output_path):
    """Save pre-computed maps to file."""
    np.savez_compressed(output_path, 
                       azimuth=azimuth_map, 
                       zenith=zenith_map)
```

### Phase 3: Plane Projection at Altitude

**File:** `plane_projection.py`

```python
import numpy as np

class PlaneProjector:
    """
    Project image pixels to a horizontal plane at specified altitude.
    """
    
    def __init__(self, calib, site_altitude_km=0.0):
        """
        Parameters:
        -----------
        calib : CameraCalibration instance
        site_altitude_km : float - altitude of camera above sea level
        """
        self.calib = calib
        self.site_altitude = site_altitude_km
        
    def pixel_to_plane(self, x_pixel, y_pixel, plane_altitude_km):
        """
        Project pixel to (X, Y) coordinates on horizontal plane.
        
        Parameters:
        -----------
        x_pixel, y_pixel : pixel coordinates
        plane_altitude_km : altitude of projection plane
        
        Returns:
        --------
        X, Y : coordinates in km (North, East) relative to camera
        valid : bool - True if ray intersects plane
        """
        # Get direction vector
        direction = self.calib.pixel_to_direction(x_pixel, y_pixel)
        
        # Ray equation: P = camera_pos + t * direction
        # Plane equation: Z = plane_altitude_km
        
        camera_z = self.site_altitude
        target_z = plane_altitude_km
        
        # Check if ray goes upward/downward correctly
        if direction[2] == 0:
            return None, None, False
        
        # Solve for t where Z = target_z
        # camera_z + t * direction[2] = target_z
        t = (target_z - camera_z) / direction[2]
        
        if t < 0:
            # Ray goes in wrong direction
            return None, None, False
        
        # Compute intersection point
        X = t * direction[0]  # North (km)
        Y = t * direction[1]  # East (km)
        
        return X, Y, True
    
    def project_image_to_plane(self, image, plane_altitude_km, 
                               grid_resolution_km=0.1,
                               grid_extent_km=50):
        """
        Project entire image to plane and create top-down view.
        
        Parameters:
        -----------
        image : np.array - input image
        plane_altitude_km : altitude of projection plane
        grid_resolution_km : resolution of output grid
        grid_extent_km : extent of grid (±extent in X and Y)
        
        Returns:
        --------
        projected_image : 2D grid with projected image values
        grid_X, grid_Y : coordinate arrays
        """
        # Create output grid
        n_points = int(2 * grid_extent_km / grid_resolution_km)
        grid_X = np.linspace(-grid_extent_km, grid_extent_km, n_points)
        grid_Y = np.linspace(-grid_extent_km, grid_extent_km, n_points)
        
        # Initialize output
        projected = np.zeros((n_points, n_points, image.shape[2] if len(image.shape) == 3 else 1))
        count = np.zeros((n_points, n_points))
        
        height, width = image.shape[:2]
        
        # For each pixel, project to plane
        for y_pixel in range(height):
            for x_pixel in range(width):
                X, Y, valid = self.pixel_to_plane(x_pixel, y_pixel, plane_altitude_km)
                
                if not valid:
                    continue
                
                # Find grid cell
                i = np.searchsorted(grid_X, X)
                j = np.searchsorted(grid_Y, Y)
                
                if 0 <= i < n_points and 0 <= j < n_points:
                    projected[j, i] += image[y_pixel, x_pixel]
                    count[j, i] += 1
        
        # Average pixels that fall in same grid cell
        mask = count > 0
        projected[mask] /= count[mask][..., np.newaxis]
        
        return projected, grid_X, grid_Y
```

---

## 5. Complete Implementation Roadmap

### Step 1: Extract and Test Calibration Parameters ✓
- [x] Read parameters from `params.csv`
- [x] Understand parameter meaning

### Step 2: Implement Standalone Functions
- [ ] Create `standalone_calibration.py`
  - [ ] Rotation matrix builder
  - [ ] Inverse polynomial solver (Newton-Raphson)
  - [ ] `pixel_to_angles()` function
  - [ ] `pixel_to_direction()` function

### Step 3: Validate Against Existing Code
- [ ] Test forward model: compare with `worldToImage()`
- [ ] Test inverse model: pixel → angles → pixel (round-trip)
- [ ] Use test data from `test_calibration2.py`

### Step 4: Create Angle Maps
- [ ] Implement `generate_angle_map.py`
- [ ] Generate maps for SIRTA camera
- [ ] Save to NPZ file for fast loading

### Step 5: Implement Plane Projection
- [ ] Create `plane_projection.py`
- [ ] Test with single pixel
- [ ] Test with full image

### Step 6: Create Example Scripts
- [ ] `example_pixel_to_angles.py`
- [ ] `example_plane_projection.py`
- [ ] Visualization utilities

### Step 7: Documentation
- [ ] API documentation
- [ ] Usage examples
- [ ] Comparison with original code

---

## 6. Key Advantages of Standalone Implementation

1. **No External Dependencies**: Pure NumPy implementation
2. **Fast Lookup**: Pre-computed angle maps for instant access
3. **Flexible**: Easy to modify for different cameras
4. **Portable**: Single file can be used anywhere
5. **Transparent**: All math is explicit and documented

---

## 7. Testing Strategy

### Test 1: Forward-Inverse Consistency
```python
# Test that pixel → angles → world → pixel gives original pixel
pixel_original = (512, 384)
azimuth, zenith = calib.pixel_to_angles(*pixel_original)
# ... use forward model ...
assert np.allclose(pixel_reconstructed, pixel_original, atol=1.0)
```

### Test 2: Bretigny Example
Use the Bretigny coordinates from `test_calibration2.py`:
- lat_brety = 48.6°
- lon_brety = 2.3°
- alt = 10 km

Verify pixel positions match the original code.

### Test 3: Zenith Angles
Verify that:
- Center pixel → zenith ≈ 0°
- Edge pixels → zenith ≈ 60° (for szamax=60)

---

## 8. Optimization Opportunities

1. **Vectorization**: Process all pixels at once using NumPy broadcasting
2. **Lookup Tables**: Pre-compute inverse polynomial for speed
3. **Parallel Processing**: Use multiprocessing for large images
4. **GPU Acceleration**: Port to CuPy for GPU processing

---

## 9. Next Steps

1. **Immediate**: Implement `standalone_calibration.py` with basic functions
2. **Short-term**: Validate against existing code
3. **Medium-term**: Create full plane projection pipeline
4. **Long-term**: Optimize for production use

---

## 10. File Structure (Proposed)

```
STEREOSTUDYIPSL/
├── standalone_calibration/
│   ├── __init__.py
│   ├── calibration.py          # Core CameraCalibration class
│   ├── inverse_models.py       # Inverse polynomial solvers
│   ├── plane_projection.py     # PlaneProjector class
│   ├── angle_maps.py           # Pre-computed angle maps
│   └── utils.py                # Helper functions
├── examples/
│   ├── example_pixel_to_angles.py
│   ├── example_plane_projection.py
│   └── test_against_original.py
├── data/
│   ├── sirta_angle_maps.npz    # Pre-computed for SIRTA
│   └── orsay_angle_maps.npz    # Pre-computed for Orsay
└── tests/
    ├── test_calibration.py
    ├── test_plane_projection.py
    └── test_bretigny.py
```

---

## Conclusion

The calibration system is well-documented in the code, and extracting a standalone implementation is straightforward. The key challenge is inverting the 9th-degree radial polynomial, which can be solved efficiently using Newton-Raphson iteration. Once the pixel→angle mapping is established, plane projection becomes a simple ray-plane intersection problem.

This approach completely eliminates dependencies on `deco` and `cloud` libraries while maintaining full compatibility with the existing calibration parameters.
