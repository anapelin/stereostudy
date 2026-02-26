"""High-level helpers to turn pixel coordinates (x, y) into (azimuth, zenith).

Coordinate Convention (matching cheick_code):
- x = VERTICAL dimension (row index, height)
- y = HORIZONTAL dimension (column index, width)
- This is OPPOSITE to standard computer vision where x=horizontal, y=vertical
- Image shape is (height, width) = (rows, cols) = (x_range, y_range)
"""

import os
import sys
from typing import Any, Dict, Optional, Tuple

try:
    import numpy as np
except ImportError as exc:
    raise ImportError("numpy is required by azimuth_zenith_calibration") from exc

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_CURRENT_DIR)
_CHEICK_CODE = os.path.join(_REPO_ROOT, "cheick_code")
if _CHEICK_CODE not in sys.path:
    sys.path.insert(0, _CHEICK_CODE)

from calibration.calibrationFripon import model as _fripon_model  # type: ignore[import]
from calibration.baseCalibration import Cartesian2Spherical, readCalParams  # type: ignore[import]


DEFAULT_IMAGE_SHAPE = (768, 1024)
SUPPORTED_FRIPON_SITES = {"SIRTA_W", "Orsay"}


class AzimuthZenithMapper:
    """Build a pixel → (azimuth, zenith) mapping for Fripon-inspired calibration."""

    def __init__(self, site: str, image_shape: Optional[Tuple[int, int]] = None) -> None:
        self.site = site
        self.image_shape = self._normalize_image_shape(image_shape or DEFAULT_IMAGE_SHAPE)
        params = readCalParams(site=self.site)
        if not params:
            raise ValueError(f"Unable to load calibration for {self.site}")
        if len(params) != 6:
            raise NotImplementedError(
                "Only Fripon-style parameter sets (6 values) are supported right now."
            )
        if self.site not in SUPPORTED_FRIPON_SITES:
            raise ValueError(
                f"{self.site} is not part of the supported Fripon subset: {SUPPORTED_FRIPON_SITES}"
            )

        # Fripon params: radial polynomial, center, Euler angles, phase corrections
        # params.csv stores [b, x0, y0, theta, K1, phi] where (in cheick_code convention):
        #   x0 = center in vertical/row dimension (should be ~768/2 = 384)
        #   y0 = center in horizontal/column dimension (should be ~1024/2 = 512)
        self._b, self._x0, self._y0, self._theta, self._K1, self._phi = params
        # No swap needed - params.csv already uses cheick_code convention
        self._spherical_method = "Simon"

    @staticmethod
    def _normalize_image_shape(image_shape: Optional[Tuple[int, int]]) -> Tuple[int, int]:
        if image_shape is None:
            return DEFAULT_IMAGE_SHAPE
        if len(image_shape) != 2:
            raise ValueError("image_shape must be (height, width)")
        return image_shape

    def _pixel_grid(self, step: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate pixel coordinate grids.
        
        Returns (x_grid, y_grid) where:
        - x_grid varies along the horizontal (column) direction
        - y_grid varies along the vertical (row) direction
        
        Note: meshgrid returns (X, Y) where X varies along columns, Y along rows
        """
        height, width = self.image_shape
        xs = np.arange(0, width, step)  # horizontal coordinates
        ys = np.arange(0, height, step)  # vertical coordinates
        return np.meshgrid(xs, ys)

    def _pixel_to_world(self, x: float, y: float) -> np.ndarray:
        """Convert pixel coordinates to 3D world vector.
        
        Parameters (using standard CV convention in this interface):
        - x: horizontal pixel coordinate (column index, 0 to width-1)
        - y: vertical pixel coordinate (row index, 0 to height-1)
        
        Note: _fripon_model uses cheick_code convention where first arg is vertical,
        second arg is horizontal, so we pass (y, x) instead of (x, y)
        """
        # Pass (vertical, horizontal) = (y, x) to match cheick_code's (x, y) expectation
        world = _fripon_model(
            y,  # vertical coordinate → cheick_code's x
            x,  # horizontal coordinate → cheick_code's y
            self._b,
            self._x0,
            self._y0,
            self._theta,
            self._K1,
            self._phi,
            site=self.site,
        )
        return np.asarray(world, dtype=float)

    def _compute_world_vectors(
        self, x_flat: np.ndarray, y_flat: np.ndarray
    ) -> np.ndarray:
        vectors = np.empty((x_flat.size, 3), dtype=float)
        for idx in range(x_flat.size):
            vectors[idx] = self._pixel_to_world(float(x_flat[idx]), float(y_flat[idx]))
        return vectors

    def _world_to_angles(self, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        theta, phi = Cartesian2Spherical(
            vectors[:, 0], vectors[:, 1], vectors[:, 2], method=self._spherical_method
        )
        # Cartesian2Spherical returns a two-element list
        return np.asarray(theta), np.asarray(phi)

    def generate_mapping(self, step: int = 1) -> Dict[str, Any]:
        """
        Compute a downsampled grid that maps pixel coordinates to azimuth/zenith.

        Parameters
        ----------
        step : int, optional
            Sampling stride along both axes. Defaults to 1 (full resolution).
        """
        if step < 1:
            raise ValueError("step must be >= 1")

        x_grid, y_grid = self._pixel_grid(step)
        x_flat = x_grid.ravel()
        y_flat = y_grid.ravel()

        world = self._compute_world_vectors(x_flat, y_flat)
        zenith, azimuth = self._world_to_angles(world)

        out_shape = x_grid.shape
        return {
            "azimuth": azimuth.reshape(out_shape),
            "zenith": zenith.reshape(out_shape),
            "x": x_grid,
            "y": y_grid,
            "site": self.site,
            "image_shape": self.image_shape,
            "step": step,
            "spherical_method": self._spherical_method,
        }
