# Azimuth/Zenith Calibration

`azimuth_zenith_calibration` exposes helpers that read the Fripon-style calibration parameters in `cheick_code/calibration/params.csv` and build a pixel-wise lookup from image coordinates to `(azimuth, zenith)` angles.

## Features

- **`AzimuthZenithMapper`**: loads the requested site (`SIRTA_W` or `Orsay`) and reuses `calibrationFripon.model` + `Cartesian2Spherical` to compute the direction vector for every sampled pixel.
- **CLI**: `python -m azimuth_zenith_calibration.cli` produces `.npz` files containing the `azimuth`, `zenith`, `x`, and `y` grids plus metadata such as `site`, `step`, and `image_shape`.

## Example

```sh
python -m azimuth_zenith_calibration.cli --site SIRTA_W --shape 901,901,3 --step 2 -o /tmp/sirta_w_lookup.npz
```

- The resulting `.npz` contains arrays named `azimuth`/`zenith` plus the sampled `x`, `y` coordinates; `step` controls the resolution (1 = full image, 2 skips every other pixel).

## Notes

- Only Fripon-style parameter rows (6 values) are supported for now; the CLI will raise if the requested site uses another model.
- Because numpy is required by both the mapper and the calibration utilities, make sure it is installed in the interpreter that runs the CLI.
