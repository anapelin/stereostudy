# Azimuth / Zenith Mapping: Run & Results

This document records the commands I ran, the environment used, and the verification steps/results when generating an azimuth/zenith lookup table for the repository.

Location of generated output
- File created: `./azimuth_zenith_map.npz` (saved to the repository root: `/data/common/STEREOSTUDYIPSL/azimuth_zenith_map.npz`).

Environment
- Conda environment: `torch_gpu_311`
- Python executable used (example on this machine): `/data/dataiku/.conda/envs/torch_gpu_311/bin/python`

Commands I ran
- Using `conda run`:
```bash
conda run -n torch_gpu_311 python -m azimuth_zenith_calibration.cli \
  --site SIRTA_W --shape 768,1024,3 --step 16 -o ./azimuth_zenith_map.npz
```

- Using the environment python directly (equivalent):
```bash
/data/dataiku/.conda/envs/torch_gpu_311/bin/python -m azimuth_zenith_calibration.cli \
  --site SIRTA_W --shape 768,1024,3 --step 16 -o ./azimuth_zenith_map.npz
```

What the code does
- `azimuth_zenith_calibration.cli` constructs an `AzimuthZenithMapper` using `site` and `--shape`, then calls `generate_mapping(step=...)` and saves the mapping as a `.npz` archive using `numpy.savez`.
- Calibration parameters are read from `cheick_code/params.csv` via `cheick_code.calibration.baseCalibration.readCalParams`.

Verification performed
- Created mapping with `step=16` and saved to `./azimuth_zenith_map.npz`.
- File size on disk: ~100,354 bytes (on the machine used).
- Contents inspected with NumPy; keys and shapes found:
  - `azimuth`: ndarray, shape (48, 64), dtype float64
  - `zenith`: ndarray, shape (48, 64), dtype float64
  - `x`: ndarray, shape (48, 64), dtype int64
  - `y`: ndarray, shape (48, 64), dtype int64
  - metadata keys: `site`, `image_shape`, `step`, `spherical_method`
- Example ranges observed:
  - `azimuth` min/max: approximately -3.14 .. 3.14
  - `zenith` min/max: approximately 0.008 .. 3.136

Quick inspect commands
```bash
# list file
ls -l azimuth_zenith_map.npz

# show keys, shapes, and min/max values
/data/dataiku/.conda/envs/torch_gpu_311/bin/python - <<'PY'
import numpy as np
d = np.load('azimuth_zenith_map.npz')
print('keys', d.files)
print('azimuth shape', d['azimuth'].shape)
print('az min/max', float(d['azimuth'].min()), float(d['azimuth'].max()))
print('zen min/max', float(d['zenith'].min()), float(d['zenith'].max()))
PY
```

Notes & suggestions
- If you want the output in a different directory, change the `-o`/`--output` path.
- To increase sampling resolution, use a smaller `--step` (e.g. `--step 1` for full resolution). Beware of memory/time costs for small steps.
- If the CLI appears to silently succeed but no file is created, run the mapping creation snippet directly with the environment python to avoid `conda run` inconsistencies.

If you want, I can:
- Save the `.npz` to another directory or a timestamped filename.
- Commit `azimuth_zenith_map.npz` (or metadata) to git (if desired).
- Produce a higher-resolution mapping and provide a download link.
