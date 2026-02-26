"""CLI for saving azimuth/zenith lookup tables from calibration files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .converter import AzimuthZenithMapper


def _parse_shape(value: str) -> tuple[int, int]:
    normalized = [int(part) for part in value.replace("x", ",").split(",") if part]
    if len(normalized) < 2:
        raise argparse.ArgumentTypeError("shape must have at least two numbers: height,width[,channels]")
    return tuple(normalized[:2])


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump pixel → (azimuth, zenith) grids.")
    parser.add_argument("--site", default="SIRTA_W", help="Calibration site name (matching params.csv)")
    parser.add_argument(
        "--shape",
        default="768,1024,3",
        type=_parse_shape,
        help="Image shape as height,width[,channels]",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Sampling stride along both axes (1=full resolution)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("azimuth_zenith_map.npz"),
        help="Destination .npz file for the lookup table",
    )

    args = parser.parse_args()

    mapper = AzimuthZenithMapper(site=args.site, image_shape=args.shape)
    mapping = mapper.generate_mapping(step=args.step)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **mapping)
    print(f"Saved {mapping['azimuth'].size} samples for {args.site} at {args.output}")
