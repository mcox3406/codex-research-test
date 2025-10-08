"""Compute and cache reference persistence diagrams for training."""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Sequence

import numpy as np

from src.tda.io import save_diagram_batch
from src.tda.persistence import compute_persistence_diagrams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=pathlib.Path, help="Path to NumPy array containing conformations or features.")
    parser.add_argument("output", type=pathlib.Path, help="Destination .npz file for cached persistence diagrams.")
    parser.add_argument(
        "--geometry",
        choices=["cartesian", "dihedral"],
        default="dihedral",
        help="Geometry/metric for the input data (default: dihedral).",
    )
    parser.add_argument(
        "--homology-dims",
        type=int,
        nargs="+",
        default=(0, 1),
        help="Homology dimensions to compute (default: 0 1).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=4000,
        help="Maximum number of samples to include when computing persistence (default: 4000).",
    )
    parser.add_argument(
        "--max-edge-length",
        type=float,
        default=None,
        help="Optional filtration truncation radius. If omitted the maximum pairwise distance is used.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "gudhi", "ripser"],
        default="auto",
        help="Persistent homology backend to use (default: auto).",
    )
    parser.add_argument(
        "--torus-metric",
        choices=["geodesic", "sincos"],
        default="sincos",
        help="Metric for dihedral geometries (default: sincos).",
    )
    parser.add_argument(
        "--torus-harmonics",
        type=int,
        default=2,
        help="Number of harmonics when using sine/cosine torus embedding (default: 2).",
    )
    parser.add_argument(
        "--summary",
        type=pathlib.Path,
        default=None,
        help="Optional JSON file summarising persistence statistics.",
    )
    return parser.parse_args()


def subsample(array: np.ndarray, sample_size: int) -> np.ndarray:
    if sample_size and array.shape[0] > sample_size:
        rng = np.random.default_rng(1234)
        indices = rng.choice(array.shape[0], size=sample_size, replace=False)
        return array[indices]
    return array


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found at {args.dataset}.")

    data = np.load(args.dataset)
    if args.geometry == "cartesian" and data.ndim != 3:
        raise ValueError("Cartesian datasets must have shape (n_samples, n_atoms, 3).")
    if args.geometry != "cartesian" and data.ndim < 2:
        raise ValueError("Feature datasets must have shape (n_samples, feature_dim).")

    filtered = subsample(data, args.sample_size)
    diagrams = compute_persistence_diagrams(
        filtered,
        homology_dims=tuple(args.homology_dims),
        max_edge_length=args.max_edge_length,
        backend=args.backend,
        center=args.geometry == "cartesian",
        geometry=args.geometry,
        torus_metric=args.torus_metric,
        torus_harmonics=args.torus_harmonics,
    )

    save_diagram_batch(diagrams, args.output)
    print(f"Saved {len(diagrams.diagrams)} persistence diagrams to {args.output}.")

    if args.summary is not None:
        summary = {
            "dataset": str(args.dataset),
            "output": str(args.output),
            "homology_dims": diagrams.homology_dims,
            "torus_metric": diagrams.metadata.get("torus_metric"),
            "torus_harmonics": diagrams.metadata.get("torus_harmonics"),
            "max_edge_length": diagrams.metadata.get("max_edge_length"),
        }
        lifetimes: dict[int, Sequence[float]] = {}
        for diag in diagrams.diagrams:
            for dim, points in diag.items():
                if points.size:
                    lifetimes.setdefault(dim, []).extend((points[:, 1] - points[:, 0]).tolist())
        summary["persistence_max"] = {dim: float(max(vals)) for dim, vals in lifetimes.items() if vals}
        summary_path = args.summary
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"Wrote summary statistics to {summary_path}.")


if __name__ == "__main__":
    main()
