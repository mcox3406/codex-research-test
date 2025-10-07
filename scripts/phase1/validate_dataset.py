"""Phase 1 data validation workflow for cyclic hexapeptide conformations.

This script orchestrates the analyses requested in Phase 1 of the experimental
plan.  It is **not executed automatically**; run it manually after generating the
raw NumPy dataset with either the OpenMM MD or Monte Carlo sampling recipes.

Outputs
-------
- `results/phase1/ground_truth_pd.png`: Persistence diagram for H0/H1.
- `results/phase1/ground_truth_barcode.png`: Barcode visualization.
- `results/phase1/rmsd_distribution.png`: Histogram of RMSD vs. reference frame.
- `results/phase1/rg_distribution.png`: Histogram of radius of gyration.
- `results/phase1/pca_projection.png`: PCA scatter colored by radius of gyration.
- `results/phase1/dataset_summary.json`: Numeric summary statistics for quick
  inspection.

These artifacts feed directly into Figure 1 and the introductory dataset
narrative in the paper.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
try:
    import mdtraj as md
except Exception:  # pragma: no cover - optional dependency at runtime
    md = None  # type: ignore

from src.data import extract_dihedrals
from src.tda.persistence import compute_persistence_diagrams


def _center_structures(coords: np.ndarray) -> np.ndarray:
    centered = coords - coords.mean(axis=1, keepdims=True)
    return centered


def _compute_rmsd(coords: np.ndarray) -> np.ndarray:
    reference = coords[0]
    diffs = coords - reference
    rmsd = np.sqrt(np.mean(np.sum(diffs**2, axis=-1), axis=-1))
    return rmsd


def _radius_of_gyration(coords: np.ndarray) -> np.ndarray:
    centered = coords - coords.mean(axis=1, keepdims=True)
    rg = np.sqrt(np.sum(centered**2, axis=(1, 2)) / coords.shape[2])
    return rg


def _pca_projection(coords: np.ndarray) -> np.ndarray:
    n_samples, n_atoms, _ = coords.shape
    flattened = coords.reshape(n_samples, n_atoms * 3)
    model = PCA(n_components=3, random_state=0)
    return model.fit_transform(flattened)


def _plot_histogram(values: np.ndarray, title: str, xlabel: str, output: pathlib.Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=50, color="#A31F34", alpha=0.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300)
    plt.close()


def _plot_pca(points: np.ndarray, color: np.ndarray, output: pathlib.Path) -> None:
    plt.figure(figsize=(5, 5))
    sc = plt.scatter(points[:, 0], points[:, 1], c=color, cmap="coolwarm", s=10, alpha=0.8)
    plt.colorbar(sc, label="Radius of gyration (Å)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of cyclic hexapeptide conformations")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300)
    plt.close()


def _plot_persistence(diagrams: Dict[int, np.ndarray], output_prefix: pathlib.Path) -> None:
    plt.figure(figsize=(6, 5))
    markers = {0: "o", 1: "s", 2: "^"}
    for dim, points in diagrams.items():
        if points.size == 0:
            continue
        births = points[:, 0]
        deaths = points[:, 1]
        plt.scatter(births, deaths, label=f"H{dim}", marker=markers.get(dim, "o"), alpha=0.8)
    finite_points = [pts for pts in diagrams.values() if pts.size > 0]
    if finite_points:
        max_val = max(points[:, 1].max() for points in finite_points)
    else:
        max_val = 1.0
    plt.plot([0, max_val], [0, max_val], linestyle="--", color="gray")
    plt.xlabel("Birth")
    plt.ylabel("Death")
    plt.title("Ground truth persistence diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_prefix.with_name(output_prefix.name + "_pd.png"), dpi=300)
    plt.close()

    # Barcode plot
    plt.figure(figsize=(6, 4))
    y_offset = 0
    for dim, points in diagrams.items():
        for birth, death in points:
            plt.hlines(y_offset, birth, death, colors="#A31F34" if dim == 1 else "#8A8B8C", linewidth=2)
            y_offset += 1
        y_offset += 1
    plt.xlabel("Filtration value")
    plt.ylabel("Features")
    plt.title("Ground truth barcode")
    plt.tight_layout()
    plt.savefig(output_prefix.with_name(output_prefix.name + "_barcode.png"), dpi=300)
    plt.close()


def _save_summary(stats: Dict[str, float], output: pathlib.Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stats, indent=2))


def validate_dataset(
    dataset_path: pathlib.Path,
    output_dir: pathlib.Path,
    topology_path: pathlib.Path,
    max_samples: int | None = None,
) -> None:
    coords = np.load(dataset_path)
    if max_samples is not None and max_samples < coords.shape[0]:
        coords = coords[:max_samples]

    coords = _center_structures(coords)
    if md is None:
        raise ImportError("mdtraj is required to extract dihedral angles for topology validation.")
    topology = md.load(str(topology_path)).topology

    dihedral_cloud = extract_dihedrals(coords, topology=topology)
    diagrams_batch = compute_persistence_diagrams(
        dihedral_cloud[None, ...],
        homology_dims=(0, 1),
        backend="auto",
        center=False,
        geometry="dihedral",
    )
    dataset_diagrams = diagrams_batch.diagrams[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_persistence(dataset_diagrams, output_dir / "ground_truth")

    rmsd = _compute_rmsd(coords)
    rg = _radius_of_gyration(coords)
    pca = _pca_projection(coords)

    _plot_histogram(rmsd, "RMSD distribution", "RMSD to reference (Å)", output_dir / "rmsd_distribution.png")
    _plot_histogram(rg, "Radius of gyration", "Rg (Å)", output_dir / "rg_distribution.png")
    _plot_pca(pca, rg, output_dir / "pca_projection.png")

    h1_diagram = dataset_diagrams.get(1, np.empty((0, 2)))
    h1_lifetimes = h1_diagram[:, 1] - h1_diagram[:, 0] if h1_diagram.size > 0 else np.array([])

    stats = {
        "num_samples": int(coords.shape[0]),
        "num_atoms": int(coords.shape[1]),
        "num_dihedral_features": int(dihedral_cloud.shape[1]),
        "rmsd_mean": float(rmsd.mean()),
        "rmsd_std": float(rmsd.std()),
        "rg_mean": float(rg.mean()),
        "rg_std": float(rg.std()),
        "h1_count": int(h1_diagram.shape[0]),
        "h1_max_persistence": float(h1_lifetimes.max()) if h1_lifetimes.size else 0.0,
        "h1_mean_persistence": float(h1_lifetimes.mean()) if h1_lifetimes.size else 0.0,
    }
    _save_summary(stats, output_dir / "dataset_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate cyclic hexapeptide dataset for Phase 1")
    parser.add_argument("--dataset", type=pathlib.Path, required=True, help="Path to NumPy array of conformations")
    parser.add_argument(
        "--topology",
        type=pathlib.Path,
        required=True,
        help="Topology file (e.g., input PDB) used to define backbone dihedrals",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/phase1"),
        help="Directory to write plots and summary statistics",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Optional cap on number of conformations to analyse (useful for smoke tests)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_dataset(args.dataset, args.output_dir, args.topology, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
