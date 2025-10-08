"""Generate a torus-based playground dataset with pronounced H₁ topology."""
from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.tda.persistence import compute_persistence_diagrams

_TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class LoopSpec:
    """Specification of a loop traced on the (ϕ, ψ) torus."""

    phi_shift: float
    psi_shift: float
    phi_orientation: float = 1.0
    psi_orientation: float = 1.0

    def sample(self, theta: np.ndarray, noise: float, rng: np.random.RandomState) -> np.ndarray:
        phi = self.phi_shift + self.phi_orientation * theta
        psi = self.psi_shift + self.psi_orientation * theta
        if noise > 0.0:
            phi += rng.normal(scale=noise, size=theta.shape)
            psi += rng.normal(scale=noise, size=theta.shape)
        return np.stack([np.mod(phi, _TWOPI), np.mod(psi, _TWOPI)], axis=1)


def _build_loops() -> List[LoopSpec]:
    """Return default loop specifications mimicking alanine pathways."""

    return [
        LoopSpec(phi_shift=0.4 * np.pi, psi_shift=1.2 * np.pi, phi_orientation=1.0, psi_orientation=1.0),
        LoopSpec(phi_shift=1.6 * np.pi, psi_shift=0.6 * np.pi, phi_orientation=1.0, psi_orientation=-1.0),
    ]


def _sample_dataset(n_samples: int, noise: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    loops = _build_loops()
    n_loops = len(loops)
    base = []
    per_loop = n_samples // n_loops
    for idx, loop in enumerate(loops):
        count = per_loop if idx < n_loops - 1 else n_samples - per_loop * (n_loops - 1)
        theta = rng.uniform(0.0, _TWOPI, size=count)
        base.append(loop.sample(theta, noise, rng))

    # Add a few bridging points to create noisy transitions between basins.
    bridge_count = max(int(0.05 * n_samples), 100)
    bridge_phi = rng.uniform(0.25 * np.pi, 1.75 * np.pi, size=bridge_count)
    bridge_psi = (bridge_phi + rng.normal(scale=0.15, size=bridge_count)) % _TWOPI
    base.append(np.stack([bridge_phi, bridge_psi], axis=1))

    data = np.concatenate(base, axis=0)
    rng.shuffle(data)
    return data


def _plot_angles(angles: np.ndarray, output: pathlib.Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    hb = ax.hexbin(angles[:, 0], angles[:, 1], gridsize=75, cmap="inferno")
    fig.colorbar(hb, ax=ax, label="Density")
    ax.set_xlabel("ϕ (rad)")
    ax.set_ylabel("ψ (rad)")
    ax.set_xlim(0, _TWOPI)
    ax.set_ylim(0, _TWOPI)
    ax.set_xticks([0, np.pi, _TWOPI])
    ax.set_xticklabels(["0", "π", "2π"])
    ax.set_yticks([0, np.pi, _TWOPI])
    ax.set_yticklabels(["0", "π", "2π"])
    ax.set_title("Playground torus loops density")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _plot_persistence(diagrams, output_prefix: pathlib.Path) -> Tuple[int, float]:
    diag = diagrams.diagrams[0]
    h1 = diag.get(1, np.empty((0, 2)))

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    if diag.get(0) is not None and diag[0].size:
        ax.scatter(diag[0][:, 0], diag[0][:, 1], color="#8A8B8C", alpha=0.6, label="H0", s=18)
    if h1.size:
        ax.scatter(h1[:, 0], h1[:, 1], color="#A31F34", alpha=0.85, label="H1", s=28)
    max_val = 1.0
    for values in diag.values():
        if values.size:
            max_val = max(max_val, float(np.max(values)))
    ax.plot([0, max_val], [0, max_val], linestyle="--", color="#333333", linewidth=1.0)
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.legend()
    ax.set_title("Persistence diagram (playground)")
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(output_prefix.name + "_pd.png"), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3))
    offset = 0
    for dim, values in diag.items():
        for birth, death in values:
            color = "#A31F34" if dim == 1 else "#8A8B8C"
            ax.hlines(offset, birth, death, colors=color, linewidth=2)
            offset += 1
        offset += 1
    ax.set_xlabel("Filtration value")
    ax.set_ylabel("Feature index")
    ax.set_title("Barcode (playground)")
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(output_prefix.name + "_barcode.png"), dpi=300)
    plt.close(fig)

    lifetimes = h1[:, 1] - h1[:, 0] if h1.size else np.array([])
    return int(h1.shape[0]), float(lifetimes.max() if lifetimes.size else 0.0)


def generate_playground(
    n_samples: int,
    noise: float,
    seed: int,
    output_dir: pathlib.Path,
    torus_harmonics: int = 2,
) -> dict:
    data = _sample_dataset(n_samples, noise, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "torus_loops.npy"
    np.save(data_path, data)

    diagrams = compute_persistence_diagrams(
        data,
        homology_dims=(0, 1),
        geometry="dihedral",
        center=False,
        torus_metric="sincos",
        torus_harmonics=torus_harmonics,
    )
    h1_count, h1_max = _plot_persistence(diagrams, output_dir / "playground_persistence")
    _plot_angles(data, output_dir / "ramachandran_density.png")

    summary = {
        "n_samples": int(data.shape[0]),
        "noise": float(noise),
        "seed": int(seed),
        "torus_harmonics": int(torus_harmonics),
        "h1_count": h1_count,
        "h1_max_persistence": h1_max,
        "data_path": str(data_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a torus playground dataset with visible H1 loops")
    parser.add_argument("--n-samples", type=int, default=4000, help="Number of samples to generate (default: 4000)")
    parser.add_argument("--noise", type=float, default=0.08, help="Gaussian noise applied to angles (default: 0.08 rad)")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed (default: 1234)")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("data/playground"),
        help="Directory to store the dataset and diagnostics",
    )
    parser.add_argument(
        "--torus-harmonics",
        type=int,
        default=2,
        help="Number of harmonics for the sincos embedding (default: 2)",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> dict:
    args = parse_args(argv)
    summary = generate_playground(
        n_samples=args.n_samples,
        noise=args.noise,
        seed=args.seed,
        output_dir=args.output_dir,
        torus_harmonics=args.torus_harmonics,
    )
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
