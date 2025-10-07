"""Generate synthetic Ramachandran-like data with guaranteed H1 topology.

Creates a toy dataset with three basins connected by transition paths that form
a loop on the torus, ensuring persistent H1 features for testing topological
regularization.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np


def sample_basin(center: tuple[float, float], std: float, n_samples: int, rng: np.random.RandomState) -> np.ndarray:
    """Sample points from a Gaussian basin on the torus."""
    angles = rng.normal(loc=center, scale=std, size=(n_samples, 2))
    return np.mod(angles, 2.0 * np.pi)


def sample_transition(start: tuple[float, float], end: tuple[float, float], width: float, n_samples: int, rng: np.random.RandomState) -> np.ndarray:
    """Sample points along a transition path between two basins."""
    t = rng.uniform(0, 1, size=n_samples)
    start_arr = np.array(start)
    end_arr = np.array(end)

    # Handle wrapping on torus
    diff = end_arr - start_arr
    diff = np.where(diff > np.pi, diff - 2 * np.pi, diff)
    diff = np.where(diff < -np.pi, diff + 2 * np.pi, diff)

    path = start_arr + t[:, None] * diff
    noise = rng.normal(0, width, size=(n_samples, 2))
    angles = path + noise
    return np.mod(angles, 2.0 * np.pi)


def generate_toy_ramachandran(
    n_total: int = 10000,
    basin_std: float = 0.3,
    transition_width: float = 0.2,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic Ramachandran data with three basins forming an H1 loop.

    Parameters
    ----------
    n_total : int
        Total number of samples to generate.
    basin_std : float
        Standard deviation of Gaussian basins (radians).
    transition_width : float
        Width of transition paths between basins (radians).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (n_total, 2) with (phi, psi) angles in [0, 2π).
    """
    rng = np.random.RandomState(seed)

    # Define three basin centers that form a triangle on the torus
    # These mimic alpha_R, alpha_L, and beta regions
    basin_centers = [
        (5.0, 5.5),  # alpha_R-like (upper right)
        (1.0, 2.5),  # alpha_L-like (lower left)
        (4.5, 2.0),  # beta-like (middle bottom)
    ]

    # Allocate samples: 60% in basins, 40% in transitions
    n_basin = int(0.6 * n_total)
    n_trans = n_total - n_basin

    # Sample from basins (20% each)
    n_per_basin = n_basin // 3
    basin_samples = []
    for center in basin_centers:
        samples = sample_basin(center, basin_std, n_per_basin, rng)
        basin_samples.append(samples)

    # Sample transitions forming a loop: 1->2, 2->3, 3->1
    n_per_trans = n_trans // 3
    transitions = [
        sample_transition(basin_centers[0], basin_centers[1], transition_width, n_per_trans, rng),
        sample_transition(basin_centers[1], basin_centers[2], transition_width, n_per_trans, rng),
        sample_transition(basin_centers[2], basin_centers[0], transition_width, n_per_trans, rng),
    ]

    # Combine all samples
    all_samples = np.vstack(basin_samples + transitions)

    # Shuffle to avoid temporal ordering artifacts
    indices = rng.permutation(len(all_samples))
    return all_samples[indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Ramachandran data with H1 topology")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("data/toy_ramachandran"),
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Total number of samples to generate",
    )
    parser.add_argument(
        "--basin-std",
        type=float,
        default=0.3,
        help="Standard deviation of basin distributions (radians)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n_samples} synthetic Ramachandran samples...", flush=True)
    phi_psi = generate_toy_ramachandran(
        n_total=args.n_samples,
        basin_std=args.basin_std,
        seed=args.seed,
    )

    output_path = args.output_dir / "phi_psi.npy"
    np.save(output_path, phi_psi.astype(np.float32))
    print(f"Saved to {output_path}", flush=True)
    print(f"Shape: {phi_psi.shape}, min: {phi_psi.min():.3f}, max: {phi_psi.max():.3f}")


if __name__ == "__main__":
    main()
