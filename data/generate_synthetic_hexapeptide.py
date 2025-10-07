"""Generate a synthetic cyclic hexapeptide conformation dataset.

This script fabricates 3D coordinates that mimic two ring-flipping pathways
connected through a loop in conformational space. It is intended purely for
local development and unit tests – **do not run inside the Codex environment**
because large binary artifacts cannot be uploaded.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

DEFAULT_NUM_SAMPLES = 5000
DEFAULT_NUM_ATOMS = 60


@dataclass
class DatasetMetadata:
    num_samples: int
    num_atoms: int
    description: str = (
        "Synthetic cyclic hexapeptide conformations with two ring-flipping loops"
    )
    seed: int = 42


def _base_ring(num_atoms: int) -> np.ndarray:
    """Construct a planar ring template representing the peptide backbone."""

    theta = np.linspace(0, 2 * np.pi, num_atoms, endpoint=False)
    radius = 4.0
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros_like(x)
    return np.stack((x, y, z), axis=-1)


def _loop_deformation(t: float) -> Tuple[float, float, float]:
    """Parametrise two coupled torsional modes forming a loop."""

    angle1 = np.sin(2 * np.pi * t)
    angle2 = np.cos(2 * np.pi * t)
    bridge = 0.5 * np.sin(4 * np.pi * t)
    return angle1, angle2, bridge


def _apply_deformation(coords: np.ndarray, t: float, noise_scale: float, rng: np.random.Generator) -> np.ndarray:
    """Bend the ring to emulate collective torsion modes."""

    angle1, angle2, bridge = _loop_deformation(t)
    deformed = coords.copy()

    axis = np.array([0.0, 0.0, 1.0])
    rot_mat_1 = _rotation_matrix(axis, 0.3 * angle1)
    rot_mat_2 = _rotation_matrix(axis, -0.3 * angle2)

    half = len(coords) // 2
    deformed[:half] = deformed[:half] @ rot_mat_1.T
    deformed[half:] = deformed[half:] @ rot_mat_2.T

    # Introduce an out-of-plane puckering to create the second degree of freedom
    z_offset = bridge * np.sin(np.linspace(0, 2 * np.pi, len(coords)))
    deformed[:, 2] += z_offset

    # Add small Gaussian noise to mimic thermal fluctuations
    deformed += rng.normal(scale=noise_scale, size=deformed.shape)

    # Recentre to remove translation
    deformed -= deformed.mean(axis=0, keepdims=True)
    return deformed


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def generate_dataset(
    num_samples: int = DEFAULT_NUM_SAMPLES,
    num_atoms: int = DEFAULT_NUM_ATOMS,
    noise_scale: float = 0.05,
    seed: int = 42,
) -> Tuple[np.ndarray, DatasetMetadata]:
    """Synthesize a dataset of cyclic peptide conformations."""

    rng = np.random.default_rng(seed)
    base = _base_ring(num_atoms)

    ts = np.linspace(0.0, 1.0, num_samples, endpoint=False)
    # Warp parameterisation to create two dominant basins connected via a loop
    mixing = 0.5 * (1 + np.sin(2 * np.pi * ts))
    ts = (mixing * ts + (1 - mixing) * (ts + 0.25)) % 1.0

    conformations = np.zeros((num_samples, num_atoms, 3), dtype=np.float32)
    for idx, t in enumerate(ts):
        conformations[idx] = _apply_deformation(base, t, noise_scale=noise_scale, rng=rng).astype(np.float32)

    metadata = DatasetMetadata(num_samples=num_samples, num_atoms=num_atoms, seed=seed)
    return conformations, metadata


def save_dataset(output: Path, conformations: np.ndarray, metadata: DatasetMetadata) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, conformations)
    meta_path = output.with_suffix(".json")
    meta_path.write_text(json.dumps(asdict(metadata), indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("synthetic_cyclohexapeptide.npy"))
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--num-atoms", type=int, default=DEFAULT_NUM_ATOMS)
    parser.add_argument("--noise-scale", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conformations, metadata = generate_dataset(
        num_samples=args.num_samples,
        num_atoms=args.num_atoms,
        noise_scale=args.noise_scale,
        seed=args.seed,
    )
    save_dataset(args.output, conformations, metadata)
    print(f"Saved synthetic dataset to {args.output} with metadata {args.output.with_suffix('.json')}")


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
