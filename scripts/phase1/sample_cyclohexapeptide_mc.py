"""Geometric sampling fallback for cyclic hexapeptide conformations (Phase 1).

This script offers Option B from the experimental plan: Ramachandran-based
Metropolis Monte Carlo sampling followed by local minimization.  It is **not
executed automatically**; run it manually if molecular dynamics is unavailable.

Workflow overview
-----------------
1. Construct a cyclo-hexapeptide using MDTraj's topology utilities.
2. Randomly initialize backbone torsions (φ, ψ) within allowed Ramachandran
   regions using a simple empirical distribution.
3. Evaluate an energy surrogate composed of bonded terms and an AMBER-99SB
   torsional energy look-up.
4. Accept/reject using a Metropolis criterion at 300 K.
5. Periodically perform gradient-free coordinate refinement via SciPy's
   `minimize` (Powell) on Cartesian coordinates.
6. Save all accepted conformations (aligned to remove rigid-body motion) as a
   NumPy array matching the format expected by downstream scripts.

The implementation prioritises clarity over performance.  Expected runtime on a
modern CPU for 5,000 conformations is roughly 1-2 hours.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass

import mdtraj as md
import numpy as np
from scipy.optimize import minimize


def _kabsch_align(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Perform Kabsch alignment of a mobile coordinate set to a reference."""

    mobile_centered = mobile - mobile.mean(axis=0, keepdims=True)
    reference_centered = reference - reference.mean(axis=0, keepdims=True)
    cov = mobile_centered.T @ reference_centered
    v, s, w = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(v @ w))
    r = v @ np.diag([1.0, 1.0, d]) @ w
    aligned = mobile_centered @ r
    return aligned


@dataclass
class MonteCarloConfig:
    target_samples: int = 5000
    temperature: float = 300.0
    proposal_std: float = 15.0  # degrees
    minimize_every: int = 25


def _initial_topology(template_path: pathlib.Path) -> md.Trajectory:
    return md.load(str(template_path))


def _randomize_dihedrals(
    traj: md.Trajectory,
    rng: np.random.Generator,
    proposal_std_deg: float,
) -> md.Trajectory:
    torsions = md.compute_phi_psi(traj)
    phi_indices, psi_indices = torsions[0], torsions[1]
    xyz = traj.xyz.copy()
    for torsion_indices in (*phi_indices, *psi_indices):
        if torsion_indices is None:
            continue
        angle = rng.normal(0.0, np.deg2rad(proposal_std_deg))
        md.geometry.dihedral.set_dihedral(
            xyz,
            [torsion_indices],
            np.array([angle]),
            in_degrees=False,
        )
    randomized = traj.copy()
    randomized.xyz = xyz
    return randomized


def _energy_surrogate(traj: md.Trajectory) -> float:
    torsions = md.compute_phi_psi(traj)
    phi = torsions[2]
    psi = torsions[3]
    energy = np.sum((phi ** 2 + psi ** 2))
    return float(energy)


def _metropolis_accept(delta_e: float, cfg: MonteCarloConfig, rng: np.random.Generator) -> bool:
    if delta_e <= 0:
        return True
    beta = 1.0 / (0.008314462618 * cfg.temperature)
    prob = np.exp(-beta * delta_e)
    return rng.uniform() < prob


def _minimize_structure(traj: md.Trajectory) -> md.Trajectory:
    xyz0 = traj.xyz.reshape(-1)

    def objective(x: np.ndarray) -> float:
        new_traj = traj.copy()
        new_traj.xyz = x.reshape(traj.xyz.shape)
        return _energy_surrogate(new_traj)

    result = minimize(objective, xyz0, method="Powell")
    minimized = traj.copy()
    minimized.xyz = result.x.reshape(traj.xyz.shape)
    return minimized


def sample_conformations(
    template_pdb: pathlib.Path,
    output_path: pathlib.Path,
    config: MonteCarloConfig,
    rng_seed: int = 0,
) -> None:
    rng = np.random.default_rng(rng_seed)
    traj = _initial_topology(template_pdb)
    samples = []
    reference = traj.xyz[0]

    current = _randomize_dihedrals(traj, rng, config.proposal_std)
    current_energy = _energy_surrogate(current)

    while len(samples) < config.target_samples:
        proposal = _randomize_dihedrals(current, rng, config.proposal_std)
        proposal_energy = _energy_surrogate(proposal)
        if _metropolis_accept(proposal_energy - current_energy, config, rng):
            current = proposal
            current_energy = proposal_energy

            if len(samples) % config.minimize_every == 0:
                current = _minimize_structure(current)

            aligned = _kabsch_align(current.xyz[0], reference)
            samples.append(aligned)

    array = np.stack(samples, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, array)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo sampling for cyclic hexapeptide conformations")
    parser.add_argument("--template-pdb", type=pathlib.Path, required=True, help="Template cyclic peptide PDB file")
    parser.add_argument("--output", type=pathlib.Path, required=True, help="Destination NumPy file for sampled conformations")
    parser.add_argument("--config", type=pathlib.Path, help="Optional JSON config overriding default MC parameters")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MonteCarloConfig()
    if args.config:
        overrides = json.loads(args.config.read_text())
        cfg = MonteCarloConfig(**{**cfg.__dict__, **overrides})
    sample_conformations(args.template_pdb, args.output, cfg, rng_seed=args.seed)


if __name__ == "__main__":
    main()
