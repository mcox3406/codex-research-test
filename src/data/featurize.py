"""Feature extraction utilities for molecular conformations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # Optional dependency used when topology information is available
    import mdtraj as md
except Exception:  # pragma: no cover - mdtraj is optional at runtime
    md = None  # type: ignore

_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class DihedralSpecification:
    """Specification of a single dihedral in terms of atom indices."""

    atom_indices: Tuple[int, int, int, int]
    label: str | None = None


def _normalize_angles(angles: np.ndarray, wrap: bool) -> np.ndarray:
    if not wrap:
        return angles
    return np.mod(angles + _TWO_PI, _TWO_PI)


def _batch_dihedral(points: np.ndarray) -> np.ndarray:
    """Compute dihedral angles for a batch of 4-point fragments."""

    p0 = points[:, 0, :]
    p1 = points[:, 1, :]
    p2 = points[:, 2, :]
    p3 = points[:, 3, :]

    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = np.linalg.norm(b1, axis=1, keepdims=True)
    # Prevent division by zero by adding an epsilon; degenerate torsions will
    # produce zero angles after normalization.
    b1_unit = np.divide(b1, np.where(b1_norm == 0.0, 1.0, b1_norm))

    v = b0 - (np.sum(b0 * b1_unit, axis=1, keepdims=True) * b1_unit)
    w = b2 - (np.sum(b2 * b1_unit, axis=1, keepdims=True) * b1_unit)

    x = np.sum(v * w, axis=1)
    y = np.sum(np.cross(b1_unit, v), w, axis=1)
    angles = np.arctan2(y, x)
    return angles


def _phi_psi_indices(
    topology: "md.Topology",
    residue_indices: Optional[Sequence[int]] = None,
    cyclic: bool = True,
) -> List[DihedralSpecification]:
    residues = list(topology.residues)
    if residue_indices is None:
        residue_indices = [res.index for res in residues]
    lookup = {(atom.residue.index, atom.name.strip()): atom.index for atom in topology.atoms}
    n_res = len(residues)

    def _get_index(res_idx: int, atom_name: str) -> int:
        key = (res_idx, atom_name)
        if key not in lookup:
            raise KeyError(f"Atom {atom_name} not found in residue {res_idx}.")
        return lookup[key]

    specs: List[DihedralSpecification] = []
    for res_idx in residue_indices:
        prev_idx = (res_idx - 1) % n_res if cyclic else res_idx - 1
        next_idx = (res_idx + 1) % n_res if cyclic else res_idx + 1

        try:
            if prev_idx >= 0:
                phi = (
                    _get_index(prev_idx, "C"),
                    _get_index(res_idx, "N"),
                    _get_index(res_idx, "CA"),
                    _get_index(res_idx, "C"),
                )
                specs.append(DihedralSpecification(phi, label=f"phi_{res_idx}"))
        except KeyError:
            pass

        try:
            if next_idx < n_res or cyclic:
                psi = (
                    _get_index(res_idx, "N"),
                    _get_index(res_idx, "CA"),
                    _get_index(res_idx, "C"),
                    _get_index(next_idx % n_res, "N"),
                )
                specs.append(DihedralSpecification(psi, label=f"psi_{res_idx}"))
        except KeyError:
            pass

    return specs


def extract_dihedrals(
    coords: np.ndarray,
    topology: "md.Topology" | None = None,
    *,
    residue_indices: Optional[Sequence[int]] = None,
    custom_dihedrals: Optional[Iterable[Sequence[int]]] = None,
    wrap: bool = True,
) -> np.ndarray:
    """Extract backbone dihedral angles from molecular conformations.

    Parameters
    ----------
    coords:
        Array of shape ``(n_samples, n_atoms, 3)`` containing Cartesian
        coordinates.
    topology:
        Optional MDTraj topology describing atom ordering. Required when
        ``custom_dihedrals`` is not provided.
    residue_indices:
        Optional iterable of residue indices for which phi/psi dihedrals should
        be computed. Defaults to all residues in ``topology``.
    custom_dihedrals:
        Optional explicit list of 4-tuples of atom indices specifying which
        dihedrals to compute. When supplied, this overrides automatic phi/psi
        inference and does not require a topology object.
    wrap:
        If ``True`` (default), returned angles are wrapped into the interval
        ``[0, 2π)``. Otherwise raw ``(-π, π]`` radians are returned.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_samples, n_dihedrals)`` containing dihedral angles
        in radians.
    """

    array = np.asarray(coords, dtype=float)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(
            "coords must have shape (n_samples, n_atoms, 3); received "
            f"{array.shape}."
        )

    n_samples = array.shape[0]

    if custom_dihedrals is not None:
        specs = [DihedralSpecification(tuple(map(int, dihedral))) for dihedral in custom_dihedrals]
    else:
        if topology is None:
            raise ValueError("topology must be provided when custom_dihedrals is None.")
        if md is None:  # pragma: no cover - happens only when mdtraj missing
            raise ImportError("mdtraj is required to infer phi/psi dihedrals from topology.")
        specs = _phi_psi_indices(topology, residue_indices=residue_indices)
        if not specs:
            raise ValueError("No dihedral definitions could be constructed from the provided topology.")

    angles = np.empty((n_samples, len(specs)), dtype=float)
    for column, spec in enumerate(specs):
        atom_indices = np.asarray(spec.atom_indices, dtype=int)
        fragment = array[:, atom_indices, :]
        angles[:, column] = _batch_dihedral(fragment)

    return _normalize_angles(angles, wrap=wrap)


def extract_phi_psi(
    coords: np.ndarray,
    topology: "md.Topology",
    *,
    wrap: bool = True,
) -> np.ndarray:
    """Convenience wrapper returning the single (ϕ, ψ) pair for alanine dipeptide.

    Parameters
    ----------
    coords:
        Coordinate array with shape ``(n_samples, n_atoms, 3)``.
    topology:
        MDTraj topology describing the atom ordering consistent with ``coords``.
    wrap:
        Whether to wrap the returned angles into ``[0, 2π)``.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_samples, 2)`` containing ``(ϕ, ψ)`` angles in radians.
    """

    if md is None:  # pragma: no cover - mdtraj optional at runtime
        raise ImportError("mdtraj is required to compute phi/psi dihedrals.")

    array = np.asarray(coords, dtype=float)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("coords must have shape (n_samples, n_atoms, 3).")

    traj = md.Trajectory(array, topology)
    _, phi = md.compute_phi(traj)
    _, psi = md.compute_psi(traj)
    if phi.shape[1] == 0 or psi.shape[1] == 0:
        raise ValueError("Topology does not define phi/psi dihedrals.")
    angles = np.stack([phi[:, 0], psi[:, 0]], axis=1)
    return _normalize_angles(angles, wrap=wrap)


__all__ = [
    "DihedralSpecification",
    "extract_dihedrals",
    "extract_phi_psi",
]
