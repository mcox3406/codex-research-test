"""Utilities for computing persistent homology on batched molecular conformations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np

try:  # Optional torch support for differentiable pipelines
    import torch
    from torch import Tensor
except Exception:  # pragma: no cover - torch is optional at import time
    torch = None  # type: ignore
    Tensor = "Tensor"  # type: ignore

try:  # Prefer GUDHI when available – stable and feature-rich
    import gudhi
except Exception:  # pragma: no cover - GUDHI is optional in lightweight environments
    gudhi = None  # type: ignore

try:  # ripser is a light-weight fallback
    from ripser import ripser
except Exception:  # pragma: no cover - optional dependency
    ripser = None  # type: ignore

ArrayLike = Union[np.ndarray, "Tensor"]


@dataclass
class DiagramBatch:
    """Container for a batch of persistence diagrams."""

    diagrams: List[Dict[int, np.ndarray]]
    homology_dims: Tuple[int, ...]
    metadata: MutableMapping[str, Union[int, float, str]] | None = None

    def __iter__(self):  # pragma: no cover
        return iter(self.diagrams)

    def __len__(self) -> int:
        return len(self.diagrams)

    def by_dimension(self, dim: int) -> List[np.ndarray]:
        """Return diagrams for a specific homology dimension across the batch."""

        if dim not in self.homology_dims:
            raise KeyError(f"Homology dimension {dim} was not computed.")
        return [diagram.get(dim, np.empty((0, 2))) for diagram in self.diagrams]

    def iter_pairs(self, dims: Optional[Iterable[int]] = None):
        """Yield ``(dim, diagram)`` pairs for convenient iteration.

        Parameters
        ----------
        dims:
            Optional iterable restricting the dimensions that are yielded. By
            default all dimensions present in :attr:`homology_dims` are used.
        """

        selected = self.homology_dims if dims is None else tuple(dims)
        for dim in selected:
            if dim not in self.homology_dims:
                raise KeyError(f"Homology dimension {dim} was not computed.")
            for diagram in self.diagrams:
                yield dim, diagram.get(dim, np.empty((0, 2)))


def _to_numpy(array: ArrayLike) -> np.ndarray:
    if torch is not None and isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _center_points(points: np.ndarray) -> np.ndarray:
    centroid = points.mean(axis=0, keepdims=True)
    return points - centroid


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    return dists


def _compute_with_gudhi(
    points: np.ndarray,
    homology_dims: Sequence[int],
    max_edge_length: Optional[float],
) -> Dict[int, np.ndarray]:
    if gudhi is None:  # pragma: no cover - guard at runtime
        raise ImportError("gudhi is required for the 'gudhi' backend but is not installed.")

    max_dim = max(homology_dims)
    rips_complex = gudhi.RipsComplex(points=points, max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
    simplex_tree.persistence()

    diagrams: Dict[int, np.ndarray] = {}
    for dim in homology_dims:
        intervals = simplex_tree.persistence_intervals_in_dimension(dim)
        if len(intervals) == 0:
            diagrams[dim] = np.empty((0, 2), dtype=np.float64)
        else:
            diagrams[dim] = np.asarray(intervals, dtype=np.float64)
    return diagrams


def _compute_with_ripser(
    points: np.ndarray,
    homology_dims: Sequence[int],
    max_edge_length: Optional[float],
    dist_matrix: Optional[np.ndarray] = None,
) -> Dict[int, np.ndarray]:
    if ripser is None:  # pragma: no cover - guard at runtime
        raise ImportError("ripser is required for the 'ripser' backend but is not installed.")

    max_dim = max(homology_dims)
    if dist_matrix is None:
        dist_matrix = _pairwise_distances(points)
    res = ripser(dist_matrix, distance_matrix=True, maxdim=max_dim, thresh=max_edge_length)
    diagrams: Dict[int, np.ndarray] = {}
    for dim in homology_dims:
        diag = res["dgms"][dim] if dim < len(res["dgms"]) else np.empty((0, 2))
        diagrams[dim] = np.asarray(diag, dtype=np.float64)
    return diagrams


def compute_persistence_diagrams(
    conformations: ArrayLike,
    homology_dims: Sequence[int] = (0, 1),
    max_edge_length: Optional[float] = None,
    backend: str = "auto",
    center: bool = True,
) -> DiagramBatch:
    """Compute persistence diagrams for a batch of conformations.

    Parameters
    ----------
    conformations:
        Array of shape ``(batch, n_atoms, 3)`` containing Cartesian coordinates.
        ``numpy.ndarray`` and ``torch.Tensor`` inputs are supported.
    homology_dims:
        Iterable of homology dimensions to compute (e.g., ``(0, 1)``).
    max_edge_length:
        Optional truncation of the Vietoris–Rips filtration. If ``None`` the
        maximum inter-point distance is used for each conformation.
    backend:
        ``"gudhi"`` to force the GUDHI implementation, ``"ripser"`` for the
        Ripser fallback, or ``"auto"`` to use the best available option.
    center:
        Whether to remove translational degrees of freedom before computing
        distances. Recommended for molecular systems.

    Returns
    -------
    DiagramBatch
        Persistence diagrams grouped per conformation and homology dimension.
    """

    homology_dims = tuple(sorted(set(int(dim) for dim in homology_dims)))
    if len(homology_dims) == 0:
        raise ValueError("At least one homology dimension must be specified.")

    coords = _to_numpy(conformations)
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(
            "Conformations must have shape (batch, n_points, 3); received "
            f"shape {coords.shape}."
        )

    batch_size = coords.shape[0]
    diagrams: List[Dict[int, np.ndarray]] = []

    if backend == "auto":
        if gudhi is not None:
            backend = "gudhi"
        elif ripser is not None:
            backend = "ripser"
        else:
            raise RuntimeError(
                "No persistent homology backend available. Install 'gudhi' or 'ripser'."
            )

    for idx in range(batch_size):
        points = coords[idx]
        if center:
            points = _center_points(points)

        max_edge = max_edge_length
        if max_edge is None:
            dmatrix = _pairwise_distances(points)
            max_edge = float(np.max(dmatrix))
        else:
            dmatrix = None

        if backend == "gudhi":
            diagrams.append(_compute_with_gudhi(points, homology_dims, max_edge))
        elif backend == "ripser":
            if dmatrix is None:
                dmatrix = _pairwise_distances(points)
            diagrams.append(_compute_with_ripser(points, homology_dims, max_edge, dmatrix))
        else:
            raise ValueError(f"Unsupported backend '{backend}'.")

    metadata: MutableMapping[str, Union[int, float, str]] = {
        "backend": backend,
        "homology_dims": homology_dims,
    }
    if max_edge_length is not None:
        metadata["max_edge_length"] = float(max_edge_length)

    return DiagramBatch(diagrams=diagrams, homology_dims=homology_dims, metadata=metadata)


__all__ = ["DiagramBatch", "compute_persistence_diagrams"]
