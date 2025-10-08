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


def _torus_pairwise_distances(points: np.ndarray) -> np.ndarray:
    """Pairwise distances on a flat torus with ``[0, 2π)`` coordinates."""

    wrapped = np.mod(points, 2.0 * np.pi)
    diff = np.abs(wrapped[:, None, :] - wrapped[None, :, :])
    diff = np.minimum(diff, 2.0 * np.pi - diff)
    return np.sqrt(np.sum(diff * diff, axis=-1))


def _torus_sincos_embedding(points: np.ndarray, harmonics: int = 1) -> np.ndarray:
    r"""Embed torus coordinates into Euclidean space using sine/cosine features.

    The embedding maps an angle ``θ`` to the vector

    .. math:: [\cos(k\theta), \sin(k\theta)]_{k=1}^{H}

    which linearises the periodic boundaries and allows Euclidean filtrations to
    approximate geodesic distances on ``\mathbb{T}^n``. Increasing
    ``harmonics`` adds higher-frequency components improving fidelity at the
    cost of dimensionality. ``harmonics`` must be a positive integer.
    """

    if harmonics <= 0:
        raise ValueError("harmonics must be a positive integer")
    wrapped = np.mod(points, 2.0 * np.pi)
    components = []
    for harmonic in range(1, harmonics + 1):
        components.append(np.cos(harmonic * wrapped))
        components.append(np.sin(harmonic * wrapped))
    return np.concatenate(components, axis=-1)


def _compute_with_gudhi(
    points: Optional[np.ndarray],
    homology_dims: Sequence[int],
    max_edge_length: Optional[float],
    dist_matrix: Optional[np.ndarray] = None,
) -> Dict[int, np.ndarray]:
    if gudhi is None:  # pragma: no cover - guard at runtime
        raise ImportError("gudhi is required for the 'gudhi' backend but is not installed.")

    max_dim = max(homology_dims)
    if dist_matrix is not None:
        rips_complex = gudhi.RipsComplex(distance_matrix=dist_matrix, max_edge_length=max_edge_length)
    else:
        if points is None:
            raise ValueError("Points must be provided when distance matrix is not supplied.")
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
    geometry: str = "cartesian",
    *,
    torus_metric: str = "geodesic",
    torus_harmonics: int = 1,
) -> DiagramBatch:
    """Compute persistence diagrams for a batch of conformations.

    Parameters
    ----------
    conformations:
        Array of shape ``(batch, n_points, d)`` containing point clouds. When
        ``geometry='cartesian'`` the final dimension ``d`` must equal 3. For
        ``geometry='dihedral'`` the coordinates are assumed to be angles in
        radians on ``[0, 2π)``.
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
        distances. Ignored when ``geometry!='cartesian'``.
    geometry:
        Metric definition for the point cloud. ``'cartesian'`` applies the usual
        Euclidean metric after optional centering. ``'dihedral'`` interprets the
        coordinates as periodic angles on a flat torus.
    torus_metric:
        Strategy used when ``geometry='dihedral'``. ``'geodesic'`` (default)
        computes exact wrapped distances on the torus. ``'sincos'`` embeds each
        angle using sine/cosine features before constructing a Euclidean
        filtration, which is often more numerically stable for sparse samples.
    torus_harmonics:
        Number of Fourier harmonics to include in the sine/cosine embedding when
        ``torus_metric='sincos'``. Higher values approximate torus geodesics more
        accurately at the cost of doubling the ambient dimension per harmonic.

    Returns
    -------
    DiagramBatch
        Persistence diagrams grouped per conformation and homology dimension.
    """

    homology_dims = tuple(sorted(set(int(dim) for dim in homology_dims)))
    if len(homology_dims) == 0:
        raise ValueError("At least one homology dimension must be specified.")

    coords = _to_numpy(conformations)
    if coords.ndim == 2:
        coords = coords[None, ...]

    if coords.ndim != 3:
        raise ValueError(
            "Conformations must have shape (batch, n_points, d); received "
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
        dist_matrix: Optional[np.ndarray] = None

        if geometry == "cartesian":
            if points.shape[-1] != 3:
                raise ValueError(
                    "Cartesian geometry expects point dimension 3; received "
                    f"{points.shape[-1]}."
                )
            if center:
                points = _center_points(points)
            if max_edge_length is None or backend == "ripser":
                dist_matrix = _pairwise_distances(points)
            max_edge = max_edge_length if max_edge_length is not None else float(np.max(dist_matrix))
        elif geometry == "dihedral":
            if torus_metric not in {"geodesic", "sincos"}:
                raise ValueError("torus_metric must be 'geodesic' or 'sincos'.")
            if torus_metric == "geodesic":
                dist_matrix = _torus_pairwise_distances(points)
                max_edge = max_edge_length if max_edge_length is not None else float(np.max(dist_matrix))
            else:  # torus_metric == "sincos"
                embedded = _torus_sincos_embedding(points, harmonics=torus_harmonics)
                if max_edge_length is None or backend == "ripser":
                    dist_matrix = _pairwise_distances(embedded)
                max_edge = max_edge_length if max_edge_length is not None else float(np.max(dist_matrix))
                points = embedded
        else:
            raise ValueError(f"Unsupported geometry '{geometry}'.")

        if backend == "gudhi":
            diagrams.append(
                _compute_with_gudhi(
                    points if geometry == "cartesian" or torus_metric == "sincos" else None,
                    homology_dims,
                    max_edge,
                    dist_matrix,
                )
            )
        elif backend == "ripser":
            if dist_matrix is None:
                dist_matrix = _pairwise_distances(points)
            diagrams.append(_compute_with_ripser(points, homology_dims, max_edge, dist_matrix))
        else:
            raise ValueError(f"Unsupported backend '{backend}'.")

    metadata: MutableMapping[str, Union[int, float, str]] = {
        "backend": backend,
        "homology_dims": homology_dims,
        "geometry": geometry,
    }
    if max_edge_length is not None:
        metadata["max_edge_length"] = float(max_edge_length)
    if geometry == "dihedral":
        metadata["torus_metric"] = torus_metric
        metadata["torus_harmonics"] = int(torus_harmonics)

    return DiagramBatch(diagrams=diagrams, homology_dims=homology_dims, metadata=metadata)


__all__ = ["DiagramBatch", "compute_persistence_diagrams"]
