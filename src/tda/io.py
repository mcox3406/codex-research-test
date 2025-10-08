"""Input/output helpers for persistence diagrams."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from .persistence import DiagramBatch


def _normalise_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def save_diagram_batch(diagrams: DiagramBatch, path: str | Path) -> Path:
    """Serialise a :class:`DiagramBatch` to a compressed ``.npz`` file."""

    path = _normalise_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata = dict(diagrams.metadata or {})
    metadata["homology_dims"] = list(diagrams.homology_dims)
    metadata_json = json.dumps(metadata)

    payload: dict[str, np.ndarray] = {
        "batch_size": np.asarray([len(diagrams.diagrams)], dtype=np.int32),
        "metadata": np.asarray(metadata_json),
    }
    for idx, diagram in enumerate(diagrams.diagrams):
        for dim in diagrams.homology_dims:
            key = f"diagram_{idx}_dim_{dim}"
            payload[key] = diagram.get(dim, np.empty((0, 2), dtype=np.float64)).astype(np.float64, copy=False)

    np.savez_compressed(path, **payload)
    return path


def load_diagram_batch(path: str | Path) -> DiagramBatch:
    """Load a :class:`DiagramBatch` from ``.npz`` created by :func:`save_diagram_batch`."""

    file = _normalise_path(path)
    with np.load(file, allow_pickle=False) as data:
        if "metadata" not in data or "batch_size" not in data:
            raise ValueError("Invalid persistence diagram file: missing metadata or batch_size entries.")
        metadata_json = str(data["metadata"].item())
        metadata = json.loads(metadata_json)
        homology_dims_iter: Iterable[int] = metadata.get("homology_dims", [])
        homology_dims: Tuple[int, ...] = tuple(int(dim) for dim in homology_dims_iter)
        if not homology_dims:
            raise ValueError("Serialized persistence diagram missing homology dimensions.")
        batch_size = int(data["batch_size"].item())
        diagrams: list[dict[int, np.ndarray]] = []
        for idx in range(batch_size):
            diag: dict[int, np.ndarray] = {}
            for dim in homology_dims:
                key = f"diagram_{idx}_dim_{dim}"
                if key in data.files:
                    diag[dim] = np.asarray(data[key], dtype=np.float64)
                else:
                    diag[dim] = np.empty((0, 2), dtype=np.float64)
            diagrams.append(diag)
    # Remove homology_dims from metadata to avoid duplication when re-saving
    metadata.pop("homology_dims", None)
    return DiagramBatch(diagrams=diagrams, homology_dims=homology_dims, metadata=metadata)


__all__ = ["save_diagram_batch", "load_diagram_batch"]
