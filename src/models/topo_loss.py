"""Topological regularisation module for molecular generative models."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from src.tda.persistence import DiagramBatch, compute_persistence_diagrams
from src.tda.pd_metrics import wasserstein_distance


def _to_numpy(batch: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(batch, torch.Tensor):
        return batch.detach().cpu().numpy()
    return np.asarray(batch)


@dataclass
class TopologicalLossState:
    """Container tracking cached loss statistics."""

    step: Optional[int] = None
    value: Optional[float] = None
    per_dimension: MutableMapping[int, float] | None = None


class TopologicalLoss(nn.Module):
    """Compute a Wasserstein-based topological loss between two batches.

    The module caches the expensive persistent homology computation and only
    recomputes it every ``update_every`` calls. ``forward`` returns a scalar
    ``Tensor`` that can be combined with other loss terms. The returned tensor
    does **not** carry gradients with respect to the input coordinates because
    persistent homology is non-differentiable for general point clouds. This is
    intentional – the tensor is safe to use within autograd graphs without
    leaking resources, but gradients will be zero. Future differentiable
    relaxations can be plugged in by overriding :meth:`_compute_distance`.
    """

    def __init__(
        self,
        homology_dims: Sequence[int] = (1,),
        max_edge_length: Optional[float] = None,
        backend: str = "auto",
        center: bool = True,
        update_every: int = 1,
        reduction: str = "mean",
        dim_weights: Optional[Mapping[int, float]] = None,
        wasserstein_kwargs: Optional[Mapping[str, float | str]] = None,
    ) -> None:
        super().__init__()
        if len(homology_dims) == 0:
            raise ValueError("At least one homology dimension must be specified.")
        self.homology_dims: Tuple[int, ...] = tuple(sorted(set(int(dim) for dim in homology_dims)))
        self.max_edge_length = max_edge_length
        self.backend = backend
        self.center = center
        self.update_every = max(1, int(update_every))
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'.")
        self.reduction = reduction
        if dim_weights is None:
            self.dim_weights = {dim: 1.0 for dim in self.homology_dims}
        else:
            self.dim_weights = {int(k): float(v) for k, v in dim_weights.items()}
        self.wasserstein_kwargs: Dict[str, float | str] = {
            "order": 2.0,
            "method": "auto",
            "epsilon": 5e-3,
            "max_iterations": 1000,
            "tolerance": 1e-6,
        }
        if wasserstein_kwargs:
            self.wasserstein_kwargs.update(wasserstein_kwargs)

        self._state = TopologicalLossState()
        self._reference: Optional[DiagramBatch] = None

    @property
    def last_step(self) -> Optional[int]:
        return self._state.step

    @property
    def last_value(self) -> Optional[float]:
        return self._state.value

    @property
    def last_per_dimension(self) -> Optional[Mapping[int, float]]:
        return self._state.per_dimension

    def reset(self) -> None:
        """Clear cached values forcing the next call to recompute diagrams."""

        self._state = TopologicalLossState()

    def set_reference(self, diagrams: DiagramBatch) -> None:
        """Provide pre-computed reference diagrams for the real batch.

        When a reference is set the ``real_batch`` passed to :meth:`forward`
        can be ``None``. This is useful for large datasets where the real
        diagrams are computed offline once.
        """

        self._reference = diagrams

    def forward(
        self,
        real_batch: Optional[Tensor | np.ndarray],
        generated_batch: Tensor | np.ndarray,
        *,
        step: Optional[int] = None,
        force: bool = False,
    ) -> Tensor:
        """Return the cached or freshly computed topological loss."""

        device = generated_batch.device if isinstance(generated_batch, torch.Tensor) else None
        dtype = generated_batch.dtype if isinstance(generated_batch, torch.Tensor) else None

        reuse = (
            not force
            and self._state.value is not None
            and self._state.step is not None
            and step is not None
            and (step - self._state.step) < self.update_every
        )
        if reuse:
            return self._to_tensor(self._state.value, device=device, dtype=dtype)

        if self._reference is None and real_batch is None:
            raise ValueError("real_batch cannot be None unless a reference is set via set_reference().")

        with torch.no_grad():
            real_diagrams = self._reference
            if real_diagrams is None:
                real_np = _to_numpy(real_batch)  # type: ignore[arg-type]
                real_diagrams = compute_persistence_diagrams(
                    real_np,
                    homology_dims=self.homology_dims,
                    max_edge_length=self.max_edge_length,
                    backend=self.backend,
                    center=self.center,
                )
            gen_np = _to_numpy(generated_batch)
            gen_diagrams = compute_persistence_diagrams(
                gen_np,
                homology_dims=self.homology_dims,
                max_edge_length=self.max_edge_length,
                backend=self.backend,
                center=self.center,
            )

            value, per_dim = self._evaluate(real_diagrams, gen_diagrams)

        self._state = TopologicalLossState(step=step, value=value, per_dimension=per_dim)
        return self._to_tensor(value, device=device, dtype=dtype)

    def _evaluate(self, real: DiagramBatch, generated: DiagramBatch) -> Tuple[float, MutableMapping[int, float]]:
        if len(real.diagrams) == 0 or len(generated.diagrams) == 0:
            return 0.0, {dim: 0.0 for dim in self.homology_dims}

        batch = min(len(real.diagrams), len(generated.diagrams))
        dim_accumulator: Dict[int, list[float]] = {dim: [] for dim in self.homology_dims}

        for idx in range(batch):
            real_diagram = real.diagrams[idx]
            gen_diagram = generated.diagrams[idx]
            for dim in self.homology_dims:
                dist = self._compute_distance(
                    real_diagram.get(dim, np.empty((0, 2))),
                    gen_diagram.get(dim, np.empty((0, 2))),
                )
                dim_accumulator[dim].append(float(dist))

        per_dimension = {
            dim: float(np.mean(values)) if len(values) > 0 else 0.0
            for dim, values in dim_accumulator.items()
        }

        total_weight = sum(self.dim_weights.get(dim, 0.0) for dim in self.homology_dims)
        if math.isclose(total_weight, 0.0):
            total_weight = float(len(self.homology_dims))
        if self.reduction == "mean":
            weighted = sum(self.dim_weights.get(dim, 1.0) * per_dimension[dim] for dim in self.homology_dims)
            loss_value = weighted / max(total_weight, 1e-12)
        else:  # sum
            loss_value = sum(self.dim_weights.get(dim, 1.0) * per_dimension[dim] for dim in self.homology_dims)

        return float(loss_value), per_dimension

    def _compute_distance(self, diag_real: np.ndarray, diag_gen: np.ndarray) -> float:
        return float(wasserstein_distance(diag_real, diag_gen, **self.wasserstein_kwargs))

    @staticmethod
    def _to_tensor(value: float, device: Optional[torch.device], dtype: Optional[torch.dtype]) -> Tensor:
        if dtype is None:
            tensor = torch.tensor(value)
        else:
            tensor = torch.tensor(value, dtype=dtype)
        if device is not None:
            tensor = tensor.to(device)
        return tensor


__all__ = ["TopologicalLoss", "TopologicalLossState"]
