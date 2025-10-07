"""Persistence diagram distance utilities with differentiable fallbacks."""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

try:  # Optional torch support for differentiable Sinkhorn computation
    import torch
    from torch import Tensor
except Exception:  # pragma: no cover - torch may be unavailable
    torch = None  # type: ignore
    Tensor = "Tensor"  # type: ignore

try:
    from gudhi import wasserstein as gudhi_wasserstein
except Exception:  # pragma: no cover - optional dependency
    gudhi_wasserstein = None  # type: ignore

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - SciPy may be unavailable
    linear_sum_assignment = None  # type: ignore

Diagram = Union[np.ndarray, "Tensor"]


def _ensure_numpy(diagram: Diagram) -> np.ndarray:
    if torch is not None and isinstance(diagram, torch.Tensor):
        array = diagram.detach().cpu().numpy()
    else:
        array = np.asarray(diagram, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 2)
    if array.shape[-1] != 2:
        raise ValueError("Persistence diagrams must have shape (n_points, 2).")
    return array


def _ensure_tensor(diagram: Diagram, like: Optional[Tensor] = None) -> Tensor:
    if torch is None:
        raise RuntimeError("PyTorch is required for differentiable Wasserstein distances.")
    if isinstance(diagram, torch.Tensor):
        if like is None:
            return diagram
        return diagram.to(device=like.device, dtype=like.dtype)
    array = np.asarray(diagram, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 2)
    if array.shape[-1] != 2:
        raise ValueError("Persistence diagrams must have shape (n_points, 2).")
    dtype = like.dtype if like is not None else torch.get_default_dtype()
    if like is not None:
        return torch.as_tensor(array, dtype=dtype, device=like.device)
    return torch.as_tensor(array, dtype=dtype)


def _project_to_diag_np(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 2), dtype=points.dtype)
    mid = 0.5 * (points[:, 0] + points[:, 1])
    return np.stack((mid, mid), axis=-1)


def _project_to_diag_torch(points: Tensor) -> Tensor:
    if points.numel() == 0:
        return torch.empty((0, 2), dtype=points.dtype, device=points.device)
    mid = 0.5 * (points[:, 0] + points[:, 1])
    return torch.stack((mid, mid), dim=-1)


def _extended_diagrams_np(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    diag_a = _project_to_diag_np(a)
    diag_b = _project_to_diag_np(b)
    ext_a = np.concatenate((a, diag_b), axis=0)
    ext_b = np.concatenate((b, diag_a), axis=0)
    return ext_a, ext_b


def _extended_diagrams_torch(a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    diag_a = _project_to_diag_torch(a)
    diag_b = _project_to_diag_torch(b)
    ext_a = torch.cat((a, diag_b), dim=0)
    ext_b = torch.cat((b, diag_a), dim=0)
    return ext_a, ext_b


def _pairwise_cost_np(points_a: np.ndarray, points_b: np.ndarray, order: float) -> np.ndarray:
    diff = points_a[:, None, :] - points_b[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    return dist ** order


def _pairwise_cost_torch(points_a: Tensor, points_b: Tensor, order: float) -> Tensor:
    if points_a.numel() == 0 or points_b.numel() == 0:
        return torch.empty(
            (points_a.shape[0], points_b.shape[0]),
            dtype=points_a.dtype,
            device=points_a.device,
        )
    dist = torch.cdist(points_a, points_b, p=2)
    return dist.pow(order)


def _prepare_cost_matrix_np(a: np.ndarray, b: np.ndarray, order: float) -> np.ndarray:
    ext_a, ext_b = _extended_diagrams_np(a, b)
    cost = _pairwise_cost_np(ext_a, ext_b, order)
    if b.shape[0] > 0 and a.shape[0] > 0:
        cost[-b.shape[0] :, -a.shape[0] :] = 0.0
    return cost


def _prepare_cost_matrix_torch(a: Tensor, b: Tensor, order: float) -> Tensor:
    ext_a, ext_b = _extended_diagrams_torch(a, b)
    cost = _pairwise_cost_torch(ext_a, ext_b, order)
    if b.shape[0] > 0 and a.shape[0] > 0:
        cost[-b.shape[0] :, -a.shape[0] :] = 0.0
    return cost


def _hungarian(cost_matrix: np.ndarray) -> float:
    if linear_sum_assignment is None:
        raise RuntimeError("SciPy is required for the exact Wasserstein solver.")
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return float(cost_matrix[row_ind, col_ind].sum())


def _uniform_weights(size: int, like: Tensor) -> Tensor:
    if size == 0:
        return torch.empty((0,), dtype=like.dtype, device=like.device)
    return torch.ones((size,), dtype=like.dtype, device=like.device)


def _uniform_weights_np(size: int) -> np.ndarray:
    if size == 0:
        return np.empty((0,), dtype=float)
    return np.ones((size,), dtype=float)


def _logsumexp_np(values: np.ndarray, axis: int) -> np.ndarray:
    max_val = np.max(values, axis=axis, keepdims=True)
    stable = values - max_val
    summed = np.log(np.sum(np.exp(stable), axis=axis, keepdims=True))
    return (max_val + summed).squeeze(axis=axis)


def _sinkhorn_numpy(
    cost_matrix: np.ndarray,
    order: float,
    epsilon: float,
    max_iterations: int,
    tolerance: float,
    weights_a: Optional[np.ndarray] = None,
    weights_b: Optional[np.ndarray] = None,
) -> float:
    if cost_matrix.size == 0:
        return 0.0

    n, m = cost_matrix.shape
    if n != m:
        raise ValueError("Extended persistence cost matrices must be square.")

    if weights_a is None:
        weights_a = _uniform_weights_np(n)
    if weights_b is None:
        weights_b = _uniform_weights_np(m)

    mass_a = float(np.sum(weights_a))
    mass_b = float(np.sum(weights_b))
    if not np.isclose(mass_a, mass_b, rtol=1e-6, atol=1e-9):
        raise ValueError("Total mass of extended diagrams must match for Sinkhorn computation.")

    norm_weights_a = weights_a / max(mass_a, 1e-12)
    norm_weights_b = weights_b / max(mass_b, 1e-12)

    log_weights_a = np.log(np.clip(norm_weights_a, 1e-12, None))
    log_weights_b = np.log(np.clip(norm_weights_b, 1e-12, None))

    log_K = -cost_matrix / epsilon

    u = np.zeros_like(log_weights_a)
    v = np.zeros_like(log_weights_b)

    for _ in range(max_iterations):
        u_prev = u
        u = log_weights_a - _logsumexp_np(log_K + v[np.newaxis, :], axis=1)
        v = log_weights_b - _logsumexp_np(log_K + u[:, np.newaxis], axis=0)
        if np.max(np.abs(u - u_prev)) < tolerance:
            break

    transport = np.exp(log_K + u[:, np.newaxis] + v[np.newaxis, :])
    total = float(np.sum(transport * cost_matrix))
    scaled_total = total * mass_a
    return float(np.clip(scaled_total, 0.0, None) ** (1.0 / order))


def _sinkhorn(
    cost_matrix: Tensor,
    order: float,
    epsilon: float,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    weights_a: Optional[Tensor] = None,
    weights_b: Optional[Tensor] = None,
) -> Tensor:
    if torch is None:
        raise RuntimeError("PyTorch is required for the Sinkhorn approximation.")

    if cost_matrix.numel() == 0:
        return torch.zeros((), dtype=cost_matrix.dtype, device=cost_matrix.device)

    n, m = cost_matrix.shape
    if n != m:
        raise ValueError("Extended persistence cost matrices must be square.")

    if weights_a is None:
        weights_a = _uniform_weights(n, cost_matrix)
    if weights_b is None:
        weights_b = _uniform_weights(m, cost_matrix)

    mass_a = weights_a.sum()
    mass_b = weights_b.sum()
    if not torch.allclose(mass_a, mass_b, rtol=1e-6, atol=1e-9):
        raise ValueError("Total mass of extended diagrams must match for Sinkhorn computation.")

    norm_weights_a = weights_a / mass_a.clamp_min(1e-12)
    norm_weights_b = weights_b / mass_b.clamp_min(1e-12)

    log_weights_a = torch.log(norm_weights_a.clamp_min(1e-12))
    log_weights_b = torch.log(norm_weights_b.clamp_min(1e-12))

    log_K = -cost_matrix / epsilon

    u = torch.zeros_like(weights_a)
    v = torch.zeros_like(weights_b)

    for _ in range(max_iterations):
        u_prev = u
        u = log_weights_a - torch.logsumexp(log_K + v.unsqueeze(0), dim=1)
        v = log_weights_b - torch.logsumexp((log_K + u.unsqueeze(1)), dim=0)
        if torch.max(torch.abs(u - u_prev)) < tolerance:
            break

    transport = torch.exp(log_K + u.unsqueeze(1) + v.unsqueeze(0))
    total = torch.sum(transport * cost_matrix)
    return total.mul(mass_a).clamp_min(0.0).pow(1.0 / order)


def wasserstein_distance(
    diagram_a: Diagram,
    diagram_b: Diagram,
    order: float = 2.0,
    method: str = "auto",
    epsilon: float = 5e-3,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
) -> Union[float, Tensor]:
    """Compute the Wasserstein distance between two persistence diagrams.

    Parameters
    ----------
    diagram_a, diagram_b:
        Persistence diagrams with shape ``(n_points, 2)``. ``numpy.ndarray`` or
        ``torch.Tensor`` inputs are supported.
    order:
        Wasserstein order ``p``. Must satisfy ``p >= 1``.
    method:
        ``"exact"`` to force the Hungarian solver, ``"sinkhorn"`` for the
        entropic relaxation, or ``"auto"`` to select an appropriate backend.
    epsilon:
        Entropic regularisation strength for ``method="sinkhorn"``.
    max_iterations:
        Maximum number of Sinkhorn iterations when using the entropic solver.
    tolerance:
        Convergence threshold for the Sinkhorn fixed-point updates.
    """

    if order < 1:
        raise ValueError("Wasserstein order must be >= 1.")

    if torch is not None and (
        isinstance(diagram_a, torch.Tensor) or isinstance(diagram_b, torch.Tensor)
    ):
        return _wasserstein_torch(
            diagram_a,
            diagram_b,
            order=order,
            method=method,
            epsilon=epsilon,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
    return _wasserstein_numpy(
        diagram_a,
        diagram_b,
        order=order,
        method=method,
        epsilon=epsilon,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )


def _wasserstein_numpy(
    diagram_a: Diagram,
    diagram_b: Diagram,
    order: float,
    method: str,
    epsilon: float,
    max_iterations: int,
    tolerance: float,
) -> float:
    a = _ensure_numpy(diagram_a)
    b = _ensure_numpy(diagram_b)

    if a.size == 0 and b.size == 0:
        return 0.0

    if a.shape == b.shape and np.allclose(a, b):
        return 0.0

    if gudhi_wasserstein is not None and method in {"auto", "exact"}:
        return float(gudhi_wasserstein.wasserstein_distance(a, b, order=order))

    if method == "auto":
        method = "exact" if max(len(a), len(b)) <= 64 and linear_sum_assignment is not None else "sinkhorn"

    if method == "exact":
        cost = _prepare_cost_matrix_np(a, b, order)
        return _hungarian(cost) ** (1.0 / order)

    cost = _prepare_cost_matrix_np(a, b, order)
    return _sinkhorn_numpy(
        cost,
        order=order,
        epsilon=epsilon,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )


def _wasserstein_torch(
    diagram_a: Diagram,
    diagram_b: Diagram,
    order: float,
    method: str,
    epsilon: float,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
) -> Tensor:
    a = _ensure_tensor(diagram_a)
    b = _ensure_tensor(diagram_b, like=a)

    if a.numel() == 0 and b.numel() == 0:
        return torch.zeros((), dtype=a.dtype, device=a.device)

    if a.shape == b.shape and torch.allclose(a, b):
        return torch.zeros((), dtype=a.dtype, device=a.device)

    if gudhi_wasserstein is not None and method in {"auto", "exact"}:
        value = gudhi_wasserstein.wasserstein_distance(a.detach().cpu().numpy(), b.detach().cpu().numpy(), order=order)
        return torch.as_tensor(value, dtype=a.dtype, device=a.device)

    if method == "auto":
        method = "exact" if max(len(a), len(b)) <= 64 and linear_sum_assignment is not None else "sinkhorn"

    if method == "exact":
        if linear_sum_assignment is None:
            raise RuntimeError("SciPy is required for the exact Wasserstein solver.")
        cost = _prepare_cost_matrix_np(a.detach().cpu().numpy(), b.detach().cpu().numpy(), order)
        value = _hungarian(cost) ** (1.0 / order)
        return torch.as_tensor(value, dtype=a.dtype, device=a.device)

    cost_matrix = _prepare_cost_matrix_torch(a, b, order)
    return _sinkhorn(
        cost_matrix,
        order=order,
        epsilon=epsilon,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )


def bottleneck_distance(
    diagram_a: Diagram,
    diagram_b: Diagram,
    method: str = "auto",
) -> Union[float, Tensor]:
    """Compute the bottleneck distance between two persistence diagrams."""

    if torch is not None and (
        isinstance(diagram_a, torch.Tensor) or isinstance(diagram_b, torch.Tensor)
    ):
        return _bottleneck_torch(diagram_a, diagram_b, method=method)
    return _bottleneck_numpy(diagram_a, diagram_b, method=method)


def _bottleneck_numpy(diagram_a: Diagram, diagram_b: Diagram, method: str) -> float:
    a = _ensure_numpy(diagram_a)
    b = _ensure_numpy(diagram_b)

    if a.size == 0 and b.size == 0:
        return 0.0

    if gudhi_wasserstein is not None and method in {"auto", "exact"}:
        return float(gudhi_wasserstein.bottleneck_distance(a, b))

    if linear_sum_assignment is None:
        raise RuntimeError("SciPy is required for bottleneck distance when GUDHI is unavailable.")

    return _bottleneck_assignment(a, b)


def _bottleneck_torch(diagram_a: Diagram, diagram_b: Diagram, method: str) -> Tensor:
    if torch is None:
        raise RuntimeError("PyTorch is not available for bottleneck distance computation.")
    a = _ensure_tensor(diagram_a)
    b = _ensure_tensor(diagram_b, like=a)

    if a.numel() == 0 and b.numel() == 0:
        return torch.zeros((), dtype=a.dtype, device=a.device)

    if gudhi_wasserstein is not None and method in {"auto", "exact"}:
        value = gudhi_wasserstein.bottleneck_distance(a.detach().cpu().numpy(), b.detach().cpu().numpy())
        return torch.as_tensor(value, dtype=a.dtype, device=a.device)

    if linear_sum_assignment is None:
        raise RuntimeError("SciPy is required for bottleneck distance when GUDHI is unavailable.")

    value = _bottleneck_assignment(a.detach().cpu().numpy(), b.detach().cpu().numpy())
    return torch.as_tensor(value, dtype=a.dtype, device=a.device)


def _bottleneck_assignment(a: np.ndarray, b: np.ndarray) -> float:
    cost = _prepare_cost_matrix_np(a, b, order=1.0)
    candidates = np.unique(cost)
    candidates = np.sort(candidates)

    def feasible(threshold: float) -> bool:
        mask = cost <= threshold
        penalty = np.where(mask, 0.0, 1e6)
        row, col = linear_sum_assignment(penalty)
        return np.all(penalty[row, col] == 0.0)

    lo, hi = 0, len(candidates) - 1
    best = candidates[-1]
    while lo <= hi:
        mid = (lo + hi) // 2
        if feasible(candidates[mid]):
            best = candidates[mid]
            hi = mid - 1
        else:
            lo = mid + 1
    return float(best)


__all__ = ["wasserstein_distance", "bottleneck_distance"]
