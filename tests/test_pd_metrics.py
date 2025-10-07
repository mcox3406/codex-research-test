"""Unit tests for persistence diagram distance utilities."""
from __future__ import annotations

import math
import unittest


try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is an optional runtime dependency
    np = None  # type: ignore

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable
    torch = None  # type: ignore

from src.tda.pd_metrics import wasserstein_distance


@unittest.skipIf(np is None, "numpy is required for persistence diagram tests")
class WassersteinDistanceTest(unittest.TestCase):
    def test_identical_diagrams_have_zero_distance(self) -> None:
        diag = np.array([[0.1, 0.6], [0.2, 0.9]], dtype=float)
        dist = wasserstein_distance(diag, diag, method="sinkhorn", epsilon=1e-2)
        self.assertAlmostEqual(dist, 0.0, places=6)

    def test_single_point_match(self) -> None:
        a = np.array([[0.0, 1.0]], dtype=float)
        b = np.array([[0.0, 2.0]], dtype=float)
        dist = wasserstein_distance(a, b, order=2.0, method="sinkhorn", epsilon=5e-3)
        self.assertAlmostEqual(dist, 1.0, places=3)

    def test_match_against_diagonal(self) -> None:
        a = np.empty((0, 2), dtype=float)
        b = np.array([[0.1, 0.7]], dtype=float)
        expected = (0.6 / math.sqrt(2.0))
        dist = wasserstein_distance(a, b, order=2.0, method="sinkhorn", epsilon=5e-3)
        self.assertAlmostEqual(dist, expected, places=3)

    @unittest.skipIf(torch is None, "torch is required for gradient-aware distance")
    def test_torch_inputs_return_tensor(self) -> None:
        a = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        b = torch.tensor([[0.0, 1.5]], dtype=torch.float64)
        dist = wasserstein_distance(a, b, order=2.0, method="sinkhorn", epsilon=1e-2)
        self.assertIsInstance(dist, torch.Tensor)
        self.assertTrue(torch.isfinite(dist))


if __name__ == "__main__":  # pragma: no cover - manual invocation
    unittest.main()
