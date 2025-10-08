import unittest
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch optional for tests
    torch = None  # type: ignore

if torch is not None:
    from src.models.topo_loss import TopologicalLoss
    from src.tda.persistence import DiagramBatch
else:  # pragma: no cover - skip tests when torch unavailable
    TopologicalLoss = None  # type: ignore
    DiagramBatch = None  # type: ignore


@unittest.skipIf(torch is None, "PyTorch is required for topological loss tests.")
class TopologicalLossTest(unittest.TestCase):
    def _diagram(self, value: float = 0.5) -> DiagramBatch:
        point = np.array([[0.0, value]], dtype=float)
        return DiagramBatch(diagrams=[{1: point}], homology_dims=(1,))

    def test_caches_loss_between_updates(self) -> None:
        topo = TopologicalLoss(homology_dims=(1,), update_every=2)
        real = torch.zeros((1, 2, 3))
        generated = torch.ones((1, 2, 3))

        diagrams = [self._diagram() for _ in range(6)]
        with mock.patch(
            "src.models.topo_loss.compute_persistence_diagrams",
            side_effect=diagrams,
        ) as mock_compute, mock.patch(
            "src.models.topo_loss.wasserstein_distance",
            return_value=0.5,
        ) as mock_metric:
            loss1 = topo(real, generated, step=0)
            self.assertAlmostEqual(loss1.item(), 0.5, places=6)
            self.assertEqual(mock_compute.call_count, 2)
            self.assertEqual(mock_metric.call_count, 1)

            loss2 = topo(real, generated, step=1)
            self.assertAlmostEqual(loss2.item(), 0.5, places=6)
            self.assertEqual(mock_compute.call_count, 2)

            loss3 = topo(real, generated, step=2)
            self.assertAlmostEqual(loss3.item(), 0.5, places=6)
            self.assertEqual(mock_compute.call_count, 4)
            self.assertEqual(mock_metric.call_count, 2)

    def test_reference_diagrams_allow_none_real_batch(self) -> None:
        topo = TopologicalLoss(homology_dims=(1,))
        topo.set_reference(self._diagram())
        generated = torch.ones((1, 2, 3))

        with mock.patch(
            "src.models.topo_loss.compute_persistence_diagrams",
            return_value=self._diagram(1.0),
        ) as mock_compute, mock.patch(
            "src.models.topo_loss.wasserstein_distance",
            return_value=1.25,
        ):
            loss = topo(None, generated, step=0, force=True)
            self.assertAlmostEqual(loss.item(), 1.25, places=6)
            mock_compute.assert_called_once()
            per_dim = topo.last_per_dimension
            self.assertIsNotNone(per_dim)
            self.assertAlmostEqual(per_dim[1], 1.25, places=6)

    def test_reference_broadcasts_across_generated_batch(self) -> None:
        topo = TopologicalLoss(homology_dims=(1,))
        topo.set_reference(self._diagram(0.4))
        generated = torch.ones((3, 2, 3))

        diagrams = DiagramBatch(
            diagrams=[{1: np.array([[0.0, 1.0]])} for _ in range(3)],
            homology_dims=(1,),
        )

        with mock.patch(
            "src.models.topo_loss.compute_persistence_diagrams",
            return_value=diagrams,
        ), mock.patch(
            "src.models.topo_loss.wasserstein_distance",
            side_effect=[0.2, 0.3, 0.4],
        ) as mock_metric:
            loss = topo(None, generated, step=0, force=True)

        self.assertAlmostEqual(loss.item(), (0.2 + 0.3 + 0.4) / 3.0, places=6)
        self.assertEqual(mock_metric.call_count, 3)

    def test_missing_reference_raises(self) -> None:
        topo = TopologicalLoss(homology_dims=(1,))
        generated = torch.ones((1, 2, 3))
        with self.assertRaises(ValueError):
            topo(None, generated)

    def test_invalid_geometry_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TopologicalLoss(homology_dims=(1,), geometry="hyperbolic")


if __name__ == "__main__":
    unittest.main()
