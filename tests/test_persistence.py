import math
import unittest
from unittest import mock

import numpy as np

from src.tda import persistence as persistence_module
from src.tda.persistence import compute_persistence_diagrams


class PersistenceGeometryTest(unittest.TestCase):
    def test_dihedral_geometry_uses_torus_distance(self) -> None:
        points = np.array([
            [0.0, 0.0],
            [2.0 * math.pi - 0.2, 0.2],
        ])

        fake_diag = {0: np.empty((0, 2))}
        with mock.patch("src.tda.persistence._compute_with_gudhi", return_value=fake_diag) as mock_gudhi, mock.patch(
            "src.tda.persistence._compute_with_ripser"
        ) as mock_ripser:
            diagrams = compute_persistence_diagrams(
                points[None, ...],
                homology_dims=(0,),
                backend="gudhi",
                center=False,
                geometry="dihedral",
            )
            self.assertEqual(len(diagrams.diagrams), 1)
            # Ensure ripser fallback wasn't used when GUDHI path succeeds
            mock_ripser.assert_not_called()

        # GUDHI helper should have received a torus-aware distance matrix where the
        # two points are ~0.2828 apart instead of ~6.18 in Euclidean space.
        _, _, _, dist_matrix = mock_gudhi.call_args[0]
        self.assertAlmostEqual(dist_matrix[0, 1], math.sqrt(0.2**2 + 0.2**2), places=6)
        self.assertAlmostEqual(dist_matrix[1, 0], dist_matrix[0, 1], places=6)

    def test_cartesian_geometry_requires_three_dims(self) -> None:
        with self.assertRaises(ValueError):
            compute_persistence_diagrams(np.zeros((1, 4, 2)), geometry="cartesian")

    @unittest.skipUnless(
        getattr(persistence_module, "gudhi", None) is not None
        or getattr(persistence_module, "ripser", None) is not None,
        "No persistent homology backend available",
    )
    def test_sincos_metric_recovers_circular_loop(self) -> None:
        theta = np.linspace(0.0, 2.0 * math.pi, num=180, endpoint=False)
        loop = np.stack([np.mod(theta + 0.3, 2.0 * math.pi), np.mod(theta + 1.1, 2.0 * math.pi)], axis=1)
        diagrams = compute_persistence_diagrams(
            loop,
            homology_dims=(1,),
            geometry="dihedral",
            center=False,
            torus_metric="sincos",
            torus_harmonics=2,
        )
        h1 = diagrams.diagrams[0].get(1, np.empty((0, 2)))
        self.assertGreaterEqual(h1.shape[0], 1)
        if h1.size:
            lifetime = float(np.max(h1[:, 1] - h1[:, 0]))
            self.assertGreater(lifetime, 0.05)


if __name__ == "__main__":
    unittest.main()
