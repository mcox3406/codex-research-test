import math
import unittest
from unittest import mock

import numpy as np

from src.data import extract_dihedrals, extract_phi_psi


class ExtractDihedralsTest(unittest.TestCase):
    def setUp(self) -> None:
        # Two simple fragments where the final atom is displaced to generate a
        # non-zero torsion. Coordinates are chosen so the first frame is planar
        # (angle = 0) and the second is rotated out of plane.
        self.coords = np.array(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0],
                ],
            ],
            dtype=float,
        )
        self.indices = [(0, 1, 2, 3)]

    def test_custom_dihedrals_shape_and_range(self) -> None:
        angles = extract_dihedrals(self.coords, custom_dihedrals=self.indices)
        self.assertEqual(angles.shape, (2, 1))
        self.assertTrue(np.all(angles >= 0.0))
        self.assertTrue(np.all(angles < 2.0 * math.pi))

    def test_wrap_false_preserves_signed_angle(self) -> None:
        wrapped = extract_dihedrals(self.coords, custom_dihedrals=self.indices, wrap=True)
        raw = extract_dihedrals(self.coords, custom_dihedrals=self.indices, wrap=False)
        self.assertAlmostEqual(float(wrapped[0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(raw[0, 0]), 0.0, places=6)
        # Second frame should be negative before wrapping because of the right-hand rule
        self.assertLess(raw[1, 0], 0.0)
        self.assertAlmostEqual(
            float(wrapped[1, 0]),
            (raw[1, 0] + 2.0 * math.pi) % (2.0 * math.pi),
            places=6,
        )

    def test_missing_topology_raises(self) -> None:
        with self.assertRaises(ValueError):
            extract_dihedrals(self.coords, topology=None)

    def test_extract_phi_psi_wraps_angles(self) -> None:
        coords = np.zeros((3, 4, 3))

        dummy_phi = np.array([[-math.pi / 2], [0.5], [3.0]])
        dummy_psi = np.array([[math.pi / 2], [1.0], [-2.0]])

        with mock.patch("src.data.featurize.md") as mock_md:
            mock_md.Trajectory.side_effect = lambda array, topology: (array, topology)
            mock_md.compute_phi.return_value = (None, dummy_phi)
            mock_md.compute_psi.return_value = (None, dummy_psi)

            angles = extract_phi_psi(coords, topology="dummy")

        self.assertEqual(angles.shape, (3, 2))
        self.assertTrue(np.all(angles >= 0.0))
        self.assertTrue(np.all(angles < 2.0 * math.pi))


if __name__ == "__main__":
    unittest.main()
