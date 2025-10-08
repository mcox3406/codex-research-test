import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.tda.io import load_diagram_batch, save_diagram_batch
from src.tda.persistence import DiagramBatch


class DiagramIOSerializationTest(unittest.TestCase):
    def test_round_trip_serialization(self) -> None:
        diagrams = DiagramBatch(
            diagrams=[
                {0: np.array([[0.0, 1.0]]), 1: np.array([[0.2, 0.8]])},
                {0: np.empty((0, 2)), 1: np.array([[0.1, 0.6], [0.15, 0.65]])},
            ],
            homology_dims=(0, 1),
            metadata={"geometry": "dihedral", "torus_metric": "sincos"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "reference.npz"
            save_diagram_batch(diagrams, path)
            self.assertTrue(path.exists())

            loaded = load_diagram_batch(path)
            self.assertEqual(loaded.homology_dims, diagrams.homology_dims)
            self.assertEqual(len(loaded.diagrams), len(diagrams.diagrams))
            self.assertEqual(loaded.metadata.get("geometry"), "dihedral")
            for original, restored in zip(diagrams.diagrams, loaded.diagrams):
                for dim in diagrams.homology_dims:
                    np.testing.assert_allclose(original.get(dim), restored.get(dim))


if __name__ == "__main__":
    unittest.main()
