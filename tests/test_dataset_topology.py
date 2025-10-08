import unittest
from pathlib import Path

import numpy as np

from src.tda.persistence import compute_persistence_diagrams


class AlanineDatasetTopologyTest(unittest.TestCase):
    DATASET = Path("data/alanine_dipeptide/phi_psi.npy")

    @unittest.skipUnless(DATASET.exists(), "Alanine dipeptide dataset not available.")
    def test_dihedral_dataset_exhibits_persistent_loop(self) -> None:
        phi_psi = np.load(self.DATASET)
        if phi_psi.shape[0] > 800:
            phi_psi = phi_psi[:800]
        diagrams = compute_persistence_diagrams(
            phi_psi,
            homology_dims=(1,),
            geometry="dihedral",
            center=False,
            torus_metric="sincos",
            torus_harmonics=2,
        )
        h1 = diagrams.diagrams[0].get(1, np.empty((0, 2)))
        self.assertGreater(h1.shape[0], 0)
        lifetime = h1[:, 1] - h1[:, 0]
        self.assertGreater(float(lifetime.max()), 0.1)


if __name__ == "__main__":
    unittest.main()
