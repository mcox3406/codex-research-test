import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

from src.tda.persistence import DiagramBatch

if torch is not None:
    from src.train.train_vae import train
    from src.train.eval_topology import evaluate
else:  # pragma: no cover - skip when torch missing
    train = None  # type: ignore
    evaluate = None  # type: ignore


@unittest.skipIf(torch is None, "PyTorch is required for the training workflow test.")
class TrainEvalWorkflowTest(unittest.TestCase):
    def test_train_and_evaluate_with_mocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.randn(12, 4, 3).astype(np.float32)
            dataset_path = Path(tmpdir) / "toy.npy"
            np.save(dataset_path, data)

            config = {
                "dataset": {
                    "path": str(dataset_path),
                    "train_fraction": 1.0,
                    "val_fraction": 0.0,
                    "test_fraction": 0.0,
                    "seed": 0,
                },
                "model": {
                    "latent_dim": 2,
                    "hidden_dims": [8],
                    "coordinate_scaling": 1.0,
                },
                "training": {
                    "epochs": 1,
                    "batch_size": 4,
                    "learning_rate": 1e-3,
                    "weight_decay": 0.0,
                    "beta": 1.0,
                    "max_steps": 3,
                    "log_every": 1,
                    "checkpoint_every": 2,
                    "output_dir": str(Path(tmpdir) / "runs"),
                    "seed": 0,
                    "device": "cpu",
                },
                "topology": {"enabled": False},
                "logging": {"backend": "stdout"},
            }

            result = train(config)
            self.assertGreater(result["state"].step, 0)
            self.assertTrue(result["checkpoint"].exists())

            diag_real = DiagramBatch(
                diagrams=[{0: np.array([[0.0, 0.5]]), 1: np.array([[0.1, 0.6]])}],
                homology_dims=(0, 1),
            )
            diag_gen = DiagramBatch(
                diagrams=[{0: np.array([[0.0, 0.55]]), 1: np.array([[0.2, 0.7]])}],
                homology_dims=(0, 1),
            )

            with mock.patch(
                "src.train.eval_topology.compute_persistence_diagrams",
                side_effect=[diag_real, diag_gen],
            ), mock.patch(
                "src.train.eval_topology.wasserstein_distance",
                return_value=0.3,
            ), mock.patch(
                "src.train.eval_topology.bottleneck_distance",
                return_value=0.2,
            ):
                eval_result = evaluate(
                    result["checkpoint"],
                    num_samples=5,
                    output_dir=Path(tmpdir) / "eval",
                    device="cpu",
                )

            self.assertIn("metrics", eval_result)
            metrics_path = Path(tmpdir) / "eval" / "metrics.json"
            self.assertTrue(metrics_path.exists())


if __name__ == "__main__":
    unittest.main()
