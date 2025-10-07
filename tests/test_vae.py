import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

if torch is not None:
    from src.models.vae import MolecularVAE, vae_loss
else:  # pragma: no cover - skip when torch missing
    MolecularVAE = None  # type: ignore
    vae_loss = None  # type: ignore


@unittest.skipIf(torch is None, "PyTorch is required for VAE tests.")
class MolecularVAETest(unittest.TestCase):
    def test_forward_shapes_and_sampling(self) -> None:
        model = MolecularVAE(feature_shape=(4, 3), latent_dim=3, hidden_dims=(16, 8))
        batch = torch.randn(2, 4, 3)
        output = model(batch)
        self.assertEqual(output.reconstruction.shape, batch.shape)
        self.assertEqual(output.mean.shape, (2, 3))
        self.assertEqual(output.logvar.shape, (2, 3))

        samples = model.sample(5)
        self.assertEqual(samples.shape, (5, 4, 3))

    def test_constraint_loss_matches_targets(self) -> None:
        model = MolecularVAE(
            feature_shape=(4, 3),
            latent_dim=2,
            hidden_dims=(8,),
            enforce_constraints=True,
            bond_indices=[(0, 1)],
            target_bond_lengths=[1.0],
        )
        coords = torch.zeros((2, 4, 3))
        coords[:, 1, 0] = 1.0
        self.assertAlmostEqual(model.constraint_loss(coords).item(), 0.0, places=6)

        coords[:, 1, 0] = 2.0
        loss = model.constraint_loss(coords)
        self.assertGreater(loss.item(), 0.0)

    def test_vae_loss_outputs(self) -> None:
        model = MolecularVAE(feature_shape=(3, 3), latent_dim=2, hidden_dims=(8,))
        batch = torch.randn(4, 3, 3)
        output = model(batch)
        losses = vae_loss(output, batch, beta=0.5)
        self.assertIn("loss", losses)
        self.assertIn("reconstruction", losses)
        self.assertIn("kl", losses)
        losses["loss"].backward()
        grads = [param.grad for param in model.parameters()]
        self.assertTrue(any(g is not None for g in grads))

    def test_feature_space_inputs_supported(self) -> None:
        model = MolecularVAE(feature_shape=(2,), latent_dim=2, hidden_dims=(8,), geometry="dihedral", center_inputs=False)
        batch = torch.rand(10, 2) * 2 * torch.pi
        output = model(batch)
        self.assertEqual(output.reconstruction.shape, batch.shape)
        losses = vae_loss(output, batch, beta=1.0)
        losses["loss"].backward()


if __name__ == "__main__":
    unittest.main()
