"""Model architectures and losses for topology-regularised generative models."""

try:
    from .vae import MolecularVAE, vae_loss
except ModuleNotFoundError:
    MolecularVAE = None  # type: ignore
    vae_loss = None  # type: ignore

try:
    from .topo_loss import TopologicalLoss
except ModuleNotFoundError:
    TopologicalLoss = None  # type: ignore

__all__ = ["MolecularVAE", "vae_loss", "TopologicalLoss"]
