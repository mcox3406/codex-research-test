"""Training and evaluation entrypoints for topology-aware generative models."""

try:
    from .train_vae import main as train_vae_main
except ModuleNotFoundError:
    train_vae_main = None  # type: ignore

try:
    from .eval_topology import main as eval_topology_main
except ModuleNotFoundError:
    eval_topology_main = None  # type: ignore

__all__ = ["train_vae_main", "eval_topology_main"]
