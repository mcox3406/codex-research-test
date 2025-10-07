"""Training loop for the molecular VAE with optional topological regularisation."""
from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from src.models.topo_loss import TopologicalLoss
from src.models.vae import MolecularVAE, vae_loss

try:  # Optional YAML support for configs
    import yaml
except Exception:  # pragma: no cover - yaml is optional
    yaml = None  # type: ignore

try:  # Optional TensorBoard logging
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - TensorBoard is optional
    SummaryWriter = None  # type: ignore

try:  # Optional Weights & Biases logging
    import wandb
except Exception:  # pragma: no cover - WandB is optional
    wandb = None  # type: ignore


DEFAULT_CONFIG: Dict[str, Any] = {
    "dataset": {
        "path": "data/synthetic_hexapeptide.npy",
        "train_fraction": 0.7,
        "val_fraction": 0.15,
        "test_fraction": 0.15,
        "seed": 123,
    },
    "model": {
        "latent_dim": 8,
        "hidden_dims": [256, 128, 64],
        "enforce_constraints": False,
        "coordinate_scaling": 1.0,
    },
    "training": {
        "epochs": 5,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "beta": 1.0,
        "max_steps": None,
        "log_every": 10,
        "checkpoint_every": 100,
        "output_dir": "runs/vae",
        "seed": 42,
        "device": "auto",
    },
    "topology": {
        "enabled": True,
        "weight": 1.0,
        "homology_dims": [1],
        "max_edge_length": 15.0,
        "update_every": 10,
        "backend": "auto",
    },
    "logging": {
        "backend": "tensorboard",
        "project": "topo-vae",
        "entity": None,
    },
}


@dataclass
class TrainState:
    epoch: int
    step: int
    best_loss: float


class ConformationDataset(Dataset[Tensor]):
    """Simple dataset wrapping a numpy array of conformations."""

    def __init__(self, array: np.ndarray) -> None:
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(
                "Conformations must have shape (n_samples, n_atoms, 3)."
            )
        self.data = torch.as_tensor(array, dtype=torch.get_default_dtype())

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.data.shape[0]

    def __getitem__(self, index: int) -> Tensor:
        return self.data[index]


class TrainingLogger:
    def __init__(self, backend: str, log_dir: Path, project: Optional[str] = None, entity: Optional[str] = None) -> None:
        self.backend = backend
        self.project = project
        self.entity = entity
        self.writer = None
        if backend == "tensorboard" and SummaryWriter is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))
        elif backend == "wandb" and wandb is not None:
            wandb.init(project=project, entity=entity, dir=str(log_dir))
        else:
            self.backend = "stdout"

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        if self.backend == "tensorboard" and self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, global_step=step)
        elif self.backend == "wandb" and wandb is not None:
            wandb.log({**metrics, "step": step})
        else:
            formatted = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"[step {step}] {formatted}")

    def close(self) -> None:  # pragma: no cover - trivial
        if self.backend == "tensorboard" and self.writer is not None:
            self.writer.flush()
            self.writer.close()
        elif self.backend == "wandb" and wandb is not None:
            wandb.finish()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - optional GPU path
        torch.cuda.manual_seed_all(seed)


def deep_update(base: MutableMapping[str, Any], new: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in new.items():
        if key in base and isinstance(base[key], MutableMapping) and isinstance(value, Mapping):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Optional[str]) -> Dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is None:
        return config
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"Config file {path} does not exist.")
    content = file.read_text()
    if file.suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML configs.")
        parsed = yaml.safe_load(content) or {}
    else:
        parsed = json.loads(content)
    if not isinstance(parsed, Mapping):
        raise ValueError("Configuration file must define a mapping.")
    deep_update(config, parsed)
    return config


def _split_dataset(dataset: ConformationDataset, train_fraction: float, val_fraction: float, seed: int) -> Tuple[Subset[Tensor], Subset[Tensor], Subset[Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    n_total = len(dataset)
    n_train = int(n_total * train_fraction)
    n_val = int(n_total * val_fraction)
    n_test = max(n_total - n_train - n_val, 0)
    return torch.utils.data.random_split(dataset, [n_train, n_val, n_test], generator=generator)


def _prepare_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader[Tensor], Optional[DataLoader[Tensor]], Optional[DataLoader[Tensor]], int]:
    dataset_cfg = config["dataset"]
    path = Path(dataset_cfg["path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Generate it with data/generate_synthetic_hexapeptide.py."
        )
    array = np.load(path)
    dataset = ConformationDataset(array)
    train_set, val_set, test_set = _split_dataset(
        dataset,
        train_fraction=float(dataset_cfg.get("train_fraction", 0.7)),
        val_fraction=float(dataset_cfg.get("val_fraction", 0.15)),
        seed=int(dataset_cfg.get("seed", 123)),
    )
    batch_size = int(config["training"].get("batch_size", 64))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False) if len(val_set) > 0 else None
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False) if len(test_set) > 0 else None
    n_atoms = dataset.data.shape[1]
    return train_loader, val_loader, test_loader, n_atoms


def _create_model(config: Dict[str, Any], n_atoms: int) -> MolecularVAE:
    model_cfg = copy.deepcopy(config["model"])
    model_cfg["n_atoms"] = n_atoms
    return MolecularVAE(**model_cfg)


def _create_topological_loss(config: Dict[str, Any]) -> Optional[TopologicalLoss]:
    topo_cfg = config.get("topology", {})
    if not topo_cfg.get("enabled", False):
        return None
    return TopologicalLoss(
        homology_dims=tuple(topo_cfg.get("homology_dims", (1,))),
        max_edge_length=topo_cfg.get("max_edge_length"),
        backend=topo_cfg.get("backend", "auto"),
        update_every=int(topo_cfg.get("update_every", 1)),
    )


def _device_from_config(config: Dict[str, Any]) -> torch.device:
    training_cfg = config["training"]
    requested = training_cfg.get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _save_checkpoint(
    directory: Path,
    model: MolecularVAE,
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    state: TrainState,
    tag: str,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
        "epoch": state.epoch,
        "step": state.step,
        "best_loss": state.best_loss,
    }
    path = directory / f"checkpoint_{tag}.pt"
    torch.save(checkpoint, path)
    return path


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint {path} does not exist.")
    return torch.load(path, map_location=device)


def train(config: Dict[str, Any], *, resume: Optional[Path] = None) -> Dict[str, Any]:
    training_cfg = config["training"]
    set_seed(int(training_cfg.get("seed", 42)))

    device = _device_from_config(config)

    train_loader, val_loader, _, n_atoms = _prepare_dataloaders(config)
    config["model"]["n_atoms"] = n_atoms

    model = _create_model(config, n_atoms).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )

    state = TrainState(epoch=0, step=0, best_loss=float("inf"))

    if resume is not None:
        checkpoint = _load_checkpoint(resume, device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        config = checkpoint.get("config", config)
        training_cfg = config["training"]
        state = TrainState(
            epoch=int(checkpoint.get("epoch", 0)),
            step=int(checkpoint.get("step", 0)),
            best_loss=float(checkpoint.get("best_loss", float("inf"))),
        )

    topo_loss_module = _create_topological_loss(config)
    topo_weight = float(config.get("topology", {}).get("weight", 1.0))
    beta = float(training_cfg.get("beta", 1.0))
    max_steps = training_cfg.get("max_steps")
    log_every = int(training_cfg.get("log_every", 10))
    checkpoint_every = int(training_cfg.get("checkpoint_every", 100))

    output_dir = Path(training_cfg.get("output_dir", "runs/vae"))
    checkpoints_dir = output_dir / "checkpoints"
    logger = TrainingLogger(
        backend=config.get("logging", {}).get("backend", "tensorboard"),
        log_dir=output_dir,
        project=config.get("logging", {}).get("project"),
        entity=config.get("logging", {}).get("entity"),
    )

    history: Dict[str, list[float]] = {"reconstruction": [], "kl": [], "topology": []}

    try:
        for epoch in range(state.epoch, int(training_cfg.get("epochs", 1))):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                output = model(batch)
                constraint_term = model.constraint_loss(output.reconstruction)
                topo_term = None
                if topo_loss_module is not None:
                    topo_term = topo_loss_module(batch.detach(), output.reconstruction.detach(), step=state.step)
                losses = vae_loss(
                    output,
                    batch,
                    beta=beta,
                    topo_term=topo_term,
                    topo_weight=topo_weight,
                    constraint_term=constraint_term,
                )
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                state.step += 1

                total_loss_value = float(losses["loss"].detach().cpu())
                history["reconstruction"].append(float(losses["reconstruction"].detach().cpu()))
                history["kl"].append(float(losses["kl"].detach().cpu()))
                history["topology"].append(float(losses["topology"].detach().cpu()))
                state.best_loss = min(state.best_loss, total_loss_value)

                if state.step % log_every == 0:
                    logger.log(
                        {
                            "loss/reconstruction": float(losses["reconstruction"].detach().cpu()),
                            "loss/kl": float(losses["kl"].detach().cpu()),
                            "loss/topology": float(losses["topology"].detach().cpu()),
                            "loss/total": total_loss_value,
                        },
                        step=state.step,
                    )

                if state.step % checkpoint_every == 0:
                    _save_checkpoint(checkpoints_dir, model, optimizer, config, state, tag=str(state.step))

                if max_steps is not None and state.step >= int(max_steps):
                    raise StopIteration

            state.epoch = epoch + 1
            _save_checkpoint(checkpoints_dir, model, optimizer, config, state, tag=f"epoch{state.epoch}")
    except StopIteration:
        pass
    finally:
        logger.close()

    final_checkpoint = _save_checkpoint(checkpoints_dir, model, optimizer, config, state, tag="final")
    return {
        "model": model,
        "optimizer": optimizer,
        "config": config,
        "state": state,
        "history": history,
        "checkpoint": final_checkpoint,
    }


def build_arg_parser() -> argparse.ArgumentParser:  # pragma: no cover - thin wrapper
    parser = argparse.ArgumentParser(description="Train the molecular VAE with topological regularisation.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML or JSON config file.")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from a checkpoint.")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> Dict[str, Any]:  # pragma: no cover - CLI wrapper
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    resume_path = Path(args.resume) if args.resume else None
    result = train(config, resume=resume_path)
    return result


__all__ = [
    "ConformationDataset",
    "TrainingLogger",
    "TrainState",
    "DEFAULT_CONFIG",
    "load_config",
    "train",
    "main",
]
