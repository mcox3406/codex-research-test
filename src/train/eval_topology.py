"""Evaluate topological metrics for a trained molecular VAE."""
from __future__ import annotations

import argparse
import json
import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.vae import MolecularVAE
from src.tda.persistence import compute_persistence_diagrams
from src.tda.pd_metrics import bottleneck_distance, wasserstein_distance
from .train_vae import deep_update, load_config


def _prepare_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_dataset(config: Mapping[str, Any]) -> np.ndarray:
    dataset_cfg = config["dataset"]
    array = np.load(dataset_cfg["path"])
    return array


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[MolecularVAE, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model_cfg = config["model"]
    model = MolecularVAE(**model_cfg)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, config


def _compute_metrics(real_diagrams, gen_diagrams, homology_dims) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    batch = min(len(real_diagrams.diagrams), len(gen_diagrams.diagrams))
    for dim in homology_dims:
        wasserstein_values = []
        bottleneck_values = []
        for idx in range(batch):
            real_diag = real_diagrams.diagrams[idx].get(dim, np.empty((0, 2)))
            gen_diag = gen_diagrams.diagrams[idx].get(dim, np.empty((0, 2)))
            wasserstein_values.append(float(wasserstein_distance(real_diag, gen_diag)))
            bottleneck_values.append(float(bottleneck_distance(real_diag, gen_diag)))
        if wasserstein_values:
            metrics[f"wasserstein/h{dim}"] = float(np.mean(wasserstein_values))
            metrics[f"bottleneck/h{dim}"] = float(np.mean(bottleneck_values))
        else:
            metrics[f"wasserstein/h{dim}"] = 0.0
            metrics[f"bottleneck/h{dim}"] = 0.0
    return metrics


def _plot_diagrams(diagrams, title: str, output_path: Path) -> None:
    cols = len(diagrams.homology_dims)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    if cols == 1:
        axes = [axes]
    for ax, dim in zip(axes, diagrams.homology_dims):
        ax.set_title(f"H_{dim} diagram")
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        for diag in diagrams.by_dimension(dim):
            if diag.size == 0:
                continue
            ax.scatter(diag[:, 0], diag[:, 1], s=12, alpha=0.6)
        ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def evaluate(
    checkpoint: Path,
    *,
    config: Optional[Dict[str, Any]] = None,
    num_samples: int = 128,
    output_dir: Optional[Path] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint_config = _load_model(checkpoint, device_obj)
    merged_config = copy.deepcopy(checkpoint_config)
    if config is not None:
        deep_update(merged_config, config)

    array = _load_dataset(merged_config)
    limit = min(len(array), num_samples)
    real_batch = array[:limit]

    with torch.no_grad():
        samples = model.sample(limit, device=device_obj).cpu().numpy()

    homology_dims = tuple(merged_config.get("topology", {}).get("homology_dims", (0, 1)))
    max_edge = merged_config.get("topology", {}).get("max_edge_length")

    real_diagrams = compute_persistence_diagrams(real_batch, homology_dims=homology_dims, max_edge_length=max_edge)
    gen_diagrams = compute_persistence_diagrams(samples, homology_dims=homology_dims, max_edge_length=max_edge)

    metrics = _compute_metrics(real_diagrams, gen_diagrams, homology_dims)

    output_path = output_dir or Path("eval_outputs")
    _prepare_output_dir(output_path)
    (output_path / "figures").mkdir(exist_ok=True)

    json_path = output_path / "metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2))

    _plot_diagrams(real_diagrams, "Real persistence diagrams", output_path / "figures" / "real.png")
    _plot_diagrams(gen_diagrams, "Generated persistence diagrams", output_path / "figures" / "generated.png")

    return {
        "metrics": metrics,
        "real_diagrams": real_diagrams,
        "generated_diagrams": gen_diagrams,
        "output_dir": output_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description="Evaluate topology of generated samples.")
    parser.add_argument("checkpoint", type=str, help="Path to a trained model checkpoint.")
    parser.add_argument("--config", type=str, default=None, help="Optional override config file (YAML/JSON).")
    parser.add_argument("--num-samples", type=int, default=128, help="Number of samples to generate for evaluation.")
    parser.add_argument("--output-dir", type=str, default="eval_outputs", help="Directory to store outputs.")
    parser.add_argument("--device", type=str, default=None, help="Evaluation device (cpu or cuda).")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> Dict[str, Any]:  # pragma: no cover - CLI wrapper
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    override_config = load_config(args.config) if args.config else None
    result = evaluate(
        Path(args.checkpoint),
        config=override_config,
        num_samples=args.num_samples,
        output_dir=Path(args.output_dir),
        device=args.device,
    )
    return result


__all__ = ["evaluate", "main"]
