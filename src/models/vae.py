"""Variational auto-encoder tailored for molecular conformations."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class VAEOutput:
    """Bundle the outputs of a forward VAE pass."""

    reconstruction: Tensor
    mean: Tensor
    logvar: Tensor


def _build_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    current = input_dim
    for width in hidden_dims:
        layers.append(nn.Linear(current, width))
        layers.append(nn.ReLU())
        current = width
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class MolecularVAE(nn.Module):
    """Simple coordinate-space VAE with optional geometric constraints."""

    def __init__(
        self,
        n_atoms: int,
        latent_dim: int = 8,
        hidden_dims: Sequence[int] = (256, 128, 64),
        enforce_constraints: bool = False,
        bond_indices: Optional[Sequence[Tuple[int, int]]] = None,
        target_bond_lengths: Optional[Sequence[float]] = None,
        coordinate_scaling: float = 1.0,
    ) -> None:
        super().__init__()
        if n_atoms <= 0:
            raise ValueError("n_atoms must be positive.")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        self.n_atoms = int(n_atoms)
        self.latent_dim = int(latent_dim)
        self.input_dim = self.n_atoms * 3
        self.enforce_constraints = enforce_constraints
        self.coordinate_scaling = float(coordinate_scaling)

        self.encoder = _build_mlp(self.input_dim, hidden_dims, 2 * self.latent_dim)
        decoder_hidden = tuple(reversed(tuple(hidden_dims)))
        self.decoder = _build_mlp(self.latent_dim, decoder_hidden, self.input_dim)

        if enforce_constraints:
            if bond_indices is None or target_bond_lengths is None:
                raise ValueError("Constraint enforcement requires bond indices and target lengths.")
            if len(bond_indices) != len(target_bond_lengths):
                raise ValueError("bond_indices and target_bond_lengths must have equal length.")
            self.register_buffer(
                "bond_indices",
                torch.tensor(bond_indices, dtype=torch.long),
            )
            self.register_buffer(
                "target_bond_lengths",
                torch.tensor(target_bond_lengths, dtype=torch.get_default_dtype()),
            )
        else:
            self.register_buffer("bond_indices", torch.empty((0, 2), dtype=torch.long))
            self.register_buffer("target_bond_lengths", torch.empty((0,), dtype=torch.get_default_dtype()))

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        flat = self._flatten(x)
        encoded = self.encoder(flat)
        mu, logvar = encoded[:, : self.latent_dim], encoded[:, self.latent_dim :]
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        flat = self.decoder(z)
        coords = flat.view(z.shape[0], self.n_atoms, 3)
        if not math.isclose(self.coordinate_scaling, 1.0, rel_tol=1e-6, abs_tol=1e-9):
            coords = coords * self.coordinate_scaling
        return coords

    def forward(self, x: Tensor) -> VAEOutput:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return VAEOutput(reconstruction=recon, mean=mu, logvar=logvar)

    def sample(self, num_samples: int, device: Optional[torch.device] = None) -> Tensor:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        device = device or next(self.parameters()).device
        z = torch.randn((num_samples, self.latent_dim), device=device)
        with torch.no_grad():
            samples = self.decode(z)
        return samples

    def constraint_loss(self, coords: Tensor) -> Tensor:
        if not self.enforce_constraints or self.bond_indices.numel() == 0:
            return coords.new_zeros(())
        pi = coords[:, self.bond_indices[:, 0]]
        pj = coords[:, self.bond_indices[:, 1]]
        distances = torch.linalg.norm(pi - pj, dim=-1)
        target = self.target_bond_lengths.unsqueeze(0).expand_as(distances)
        loss = F.mse_loss(distances, target)
        return loss

    def _flatten(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[1] != self.n_atoms or x.shape[2] != 3:
            raise ValueError(
                f"Input must have shape (batch, {self.n_atoms}, 3); received {tuple(x.shape)}."
            )
        centered = x - x.mean(dim=1, keepdim=True)
        return centered.view(x.shape[0], -1)


def vae_loss(
    output: VAEOutput,
    inputs: Tensor,
    *,
    beta: float = 1.0,
    topo_term: Optional[Tensor] = None,
    topo_weight: float = 1.0,
    constraint_term: Optional[Tensor] = None,
    constraint_weight: float = 1.0,
    reduction: str = "mean",
) -> Dict[str, Tensor]:
    """Assemble the VAE objective and optional regularisers."""

    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'.")

    if reduction == "mean":
        recon = F.mse_loss(output.reconstruction, inputs, reduction="mean")
    else:
        recon = F.mse_loss(output.reconstruction, inputs, reduction="sum")

    kl = -0.5 * torch.sum(1 + output.logvar - output.mean.pow(2) - output.logvar.exp(), dim=1)
    if reduction == "mean":
        kl = kl.mean()
    else:
        kl = kl.sum()

    total = recon + beta * kl

    topo = torch.zeros_like(total)
    if topo_term is not None:
        topo = topo_term if topo_term.ndim == 0 else topo_term.squeeze()
        total = total + topo_weight * topo

    constraint = torch.zeros_like(total)
    if constraint_term is not None:
        constraint = constraint_term if constraint_term.ndim == 0 else constraint_term.squeeze()
        total = total + constraint_weight * constraint

    return {
        "loss": total,
        "reconstruction": recon.detach(),
        "kl": kl.detach(),
        "topology": topo.detach(),
        "constraint": constraint.detach(),
    }


__all__ = ["MolecularVAE", "VAEOutput", "vae_loss"]
