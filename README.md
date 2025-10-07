# topo-gen

**Topological regularization for molecular generative models.**

Enforce that generated molecular conformations preserve the topological structure (loops, voids) of the true conformational landscape using persistent homology.

## Quick Start

```bash
# Setup
conda env create -f environment.yml
conda activate topo-gen

# Generate data (cyclic peptide MD → 5K conformations)
python src/data/generate_cyclopeptide.py --output data/cyclo_hexa_ala.npy

# Train baseline VAE
python src/train/train_vae.py --config configs/vae_base.yaml

# Train with topological regularization
python src/train/train_vae.py --config configs/topo_reg.yaml

# Evaluate topology preservation
python src/train/eval_topology.py --model checkpoints/topo_vae.pt --n_samples 1000
```

## Core Idea

**Problem:** Standard VAEs/diffusion models generate unrealistic conformations because they don't preserve the global topology of conformational space (e.g., loop structures from cyclic transitions).

**Solution:** Add topological loss $L_\text{topo} = W(PD_\text{real}, PD_\text{gen})$ where $W$ is Wasserstein distance between persistence diagrams.

**Result:** Generated conformations have correct topological features → better ensemble docking, realistic interpolations.

## Key Files

- `src/models/topo_loss.py` - Differentiable topological regularization
- `src/tda/persistence.py` - Compute persistence diagrams from conformations
- `configs/topo_reg.yaml` - Tune $\lambda$, homology dimensions, filtration scales

## Citation

```
[Paper coming soon]
```

See `agents.md` for mathematical background and `project_overview.md` for detailed motivation.
