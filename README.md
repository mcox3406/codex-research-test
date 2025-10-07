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

## Phase 1: Data generation & validation

The Phase 1 workflow prepares the cyclic hexapeptide dataset used throughout the
paper. Run these commands manually (no heavy jobs are executed by default):

```bash
# Preferred: implicit-solvent OpenMM MD (5 × 10 ns)
python scripts/phase1/run_openmm_cyclohexapeptide.py \
  --input-pdb data/templates/cyclo_ala6_initial.pdb \
  --output-npy data/cyclo_ala6_md.npy

# Fallback: geometric Monte Carlo sampling
python scripts/phase1/sample_cyclohexapeptide_mc.py \
  --template-pdb data/templates/cyclo_ala6_initial.pdb \
  --output data/cyclo_ala6_mc.npy

# Validate whichever dataset you generated
python scripts/phase1/validate_dataset.py \
  --dataset data/cyclo_ala6_md.npy \
  --output-dir results/phase1
```

The validation script produces persistence diagrams, barcodes, RMSD/Rg
histograms, PCA projections, and a JSON summary that can be inserted directly
into Section 4 of the manuscript.

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
