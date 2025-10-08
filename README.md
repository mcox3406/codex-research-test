# topo-gen

**Topological regularization for molecular generative models.**

Enforce that generated molecular conformations preserve the topological structure (loops, voids) of the true conformational landscape using persistent homology.

## Quick Start

```bash
# Setup
conda env create -f environment.yml
conda activate topo-gen

# Generate alanine dipeptide φ/ψ dataset (manual heavy job)
python scripts/phase1/run_openmm_alanine_dipeptide.py \
  --input-pdb data/templates/alanine_dipeptide.pdb \
  --output-dir data/alanine_dipeptide

# Validate topology in φ/ψ space (produces Ramachandran + PD plots)
python scripts/phase1/validate_alanine_dipeptide.py \
  --phi-psi data/alanine_dipeptide/phi_psi.npy \
  --output-dir results/alanine_dipeptide \
  --torus-metric sincos \
  --compare-geodesic

# Train baseline and topo-regularised VAEs
python src/train/train_vae.py --config configs/alanine_dipeptide_baseline.yaml
python src/train/train_vae.py --config configs/alanine_dipeptide_high_beta.yaml
python src/train/train_vae.py --config configs/alanine_dipeptide_topo.yaml

# Evaluate topology preservation
python src/train/eval_topology.py \
  checkpoints/alanine_dipeptide/topo/checkpoint_final.pt \
  --config configs/alanine_dipeptide_topo.yaml \
  --num-samples 2000 \
  --output-dir results/alanine_eval
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

> **Runtime note:** Running the preferred OpenMM workflow with five replicas of
> 10~ns each (50~ns total for the $\sim$60-atom cyclic hexapeptide) took roughly
> 50--55 minutes on commodity GPU hardware (equivalent to about 1340~ns/day).

### Alanine dipeptide (Ramachandran topology)

Because the cyclic hexapeptide dataset exhibited weak $H_1$ structure in
dihedral space, we provide an alanine dipeptide pipeline tailored for
Ramachandran analyses:

```bash
# 20 ns implicit-solvent MD at 500 K (produces 20,000 frames / 2D angles)
python scripts/phase1/run_openmm_alanine_dipeptide.py \
  --input-pdb data/templates/alanine_dipeptide.pdb \
  --output-dir data/alanine_dipeptide \
  --ns-total 20 \
  --save-interval-ps 1

# Validate φ/ψ coverage, persistence, and basin occupancies
python scripts/phase1/validate_alanine_dipeptide.py \
  --phi-psi data/alanine_dipeptide/phi_psi.npy \
  --output-dir results/alanine_dipeptide \
  --torus-metric sincos \
  --compare-geodesic

# Cache reference persistence diagrams for fast topo loss
python scripts/phase2/cache_reference_persistence.py \
  data/alanine_dipeptide/phi_psi.npy \
  results/alanine_dipeptide/reference_pd_sincos.npz \
  --geometry dihedral \
  --torus-metric sincos \
  --torus-harmonics 2 \
  --homology-dims 1 \
  --sample-size 4000 \
  --summary results/alanine_dipeptide/reference_pd_sincos.json

# Train/evaluate VAEs directly in φ/ψ space
python src/train/train_vae.py --config configs/alanine_dipeptide_baseline.yaml
python src/train/train_vae.py --config configs/alanine_dipeptide_high_beta.yaml
python src/train/train_vae.py --config configs/alanine_dipeptide_topo.yaml
```

The validation routine outputs a Ramachandran density, torus-aware persistence
diagrams (using a sine/cosine embedding of the torus by default), and occupancy
statistics for the canonical $\alpha_R$, $C7_{eq}$, and $\alpha_L$ basins. Use
`--compare-geodesic` to additionally contrast the wrapped geodesic metric when
diagnosing weak $H_1$ signals.

### Torus playground dataset

For rapid experimentation or debugging persistent homology settings without
running MD, generate a synthetic torus dataset with clear $H_1$ loops:

```bash
python scripts/playground/generate_dihedral_playground.py \
  --output-dir data/playground \
  --n-samples 4000 \
  --noise 0.08

# Inspect the resulting persistence diagram
python scripts/phase1/validate_alanine_dipeptide.py \
  --phi-psi data/playground/torus_loops.npy \
  --output-dir results/playground \
  --torus-metric sincos \
  --compare-geodesic
```

The playground workflow produces Ramachandran density plots and persistence
diagrams that exhibit at least one long-lived $H_1$ feature, providing a
lightweight regression target for topology-aware pipelines.

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
