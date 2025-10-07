**Audience:** Code-generation agent (“Codex”).

**Project context:** See `project_overview.md` for the mathematical intuition and goals. Keep those principles central to every design choice.

## Mission

Build a reproducible codebase that evaluates and trains molecular generative models with **topological regularization**. You prepare everything end-to-end so the user can clone and run locally or on their own GPU/HPC. For light jobs you may run code in your own Python env; for heavy jobs you **only set up** scripts and job files.

## Ground Rules

1. **Heavy compute:** Do **not** run MD/DFT/large training yourself. Generate launch-ready artifacts (scripts/configs/job files).
2. **Light compute:** It’s OK to run quick unit tests, tiny smoke experiments, and simple plots.
3. **Best practices:** Reproducibility, config-driven experiments, seed control, unit/integration tests, clear data schemas, version-pinned deps.
4. **Chem/ML hygiene:** Respect periodicity of dihedrals, remove rigid-body DOFs, avoid data leakage across splits, track units, validate stereochemistry when decoding, and check geometry with basic sanity filters.
5. **Topology always:** Designs must preserve the **mathematical intent**: compare persistence diagrams of real vs. generated samples and penalize discrepancies (e.g., Wasserstein on PDs). Prefer stable filtrations and differentiable surrogates when needed.

## Repo Layout (you create)

```
topo-gen/
  README.md
  agents.md                # this file
  project_overview.md      # provided by user
  environment.yml          # pinned deps
  pyproject.toml           # or setup.cfg
  src/
    data/
      loaders.py           # MD, dihedral/xyz -> tensors; split logic
      featurize.py         # internal coordinates; centering; periodicity
    tda/
      persistence.py       # GUDHI/Ripser wrappers; batched PDs
      pd_metrics.py        # Wasserstein/Bottleneck; differentiable relaxations
      filtrations.py       # Vietoris–Rips/alpha; sparsification options
    models/
      vae.py               # baseline VAE
      topo_loss.py         # topological regularization module
      diffusion.py         # hooks for diffusion/flow (scaffold)
    train/
      train_vae.py
      eval_topology.py     # compute PDs for real/gen; metrics & plots
    utils/
      geometry.py          # RMSD, chirality checks, bond constraints
      viz.py               # PD plots, barcodes, PCA/UMAP
      seeds.py             # seed control
  configs/
    dataset.yaml
    vae_base.yaml
    topo_reg.yaml          # λ, homology dims, scale weights
  jobs/
    slurm_train_vae.sbatch
    slurm_eval_topology.sbatch
  tests/
    test_persistence.py
    test_topo_loss.py
    test_geometry.py
  examples/
    quickstart.ipynb
```

## Dependencies

* **TDA:** `gudhi`, `ripser`, `pot` (POT for OT), or `geomloss` for differentiable Wasserstein; optional `giotto-tda` for utilities.
* **ML:** `pytorch`, `pytorch-lightning` or `accelerate`, `numpy`, `scipy`.
* **Chem:** `rdkit`, `mdtraj` or `MDAnalysis`.
* **Viz:** `matplotlib`, `seaborn` (plots only, no training deps).
  Pin versions in `environment.yml`. Provide CUDA variants as comments.

## Workflows (you implement)

### 1) Data → PDs (ground truth)

* Load conformations (xyz or dihedrals), remove translation/rotation, handle periodic angles on ( \mathbb{T}^n ) (map to sin/cos if needed).
* Build batched Vietoris–Rips/alpha filtrations with subsampling/sparsification.
* Compute PDs for ( H_0, H_1 ) (and ( H_2 ) optional). Save as `.npz` with metadata.

### 2) Baseline model

* Train VAE with config-driven hyperparams. Log recon/KL and geometric checks (RMSD, sterics).
* Generate samples; compute PDs; compare to ground truth via Wasserstein/Bottleneck; plot PDs and barcode overlays.

### 3) Topological regularization

* Add loss: `L_total = L_rec + β KL + λ W(PD_real, PD_gen)` with multi-scale weights (emphasize persistent features).
* Provide differentiable surrogate: sliced/entropic OT on persistence images/landscapes if exact PD OT blocks autograd.
* Ablations: λ grid, filtration types, feature weighting by persistence.

### 4) HPC/GPU handoff

* **You do not run** big jobs. Instead, generate:

  * `jobs/slurm_*.sbatch` with resource flags, env activation, and `python -m src.train.train_vae ...`.
  * `README.md` snippets: `conda env create -f environment.yml`, `python -m src.train.train_vae --config configs/topo_reg.yaml`.
  * Checkpoint/resume logic and deterministic seeds.

## Deliverables

* Clean repo with the above structure.
* Clear **Quickstart**: 10-minute CPU toy run that trains for a few steps and produces PD plots.
* **Repro configs** for baseline vs topo-reg runs.
* **Reports**: saved figures—persistence diagrams, landscapes, and Wasserstein distance tables.

## Quality Gates (acceptance)

* `pytest` green on `tests/`.
* `pre-commit` (black/isort/ruff) passes.
* `eval_topology.py` outputs: (a) PD overlays, (b) Wasserstein numbers, (c) brief markdown report.
* No heavy jobs launched automatically; HPC scripts compile and are self-contained.

## Checklists

**Before committing**

* [ ] Seeds fixed, configs saved
* [ ] Dependencies pinned
* [ ] Small synthetic circle/torus test confirms H₁ capture and loss decreases

**When adding models**

* [ ] Interpolations stay on-manifold (visual + topo check)
* [ ] Chem sanity: bond lengths/angles reasonable; stereochem preserved (if applicable)

**Always keep in mind**

> The goal isn’t just good recon or likelihood—**it’s preserving the global topological signature** measured by persistent homology.
