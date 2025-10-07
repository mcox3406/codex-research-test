# Phase 1: Data generation and validation

This directory contains **launch-ready scripts** for the Phase 1 workflow
outlined in the project brief. None of these scripts are executed
automatically—run them manually on suitable hardware.

## 1. Molecular dynamics (preferred)

```
# 1) Prepare an initial cyclic hexapeptide PDB (e.g., cyclo-(Ala)_6)
# 2) Run the OpenMM driver
python scripts/phase1/run_openmm_cyclohexapeptide.py \
    --input-pdb data/templates/cyclo_ala6_initial.pdb \
    --output-npy data/cyclo_ala6_md.npy \
    --replicas 5 \
    --ns-per-replica 10 \
    --save-interval-ps 10
```

The script writes per-replica DCD + log files into a temporary directory and a
consolidated, rigid-body-aligned NumPy archive at `--output-npy` with shape
`(n_frames, n_atoms, 3)`.

## 2. Geometric sampling fallback

```
python scripts/phase1/sample_cyclohexapeptide_mc.py \
    --template-pdb data/templates/cyclo_ala6_initial.pdb \
    --output data/cyclo_ala6_mc.npy \
    --seed 0
```

Optional overrides can be provided via `--config config/mc_overrides.json`
containing any subset of the fields in `MonteCarloConfig`.

## 3. Dataset validation

After generating a dataset (either MD or MC), run:

```
python scripts/phase1/validate_dataset.py \
    --dataset data/cyclo_ala6_md.npy \
    --topology data/templates/cyclo_ala6_initial.pdb \
    --output-dir results/phase1
```

This produces the figures and summary statistics required for Figure 1 and the
paper narrative. The persistence calculations now operate on backbone dihedral
angles (φ/ψ) to capture cyclic transitions that are invisible in Cartesian
space:

- `ground_truth_pd.png` / `ground_truth_barcode.png`
- `rmsd_distribution.png`, `rg_distribution.png`
- `pca_projection.png`
- `dataset_summary.json`

All outputs land in `results/phase1/` by default.

## Tips

- Install dependencies via `conda env create -f environment.yml` before running.
- For HPC clusters, wrap the MD script in an sbatch file; the script is
  thread-safe and can be parallelised across replicas.
- The validation script requires the same topology (PDB) used for data
  generation so it can compute φ/ψ dihedrals prior to persistence analysis.
- Validate a small subset first using `--max-samples 500` when running the
  validation script.
