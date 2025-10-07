"""OpenMM workflow for generating cyclic hexapeptide conformations (Phase 1).

This script is provided as an executable recipe; it is **not executed
automatically** by the repository.  Run it on your workstation or cluster to
produce the ~5,000 conformations required for the Phase 1 analyses.

Key features
------------
- Supports both cyclo-(Ala)_6 and cyclo-(Gly)_6 via an input PDB file.
- Launches multiple replicas sequentially (or in parallel when combined with a
  job scheduler) to gather diverse conformations.
- Uses an implicit-solvent GBSA model at 300 K with a 2 fs timestep.
- Saves aligned Cartesian coordinates into a consolidated NumPy archive ready for
  downstream topology computations.

Example command
---------------
python scripts/phase1/run_openmm_cyclohexapeptide.py \
    --input-pdb data/templates/cyclo_ala6_initial.pdb \
    --output-npy data/cyclo_ala6_5000.npy

Input structure preparation
---------------------------
You can build a starting cyclic peptide using external tools such as:
- `tleap` (AMBER Tools): load a linear peptide and manually bond the termini.
- `rdkit` or `pyrosetta` cyclization scripts.
- Download a crystallographic structure (e.g., PDB ID 1P79 for cyclic hexapeptides)
  and trim to the desired sequence.

Dependencies
------------
- openmm >= 8.0
- mdtraj
- numpy, tqdm

The script assumes OpenMM and MDTraj are installed in the active environment but
performs no imports from optional GPU-only packages beyond OpenMM's CUDA
platform detection.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile
from typing import Iterable, List

import mdtraj as md
import numpy as np
from openmm import LangevinIntegrator, Platform
from openmm.app import Modeller, PDBFile, Simulation
from openmm.app import ForceField, NoCutoff
from openmm.unit import kelvin, picosecond, femtosecond

_DEFAULT_TEMPERATURE = 300.0  # Kelvin
_DEFAULT_TIMESTEP = 2.0 * femtosecond
_DEFAULT_FRICTION = 1.0 / picosecond
_DEFAULT_NS_PER_REPLICA = 10.0
_DEFAULT_SAVE_EVERY_PS = 10.0
_DEFAULT_FORCEFIELD = ("amber99sbildn.xml", "amber99_obc.xml")


def _load_modeller(pdb_path: pathlib.Path) -> Modeller:
    """Load the input PDB file into an OpenMM `Modeller` instance.

    For cyclic peptides, manually adds the bond between the C-terminus
    and N-terminus to close the ring.
    """

    pdb = PDBFile(str(pdb_path))
    modeller = Modeller(pdb.topology, pdb.positions)

    # Add cyclic bond between last residue's C and first residue's N
    residues = list(modeller.topology.residues())
    if len(residues) > 1:
        first_res = residues[0]
        last_res = residues[-1]

        # Find the C atom in the last residue
        c_atom = None
        for atom in last_res.atoms():
            if atom.name == 'C':
                c_atom = atom
                break

        # Find the N atom in the first residue
        n_atom = None
        for atom in first_res.atoms():
            if atom.name == 'N':
                n_atom = atom
                break

        # Add the bond if both atoms were found
        if c_atom is not None and n_atom is not None:
            modeller.topology.addBond(c_atom, n_atom)

    return modeller


def _configure_simulation(modeller: Modeller, temperature: float) -> Simulation:
    """Create an implicit-solvent OpenMM `Simulation` from the modeller."""

    ff = ForceField(*_DEFAULT_FORCEFIELD)
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=NoCutoff,
        constraints=None,
        removeCMMotion=True,
    )

    integrator = LangevinIntegrator(temperature * kelvin, _DEFAULT_FRICTION, _DEFAULT_TIMESTEP)

    platform = None
    for candidate in ("CUDA", "OpenCL", "CPU", "Reference"):
        try:
            platform = Platform.getPlatformByName(candidate)
            break
        except Exception:  # noqa: BLE001
            continue
    if platform is None:
        raise RuntimeError("No compatible OpenMM platform detected. Check your installation.")

    simulation = Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    return simulation


def _production_steps(ns_total: float) -> int:
    return int((ns_total * 1_000_000.0) / (_DEFAULT_TIMESTEP.value_in_unit(femtosecond)))


def _report_interval(save_interval_ps: float) -> int:
    return int((save_interval_ps * 1000.0) / (_DEFAULT_TIMESTEP.value_in_unit(femtosecond)))


def _align_trajectory(traj: md.Trajectory) -> md.Trajectory:
    traj.center_coordinates()
    traj.superpose(traj, frame=0)
    return traj


def _load_and_align(dcd_paths: Iterable[pathlib.Path], pdb_path: pathlib.Path) -> np.ndarray:
    frames: List[np.ndarray] = []
    for dcd in dcd_paths:
        traj = md.load_dcd(str(dcd), top=str(pdb_path))
        traj = _align_trajectory(traj)
        frames.append(traj.xyz)
    return np.concatenate(frames, axis=0)


def generate_conformations(
    input_pdb: pathlib.Path,
    output_npy: pathlib.Path,
    replicas: int,
    ns_per_replica: float,
    temperature: float,
    save_interval_ps: float,
) -> None:
    """Run multiple replicas and save aligned Cartesian coordinates."""

    modeller = _load_modeller(input_pdb)
    n_steps = _production_steps(ns_per_replica)
    report_interval = _report_interval(save_interval_ps)

    all_dcds: List[pathlib.Path] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = pathlib.Path(tmpdir)
        for replica in range(replicas):
            print(f"Starting replica {replica + 1}/{replicas} ({ns_per_replica} ns)...", flush=True)
            simulation = _configure_simulation(modeller, temperature)
            dcd_path = tmpdir_path / f"replica_{replica:02d}.dcd"
            log_path = tmpdir_path / f"replica_{replica:02d}.log"

            from openmm.app import DCDReporter, StateDataReporter

            simulation.reporters.append(DCDReporter(str(dcd_path), report_interval))
            simulation.reporters.append(
                StateDataReporter(
                    str(log_path),
                    report_interval,
                    step=True,
                    potentialEnergy=True,
                    temperature=True,
                    progress=True,
                    elapsedTime=True,
                    totalSteps=n_steps,
                )
            )
            simulation.reporters.append(
                StateDataReporter(
                    sys.stdout,
                    report_interval,
                    step=True,
                    progress=True,
                    remainingTime=True,
                    speed=True,
                    totalSteps=n_steps,
                )
            )

            simulation.step(n_steps)
            all_dcds.append(dcd_path)

        coords = _load_and_align(all_dcds, input_pdb)

    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, coords)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cyclic hexapeptide conformations with OpenMM")
    parser.add_argument("--input-pdb", type=pathlib.Path, required=True, help="Path to initial cyclic peptide PDB file")
    parser.add_argument("--output-npy", type=pathlib.Path, required=True, help="Destination for aligned NumPy array")
    parser.add_argument("--replicas", type=int, default=5, help="Number of MD replicas to run (default: 5)")
    parser.add_argument(
        "--ns-per-replica",
        type=float,
        default=_DEFAULT_NS_PER_REPLICA,
        help="Nanoseconds simulated per replica (default: 10 ns)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=_DEFAULT_TEMPERATURE,
        help="Simulation temperature in Kelvin (default: 300)",
    )
    parser.add_argument(
        "--save-interval-ps",
        type=float,
        default=_DEFAULT_SAVE_EVERY_PS,
        help="Stride between saved frames in picoseconds (default: 10 ps)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_conformations(
        input_pdb=args.input_pdb,
        output_npy=args.output_npy,
        replicas=args.replicas,
        ns_per_replica=args.ns_per_replica,
        temperature=args.temperature,
        save_interval_ps=args.save_interval_ps,
    )


if __name__ == "__main__":
    main()
