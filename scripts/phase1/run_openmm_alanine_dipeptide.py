"""Generate alanine dipeptide conformations via high-temperature implicit-solvent MD.

The script mirrors the cyclic-hexapeptide workflow but targets alanine dipeptide,
a canonical two-dihedral system that exhibits rich Ramachandran topology.
It performs a single long trajectory (default 20 ns) at 500 K using OpenMM's
Amber force fields and extracts both Cartesian coordinates and (phi, psi)
dihedral angles at a 1 ps interval (~20,000 frames).

Example usage
-------------
python scripts/phase1/run_openmm_alanine_dipeptide.py \
    --input-pdb data/templates/alanine_dipeptide.pdb \
    --output-dir data/alanine_dipeptide \
    --ns-total 20 \
    --temperature 500 \
    --save-interval-ps 1

Outputs
-------
* ``output_dir/coordinates.npy``: Aligned Cartesian coordinates with shape
  ``(n_frames, n_atoms, 3)``.
* ``output_dir/phi_psi.npy``: Wrapped dihedral angles with shape ``(n_frames, 2)``
  in radians on ``[0, 2π)``.
* ``output_dir/metadata.json``: Simulation metadata and provenance, including the
  mdtraj topology hash used for reproducibility.

The script **does not run automatically**. Invoke it manually on a machine with
OpenMM, MDTraj, and an appropriate GPU/CPU setup. The heavy simulation is kept
configurable via CLI flags so it can be submitted to HPC job schedulers.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile
from typing import Iterable, Tuple

import mdtraj as md
import numpy as np
from openmm import LangevinIntegrator, Platform
from openmm.app import DCDReporter, PDBFile, Simulation, StateDataReporter
from openmm.app import ForceField, NoCutoff
from openmm.unit import femtosecond, kelvin, picosecond

_DEFAULT_TIMESTEP = 2.0 * femtosecond
_DEFAULT_FRICTION = 1.0 / picosecond
_DEFAULT_FORCEFIELD = ("amber99sbildn.xml", "amber99_obc.xml")


def _production_steps(ns_total: float) -> int:
    return int((ns_total * 1_000_000.0) / (_DEFAULT_TIMESTEP.value_in_unit(femtosecond)))


def _report_interval(save_interval_ps: float) -> int:
    return int((save_interval_ps * 1000.0) / (_DEFAULT_TIMESTEP.value_in_unit(femtosecond)))


def _load_simulation(pdb_path: pathlib.Path, temperature: float) -> Tuple[Simulation, md.Topology]:
    pdb = PDBFile(str(pdb_path))
    forcefield = ForceField(*_DEFAULT_FORCEFIELD)
    system = forcefield.createSystem(
        pdb.topology,
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
        raise RuntimeError("No compatible OpenMM platform detected.")

    simulation = Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    return simulation, pdb.topology


def _align_frames(dcd_files: Iterable[pathlib.Path], topology: pathlib.Path) -> md.Trajectory:
    frames = []
    for path in dcd_files:
        traj = md.load_dcd(str(path), top=str(topology))
        traj.center_coordinates()
        traj.superpose(traj, frame=0)
        frames.append(traj)
    return frames[0].join(frames[1:]) if len(frames) > 1 else frames[0]


def _wrap_angles(phi: np.ndarray, psi: np.ndarray) -> np.ndarray:
    two_pi = 2.0 * np.pi
    phi_wrapped = np.mod(phi + two_pi, two_pi)
    psi_wrapped = np.mod(psi + two_pi, two_pi)
    return np.stack([phi_wrapped, psi_wrapped], axis=1)


def run_simulation(
    input_pdb: pathlib.Path,
    output_dir: pathlib.Path,
    temperature: float,
    ns_total: float,
    save_interval_ps: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    simulation, topology = _load_simulation(input_pdb, temperature)

    total_steps = _production_steps(ns_total)
    report_interval = _report_interval(save_interval_ps)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        dcd_path = tmp / "ala_dipeptide.dcd"
        log_path = tmp / "ala_dipeptide.log"

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
                totalSteps=total_steps,
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
                totalSteps=total_steps,
            )
        )

        print(
            f"Running alanine dipeptide MD: {ns_total} ns at {temperature} K "
            f"with frames every {save_interval_ps} ps.",
            flush=True,
        )
        simulation.minimizeEnergy()
        simulation.context.setVelocitiesToTemperature(temperature * kelvin)
        simulation.step(total_steps)

        traj = _align_frames([dcd_path], input_pdb)

    coords = traj.xyz.astype(np.float32)
    np.save(output_dir / "coordinates.npy", coords)

    _, phi = md.compute_phi(traj)
    _, psi = md.compute_psi(traj)
    if phi.shape[1] != 1 or psi.shape[1] != 1:
        raise RuntimeError("Alanine dipeptide should yield a single phi and psi angle per frame.")
    angles = _wrap_angles(phi[:, 0], psi[:, 0])
    np.save(output_dir / "phi_psi.npy", angles.astype(np.float32))

    metadata = {
        "input_pdb": str(input_pdb),
        "temperature_K": temperature,
        "ns_total": ns_total,
        "save_interval_ps": save_interval_ps,
        "timestep_fs": float(_DEFAULT_TIMESTEP.value_in_unit(femtosecond)),
        "n_frames": int(coords.shape[0]),
        "n_atoms": int(coords.shape[1]),
        "topology_hash": md.load(str(input_pdb)).topology.md5(),
        "forcefield": list(_DEFAULT_FORCEFIELD),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run alanine dipeptide MD and extract φ/ψ angles.")
    parser.add_argument("--input-pdb", type=pathlib.Path, required=True, help="Starting structure of alanine dipeptide.")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("data/alanine_dipeptide"),
        help="Directory for outputs (coordinates.npy, phi_psi.npy, metadata.json).",
    )
    parser.add_argument("--temperature", type=float, default=500.0, help="Simulation temperature in Kelvin.")
    parser.add_argument("--ns-total", type=float, default=20.0, help="Total simulation length in nanoseconds.")
    parser.add_argument(
        "--save-interval-ps",
        type=float,
        default=1.0,
        help="Trajectory save interval in picoseconds (1 ps → 20,000 frames for 20 ns).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run_simulation(
        input_pdb=args.input_pdb,
        output_dir=args.output_dir,
        temperature=args.temperature,
        ns_total=args.ns_total,
        save_interval_ps=args.save_interval_ps,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point only
    main()
