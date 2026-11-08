#!/usr/bin/env python3
"""Run the staged FP32/FP64 HydraGNN structural-relaxation benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ase import Atoms
from ase.build import bulk, molecule
from ase.geometry import find_mic, get_distances
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS, FIRE
from ase.filters import UnitCellFilter

from hydragnn_ase_calculator import (
    BACKBONE_SHA256,
    MLP_SHA256,
    NUM_BRANCHES,
    FusedHydraGNNCalculator,
    load_fused_calculator,
    load_single_head_calculator,
    sha256_file,
)


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "benchmark_config.json"
BENCHMARK_CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
SYSTEM_CONFIG = BENCHMARK_CONFIG["systems"]
INFERENCE_CONFIG = BENCHMARK_CONFIG["inference"]
RELAXATION_CONFIG = BENCHMARK_CONFIG["relaxation"]
PERTURBATION_CONFIG = BENCHMARK_CONFIG["initial_perturbation"]
FD_CONFIG = BENCHMARK_CONFIG["finite_difference_check"]
BRANCH_DATASETS = tuple(BENCHMARK_CONFIG["branch_dataset_mapping"])
SINGLE_HEAD_CONFIG = BENCHMARK_CONFIG["single_head_selection"]
SC26_DIR = HERE.parent
REPO_ROOT = HERE.parents[2]
LOGDIR = SC26_DIR / "multidataset_hpo-BEST6-fp64"
BACKBONE_PATH = LOGDIR / "multidataset_hpo-BEST6-fp64_epoch_97.pk"
MLP_PATH = SC26_DIR / "mlp_branch_weights.pt"
PRECISIONS = ("fp32", "fp64")
SYSTEM_NAMES = tuple(SYSTEM_CONFIG)
REPRESENTATIVE_SYSTEMS = ("H2O", "Si", "MgO")
DISPLACEMENT_SEED = int(PERTURBATION_CONFIG["seed"])
DISPLACEMENT_MIN, DISPLACEMENT_MAX = map(
    float, PERTURBATION_CONFIG["per_atom_displacement_magnitude_angstrom"]
)
FORCE_TOLERANCE = float(RELAXATION_CONFIG["force_convergence_tolerance_ev_per_angstrom"])
FULL_MAX_STEPS = int(RELAXATION_CONFIG["maximum_steps"])
PILOT_MAX_STEPS = int(RELAXATION_CONFIG["pilot_maximum_steps"])
FD_STEPS = tuple(
    float(step)
    for step in FD_CONFIG["central_difference_step_sizes_angstrom"]
)
STRESS_FD_TOLERANCES = {
    precision: float(value)
    for precision, value in FD_CONFIG[
        "maximum_absolute_stress_error_ev_per_angstrom3"
    ].items()
}

if int(INFERENCE_CONFIG["branch_count"]) != 16:
    raise ValueError("This benchmark requires exactly 16 source-specific branches")
if len(BRANCH_DATASETS) != int(INFERENCE_CONFIG["branch_count"]):
    raise ValueError("Branch-to-dataset mapping does not match configured branch count")
if BENCHMARK_CONFIG["expected_checkpoint_sha256"] != {
    "branch_weight_mlp": MLP_SHA256,
    "hydragNN": BACKBONE_SHA256,
}:
    raise ValueError("Configured checkpoint hashes do not match the verified checkpoints")


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _reference_structures() -> dict[str, Atoms]:
    structures: dict[str, Atoms] = {}
    for name, config in SYSTEM_CONFIG.items():
        if config["kind"] == "molecule":
            structures[name] = molecule(config["formula"])
        elif config["kind"] == "bulk":
            structures[name] = bulk(
                config["symbol"],
                config["crystal_structure"],
                a=float(config["lattice_constant_angstrom"]),
                cubic=bool(config["cubic"]),
            )
        else:
            raise ValueError(f"Unknown structure kind for {name}: {config['kind']}")
    return structures


def _minimum_interatomic_distance(atoms: Atoms) -> float:
    distances = get_distances(atoms.positions, cell=atoms.cell, pbc=atoms.pbc)[1]
    distances[distances < 1e-8] = np.inf
    return float(np.min(distances))


def _prepare_inputs(output_dir: Path) -> dict[str, Any]:
    input_dir = output_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    references = _reference_structures()
    rng = np.random.default_rng(DISPLACEMENT_SEED)
    metadata: dict[str, Any] = {
        "seed": DISPLACEMENT_SEED,
        "displacement_magnitude_angstrom": [DISPLACEMENT_MIN, DISPLACEMENT_MAX],
        "systems": {},
    }
    for name in SYSTEM_NAMES:
        reference_path = input_dir / f"{name}_reference.extxyz"
        initial_path = input_dir / f"{name}_initial.extxyz"
        if reference_path.exists() or initial_path.exists():
            if not (reference_path.is_file() and initial_path.is_file()):
                raise FileExistsError(
                    f"Partial input set exists for {name}; refusing to overwrite: {input_dir}"
                )
            reference = read(reference_path)
            initial = read(initial_path)
        else:
            reference = references[name].copy()
            initial = reference.copy()
            vectors = rng.normal(size=(len(initial), 3))
            lengths = np.linalg.norm(vectors, axis=1)
            if np.any(lengths == 0):
                raise ValueError(f"Zero-length displacement vector generated for {name}")
            magnitudes = rng.uniform(
                DISPLACEMENT_MIN, DISPLACEMENT_MAX, size=len(initial)
            )
            displacement = vectors / lengths[:, None] * magnitudes[:, None]
            initial.positions += displacement
            write(reference_path, reference, format="extxyz")
            write(initial_path, initial, format="extxyz")
        if len(reference) != len(initial):
            raise ValueError(f"Atom count changed between reference and initial {name}")
        if len(reference) != int(SYSTEM_CONFIG[name]["atom_count"]):
            raise ValueError(f"Reference atom count does not match configuration for {name}")
        if not np.array_equal(reference.numbers, initial.numbers):
            raise ValueError(f"Element ordering changed between reference and initial {name}")
        if not np.array_equal(
            reference.pbc, np.full(3, bool(SYSTEM_CONFIG[name]["pbc"]))
        ):
            raise ValueError(f"Reference PBC does not match configuration for {name}")
        if not np.array_equal(reference.pbc, initial.pbc) or not np.allclose(
            reference.cell.array, initial.cell.array, rtol=0, atol=1e-12
        ):
            raise ValueError(f"Cell or PBC changed for {name}")
        if not np.isfinite(initial.positions).all():
            raise ValueError(f"Nonfinite initial coordinates for {name}")
        if np.max(np.linalg.norm(initial.positions - reference.positions, axis=1)) > 0.050001:
            raise ValueError(f"Initial displacement exceeds 0.05 A for {name}")
        displacement_lengths = np.linalg.norm(
            initial.positions - reference.positions, axis=1
        )
        if np.any(displacement_lengths < DISPLACEMENT_MIN - 1e-8):
            raise ValueError(f"Initial displacement is below 0.03 A for {name}")
        minimum_distance = _minimum_interatomic_distance(initial)
        if minimum_distance < 0.5:
            raise ValueError(
                f"Unreasonable initial overlap in {name}: {minimum_distance:.4f} A"
            )
        metadata["systems"][name] = {
            "formula": initial.get_chemical_formula(),
            "atom_count": len(initial),
            "atomic_numbers": initial.numbers.tolist(),
            "pbc": initial.pbc.tolist(),
            "cell_angstrom": initial.cell.array.tolist(),
            "minimum_image_separation_angstrom": minimum_distance,
            "reference_geometry": str(reference_path.relative_to(output_dir)),
            "perturbed_geometry": str(initial_path.relative_to(output_dir)),
            "geometry_sha256": sha256_file(initial_path),
        }
    _json_dump(input_dir / "input_manifest.json", metadata)
    return metadata


def _read_initial(output_dir: Path, system: str) -> Atoms:
    path = output_dir / "inputs" / f"{system}_initial.extxyz"
    if not path.is_file():
        raise FileNotFoundError(f"Required saved initial structure is missing: {path}")
    return read(path, format="extxyz")


def _runtime_provenance() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Unable to record the repository commit") from error
    return {
        "repository_commit": commit,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "ase_version": __import__("ase").__version__,
        "torch_geometric_version": __import__("torch_geometric").__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "checkpoint_paths": {
            "hydragNN": str(BACKBONE_PATH),
            "branch_weight_mlp": str(MLP_PATH),
        },
        "checkpoint_sha256": {
            "hydragNN": sha256_file(BACKBONE_PATH),
            "branch_weight_mlp": sha256_file(MLP_PATH),
        },
        "expected_checkpoint_sha256": BENCHMARK_CONFIG["expected_checkpoint_sha256"],
        "benchmark_config_sha256": sha256_file(CONFIG_PATH),
        "energy_units": "eV (inference output documented by inference_fused_write_adios.py)",
        "force_units": "eV/angstrom (inference output documented by inference_fused_write_adios.py)",
    }


def _edge_rows(result: dict[str, Any]) -> np.ndarray:
    return np.asarray(result["edge_index"], dtype=np.int64).T


def _same_edge_set(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_edges = _edge_rows(first)
    second_edges = _edge_rows(second)
    if first_edges.shape != second_edges.shape:
        return False
    if not len(first_edges):
        return True
    first_order = np.lexsort((first_edges[:, 1], first_edges[:, 0]))
    second_order = np.lexsort((second_edges[:, 1], second_edges[:, 0]))
    return np.array_equal(first_edges[first_order], second_edges[second_order])


def _finite_difference_check(
    calculator: FusedHydraGNNCalculator, atoms: Atoms
) -> dict[str, Any]:
    baseline = calculator.evaluate_atoms(atoms)
    max_force_index = np.unravel_index(
        np.argmax(np.abs(baseline["forces"])), baseline["forces"].shape
    )
    atom_index, axis = (int(max_force_index[0]), int(max_force_index[1]))
    autograd_force = float(baseline["forces"][atom_index, axis])
    differences: list[dict[str, Any]] = []
    for step_size in FD_STEPS:
        plus = atoms.copy()
        minus = atoms.copy()
        plus.positions[atom_index, axis] += step_size
        minus.positions[atom_index, axis] -= step_size
        plus_result = calculator.evaluate_atoms(plus)
        minus_result = calculator.evaluate_atoms(minus)
        same_topology = _same_edge_set(baseline, plus_result) and _same_edge_set(
            baseline, minus_result
        )
        finite_difference_force = -(
            plus_result["energy"] - minus_result["energy"]
        ) / (2 * step_size)
        differences.append(
            {
                "step_angstrom": step_size,
                "finite_difference_force_ev_per_angstrom": float(
                    finite_difference_force
                ),
                "autograd_force_ev_per_angstrom": autograd_force,
                "absolute_error_ev_per_angstrom": float(
                    abs(finite_difference_force - autograd_force)
                ),
                "relative_error": float(
                    abs(finite_difference_force - autograd_force)
                    / max(abs(finite_difference_force), abs(autograd_force), 1e-8)
                ),
                "same_neighbor_topology": bool(same_topology),
            }
        )
    best_error = min(row["absolute_error_ev_per_angstrom"] for row in differences)
    threshold = float(
        FD_CONFIG["maximum_absolute_force_error_ev_per_angstrom"][
            calculator.precision
        ]
    )
    edge_stable = all(row["same_neighbor_topology"] for row in differences)
    return {
        "precision": calculator.precision,
        "atom_index_zero_based": atom_index,
        "cartesian_axis": ("x", "y", "z")[axis],
        "autograd_force_ev_per_angstrom": autograd_force,
        "step_results": differences,
        "best_absolute_error_ev_per_angstrom": best_error,
        "acceptance_tolerance_ev_per_angstrom": threshold,
        "neighbor_topology_stable": edge_stable,
        "passed": bool(edge_stable and best_error <= threshold),
    }


def _finite_difference_stress_check(
    calculator: FusedHydraGNNCalculator,
    atoms: Atoms,
    *,
    strain_step: float = float(FD_CONFIG["stress_strain_step"]),
) -> dict[str, Any]:
    if not np.all(atoms.pbc):
        raise ValueError("Stress finite differences require a fully periodic structure")
    baseline = calculator.evaluate_atoms(atoms)
    plus = atoms.copy()
    minus = atoms.copy()
    plus_strain = np.eye(3)
    minus_strain = np.eye(3)
    plus_strain[0, 0] += strain_step
    minus_strain[0, 0] -= strain_step
    plus.set_cell(atoms.cell.array @ plus_strain, scale_atoms=True)
    minus.set_cell(atoms.cell.array @ minus_strain, scale_atoms=True)
    plus_result = calculator.evaluate_atoms(plus)
    minus_result = calculator.evaluate_atoms(minus)
    finite_difference_stress = (
        plus_result["energy"] - minus_result["energy"]
    ) / (2 * strain_step * atoms.get_volume())
    autograd_stress = float(baseline["stress"][0])
    absolute_error = abs(finite_difference_stress - autograd_stress)
    tolerance = STRESS_FD_TOLERANCES[calculator.precision]
    return {
        "strain_component": "xx",
        "strain_step": strain_step,
        "finite_difference_stress_ev_per_angstrom3": float(finite_difference_stress),
        "autograd_stress_ev_per_angstrom3": autograd_stress,
        "absolute_error_ev_per_angstrom3": float(absolute_error),
        "acceptance_tolerance_ev_per_angstrom3": tolerance,
        "same_neighbor_topology": bool(
            _same_edge_set(baseline, plus_result)
            and _same_edge_set(baseline, minus_result)
        ),
        "passed": bool(absolute_error <= tolerance),
    }


def _state_row(
    atoms: Atoms,
    calculator: FusedHydraGNNCalculator,
    step: int,
    *,
    converged: bool,
    optimization_target: Atoms | UnitCellFilter,
) -> dict[str, Any]:
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    if not np.isfinite(energy) or not np.isfinite(forces).all():
        raise FloatingPointError("Nonfinite energy or forces during relaxation")
    force_norms = np.linalg.norm(forces, axis=1)
    stress = np.asarray(atoms.get_stress(voigt=False), dtype=np.float64)
    optimizer_forces = np.asarray(
        optimization_target.get_forces(), dtype=np.float64
    )
    if calculator.last_branch_weights is None or calculator.last_branch_energies is None:
        raise RuntimeError("Calculator did not retain branch results")
    return {
        "step": int(step),
        "head_mode": (
            "single" if calculator.single_branch_id is not None else "fused"
        ),
        "energy_ev": energy,
        "energy_ev_per_atom": energy / len(atoms),
        "maximum_force_ev_per_angstrom": float(force_norms.max()),
        "rms_force_ev_per_angstrom": float(np.sqrt(np.mean(forces**2))),
        "maximum_optimization_force": float(
            np.linalg.norm(optimizer_forces, axis=1).max()
        ),
        "stress_ev_per_angstrom3_json": json.dumps(stress.tolist()),
        "maximum_absolute_stress_ev_per_angstrom3": float(np.abs(stress).max()),
        "cell_angstrom_json": json.dumps(np.asarray(atoms.cell).tolist()),
        "coordinates_angstrom_json": json.dumps(atoms.positions.tolist()),
        "graph_edges": int(calculator.last_graph_edges),
        "evaluated_branch_ids_json": json.dumps(
            [calculator.single_branch_id]
            if calculator.single_branch_id is not None
            else list(range(calculator.num_branches))
        ),
        "branch_weight_semantics": (
            "single_head_selection_mask"
            if calculator.single_branch_id is not None
            else "mlp_softmax"
        ),
        "branch_weights_json": json.dumps(
            calculator.last_branch_weights.tolist(), allow_nan=False
        ),
        "converged": bool(converged),
    }


def _run_relaxation(
    atoms: Atoms,
    calculator: FusedHydraGNNCalculator,
    output_dir: Path,
    *,
    max_steps: int,
    force_tolerance: float,
    relax_cell: bool = False,
    optimizer_name: str = "bfgs",
    optimizer_maxstep: float = 0.2,
    hydrostatic_cell: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    atoms.calc = calculator
    if relax_cell and not np.all(atoms.pbc):
        raise ValueError("Variable-cell relaxation requires periodicity in all directions")
    if hydrostatic_cell and not relax_cell:
        raise ValueError("Hydrostatic cell constraints require variable-cell relaxation")
    optimizer_classes = {"bfgs": BFGS, "fire": FIRE}
    if optimizer_name not in optimizer_classes:
        raise ValueError(f"Unsupported optimizer {optimizer_name!r}")
    optimization_target: Atoms | UnitCellFilter = (
        UnitCellFilter(
            atoms,
            cell_factor=float(len(atoms)),
            hydrostatic_strain=hydrostatic_cell,
        )
        if relax_cell
        else atoms
    )
    log_path = output_dir / f"{optimizer_name}.log"
    trajectory_path = output_dir / "trajectory.traj"
    csv_path = output_dir / "trajectory.csv"
    rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    error: str | None = None
    status = "not_started"
    converged = False
    steps_taken = 0

    with Trajectory(str(trajectory_path), mode="w", atoms=atoms) as trajectory:
        try:
            initial_row = _state_row(
                atoms,
                calculator,
                0,
                converged=False,
                optimization_target=optimization_target,
            )
            rows.append(initial_row)
            trajectory.write(atoms)
            optimizer = optimizer_classes[optimizer_name](
                optimization_target,
                logfile=str(log_path),
                maxstep=optimizer_maxstep,
            )
            optimizer.attach(
                lambda: _record_optimizer_step(
                    atoms,
                    calculator,
                    optimization_target,
                    optimizer,
                    rows,
                    trajectory,
                    force_tolerance,
                ),
                interval=1,
            )
            status = "running"
            converged = bool(optimizer.run(fmax=force_tolerance, steps=max_steps))
            steps_taken = int(optimizer.get_number_of_steps())
            if rows:
                rows[-1]["converged"] = converged
            status = "converged" if converged else "max_steps_without_convergence"
        except Exception as caught:
            status = "failed"
            error = f"{type(caught).__name__}: {caught}"
            log_path.open("a", encoding="utf-8").write(
                "\nBENCHMARK ERROR\n" + traceback.format_exc()
            )
            if rows:
                rows[-1]["converged"] = False
        finally:
            recorded_steps = max((int(row["step"]) for row in rows), default=0)
            steps_taken = max(steps_taken, recorded_steps)
            if rows and (
                not np.allclose(
                    np.asarray(json.loads(rows[-1]["coordinates_angstrom_json"])),
                    atoms.positions,
                    rtol=0,
                    atol=1e-12,
                )
                or not np.allclose(
                    np.asarray(json.loads(rows[-1]["cell_angstrom_json"])),
                    atoms.cell.array,
                    rtol=0,
                    atol=1e-12,
                )
            ):
                trajectory.write(atoms)

    elapsed = time.perf_counter() - start_time
    if not rows or rows[-1]["step"] != steps_taken:
        try:
            rows.append(
                _state_row(
                    atoms,
                    calculator,
                    steps_taken,
                    converged=converged,
                    optimization_target=optimization_target,
                )
            )
        except Exception as caught:
            status = "failed"
            error = error or f"{type(caught).__name__}: {caught}"
    if rows:
        _write_csv(csv_path, rows)
    final_geometry = atoms.copy()
    final_geometry.calc = None
    write(output_dir / "final.extxyz", final_geometry, format="extxyz")
    summary = {
        "precision": calculator.precision,
        "head_mode": "single" if calculator.single_branch_id is not None else "fused",
        "selected_branch_id": calculator.single_branch_id,
        "status": status,
        "converged": bool(converged and status == "converged"),
        "force_tolerance_ev_per_angstrom": force_tolerance,
        "cell_relaxation": relax_cell,
        "hydrostatic_cell": hydrostatic_cell,
        "cell_factor": float(len(atoms)) if relax_cell else None,
        "optimizer": optimizer_name,
        "optimizer_maxstep_angstrom": optimizer_maxstep,
        "maximum_steps": max_steps,
        "steps_taken": steps_taken,
        "total_calculation_time_seconds": elapsed,
        "initial_energy_ev": rows[0]["energy_ev"] if rows else None,
        "final_energy_ev": rows[-1]["energy_ev"] if rows else None,
        "final_maximum_force_ev_per_angstrom": (
            rows[-1]["maximum_force_ev_per_angstrom"] if rows else None
        ),
        "final_rms_force_ev_per_angstrom": (
            rows[-1]["rms_force_ev_per_angstrom"] if rows else None
        ),
        "final_maximum_optimization_force": (
            rows[-1]["maximum_optimization_force"] if rows else None
        ),
        "final_maximum_absolute_stress_ev_per_angstrom3": (
            rows[-1]["maximum_absolute_stress_ev_per_angstrom3"] if rows else None
        ),
        "initial_branch_weights_json": (
            rows[0]["branch_weights_json"] if rows else None
        ),
        "final_branch_weights_json": (
            rows[-1]["branch_weights_json"] if rows else None
        ),
        "graph_edges_final": rows[-1]["graph_edges"] if rows else None,
        "trajectory_csv": csv_path.name if rows else None,
        "trajectory_file": trajectory_path.name,
        "error": error,
    }
    _json_dump(output_dir / "run_summary.json", summary)
    return summary


def _record_optimizer_step(
    atoms: Atoms,
    calculator: FusedHydraGNNCalculator,
    optimization_target: Atoms | UnitCellFilter,
    optimizer: BFGS | FIRE,
    rows: list[dict[str, Any]],
    trajectory: Trajectory,
    force_tolerance: float,
) -> None:
    if optimizer.get_number_of_steps() == 0:
        return
    forces = np.asarray(optimization_target.get_forces(), dtype=np.float64)
    max_force = float(np.linalg.norm(forces, axis=1).max())
    converged = bool(np.isfinite(max_force) and max_force < force_tolerance)
    rows.append(
        _state_row(
            atoms,
            calculator,
            int(optimizer.get_number_of_steps()),
            converged=converged,
            optimization_target=optimization_target,
        )
    )
    trajectory.write(atoms)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _single_point_comparison(
    output_dir: Path,
    initial_atoms: dict[str, Atoms],
    calculators: dict[str, dict[str, FusedHydraGNNCalculator]],
    *,
    head_mode: str,
    single_head_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for system, initial in initial_atoms.items():
        results[system] = {}
        for precision in PRECISIONS:
            atoms = initial.copy()
            calculator = calculators[system][precision]
            atoms.calc = calculator
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=np.float64)
            if calculator.last_branch_weights is None or calculator.last_branch_energies is None:
                raise RuntimeError(f"Missing branch outputs for {system}/{precision}")
            results[system][precision] = {
                "energy_ev": energy,
                "forces": forces,
                "weights": calculator.last_branch_weights.copy(),
                "branch_energies": calculator.last_branch_energies.copy(),
                "edge_count": calculator.last_graph_edges,
                "stress": calculator.results["stress"].copy(),
            }
        fp32 = results[system]["fp32"]
        fp64 = results[system]["fp64"]
        force_delta = fp32["forces"] - fp64["forces"]
        row = {
            "system": system,
            "formula": initial.get_chemical_formula(),
            "atom_count": len(initial),
            "fp32_energy_ev": fp32["energy_ev"],
            "fp64_energy_ev": fp64["energy_ev"],
            "absolute_energy_difference_ev": abs(
                fp32["energy_ev"] - fp64["energy_ev"]
            ),
            "absolute_energy_difference_ev_per_atom": abs(
                fp32["energy_ev"] - fp64["energy_ev"]
            )
            / len(initial),
            "fp32_maximum_force_ev_per_angstrom": float(
                np.linalg.norm(fp32["forces"], axis=1).max()
            ),
            "fp64_maximum_force_ev_per_angstrom": float(
                np.linalg.norm(fp64["forces"], axis=1).max()
            ),
            "force_mae_ev_per_angstrom": float(np.mean(np.abs(force_delta))),
            "force_rmse_ev_per_angstrom": float(np.sqrt(np.mean(force_delta**2))),
            "maximum_force_component_difference_ev_per_angstrom": float(
                np.abs(force_delta).max()
            ),
            "evaluated_branch_ids": (
                [int(single_head_map[system]["branch_id"])]
                if head_mode == "single"
                else list(range(NUM_BRANCHES))
            ),
            "selected_dataset": (
                single_head_map[system]["dataset"] if head_mode == "single" else None
            ),
            "branch_weight_semantics": (
                "single_head_selection_mask" if head_mode == "single" else "mlp_softmax"
            ),
            "branch_weight_maximum_absolute_difference": (
                None
                if head_mode == "single"
                else float(np.abs(fp32["weights"] - fp64["weights"]).max())
            ),
            "fp32_graph_edges": fp32["edge_count"],
            "fp64_graph_edges": fp64["edge_count"],
            "fp32_stress_ev_per_angstrom3_json": json.dumps(
                fp32["stress"].tolist()
            ),
            "fp64_stress_ev_per_angstrom3_json": json.dumps(
                fp64["stress"].tolist()
            ),
            "fp32_branch_energies_ev_json": json.dumps(
                fp32["branch_energies"].tolist()
            ),
            "fp64_branch_energies_ev_json": json.dumps(
                fp64["branch_energies"].tolist()
            ),
            "fp32_branch_weights_json": json.dumps(fp32["weights"].tolist()),
            "fp64_branch_weights_json": json.dumps(fp64["weights"].tolist()),
            "starting_coordinates_sha256": _coordinates_hash(initial.positions),
        }
        rows.append(row)
    _write_csv(output_dir / "single_point_comparison.csv", rows)
    return results


def _coordinates_hash(positions: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.asarray(positions, dtype=np.float64).tobytes()).hexdigest()


def _aligned_rms_displacement(first: Atoms, second: Atoms) -> float:
    if len(first) != len(second) or not np.array_equal(first.numbers, second.numbers):
        raise ValueError("Final structures do not have matching atom correspondence")
    delta = second.positions - first.positions
    if np.any(first.pbc):
        minimum_image, _ = find_mic(delta, first.cell, pbc=first.pbc)
        return float(np.sqrt(np.mean(np.sum(minimum_image**2, axis=1))))
    first_centered = first.positions - first.positions.mean(axis=0)
    second_centered = second.positions - second.positions.mean(axis=0)
    covariance = first_centered.T @ second_centered
    left, _, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(left @ right_t)
    rotation = left @ correction @ right_t
    aligned = first_centered @ rotation
    return float(np.sqrt(np.mean(np.sum((aligned - second_centered) ** 2, axis=1))))


def _resolve_single_head_map(overrides: list[str]) -> dict[str, dict[str, Any]]:
    selection = {
        system: dict(value)
        for system, value in SINGLE_HEAD_CONFIG["selected_heads"].items()
    }
    for override in overrides:
        try:
            system, raw_branch_id = override.split("=", maxsplit=1)
            branch_id = int(raw_branch_id)
        except ValueError as error:
            raise ValueError(
                f"Invalid --head-override {override!r}; expected SYSTEM=BRANCH_ID"
            ) from error
        if system not in SYSTEM_CONFIG:
            raise ValueError(f"Unknown system in --head-override: {system}")
        if not 0 <= branch_id < len(BRANCH_DATASETS):
            raise ValueError(
                f"Branch ID for {system} must be in [0, {len(BRANCH_DATASETS) - 1}]"
            )
        selection[system] = {
            "branch_id": branch_id,
            "dataset": BRANCH_DATASETS[branch_id],
        }
    for system in SYSTEM_NAMES:
        selected = selection.get(system)
        if selected is None:
            raise ValueError(f"No single head is configured for {system}")
        branch_id = int(selected["branch_id"])
        if not 0 <= branch_id < len(BRANCH_DATASETS):
            raise ValueError(f"Configured branch ID for {system} is out of range")
        if selected["dataset"] != BRANCH_DATASETS[branch_id]:
            raise ValueError(
                f"Configured dataset {selected['dataset']} for {system} does not "
                f"match branch-{branch_id} dataset {BRANCH_DATASETS[branch_id]}"
            )
    return selection


def _load_calculator(
    precision: str,
    *,
    head_mode: str,
    system: str,
    single_head_map: dict[str, dict[str, Any]],
) -> tuple[FusedHydraGNNCalculator, dict[str, Any]]:
    common = {
        "logdir": LOGDIR,
        "backbone_checkpoint": BACKBONE_PATH,
        "precision": precision,
        "charge": float(INFERENCE_CONFIG["charge"]),
        "spin": float(INFERENCE_CONFIG["spin"]),
    }
    if head_mode == "single":
        return load_single_head_calculator(
            **common, branch_id=int(single_head_map[system]["branch_id"])
        )
    if head_mode == "fused":
        return load_fused_calculator(**common, mlp_checkpoint=MLP_PATH)
    raise ValueError(f"Unsupported head mode: {head_mode}")


def _run_stage_a(
    output_dir: Path,
    initial_atoms: dict[str, Atoms],
    *,
    head_mode: str,
    single_head_map: dict[str, dict[str, Any]],
    relax_cell: bool,
    optimizer_name: str,
    optimizer_maxstep: float,
    hydrostatic_cell: bool,
) -> None:
    stage_dir = output_dir / "stage_A"
    if stage_dir.exists() and any(stage_dir.iterdir()):
        raise FileExistsError(f"Stage A output already exists: {stage_dir}")
    stage_dir.mkdir(parents=True, exist_ok=True)
    pilot_system = "Si" if relax_cell else "H2O"
    calculators: dict[str, FusedHydraGNNCalculator] = {}
    provenance: dict[str, Any] = {}
    for precision in PRECISIONS:
        calculators[precision], provenance[precision] = _load_calculator(
            precision=precision,
            head_mode=head_mode,
            system=pilot_system,
            single_head_map=single_head_map,
        )

    pilot_atoms = initial_atoms[pilot_system]
    pilot_results: dict[str, Any] = {}
    for precision in PRECISIONS:
        calculator = calculators[precision]
        single_point = calculator.evaluate_atoms(pilot_atoms)
        expected_evaluated_branches = 1 if head_mode == "single" else NUM_BRANCHES
        if len(single_point["branch_energies"]) != expected_evaluated_branches:
            raise RuntimeError(
                f"{precision} evaluated {len(single_point['branch_energies'])} branches, "
                f"expected {expected_evaluated_branches} in {head_mode} mode"
            )
        if not np.isfinite(single_point["forces"]).all():
            raise FloatingPointError(f"Nonfinite {precision} pilot forces")
        fd = _finite_difference_check(calculator, pilot_atoms)
        if not fd["passed"]:
            raise RuntimeError(
                f"{precision} finite-difference force check failed: "
                f"best discrepancy {fd['best_absolute_error_ev_per_angstrom']:.6g} "
                f"eV/A; topology stable={fd['neighbor_topology_stable']}"
            )
        stress_fd = (
            _finite_difference_stress_check(calculator, pilot_atoms)
            if relax_cell
            else None
        )
        if relax_cell and (
            stress_fd is None
            or not stress_fd["same_neighbor_topology"]
            or stress_fd["absolute_error_ev_per_angstrom3"]
            > STRESS_FD_TOLERANCES[precision]
        ):
            raise RuntimeError(
                f"{precision} stress finite-difference check failed: {stress_fd}"
            )
        relaxed_pilot_atoms = pilot_atoms.copy()
        pilot = _run_relaxation(
            relaxed_pilot_atoms,
            calculator,
            stage_dir / pilot_system / precision,
            max_steps=PILOT_MAX_STEPS,
            force_tolerance=FORCE_TOLERANCE,
            relax_cell=relax_cell,
            optimizer_name=optimizer_name,
            optimizer_maxstep=optimizer_maxstep,
            hydrostatic_cell=hydrostatic_cell,
        )
        if pilot["status"] == "failed":
            raise RuntimeError(
                f"{precision} {pilot_system} pilot relaxation failed: {pilot['error']}"
            )
        if calculator.graph_build_count <= 1 or calculator.last_graph_positions is None:
            raise RuntimeError(f"{precision} graph was not rebuilt during relaxation")
        if not np.allclose(
            calculator.last_graph_positions,
            relaxed_pilot_atoms.positions,
            rtol=0,
            atol=1e-12,
        ):
            raise RuntimeError(f"{precision} calculator graph is stale for the current coordinates")
        if calculator.last_graph_cell is None or not np.allclose(
            np.asarray(calculator.last_graph_cell),
            relaxed_pilot_atoms.cell.array,
            rtol=0,
            atol=1e-12,
        ):
            raise RuntimeError(f"{precision} calculator graph is stale for the current cell")
        pilot_results[precision] = {
            "single_point_energy_ev": single_point["energy"],
            "force_maximum_ev_per_angstrom": float(
                np.linalg.norm(single_point["forces"], axis=1).max()
            ),
            "branch_weights": single_point["branch_weights"].tolist(),
            "branch_energies_ev": single_point["branch_energies"].tolist(),
            "evaluated_branch_ids": single_point["evaluated_branch_ids"],
            "graph_edges": single_point["edge_count"],
            "finite_difference_check": fd,
            "finite_difference_stress_check": stress_fd,
            "pilot_relaxation": pilot,
            "graph_rebuild_count": calculator.graph_build_count,
            "graph_matches_latest_positions": True,
            "precision_provenance": provenance[precision],
        }
    max_weight_difference = (
        float(
            np.max(
                np.abs(
                    np.asarray(pilot_results["fp32"]["branch_weights"])
                    - np.asarray(pilot_results["fp64"]["branch_weights"])
                )
            )
        )
        if head_mode == "fused"
        else None
    )
    status = {
        "status": "passed",
        "head_mode": head_mode,
        "cell_relaxation": relax_cell,
        "hydrostatic_cell": hydrostatic_cell,
        "optimizer": optimizer_name,
        "optimizer_maxstep_angstrom": optimizer_maxstep,
        "pilot_system": pilot_system,
        "single_head_selection": (
            single_head_map if head_mode == "single" else None
        ),
        "technical_validation_passed": True,
        "scientific_convergence_claim": False,
        "note": "The pilot is a software validation only; a 10-step limit is not a full-convergence result.",
        "pilot_precision_results": pilot_results,
        "fp32_fp64_branch_weight_maximum_absolute_difference": max_weight_difference,
        "checks": {
            "checkpoint_hashes_verified": True,
            "both_precisions_loaded_and_frozen": True,
            "mlp_loaded": head_mode == "fused",
            "all_16_branches_evaluated": head_mode == "fused",
            "single_head_evaluated_only": head_mode == "single",
            "branch_weights_or_selection_mask_sum_to_one": True,
            "single_point_energy_and_forces_finite": True,
            "finite_difference_force_check_passed": True,
            "finite_difference_stress_check_passed": not relax_cell
            or all(
                value["finite_difference_stress_check"]["same_neighbor_topology"]
                and value["finite_difference_stress_check"][
                    "absolute_error_ev_per_angstrom3"
                ]
                <= STRESS_FD_TOLERANCES[precision]
                for precision, value in pilot_results.items()
            ),
            "neighbor_topology_unchanged_in_finite_difference": True,
            "graph_rebuilt_at_moving_geometry": True,
            "optimizer_executed_without_error": True,
            "periodic_and_molecular_construction_implemented": True,
            "branch_skipping_disabled": head_mode == "fused",
            "only_selected_branch_evaluated": head_mode == "single",
        },
    }
    _json_dump(stage_dir / "stage_a_status.json", status)
    print(
        f"Stage A passed. The pilot used {pilot_system} only and is not a scientific convergence result."
    )


def _capture_failure(path: Path, error: Exception) -> dict[str, Any]:
    failure = {
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }
    _json_dump(path, failure)
    print(f"FAILED: {path.parent}: {failure['error_type']}: {failure['error']}", file=sys.stderr)
    return failure


def _run_stage_b(
    output_dir: Path,
    initial_atoms: dict[str, Atoms],
    *,
    head_mode: str,
    single_head_map: dict[str, dict[str, Any]],
    relax_cell: bool,
    optimizer_name: str,
    optimizer_maxstep: float,
    hydrostatic_cell: bool,
) -> None:
    stage_a_path = output_dir / "stage_A" / "stage_a_status.json"
    if not stage_a_path.is_file():
        raise RuntimeError("Stage B is blocked until Stage A has completed")
    stage_a = json.loads(stage_a_path.read_text(encoding="utf-8"))
    if stage_a.get("status") != "passed":
        raise RuntimeError("Stage B is blocked because Stage A did not pass")
    if stage_a.get("head_mode") != head_mode:
        raise RuntimeError("Stage B head mode must match the validated Stage A pilot")
    if stage_a.get("cell_relaxation", False) != relax_cell:
        raise RuntimeError("Stage B cell-relaxation setting must match Stage A")
    if stage_a.get("hydrostatic_cell", False) != hydrostatic_cell:
        raise RuntimeError("Stage B cell constraint must match Stage A")
    if stage_a.get("optimizer") != optimizer_name or not np.isclose(
        stage_a.get("optimizer_maxstep_angstrom", np.nan), optimizer_maxstep
    ):
        raise RuntimeError("Stage B optimizer settings must match Stage A")
    if head_mode == "single" and stage_a.get("single_head_selection") != single_head_map:
        raise RuntimeError("Stage B single-head selection must match the Stage A pilot")
    stage_dir = output_dir / "stage_B"
    if stage_dir.exists() and any(stage_dir.iterdir()):
        raise FileExistsError(f"Stage B output already exists: {stage_dir}")
    stage_dir.mkdir(parents=True, exist_ok=True)

    calculators: dict[str, dict[str, FusedHydraGNNCalculator]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for system in SYSTEM_NAMES:
        calculators[system] = {}
        provenance[system] = {}
        for precision in PRECISIONS:
            calculators[system][precision], provenance[system][precision] = (
                _load_calculator(
                    precision,
                    head_mode=head_mode,
                    system=system,
                    single_head_map=single_head_map,
                )
            )
    _json_dump(stage_dir / "precision_provenance.json", provenance)
    single_points = _single_point_comparison(
        stage_dir,
        initial_atoms,
        calculators,
        head_mode=head_mode,
        single_head_map=single_head_map,
    )
    run_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    finals: dict[str, dict[str, Atoms]] = {}
    for system in SYSTEM_NAMES:
        run_summaries[system] = {}
        finals[system] = {}
        for precision in PRECISIONS:
            run_dir = stage_dir / system / precision
            relax_this_cell = relax_cell and bool(np.all(initial_atoms[system].pbc))
            start = _read_initial(output_dir, system)
            failure_path = run_dir / "failure.json"
            try:
                summary = _run_relaxation(
                    start,
                    calculators[system][precision],
                    run_dir,
                    max_steps=FULL_MAX_STEPS,
                    force_tolerance=FORCE_TOLERANCE,
                    relax_cell=relax_this_cell,
                    optimizer_name=optimizer_name,
                    optimizer_maxstep=optimizer_maxstep,
                    hydrostatic_cell=hydrostatic_cell and relax_this_cell,
                )
                if summary["status"] == "failed":
                    run_summaries[system][precision] = summary
                    _capture_failure(
                        failure_path,
                        RuntimeError(summary["error"] or "Unknown optimizer failure"),
                    )
                else:
                    run_summaries[system][precision] = summary
                final_path = run_dir / "final.extxyz"
                if final_path.is_file():
                    finals[system][precision] = read(final_path)
            except Exception as error:
                run_summaries[system][precision] = _capture_failure(failure_path, error)
                if (run_dir / "final.extxyz").is_file():
                    finals[system][precision] = read(run_dir / "final.extxyz")

    comparisons: list[dict[str, Any]] = []
    for system in SYSTEM_NAMES:
        fp32_final = finals.get(system, {}).get("fp32")
        fp64_final = finals.get(system, {}).get("fp64")
        fp32_summary = run_summaries[system].get("fp32", {"status": "not_run"})
        fp64_summary = run_summaries[system].get("fp64", {"status": "not_run"})
        row: dict[str, Any] = {
            "system": system,
            "head_mode": head_mode,
            "cell_relaxation": relax_cell and bool(np.all(initial_atoms[system].pbc)),
            "selected_branch_id": (
                int(single_head_map[system]["branch_id"])
                if head_mode == "single"
                else None
            ),
            "selected_dataset": (
                single_head_map[system]["dataset"] if head_mode == "single" else None
            ),
            "fp32_status": fp32_summary.get("status"),
            "fp64_status": fp64_summary.get("status"),
            "fp32_converged": bool(fp32_summary.get("converged", False)),
            "fp64_converged": bool(fp64_summary.get("converged", False)),
            "fp32_steps": fp32_summary.get("steps_taken"),
            "fp64_steps": fp64_summary.get("steps_taken"),
            "rms_displacement_angstrom": None,
            "fp32_final_energy_ev": fp32_summary.get("final_energy_ev"),
            "fp64_final_energy_ev": fp64_summary.get("final_energy_ev"),
            "fp32_relaxed_common_fp64_energy_ev": None,
            "fp64_relaxed_common_fp64_energy_ev": None,
            "fp64_force_on_fp32_relaxed_max_ev_per_angstrom": None,
            "fp32_force_on_fp64_relaxed_max_ev_per_angstrom": None,
            "maximum_cross_precision_force_ev_per_angstrom": None,
            "maximum_final_branch_weight_difference": None,
            "cross_evaluation_error": None,
        }
        if fp32_final is not None and fp64_final is not None:
            row["rms_displacement_angstrom"] = _aligned_rms_displacement(
                fp32_final, fp64_final
            )
            cross_results: dict[str, dict[str, Any]] = {}
            for label, geometry, precision in (
                ("fp32_geometry_as_fp64", fp32_final, "fp64"),
                ("fp64_geometry_as_fp32", fp64_final, "fp32"),
            ):
                try:
                    cross_results[label] = calculators[system][precision].evaluate_atoms(geometry)
                except Exception as error:
                    row["cross_evaluation_error"] = (
                        f"{type(error).__name__}: {error}"
                    )
            fp32_as_fp64 = cross_results.get("fp32_geometry_as_fp64")
            fp64_as_fp32 = cross_results.get("fp64_geometry_as_fp32")
            if fp32_as_fp64 is not None:
                row["fp32_relaxed_common_fp64_energy_ev"] = fp32_as_fp64["energy"]
                row["fp64_force_on_fp32_relaxed_max_ev_per_angstrom"] = float(
                    np.linalg.norm(fp32_as_fp64["forces"], axis=1).max()
                )
            if fp64_as_fp32 is not None:
                row["fp64_relaxed_common_fp64_energy_ev"] = calculators[system][
                    "fp64"
                ].evaluate_atoms(fp64_final)["energy"]
                row["fp32_force_on_fp64_relaxed_max_ev_per_angstrom"] = float(
                    np.linalg.norm(fp64_as_fp32["forces"], axis=1).max()
                )
            residuals = [
                value
                for value in (
                    row["fp64_force_on_fp32_relaxed_max_ev_per_angstrom"],
                    row["fp32_force_on_fp64_relaxed_max_ev_per_angstrom"],
                )
                if value is not None
            ]
            row["maximum_cross_precision_force_ev_per_angstrom"] = (
                max(residuals) if residuals else None
            )
            if head_mode == "fused":
                weights32 = single_points[system]["fp32"]["weights"]
                weights64 = single_points[system]["fp64"]["weights"]
                row["maximum_final_branch_weight_difference"] = float(
                    np.abs(weights32 - weights64).max()
                )
                weight_checks = []
                for precision, weights in (("fp32", weights32), ("fp64", weights64)):
                    summary = run_summaries[system].get(precision, {})
                    initial_weights = summary.get("initial_branch_weights_json")
                    final_weights = summary.get("final_branch_weights_json")
                    if initial_weights is not None and final_weights is not None:
                        weight_checks.append(
                            np.allclose(
                                json.loads(initial_weights), weights, rtol=0, atol=2e-6
                            )
                            and np.allclose(
                                json.loads(final_weights), weights, rtol=0, atol=2e-6
                            )
                        )
                row["same_composition_weights_match_initial"] = (
                    bool(all(weight_checks)) if weight_checks else None
                )
        comparisons.append(row)
    _write_csv(stage_dir / "final_structure_comparison.csv", comparisons)
    _json_dump(
        stage_dir / "full_benchmark_summary.json",
        {
            "stage": "B",
            "head_mode": head_mode,
            "mlp_used_for_relaxation": head_mode == "fused",
            "single_head_selection": (
                single_head_map if head_mode == "single" else None
            ),
            "force_tolerance_ev_per_angstrom": FORCE_TOLERANCE,
            "maximum_optimizer_steps": FULL_MAX_STEPS,
            "cell_relaxation": relax_cell,
            "hydrostatic_cell": hydrostatic_cell,
            "optimizer": optimizer_name,
            "optimizer_maxstep_angstrom": optimizer_maxstep,
            "branch_skipping": head_mode == "fused",
            "evaluated_branch_count": (
                NUM_BRANCHES if head_mode == "fused" else 1
            ),
            "single_point_comparison_csv": "single_point_comparison.csv",
            "final_structure_comparison_csv": "final_structure_comparison.csv",
            "systems": run_summaries,
            "final_structure_comparisons": comparisons,
        },
    )
    _make_figures(stage_dir, run_summaries, optimizer_name=optimizer_name)
    _write_report(
        output_dir,
        stage_dir,
        run_summaries,
        comparisons,
        provenance,
        head_mode=head_mode,
        single_head_map=single_head_map,
        relax_cell=relax_cell,
        optimizer_name=optimizer_name,
        optimizer_maxstep=optimizer_maxstep,
        hydrostatic_cell=hydrostatic_cell,
    )
    print(f"Stage B outputs written under {stage_dir}")


def _make_figures(
    stage_dir: Path,
    run_summaries: dict[str, dict[str, dict[str, Any]]],
    *,
    optimizer_name: str,
) -> None:
    figure_dir = stage_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {"fp32": "#e45756", "fp64": "#4c78a8"}
    labels = {"fp32": "FP32", "fp64": "FP64"}
    for figure_type in ("force", "energy"):
        fig, axes = plt.subplots(1, len(REPRESENTATIVE_SYSTEMS), figsize=(15, 4.8))
        for axis, system in zip(axes, REPRESENTATIVE_SYSTEMS):
            for precision in PRECISIONS:
                csv_path = stage_dir / system / precision / "trajectory.csv"
                if not csv_path.is_file():
                    continue
                with csv_path.open(encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                steps = np.asarray([int(row["step"]) for row in rows])
                if figure_type == "force":
                    values = np.asarray(
                        [
                            float(row["maximum_force_ev_per_angstrom"])
                            for row in rows
                        ]
                    )
                    axis.semilogy(
                        steps, values, marker=".", color=colors[precision],
                        label=labels[precision],
                    )
                else:
                    energies = np.asarray([float(row["energy_ev"]) for row in rows])
                    axis.plot(
                        steps,
                        energies - energies[0],
                        marker=".",
                        color=colors[precision],
                        label=labels[precision],
                    )
            axis.set_title(system)
            axis.set_xlabel(f"{optimizer_name.upper()} step")
            axis.grid(alpha=0.25)
            if figure_type == "force":
                axis.set_ylabel("Maximum atomic force (eV/Å)")
                axis.axhline(
                    FORCE_TOLERANCE,
                    linestyle="--",
                    linewidth=1.2,
                    color="black",
                    label="0.05 eV/Å target",
                )
            else:
                axis.set_ylabel(r"$E_n - E_0$ (eV)")
            axis.legend(fontsize=8)
        fig.suptitle(
            "FP32 vs FP64 structural relaxation — "
            + ("maximum force" if figure_type == "force" else "energy change")
        )
        fig.tight_layout()
        stem = "figure_1_relaxation_convergence" if figure_type == "force" else "figure_2_energy_evolution"
        fig.savefig(figure_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)


def _write_report(
    output_dir: Path,
    stage_dir: Path,
    runs: dict[str, dict[str, dict[str, Any]]],
    comparisons: list[dict[str, Any]],
    provenance: dict[str, Any],
    *,
    head_mode: str,
    single_head_map: dict[str, dict[str, Any]],
    relax_cell: bool,
    optimizer_name: str,
    optimizer_maxstep: float,
    hydrostatic_cell: bool,
) -> None:
    provenance_by_precision = provenance["H2O"]
    single_points = list(csv.DictReader((stage_dir / "single_point_comparison.csv").open()))
    single_mode = head_mode == "single"
    table_header = (
        "| System | Head | FP32 converged? | FP64 converged? | FP32 steps | "
        "FP64 steps | RMS displacement (Å) | Max cross-precision force (eV/Å) |"
        if single_mode
        else "| System | FP32 converged? | FP64 converged? | FP32 steps | FP64 steps | RMS displacement (Å) | Max cross-precision force (eV/Å) |"
    )
    table_separator = (
        "|---|---|---:|---:|---:|---:|---:|---:|"
        if single_mode
        else "|---|---:|---:|---:|---:|---:|---:|"
    )
    table_lines = [table_header, table_separator]
    for row in comparisons:
        head_cell = (
            f"branch-{row['selected_branch_id']} ({row['selected_dataset']}) | "
            if single_mode
            else ""
        )
        table_lines.append(
            f"| {row['system']} | {head_cell}{row['fp32_converged']} | "
            f"{row['fp64_converged']} | {row['fp32_steps']} | {row['fp64_steps']} | "
            f"{row['rms_displacement_angstrom']} | "
            f"{row['maximum_cross_precision_force_ev_per_angstrom']} |"
        )
    weight_header = "MLP max Δweight" if not single_mode else "MLP weight difference"
    sp_lines = [
        f"| System | ΔE (eV) | ΔE/N (eV/atom) | Force MAE (eV/Å) | "
        f"Force RMSE (eV/Å) | Max component ΔF (eV/Å) | {weight_header} |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in single_points:
        branch_weight_difference = (
            "n/a"
            if row["branch_weight_maximum_absolute_difference"] in (None, "")
            else f"{float(row['branch_weight_maximum_absolute_difference']):.6g}"
        )
        sp_lines.append(
            f"| {row['system']} | {float(row['absolute_energy_difference_ev']):.6g} "
            f"| {float(row['absolute_energy_difference_ev_per_atom']):.6g} "
            f"| {float(row['force_mae_ev_per_angstrom']):.6g} "
            f"| {float(row['force_rmse_ev_per_angstrom']):.6g} "
            f"| {float(row['maximum_force_component_difference_ev_per_angstrom']):.6g} "
            f"| {branch_weight_difference} |"
        )
    convergence_notes = []
    for system in SYSTEM_NAMES:
        for precision in PRECISIONS:
            result = runs.get(system, {}).get(precision, {})
            if result.get("status") != "converged":
                convergence_notes.append(
                    f"- **{system}/{precision}:** {result.get('status', 'not run')}; "
                    f"steps={result.get('steps_taken', 'n/a')}; "
                    f"final max force={result.get('final_maximum_force_ev_per_angstrom', 'n/a')} "
                    f"eV/Å; error={result.get('error', 'none')}."
                )
    if not convergence_notes:
        convergence_notes = ["All ten full runs reached the stated force tolerance."]
    converged_systems = sum(
        bool(
            runs.get(system, {}).get("fp32", {}).get("converged", False)
            and runs.get(system, {}).get("fp64", {}).get("converged", False)
        )
        for system in SYSTEM_NAMES
    )
    max_energy_difference = max(
        float(row["absolute_energy_difference_ev"]) for row in single_points
    )
    max_force_component_difference = max(
        float(row["maximum_force_component_difference_ev_per_angstrom"])
        for row in single_points
    )
    max_force_rmse = max(
        float(row["force_rmse_ev_per_angstrom"]) for row in single_points
    )
    max_branch_weight_difference = (
        max(
            float(row["branch_weight_maximum_absolute_difference"])
            for row in single_points
        )
        if not single_mode
        else None
    )
    max_final_displacement = max(
        float(row["rms_displacement_angstrom"])
        for row in comparisons
        if row["rms_displacement_angstrom"] is not None
    )
    max_cross_precision_force = max(
        float(row["maximum_cross_precision_force_ev_per_angstrom"])
        for row in comparisons
        if row["maximum_cross_precision_force_ev_per_angstrom"] is not None
    )
    nonconverged_runs = [
        f"{system}/{precision}"
        for system in SYSTEM_NAMES
        for precision in PRECISIONS
        if not runs.get(system, {}).get(precision, {}).get("converged", False)
    ]
    branch_policy_text = (
        "For each system, the selected branch is fixed by the configured "
        "single-head selection rule; only that decoder head is evaluated during "
        "each relaxation force call. The BranchWeightMLP is not loaded or invoked."
        if single_mode
        else "All 16 source-specific prediction heads were included at every "
        "force step; the BranchWeightMLP softmax weights were fixed during a "
        "relaxation because composition does not change."
    )
    branch_selection_rows = "\n".join(
        f"| {system} | branch-{single_head_map[system]['branch_id']} | "
        f"{single_head_map[system]['dataset']} |"
        for system in SYSTEM_NAMES
    )
    branch_selection_section = (
        "## Single-head selection\n\n"
        "The selected heads below were fixed before relaxation using the "
        "highest-weight head in the already-validated FP64 fused single-point "
        "results. This uses the MLP only as an offline selection diagnostic; "
        "the single-head relaxation run does not load or call it. Branch/dataset "
        "IDs follow the training launch's `--multi_model_list` order.\n\n"
        "| System | Branch | Dataset |\n|---|---|---|\n"
        + branch_selection_rows
        if single_mode
        else ""
    )
    mlp_provenance_line = (
        "- BranchWeightMLP: not loaded or used in single-head mode.\n"
        if single_mode
        else f"- BranchWeightMLP checkpoint SHA-256: `{provenance_by_precision['fp64']['checkpoint_sha256']['branch_weight_mlp']}`\n"
    )
    weight_interpretation = (
        "Branch-weight differences are not applicable in single-head mode; "
        "the one-hot branch selection mask is not an MLP output."
        if single_mode
        else f"The maximum FP32–FP64 MLP branch-weight difference at the starting geometries was {max_branch_weight_difference:.3g}."
    )
    manuscript_branch_text = (
        "Each structure was evaluated using its fixed, selected source-specific "
        "head; the MLP was not loaded or invoked in the relaxation."
        if single_mode
        else "All 16 prediction branches were combined using the frozen BranchWeightMLP."
    )
    reviewer_branch_text = (
        "we selected one source-specific head per composition using the "
        "highest-weight FP64 fused single-point route, then ran the relaxation "
        "with that head alone and without loading or invoking the MLP."
        if single_mode
        else "all 16 source-specific heads and the frozen BranchWeightMLP were used."
    )
    branch_weight_claim = (
        ""
        if single_mode
        else f"; the maximum branch-weight difference was {max_branch_weight_difference:.3g}"
    )
    si_converged = bool(
        runs.get("Si", {}).get("fp32", {}).get("converged", False)
        and runs.get("Si", {}).get("fp64", {}).get("converged", False)
    )
    si_interpretation = (
        "Both Si runs met the force criterion."
        if si_converged
        else "At least one Si precision did not meet the force criterion."
    )
    si_manuscript_text = (
        "Both Si runs converged."
        if si_converged
        else (
            f"Si did not reach the 0.05 eV/Å force criterion within "
            f"{FULL_MAX_STEPS} ASE {optimizer_name.upper()} steps in either "
            "precision (final maximum forces "
            f"{runs['Si']['fp32']['final_maximum_force_ev_per_angstrom']:.3f} "
            "and "
            f"{runs['Si']['fp64']['final_maximum_force_ev_per_angstrom']:.3f} "
            "eV/Å, respectively)."
        )
    )
    if relax_cell:
        cell_protocol_text = (
            "Cells were variable for fully periodic systems and fixed for molecules. "
            f"ASE UnitCellFilter used {'hydrostatic' if hydrostatic_cell else 'full'} "
            "cell strain and a cell factor equal to the atom count; the optimizer "
            "criterion therefore includes atomic and generalized cell forces. "
            "Per-step stress tensors are in eV/Å³."
        )
    else:
        cell_protocol_text = "Cells and PBC were fixed."
    convergence_force_description = (
        "generalized optimization force"
        if relax_cell
        else "atomic force"
    )
    crystal_alignment_text = (
        "minimum-image mapping using the FP32 final cell for crystals"
        if relax_cell
        else "minimum-image mapping in the fixed cell for crystals"
    )

    report = f"""# FP32 vs FP64 HydraGNN structural-relaxation benchmark

## Scope and provenance

This benchmark evaluates only numerical precision robustness of the frozen
HydraGNN inference stack. It does not evaluate DFT accuracy, physical accuracy,
or molecular-dynamics stability. {branch_policy_text}

- HydraGNN checkpoint SHA-256: `{provenance_by_precision['fp64']['checkpoint_sha256']['hydragNN']}`
{mlp_provenance_line}- HydraGNN parameter dtypes: FP32 `{provenance_by_precision['fp32']['parameter_dtype']}`, FP64 `{provenance_by_precision['fp64']['parameter_dtype']}`
- Devices: FP32 `{provenance_by_precision['fp32']['device']}`, FP64 `{provenance_by_precision['fp64']['device']}`
- Branch order: `{provenance_by_precision['fp64']['branch_output_order']}`
- Graph radius: {provenance_by_precision['fp64']['radius_angstrom']} Å; maximum neighbors: {provenance_by_precision['fp64']['max_neighbours']}
- Energy and force units follow the existing inference output convention: eV and eV/Å.
- Initial structures: standard ASE H₂O/CH₄ molecules, diamond Si conventional cell,
  rocksalt MgO conventional cell, and FCC Al conventional cell. Each was
  perturbed once using seed 42, then the saved identical coordinates were used
  for both precisions. {cell_protocol_text}
- Optimizer: ASE {optimizer_name.upper()} (maximum step {optimizer_maxstep} Å),
  maximum 200 steps, convergence at a maximum {convergence_force_description}
  below 0.05 eV/Å. Failed and capped runs are retained in their trajectory folders.

{branch_selection_section}

## Initial single-point precision comparison

{chr(10).join(sp_lines)}

## Structural relaxation summary

{chr(10).join(table_lines)}

Unsuccessful runs:

{chr(10).join(convergence_notes)}

## Interpretation and limitations

Energy and force differences at identical geometries are reported above and in
`single_point_comparison.csv`. Final-geometry displacement is calculated after
translation/rotation alignment for molecules and {crystal_alignment_text}.
{("Final lattice vectors and per-step stress are also preserved in the trajectory CSVs." if relax_cell else "")} Both final geometries are also evaluated at a common
FP64 precision; cross-precision maximum residual forces are reported in the
summary table. Every row reports each precision's independent convergence
status and step count; the 0.05 eV/Å threshold is unchanged.

Across the five initial geometries, the largest absolute FP32–FP64 energy
difference was {max_energy_difference:.6g} eV and the largest force-component
difference was {max_force_component_difference:.6g} eV/Å. Both precisions
converged for {converged_systems}/5 systems. The nonconverged runs were
{", ".join(nonconverged_runs) if nonconverged_runs else "none"}; capped runs
must not be described as relaxed structures. The largest aligned final-geometry
RMS difference was {max_final_displacement:.6g} Å, and the largest
cross-precision maximum force was {max_cross_precision_force:.6g} eV/Å.
{weight_interpretation}
{si_interpretation} Agreement therefore supports numerical
consistency of this frozen model on these probes, not structural validity or
physical accuracy. No reference calculations were run, and the selected
compositions are not validated against the training distribution.

## Manuscript-ready wording

We compared FP32 and FP64 structural relaxations using the same frozen
HydraGNN checkpoint and identical perturbed starting geometries. {manuscript_branch_text}
Across the
five initial geometries, the largest absolute energy difference was
{max_energy_difference:.3g} eV and the largest force-component difference was
{max_force_component_difference:.3g} eV/Å (maximum force RMSE
{max_force_rmse:.3g} eV/Å){branch_weight_claim}. Both precisions converged for
{converged_systems}/5 systems; {si_manuscript_text} The maximum aligned FP32–FP64 final-geometry RMS difference was
{max_final_displacement:.3g} Å. These results characterize numerical precision
sensitivity of the pretrained potential on the tested structures; they do not
establish DFT-level accuracy or structural validity.

## Draft response to Reviewer 2, Comment 6

We thank the reviewer for requesting a direct FP32/FP64 relaxation comparison.
We performed paired relaxations of H₂O, CH₄, Si, MgO, and Al using identical
starting coordinates and the same frozen HydraGNN checkpoint; {reviewer_branch_text}
ASE {optimizer_name.upper()} used a 0.05 eV/Å criterion on the maximum
{convergence_force_description}.
We additionally compared initial energy and force predictions, checked the
energy-gradient force implementation by finite differences, and cross-evaluated
the final structures at the opposite precision. Both precisions converged for
{converged_systems}/5 systems. {si_manuscript_text}
The largest initial FP32–FP64 energy difference was {max_energy_difference:.3g}
eV, the largest force-component difference was
{max_force_component_difference:.3g} eV/Å (maximum force RMSE
{max_force_rmse:.3g} eV/Å){branch_weight_claim}. The largest aligned final geometry RMS
difference was {max_final_displacement:.3g} Å. We interpret these data as a
test of numerical consistency only, not as evidence of DFT accuracy or
structural validity.

## Reproducibility

- Single-point results: `stage_B/single_point_comparison.csv`
- Final geometry table: `stage_B/final_structure_comparison.csv`
- Per-run complete ASE trajectories (`.traj`), coordinate/force logs (`.csv`),
  optimizer logs, final `.extxyz`, and `run_summary.json`: `stage_B/<system>/<precision>/`
- Figures: `stage_B/figures/`
- Input references and saved perturbed structures: `inputs/`
- Runtime and checkpoint provenance: `provenance.json`, `stage_B/precision_provenance.json`
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "full"), required=True)
    parser.add_argument(
        "--head-mode",
        choices=("fused", "single"),
        default="fused",
        help="Use the BranchWeightMLP-weighted 16-head ensemble (fused) or one fixed head without loading the MLP (single).",
    )
    parser.add_argument(
        "--head-override",
        action="append",
        default=[],
        metavar="SYSTEM=BRANCH_ID",
        help="Override the configured single-head selection; repeat per system. Example: --head-override H2O=8",
    )
    parser.add_argument(
        "--relax-cell",
        action="store_true",
        help="Relax the lattice vectors for fully periodic systems; molecular cells remain fixed.",
    )
    parser.add_argument(
        "--hydrostatic-cell",
        action="store_true",
        help="Constrain --relax-cell to hydrostatic (volume-only) strain.",
    )
    parser.add_argument(
        "--optimizer",
        choices=("bfgs", "fire"),
        default="bfgs",
        help="ASE optimizer to use (default: bfgs).",
    )
    parser.add_argument(
        "--optimizer-maxstep",
        type=float,
        default=0.2,
        help="Maximum optimizer step in angstroms (default: 0.2).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Run directory. Pilot creates it; full resumes the same directory after pilot.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.hydrostatic_cell and not args.relax_cell:
        raise ValueError("--hydrostatic-cell requires --relax-cell")
    if not np.isfinite(args.optimizer_maxstep) or args.optimizer_maxstep <= 0:
        raise ValueError("--optimizer-maxstep must be a positive finite number")
    if args.head_mode == "single":
        single_head_map = _resolve_single_head_map(args.head_override)
    else:
        if args.head_override:
            raise ValueError("--head-override can only be used with --head-mode single")
        single_head_map = _resolve_single_head_map([])
    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = HERE / "results" / f"run_{stamp}"
    else:
        output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "pilot":
        if (output_dir / "stage_A" / "stage_a_status.json").exists():
            raise FileExistsError(f"Stage A already exists in {output_dir}; refusing to overwrite")
        inputs = _prepare_inputs(output_dir)
        provenance = _runtime_provenance()
        _json_dump(output_dir / "provenance.json", provenance)
        _json_dump(output_dir / "input_manifest.json", inputs)
        run_config = json.loads(json.dumps(BENCHMARK_CONFIG))
        run_config["relaxation"]["cell_relaxation"] = args.relax_cell
        run_config["relaxation"]["hydrostatic_cell"] = args.hydrostatic_cell
        run_config["relaxation"]["optimizer"] = args.optimizer
        run_config["relaxation"]["optimizer_maxstep_angstrom"] = args.optimizer_maxstep
        _json_dump(output_dir / "benchmark_config.json", run_config)
        _run_stage_a(
            output_dir,
            {name: _read_initial(output_dir, name) for name in SYSTEM_NAMES},
            head_mode=args.head_mode,
            single_head_map=single_head_map,
            relax_cell=args.relax_cell,
            optimizer_name=args.optimizer,
            optimizer_maxstep=args.optimizer_maxstep,
            hydrostatic_cell=args.hydrostatic_cell,
        )
        return

    stage_a_path = output_dir / "stage_A" / "stage_a_status.json"
    if not stage_a_path.is_file():
        raise RuntimeError(
            "Stage B requires a passing Stage A pilot in the same --output-dir"
        )
    initial_atoms = {name: _read_initial(output_dir, name) for name in SYSTEM_NAMES}
    _run_stage_b(
        output_dir,
        initial_atoms,
        head_mode=args.head_mode,
        single_head_map=single_head_map,
        relax_cell=args.relax_cell,
        optimizer_name=args.optimizer,
        optimizer_maxstep=args.optimizer_maxstep,
        hydrostatic_cell=args.hydrostatic_cell,
    )


if __name__ == "__main__":
    main()
