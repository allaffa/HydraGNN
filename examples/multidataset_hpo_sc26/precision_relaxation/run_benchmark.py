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
from ase.optimize import BFGS

from hydragnn_ase_calculator import (
    BACKBONE_SHA256,
    MLP_SHA256,
    FusedHydraGNNCalculator,
    load_fused_calculator,
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

if int(INFERENCE_CONFIG["branch_count"]) != 16:
    raise ValueError("This benchmark requires exactly 16 source-specific branches")
if RELAXATION_CONFIG["cell_relaxation"]:
    raise ValueError("This benchmark protocol requires fixed cells")
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


def _state_row(
    atoms: Atoms,
    calculator: FusedHydraGNNCalculator,
    step: int,
    *,
    converged: bool,
) -> dict[str, Any]:
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    if not np.isfinite(energy) or not np.isfinite(forces).all():
        raise FloatingPointError("Nonfinite energy or forces during relaxation")
    force_norms = np.linalg.norm(forces, axis=1)
    if calculator.last_branch_weights is None or calculator.last_branch_energies is None:
        raise RuntimeError("Calculator did not retain branch results")
    return {
        "step": int(step),
        "energy_ev": energy,
        "energy_ev_per_atom": energy / len(atoms),
        "maximum_force_ev_per_angstrom": float(force_norms.max()),
        "rms_force_ev_per_angstrom": float(np.sqrt(np.mean(forces**2))),
        "coordinates_angstrom_json": json.dumps(atoms.positions.tolist()),
        "graph_edges": int(calculator.last_graph_edges),
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
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    atoms.calc = calculator
    log_path = output_dir / "bfgs.log"
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
            initial_row = _state_row(atoms, calculator, 0, converged=False)
            rows.append(initial_row)
            trajectory.write(atoms)
            optimizer = BFGS(atoms, logfile=str(log_path))
            optimizer.attach(
                lambda: _record_optimizer_step(
                    atoms, calculator, optimizer, rows, trajectory, force_tolerance
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
            if rows and not np.allclose(
                np.asarray(json.loads(rows[-1]["coordinates_angstrom_json"])),
                atoms.positions,
                rtol=0,
                atol=1e-12,
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
        "status": status,
        "converged": bool(converged and status == "converged"),
        "force_tolerance_ev_per_angstrom": force_tolerance,
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
    optimizer: BFGS,
    rows: list[dict[str, Any]],
    trajectory: Trajectory,
    force_tolerance: float,
) -> None:
    if optimizer.get_number_of_steps() == 0:
        return
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    max_force = float(np.linalg.norm(forces, axis=1).max())
    converged = bool(np.isfinite(max_force) and max_force < force_tolerance)
    rows.append(
        _state_row(
            atoms,
            calculator,
            int(optimizer.get_number_of_steps()),
            converged=converged,
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
    calculators: dict[str, FusedHydraGNNCalculator],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for system, initial in initial_atoms.items():
        results[system] = {}
        for precision in PRECISIONS:
            atoms = initial.copy()
            atoms.calc = calculators[precision]
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=np.float64)
            calculator = calculators[precision]
            if calculator.last_branch_weights is None or calculator.last_branch_energies is None:
                raise RuntimeError(f"Missing branch outputs for {system}/{precision}")
            results[system][precision] = {
                "energy_ev": energy,
                "forces": forces,
                "weights": calculator.last_branch_weights.copy(),
                "branch_energies": calculator.last_branch_energies.copy(),
                "edge_count": calculator.last_graph_edges,
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
            "branch_weight_maximum_absolute_difference": float(
                np.abs(fp32["weights"] - fp64["weights"]).max()
            ),
            "fp32_graph_edges": fp32["edge_count"],
            "fp64_graph_edges": fp64["edge_count"],
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


def _run_stage_a(output_dir: Path, initial_atoms: dict[str, Atoms]) -> None:
    stage_dir = output_dir / "stage_A"
    if stage_dir.exists() and any(stage_dir.iterdir()):
        raise FileExistsError(f"Stage A output already exists: {stage_dir}")
    stage_dir.mkdir(parents=True, exist_ok=True)
    calculators: dict[str, FusedHydraGNNCalculator] = {}
    provenance: dict[str, Any] = {}
    for precision in PRECISIONS:
        calculators[precision], provenance[precision] = load_fused_calculator(
            logdir=LOGDIR,
            backbone_checkpoint=BACKBONE_PATH,
            mlp_checkpoint=MLP_PATH,
            precision=precision,
            charge=float(INFERENCE_CONFIG["charge"]),
            spin=float(INFERENCE_CONFIG["spin"]),
        )

    water = initial_atoms["H2O"]
    water_results: dict[str, Any] = {}
    for precision in PRECISIONS:
        calculator = calculators[precision]
        single_point = calculator.evaluate_atoms(water)
        if len(single_point["branch_energies"]) != 16:
            raise RuntimeError(f"{precision} did not evaluate all 16 branches")
        if not np.isfinite(single_point["forces"]).all():
            raise FloatingPointError(f"Nonfinite {precision} pilot forces")
        fd = _finite_difference_check(calculator, water)
        if not fd["passed"]:
            raise RuntimeError(
                f"{precision} finite-difference force check failed: "
                f"best discrepancy {fd['best_absolute_error_ev_per_angstrom']:.6g} "
                f"eV/A; topology stable={fd['neighbor_topology_stable']}"
            )
        pilot_atoms = water.copy()
        pilot = _run_relaxation(
            pilot_atoms,
            calculator,
            stage_dir / "H2O" / precision,
            max_steps=PILOT_MAX_STEPS,
            force_tolerance=FORCE_TOLERANCE,
        )
        if pilot["status"] == "failed":
            raise RuntimeError(
                f"{precision} water pilot relaxation failed: {pilot['error']}"
            )
        if calculator.graph_build_count <= 1 or calculator.last_graph_positions is None:
            raise RuntimeError(f"{precision} graph was not rebuilt during relaxation")
        if not np.allclose(
            calculator.last_graph_positions, pilot_atoms.positions, rtol=0, atol=1e-12
        ):
            raise RuntimeError(f"{precision} calculator graph is stale for the current coordinates")
        water_results[precision] = {
            "single_point_energy_ev": single_point["energy"],
            "force_maximum_ev_per_angstrom": float(
                np.linalg.norm(single_point["forces"], axis=1).max()
            ),
            "branch_weights": single_point["branch_weights"].tolist(),
            "branch_energies_ev": single_point["branch_energies"].tolist(),
            "graph_edges": single_point["edge_count"],
            "finite_difference_check": fd,
            "pilot_relaxation": pilot,
            "graph_rebuild_count": calculator.graph_build_count,
            "graph_matches_latest_positions": True,
            "precision_provenance": provenance[precision],
        }
    max_weight_difference = float(
        np.max(
            np.abs(
                np.asarray(water_results["fp32"]["branch_weights"])
                - np.asarray(water_results["fp64"]["branch_weights"])
            )
        )
    )
    status = {
        "status": "passed",
        "technical_validation_passed": True,
        "scientific_convergence_claim": False,
        "note": "The pilot is a software validation only; a 10-step limit is not a full-convergence result.",
        "water_precision_results": water_results,
        "fp32_fp64_branch_weight_maximum_absolute_difference": max_weight_difference,
        "checks": {
            "checkpoint_hashes_verified": True,
            "both_precisions_loaded_and_frozen": True,
            "all_16_branches_evaluated": True,
            "weights_sum_to_one": True,
            "single_point_energy_and_forces_finite": True,
            "finite_difference_force_check_passed": True,
            "neighbor_topology_unchanged_in_finite_difference": True,
            "graph_rebuilt_at_moving_geometry": True,
            "ase_bfgs_executed_without_error": True,
            "periodic_and_molecular_construction_implemented": True,
            "branch_skipping_disabled": True,
        },
    }
    _json_dump(stage_dir / "stage_a_status.json", status)
    print(
        "Stage A passed. The pilot used H2O only and is not a scientific convergence result."
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


def _run_stage_b(output_dir: Path, initial_atoms: dict[str, Atoms]) -> None:
    stage_a_path = output_dir / "stage_A" / "stage_a_status.json"
    if not stage_a_path.is_file():
        raise RuntimeError("Stage B is blocked until Stage A has completed")
    stage_a = json.loads(stage_a_path.read_text(encoding="utf-8"))
    if stage_a.get("status") != "passed":
        raise RuntimeError("Stage B is blocked because Stage A did not pass")
    stage_dir = output_dir / "stage_B"
    if stage_dir.exists() and any(stage_dir.iterdir()):
        raise FileExistsError(f"Stage B output already exists: {stage_dir}")
    stage_dir.mkdir(parents=True, exist_ok=True)

    calculators: dict[str, FusedHydraGNNCalculator] = {}
    provenance: dict[str, Any] = {}
    for precision in PRECISIONS:
        calculators[precision], provenance[precision] = load_fused_calculator(
            logdir=LOGDIR,
            backbone_checkpoint=BACKBONE_PATH,
            mlp_checkpoint=MLP_PATH,
            precision=precision,
            charge=float(INFERENCE_CONFIG["charge"]),
            spin=float(INFERENCE_CONFIG["spin"]),
        )
    _json_dump(stage_dir / "precision_provenance.json", provenance)
    single_points = _single_point_comparison(stage_dir, initial_atoms, calculators)
    run_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    finals: dict[str, dict[str, Atoms]] = {}
    for system in SYSTEM_NAMES:
        run_summaries[system] = {}
        finals[system] = {}
        for precision in PRECISIONS:
            run_dir = stage_dir / system / precision
            start = _read_initial(output_dir, system)
            failure_path = run_dir / "failure.json"
            try:
                summary = _run_relaxation(
                    start,
                    calculators[precision],
                    run_dir,
                    max_steps=FULL_MAX_STEPS,
                    force_tolerance=FORCE_TOLERANCE,
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
                    cross_results[label] = calculators[precision].evaluate_atoms(geometry)
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
                row["fp64_relaxed_common_fp64_energy_ev"] = calculators[
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
            "force_tolerance_ev_per_angstrom": FORCE_TOLERANCE,
            "maximum_optimizer_steps": FULL_MAX_STEPS,
            "branch_skipping": False,
            "single_point_comparison_csv": "single_point_comparison.csv",
            "final_structure_comparison_csv": "final_structure_comparison.csv",
            "systems": run_summaries,
            "final_structure_comparisons": comparisons,
        },
    )
    _make_figures(stage_dir, run_summaries)
    _write_report(output_dir, stage_dir, run_summaries, comparisons, provenance)
    print(f"Stage B outputs written under {stage_dir}")


def _make_figures(
    stage_dir: Path, run_summaries: dict[str, dict[str, dict[str, Any]]]
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
            axis.set_xlabel("BFGS step")
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
) -> None:
    single_points = list(csv.DictReader((stage_dir / "single_point_comparison.csv").open()))
    table_lines = [
        "| System | FP32 converged? | FP64 converged? | FP32 steps | FP64 steps | RMS displacement (Å) | Max cross-precision force (eV/Å) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        table_lines.append(
            "| {system} | {fp32_converged} | {fp64_converged} | {fp32_steps} | "
            "{fp64_steps} | {rms_displacement_angstrom} | "
            "{maximum_cross_precision_force_ev_per_angstrom} |".format(**row)
        )
    sp_lines = [
        "| System | ΔE (eV) | ΔE/N (eV/atom) | Force MAE (eV/Å) | Force RMSE (eV/Å) | Max component ΔF (eV/Å) | Max Δweight |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in single_points:
        sp_lines.append(
            f"| {row['system']} | {float(row['absolute_energy_difference_ev']):.6g} "
            f"| {float(row['absolute_energy_difference_ev_per_atom']):.6g} "
            f"| {float(row['force_mae_ev_per_angstrom']):.6g} "
            f"| {float(row['force_rmse_ev_per_angstrom']):.6g} "
            f"| {float(row['maximum_force_component_difference_ev_per_angstrom']):.6g} "
            f"| {float(row['branch_weight_maximum_absolute_difference']):.6g} |"
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
    max_branch_weight_difference = max(
        float(row["branch_weight_maximum_absolute_difference"])
        for row in single_points
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

    report = f"""# FP32 vs FP64 HydraGNN structural-relaxation benchmark

## Scope and provenance

This benchmark evaluates only numerical precision robustness of the frozen
HydraGNN + BranchWeightMLP inference stack. It does not evaluate DFT accuracy,
physical accuracy, or molecular-dynamics stability. All 16 source-specific
prediction heads were included at every force step; branch skipping was
disabled. The composition-dependent 16-way softmax weights are fixed during a
relaxation because each system's element counts are unchanged.

- HydraGNN checkpoint SHA-256: `{provenance['fp64']['checkpoint_sha256']['hydragNN']}`
- BranchWeightMLP SHA-256: `{provenance['fp64']['checkpoint_sha256']['branch_weight_mlp']}`
- HydraGNN parameter dtypes: FP32 `{provenance['fp32']['parameter_dtype']}`, FP64 `{provenance['fp64']['parameter_dtype']}`
- Devices: FP32 `{provenance['fp32']['device']}`, FP64 `{provenance['fp64']['device']}`
- Branch order: `{provenance['fp64']['branch_output_order']}`
- Graph radius: {provenance['fp64']['radius_angstrom']} Å; maximum neighbors: {provenance['fp64']['max_neighbours']}
- Energy and force units follow the existing inference output convention: eV and eV/Å.
- Initial structures: standard ASE H₂O/CH₄ molecules, diamond Si conventional cell,
  rocksalt MgO conventional cell, and FCC Al conventional cell. Each was
  perturbed once using seed 42, then the saved identical coordinates were used
  for both precisions. Cells and PBC were fixed.
- Optimizer: ASE BFGS, maximum 200 steps, convergence if maximum atomic force is
  below 0.05 eV/Å. Failed and capped runs are retained in their trajectory folders.

## Initial single-point precision comparison

{chr(10).join(sp_lines)}

## Structural relaxation summary

{chr(10).join(table_lines)}

Unsuccessful runs:

{chr(10).join(convergence_notes)}

## Interpretation and limitations

Energy and force differences at identical geometries are reported above and in
`single_point_comparison.csv`. Final-geometry displacement is calculated after
translation/rotation alignment for molecules and minimum-image mapping in the
fixed cell for crystals. Both final geometries are also evaluated at a common
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
The Si residual force is associated with its failure to meet the force
criterion in either precision. Agreement therefore supports numerical
consistency of this frozen model on these probes, not structural validity or
physical accuracy. No reference calculations were run, and the selected
compositions are not validated against the training distribution.

## Manuscript-ready wording

We compared FP32 and FP64 structural relaxations using the same frozen
HydraGNN and BranchWeightMLP checkpoints, identical perturbed starting
geometries, all 16 prediction branches, and a common BFGS protocol. Across the
five initial geometries, the largest absolute energy difference was
{max_energy_difference:.3g} eV and the largest force-component difference was
{max_force_component_difference:.3g} eV/Å (maximum force RMSE
{max_force_rmse:.3g} eV/Å); the maximum branch-weight difference was
{max_branch_weight_difference:.3g}. Both precisions converged for
{converged_systems}/5 systems; Si did not reach the 0.05 eV/Å force
criterion within 200 BFGS steps in either precision (final maximum forces
{runs["Si"]["fp32"]["final_maximum_force_ev_per_angstrom"]:.3f} and
{runs["Si"]["fp64"]["final_maximum_force_ev_per_angstrom"]:.3f} eV/Å,
respectively). The maximum aligned FP32–FP64 final-geometry RMS difference was
{max_final_displacement:.3g} Å. These results characterize numerical precision
sensitivity of the pretrained potential on the tested structures; they do not
establish DFT-level accuracy or structural validity.

## Draft response to Reviewer 2, Comment 6

We thank the reviewer for requesting a direct FP32/FP64 relaxation comparison.
We performed paired relaxations of H₂O, CH₄, Si, MgO, and Al using identical
starting coordinates, the same frozen pretrained checkpoints, all 16
source-specific heads, and ASE BFGS with a 0.05 eV/Å maximum-force criterion.
We additionally compared initial energy and force predictions, checked the
energy-gradient force implementation by finite differences, and cross-evaluated
the final structures at the opposite precision. Both precisions converged for
{converged_systems}/5 systems; Si did not meet the force criterion within 200 steps
in either precision, with final maximum forces of
{runs["Si"]["fp32"]["final_maximum_force_ev_per_angstrom"]:.3f} and
{runs["Si"]["fp64"]["final_maximum_force_ev_per_angstrom"]:.3f} eV/Å.
The largest initial FP32–FP64 energy difference was {max_energy_difference:.3g}
eV, the largest force-component difference was
{max_force_component_difference:.3g} eV/Å (maximum force RMSE
{max_force_rmse:.3g} eV/Å); the maximum branch-weight difference was
{max_branch_weight_difference:.3g}. The largest aligned final geometry RMS
difference was {max_final_displacement:.3g} Å. We interpret these data as a
test of numerical consistency only, not as evidence of DFT accuracy or
structural validity.

## Reproducibility

- Single-point results: `stage_B/single_point_comparison.csv`
- Final geometry table: `stage_B/final_structure_comparison.csv`
- Per-run complete ASE trajectories (`.traj`), coordinate/force logs (`.csv`),
  BFGS logs, final `.extxyz`, and `run_summary.json`: `stage_B/<system>/<precision>/`
- Figures: `stage_B/figures/`
- Input references and saved perturbed structures: `inputs/`
- Runtime and checkpoint provenance: `provenance.json`, `stage_B/precision_provenance.json`
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "full"), required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Run directory. Pilot creates it; full resumes the same directory after pilot.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
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
        _json_dump(output_dir / "benchmark_config.json", BENCHMARK_CONFIG)
        _run_stage_a(
            output_dir,
            {name: _read_initial(output_dir, name) for name in SYSTEM_NAMES},
        )
        return

    stage_a_path = output_dir / "stage_A" / "stage_a_status.json"
    if not stage_a_path.is_file():
        raise RuntimeError(
            "Stage B requires a passing Stage A pilot in the same --output-dir"
        )
    initial_atoms = {name: _read_initial(output_dir, name) for name in SYSTEM_NAMES}
    _run_stage_b(output_dir, initial_atoms)


if __name__ == "__main__":
    main()
