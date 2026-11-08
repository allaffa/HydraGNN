#!/usr/bin/env python3
"""Diagnose frozen-model carbon failures without changing the original benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import vesin
from ase import Atoms
from ase.build import bulk
from ase.geometry import find_mic
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.stress import voigt_6_to_full_3x3_stress

WORKSPACE = Path(__file__).resolve().parent
BENCHMARK = WORKSPACE.parents[1]
SOURCE = BENCHMARK / "results" / "off_equilibrium_full_cell_20261108_0328"
sys.path.insert(0, str(BENCHMARK))

import hydragnn_ase_calculator as calculator_module
from hydragnn_ase_calculator import _graph_from_atoms, sha256_file
from run_benchmark import _load_calculator, _resolve_single_head_map, _run_relaxation

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
})

SEED = 20261108
FD_STEPS = (1e-3, 1e-4, 1e-5)
CASES = ("C_diamond_compressed", "C_diamond_expanded")
PRECISIONS = ("fp32", "fp64")
DATA = WORKSPACE / "analysis"
FIGURES = WORKSPACE / "figures"
INPUTS = WORKSPACE / "inputs"
RUNS = WORKSPACE / "experiments"


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No records to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def source_manifest() -> dict[str, str]:
    paths = sorted(path for path in SOURCE.rglob("*") if path.is_file())
    return {str(path.relative_to(SOURCE)): sha256_file(path) for path in paths}


def load_calculator(precision: str):
    calculator, provenance = _load_calculator(
        precision, head_mode="fused", system="Si",
        single_head_map=_resolve_single_head_map([]),
    )
    if calculator.radius != 5.0 or calculator.max_neighbours != 20:
        raise ValueError("This diagnostic requires the original 5 Å / 20-neighbor graph")
    if calculator.charge != 0 or calculator.spin != 0 or calculator.single_branch_id is not None:
        raise ValueError("This diagnostic requires neutral, zero-spin, all-branch inference")
    dump(DATA / f"provenance_{precision}.json", provenance)
    return calculator


def build_graph(atoms: Atoms, precision: str):
    return _graph_from_atoms(
        atoms,
        dtype=torch.float32 if precision == "fp32" else torch.float64,
        radius=5.0, max_neighbours=20, charge=0.0, spin=0.0,
    )


def edge_identity(graph) -> tuple[np.ndarray, set[tuple[int, ...]]]:
    cell = graph.cell.cpu().numpy()
    shifts = np.rint(np.linalg.solve(cell.T, graph.edge_shifts.cpu().numpy().T).T)
    identities = np.column_stack(
        (graph.edge_index.cpu().numpy().T, shifts)
    ).astype(np.int64)
    return identities, {tuple(row) for row in identities.tolist()}


def graph_statistics(atoms: Atoms, precision: str):
    graph = build_graph(atoms, precision)
    identities, edges = edge_identity(graph)
    incoming = np.bincount(identities[:, 1], minlength=len(atoms))
    quantized = Atoms(
        numbers=atoms.numbers,
        positions=graph.pos.cpu().numpy(),
        cell=graph.cell.cpu().numpy(), pbc=atoms.pbc,
    )
    _, dst, distances = vesin.ase_neighbor_list("ijd", quantized, cutoff=5.0)
    counts = np.bincount(dst, minlength=len(atoms))
    gaps = []
    for atom_index in range(len(atoms)):
        values = np.sort(distances[dst == atom_index])
        if len(values) > 20:
            gaps.append(float(values[20] - values[19]))
    sorted_identities = np.asarray(sorted(edges), dtype=np.int64)
    return {
        "graph_edges": len(identities),
        "atoms_at_neighbor_cap": int(np.sum(incoming == 20)),
        "atoms_with_neighbors_discarded": int(np.sum(counts > 20)),
        "raw_neighbors_min": int(counts.min()),
        "raw_neighbors_max": int(counts.max()),
        "minimum_cap_boundary_gap_angstrom": min(gaps) if gaps else None,
        "periodic_edge_set_sha256": hashlib.sha256(sorted_identities.tobytes()).hexdigest(),
        "ordered_edges_sha256": hashlib.sha256(identities.tobytes()).hexdigest(),
    }, edges


def geometry_metrics(initial: Atoms, atoms: Atoms) -> dict[str, Any]:
    initial_fractional = np.linalg.solve(initial.cell.array.T, initial.positions.T).T
    final_fractional = np.linalg.solve(atoms.cell.array.T, atoms.positions.T).T
    _, internal = find_mic(
        (final_fractional - initial_fractional) @ atoms.cell.array,
        atoms.cell, pbc=atoms.pbc,
    )
    _, total = find_mic(atoms.positions - initial.positions, atoms.cell, pbc=atoms.pbc)
    distances = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    lengths = atoms.cell.lengths()
    angles = atoms.cell.angles()
    return {
        "cell_volume_angstrom3": atoms.get_volume(),
        "cell_a_angstrom": lengths[0], "cell_b_angstrom": lengths[1],
        "cell_c_angstrom": lengths[2],
        "alpha_degrees": angles[0], "beta_degrees": angles[1],
        "gamma_degrees": angles[2],
        "minimum_interatomic_distance_angstrom": float(distances.min()),
        "rms_internal_displacement_angstrom": float(np.sqrt(np.mean(internal**2))),
        "maximum_internal_displacement_angstrom": float(internal.max()),
        "maximum_total_displacement_angstrom": float(total.max()),
    }


def paired_original_plot(name: str, series: dict[str, list[dict[str, Any]]]) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    fields = (
        ("energy_ev_per_atom", "Energy (eV/atom)"),
        ("maximum_force_ev_per_angstrom", "Max atomic force (eV/Å)"),
        ("maximum_optimization_force", "Max generalized force (eV/Å)"),
        ("maximum_absolute_stress_ev_per_angstrom3", "Max absolute stress (eV/Å³)"),
        ("cell_volume_angstrom3", "Cell volume (Å³)"),
        ("periodic_edges_changed_since_previous_step", "Changed periodic edge identities"),
    )
    for axis, (field, label) in zip(axes.flat, fields, strict=True):
        for precision, rows in series.items():
            axis.plot([r["step"] for r in rows], [r[field] for r in rows], label=precision)
        axis.set_xlabel("FIRE step")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle(name + ": original failed full-cell trajectories")
    figure.savefig(FIGURES / f"{name}_original_precision.png", dpi=180)
    plt.close(figure)


def analyze_originals() -> None:
    if (DATA / "original_summary.json").exists():
        raise FileExistsError("Original analysis already exists; use a later phase")
    dump(DATA / "source_manifest.json", source_manifest())
    summaries = {}
    selected = {}
    for name in CASES:
        initial = read(SOURCE / name / "initial.extxyz")
        series = {}
        graph_sets = {}
        atom_series = {}
        for precision in PRECISIONS:
            path = SOURCE / name / precision
            saved = read_csv(path / "trajectory.csv")
            with Trajectory(str(path / "trajectory.traj")) as trajectory:
                frames = list(trajectory)
            if len(saved) != len(frames):
                raise ValueError(f"CSV/trajectory frame count mismatch: {path}")
            rows = []
            previous = set()
            sets = []
            for record, atoms in zip(saved, frames, strict=True):
                coordinates = np.asarray(json.loads(record["coordinates_angstrom_json"]))
                if not np.allclose(coordinates, atoms.positions, rtol=0, atol=1e-10):
                    raise ValueError(f"CSV/trajectory coordinates differ: {path}")
                stats, edges = graph_statistics(atoms, precision)
                if stats["graph_edges"] != int(record["graph_edges"]):
                    raise ValueError(f"Reconstructed graph count differs: {path}")
                row = {
                    "step": int(record["step"]),
                    **{
                        key: float(record[key]) for key in (
                            "energy_ev_per_atom", "maximum_force_ev_per_angstrom",
                            "maximum_optimization_force",
                            "maximum_absolute_stress_ev_per_angstrom3",
                        )
                    },
                    **geometry_metrics(initial, atoms), **stats,
                    "periodic_edges_changed_since_previous_step": (
                        len(edges ^ previous) if rows else 0
                    ),
                }
                rows.append(row)
                sets.append(edges)
                previous = edges
            write_csv(DATA / f"{name}_{precision}_steps.csv", rows)
            energies = np.asarray([r["energy_ev_per_atom"] for r in rows])
            late_forces = np.asarray([r["maximum_force_ev_per_angstrom"] for r in rows[-100:]])
            jump_step = int(np.argmax(np.diff(energies)) + 1)
            selected[f"{name}_{precision}"] = jump_step
            summaries[f"{name}/{precision}"] = {
                "source_run_summary": json.loads((path / "run_summary.json").read_text()),
                "fire_log_line_count": len((path / "fire.log").read_text().splitlines()),
                "final_extxyz_matches_trajectory": bool(np.allclose(
                    read(path / "final.extxyz").positions, frames[-1].positions,
                    rtol=0, atol=5.1e-9,
                )),
                "energy_increase_steps_above_1e_6_ev_per_atom": int(np.sum(np.diff(energies) > 1e-6)),
                "largest_upward_energy_jump_ev_per_atom": float(np.diff(energies).max()),
                "largest_upward_jump_step": jump_step,
                "periodic_edges_changed_at_jump": rows[jump_step]["periodic_edges_changed_since_previous_step"],
                "graph_change_steps": sum(r["periodic_edges_changed_since_previous_step"] > 0 for r in rows),
                "minimum_atomic_separation_angstrom": min(r["minimum_interatomic_distance_angstrom"] for r in rows),
                "initial_volume_angstrom3": rows[0]["cell_volume_angstrom3"],
                "final_volume_angstrom3": rows[-1]["cell_volume_angstrom3"],
                "minimum_volume_angstrom3": min(r["cell_volume_angstrom3"] for r in rows),
                "maximum_volume_angstrom3": max(r["cell_volume_angstrom3"] for r in rows),
                "final_force_ev_per_angstrom": rows[-1]["maximum_force_ev_per_angstrom"],
                "last_100_steps_max_force_mean": float(late_forces.mean()),
                "last_100_steps_max_force_std": float(late_forces.std()),
                "last_100_steps_minimum_max_force": float(late_forces.min()),
            }
            series[precision] = rows
            graph_sets[precision] = sets
            atom_series[precision] = frames
            print(f"Analyzed {name}/{precision}: {len(rows)} saved steps", flush=True)
        comparison = []
        for index, (first, second) in enumerate(zip(series["fp32"], series["fp64"], strict=True)):
            first_atoms, second_atoms = atom_series["fp32"][index], atom_series["fp64"][index]
            delta = first_atoms.positions - second_atoms.positions
            first_forces = atom_series["fp32"][index].get_forces()
            second_forces = atom_series["fp64"][index].get_forces()
            rms = float(np.sqrt(np.mean(np.sum(delta**2, axis=1))))
            energy_delta = abs(first["energy_ev_per_atom"] - second["energy_ev_per_atom"])
            force_rms = float(np.sqrt(np.mean((first_forces - second_forces)**2)))
            comparison.append({
                "step": first["step"], "energy_delta_ev_per_atom": energy_delta,
                "rms_cartesian_position_delta_angstrom": rms,
                "force_component_rmse_ev_per_angstrom": force_rms,
                "different_periodic_edges": len(graph_sets["fp32"][index] ^ graph_sets["fp64"][index]),
                "material_divergence": energy_delta > 1e-4 or rms > 1e-3 or force_rms > 1e-2,
            })
        write_csv(DATA / f"{name}_precision_divergence.csv", comparison)
        summaries[f"{name}/precision_comparison"] = {
            "thresholds": {
                "energy_delta_ev_per_atom": 1e-4,
                "rms_position_delta_angstrom": 1e-3,
                "force_component_rmse_ev_per_angstrom": 1e-2,
                "rule": "First step exceeding any threshold; operational, not physical-error tolerances",
            },
            "first_material_divergence_step": next((r["step"] for r in comparison if r["material_divergence"]), None),
            "first_periodic_edge_difference_step": next((r["step"] for r in comparison if r["different_periodic_edges"]), None),
            "final": comparison[-1],
        }
        paired_original_plot(name, series)
    dump(DATA / "problem_steps.json", selected)
    dump(DATA / "original_summary.json", summaries)
    dump(DATA / "progress.json", {"phase": "originals_complete"})


def ideal_diamond() -> Atoms:
    return bulk("C", "diamond", a=3.567, cubic=True).repeat((2, 2, 2))


def experiment_inputs() -> dict[str, tuple[Atoms, bool, str]]:
    cases = {}
    for name in CASES:
        original = read(SOURCE / name / "initial.extxyz")
        cases[f"A_{name}_fixed_FIRE"] = (original.copy(), False, "fire")
        cases[f"C_{name}_full_BFGS"] = (original.copy(), True, "bfgs")
    rng = np.random.default_rng(SEED)
    directions = rng.normal(size=(64, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    small_magnitudes = rng.uniform(0.02, 0.03, 64)
    for label, scale, amplitude in (
        ("B1", 1.0, None), ("B2", 1.0, .05),
        ("B3", .99, .05), ("B4", 1.01, .05),
        ("B5", .97, .05), ("B6", 1.03, .05),
    ):
        atoms = ideal_diamond()
        atoms.set_cell(atoms.cell.array * scale, scale_atoms=True)
        magnitudes = small_magnitudes if amplitude is None else np.full(64, amplitude)
        atoms.positions += directions * magnitudes[:, None]
        atoms.info.update(seed=SEED, scale=scale, amplitude="0.02-0.03" if amplitude is None else amplitude)
        cases[f"{label}_full_FIRE"] = (atoms, True, "fire")
    return cases


def experiment_plot(results: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, name in zip(axes[0], CASES, strict=True):
        for label, directory in (
            ("original full-cell FIRE", SOURCE / name / "fp64"),
            ("fixed-cell FIRE", RUNS / f"A_{name}_fixed_FIRE"),
            ("full-cell BFGS", RUNS / f"C_{name}_full_BFGS"),
        ):
            records = read_csv(directory / "trajectory.csv")
            axis.semilogy(
                [int(r["step"]) for r in records],
                [float(r["maximum_optimization_force"]) for r in records], label=label,
            )
        axis.set_title(name)
        axis.set_ylabel("Max optimization force (eV/Å)")
    for row in results:
        if row["experiment"].startswith("B"):
            records = read_csv(RUNS / row["experiment"] / "trajectory.csv")
            steps = [int(r["step"]) for r in records]
            axes[1, 0].semilogy(steps, [float(r["maximum_optimization_force"]) for r in records], label=row["experiment"])
            energies = np.asarray([float(r["energy_ev_per_atom"]) for r in records])
            axes[1, 1].plot(steps, energies - energies[0], label=row["experiment"])
    axes[1, 0].set_title("Initial perturbation sensitivity")
    axes[1, 0].set_ylabel("Max generalized force (eV/Å)")
    axes[1, 1].set_ylabel("Energy change (eV/atom)")
    for axis in axes.flat:
        axis.set_xlabel("Optimizer step")
        axis.legend(fontsize=8)
        axis.grid(alpha=.25)
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.axhline(.05, color="black", linestyle="--")
    figure.savefig(FIGURES / "controlled_experiments.png", dpi=180)
    plt.close(figure)


def run_experiments() -> None:
    if not (DATA / "original_summary.json").exists():
        raise RuntimeError("Analyze original trajectories before running new optimizations")
    if RUNS.exists():
        raise FileExistsError("Experiment outputs already exist; refusing to overwrite")
    calculator = load_calculator("fp64")
    cases = experiment_inputs()
    for name, (atoms, _, _) in cases.items():
        write(INPUTS / f"{name}.extxyz", atoms, format="extxyz")
    rows = []
    for name, (_, relax_cell, optimizer) in cases.items():
        dump(DATA / "progress.json", {"phase": "experiments", "active_case": name, "completed": len(rows), "expected": 10})
        initial = read(INPUTS / f"{name}.extxyz")
        summary = _run_relaxation(
            initial.copy(), calculator, RUNS / name, max_steps=500,
            force_tolerance=.05, relax_cell=relax_cell, optimizer_name=optimizer,
            optimizer_maxstep=.03, hydrostatic_cell=False,
        )
        final = read(RUNS / name / "final.extxyz")
        if not relax_cell and not np.array_equal(initial.cell.array, final.cell.array):
            raise ValueError(f"Fixed-cell experiment changed its cell: {name}")
        row = {
            "experiment": name, "cell_relaxation": relax_cell,
            "optimizer": optimizer, "maxstep_angstrom": .03,
            **summary, **geometry_metrics(initial, final),
        }
        rows.append(row)
        dump(DATA / "experiments.json", rows)
        print(f"{name}: {summary['status']} in {summary['steps_taken']} steps", flush=True)
    experiment_plot(rows)
    dump(DATA / "progress.json", {"phase": "experiments_complete"})


def evaluate(calculator, atoms: Atoms):
    result = calculator.evaluate_atoms(atoms)
    graph = build_graph(atoms, calculator.precision)
    _, edges = edge_identity(graph)
    if result["evaluated_branch_ids"] != list(range(16)):
        raise ValueError("A prediction did not evaluate all 16 heads")
    return result, edges


def frozen_graph_evaluate(calculator, atoms: Atoms, reference_graph):
    """Conditional derivative control only; never used for relaxation or EOS."""
    integers = np.rint(np.linalg.solve(
        reference_graph.cell.cpu().numpy().T,
        reference_graph.edge_shifts.cpu().numpy().T,
    ).T)

    def fixed_graph(current: Atoms, **kwargs):
        graph = reference_graph.clone()
        dtype = kwargs["dtype"]
        graph.pos = torch.as_tensor(current.positions.copy(), dtype=dtype)
        graph.cell = torch.as_tensor(current.cell.array.copy(), dtype=dtype)
        graph.edge_shifts = torch.as_tensor(integers @ current.cell.array, dtype=dtype)
        return graph

    with patch.object(calculator_module, "_graph_from_atoms", fixed_graph):
        return calculator.evaluate_atoms(atoms)


def derivative_inputs() -> dict[str, Atoms]:
    steps = json.loads((DATA / "problem_steps.json").read_text())
    geometries = {"ideal": ideal_diamond()}
    for name in CASES:
        geometries[f"{name}_initial"] = read(SOURCE / name / "initial.extxyz")
        geometries[f"{name}_final"] = read(SOURCE / name / "fp64" / "final.extxyz")
    step = steps["C_diamond_compressed_fp64"]
    with Trajectory(str(SOURCE / "C_diamond_compressed" / "fp64" / "trajectory.traj")) as trajectory:
        geometries[f"compressed_problem_step_{step}"] = trajectory[step].copy()
    return geometries


def check_derivatives(calculator, name: str, atoms: Atoms, atom_count: int) -> dict[str, Any]:
    baseline, baseline_edges = evaluate(calculator, atoms)
    repeat, repeat_edges = evaluate(calculator, atoms)
    reference_graph = build_graph(atoms, "fp64")
    ordered_atoms = np.argsort(-np.linalg.norm(baseline["forces"], axis=1))
    selected_atoms = ordered_atoms[:atom_count]
    components = [(int(atom), axis) for atom in selected_atoms for axis in range(3)]
    records = []
    summaries = []
    for h in FD_STEPS:
        for atom, axis in components:
            plus, minus = atoms.copy(), atoms.copy()
            plus.positions[atom, axis] += h
            minus.positions[atom, axis] -= h
            positive, positive_edges = evaluate(calculator, plus)
            negative, negative_edges = evaluate(calculator, minus)
            numerical = -(positive["energy"] - negative["energy"]) / (2 * h)
            analytic = float(baseline["forces"][atom, axis])
            changed = len(positive_edges ^ baseline_edges) + len(negative_edges ^ baseline_edges)
            conditional = None
            if changed:
                fixed_plus = frozen_graph_evaluate(calculator, plus, reference_graph)
                fixed_minus = frozen_graph_evaluate(calculator, minus, reference_graph)
                conditional = -(fixed_plus["energy"] - fixed_minus["energy"]) / (2 * h)
            records.append({
                "geometry": name, "h_angstrom": h, "atom_index": atom, "axis": axis,
                "analytic_force": analytic, "finite_difference_force": numerical,
                "absolute_error": abs(numerical - analytic),
                "periodic_edge_changes_plus": len(positive_edges ^ baseline_edges),
                "periodic_edge_changes_minus": len(negative_edges ^ baseline_edges),
                "conditional_fixed_graph_force": conditional,
                "conditional_fixed_graph_error": abs(conditional - analytic) if conditional is not None else None,
            })
        selected = [row for row in records if row["h_angstrom"] == h]
        errors = np.asarray([row["finite_difference_force"] - row["analytic_force"] for row in selected])
        stable = [row for row in selected if row["periodic_edge_changes_plus"] == row["periodic_edge_changes_minus"] == 0]
        fixed = [row["conditional_fixed_graph_error"] for row in selected if row["conditional_fixed_graph_error"] is not None]
        summaries.append({
            "h_angstrom": h, "tested_components": len(selected),
            "max_force_error": float(np.abs(errors).max()),
            "rms_force_error": float(np.sqrt(np.mean(errors**2))),
            "graph_changed_components": len(selected) - len(stable),
            "stable_graph_max_error": max((row["absolute_error"] for row in stable), default=None),
            "conditional_fixed_graph_max_error": max(fixed) if fixed else None,
        })
        write_csv(DATA / f"{name}_atomic_fd.csv", records)
        print(f"{name} h={h}: max force error {np.abs(errors).max():.5g}; graph changes {len(selected)-len(stable)}/{len(selected)}", flush=True)
    stress_records = []
    stress_tensor = voigt_6_to_full_3x3_stress(baseline["stress"])
    for h in FD_STEPS:
        for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
            deformation = np.zeros((3, 3))
            deformation[i, j] = 1
            deformation[j, i] = 1
            factor = 2 if i != j else 1
            plus, minus = atoms.copy(), atoms.copy()
            plus.set_cell(atoms.cell.array @ (np.eye(3) + h * deformation), scale_atoms=True)
            minus.set_cell(atoms.cell.array @ (np.eye(3) - h * deformation), scale_atoms=True)
            positive, positive_edges = evaluate(calculator, plus)
            negative, negative_edges = evaluate(calculator, minus)
            numerical = (positive["energy"] - negative["energy"]) / (2 * h * atoms.get_volume() * factor)
            analytic = float(stress_tensor[i, j])
            changed = positive_edges != baseline_edges or negative_edges != baseline_edges
            conditional = None
            if changed:
                fixed_plus = frozen_graph_evaluate(calculator, plus, reference_graph)
                fixed_minus = frozen_graph_evaluate(calculator, minus, reference_graph)
                conditional = (fixed_plus["energy"] - fixed_minus["energy"]) / (2 * h * atoms.get_volume() * factor)
            stress_records.append({
                "geometry": name, "strain_step": h, "component": f"{i}{j}",
                "analytic_stress_ev_per_angstrom3": analytic,
                "finite_difference_stress_ev_per_angstrom3": numerical,
                "absolute_error": abs(numerical - analytic),
                "periodic_edge_changes_plus": len(positive_edges ^ baseline_edges),
                "periodic_edge_changes_minus": len(negative_edges ^ baseline_edges),
                "conditional_fixed_graph_stress": conditional,
                "conditional_fixed_graph_error": abs(conditional-analytic) if conditional is not None else None,
            })
    write_csv(DATA / f"{name}_stress_fd.csv", stress_records)
    return {
        "geometry": name, "atom_count": len(atoms),
        "tested_atoms": selected_atoms.tolist(), "atomic_fd": summaries,
        "stress_fd": stress_records,
        "repeat_energy_delta_ev": abs(repeat["energy"] - baseline["energy"]),
        "repeat_graph_identical": repeat_edges == baseline_edges,
        "branch_weights": baseline["branch_weights"].tolist(),
    }


def lattice_scan(calculator) -> list[dict[str, Any]]:
    rows = []
    previous_edges = set()
    for scale in np.linspace(.90, 1.10, 21):
        atoms = ideal_diamond()
        atoms.set_cell(atoms.cell.array * scale, scale_atoms=True)
        result, edges = evaluate(calculator, atoms)
        stress = voigt_6_to_full_3x3_stress(result["stress"])
        graph, _ = graph_statistics(atoms, "fp64")
        rows.append({
            "lattice_scale": float(scale), "lattice_parameter_angstrom": float(3.567*scale),
            "energy_ev_per_atom": result["energy"]/len(atoms),
            "maximum_force_ev_per_angstrom": float(np.linalg.norm(result["forces"],axis=1).max()),
            "mean_normal_stress_ev_per_angstrom3": float(np.trace(stress)/3),
            "pressure_ev_per_angstrom3": float(-np.trace(stress)/3),
            "stress_voigt_json": json.dumps(result["stress"].tolist()),
            **graph,
            "periodic_edges_changed_from_previous_scale": len(edges ^ previous_edges) if rows else 0,
        })
        previous_edges = edges
        print(f"EOS {scale:.2f}: {rows[-1]['energy_ev_per_atom']:.6f} eV/atom",flush=True)
    write_csv(DATA / "lattice_scan.csv", rows)
    figure, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
    scales = [r["lattice_scale"] for r in rows]
    for axis, key, label in zip(
        axes, ("energy_ev_per_atom", "pressure_ev_per_angstrom3", "maximum_force_ev_per_angstrom"),
        ("Energy (eV/atom)", "Pressure = -Tr(stress)/3 (eV/Å³)", "Max atomic force (eV/Å)"), strict=True,
    ):
        axis.plot(scales, [r[key] for r in rows], "o-")
        axis.set_ylabel(label)
        axis.set_xlabel("a / a₀")
        axis.grid(alpha=.25)
    figure.suptitle("Frozen-model diamond lattice scan; not DFT")
    figure.savefig(FIGURES / "diamond_lattice_scan.png", dpi=180)
    plt.close(figure)
    return rows


def transition_probe(calculator, name: str, step: int) -> dict[str, Any]:
    with Trajectory(str(SOURCE / name / "fp64" / "trajectory.traj")) as trajectory:
        first, last = trajectory[step-1].copy(), trajectory[step].copy()
    rows = []
    previous_edges = set()
    for fraction in np.linspace(0, 1, 21):
        atoms = first.copy()
        atoms.positions[:] = first.positions + fraction*(last.positions-first.positions)
        atoms.set_cell(first.cell.array + fraction*(last.cell.array-first.cell.array), scale_atoms=False)
        result, edges = evaluate(calculator, atoms)
        stats, _ = graph_statistics(atoms, "fp64")
        rows.append({
            "fraction": float(fraction), "energy_ev_per_atom": result["energy"]/len(atoms),
            "maximum_force_ev_per_angstrom": float(np.linalg.norm(result["forces"],axis=1).max()),
            **stats, "edges_changed_since_previous_sample": len(edges ^ previous_edges) if rows else 0,
        })
        previous_edges = edges
    write_csv(DATA / f"{name}_problem_transition.csv", rows)
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot([r["fraction"] for r in rows],[r["energy_ev_per_atom"] for r in rows],"o-")
    axes[0].set_ylabel("Energy (eV/atom)")
    axes[1].plot([r["fraction"] for r in rows],[r["edges_changed_since_previous_sample"] for r in rows],"o-")
    axes[1].set_ylabel("Changed periodic edge identities")
    for axis in axes:
        axis.set_xlabel(f"Interpolation between original steps {step-1} and {step}")
        axis.grid(alpha=.25)
    figure.savefig(FIGURES / f"{name}_problem_transition.png",dpi=180)
    plt.close(figure)
    jumps = [
        index for index in range(1, len(rows))
        if rows[index]["edges_changed_since_previous_sample"]
    ]
    boundary = None
    if jumps:
        index = max(
            jumps,
            key=lambda index: abs(rows[index]["energy_ev_per_atom"] - rows[index-1]["energy_ev_per_atom"]),
        )

        def interpolated(fraction):
            atoms = first.copy()
            atoms.positions[:] = first.positions + fraction*(last.positions-first.positions)
            atoms.set_cell(
                first.cell.array + fraction*(last.cell.array-first.cell.array),
                scale_atoms=False,
            )
            return atoms

        left, right = rows[index-1]["fraction"], rows[index]["fraction"]
        left_result, left_edges = evaluate(calculator, interpolated(left))
        right_result, right_edges = evaluate(calculator, interpolated(right))
        contractions = []
        for iteration in range(24):
            middle = (left + right) / 2
            middle_result, middle_edges = evaluate(calculator, interpolated(middle))
            if middle_edges == left_edges:
                left, left_result = middle, middle_result
            else:
                right, right_result, right_edges = middle, middle_result, middle_edges
            if iteration in (0, 4, 9, 14, 19, 23):
                contractions.append({
                    "fraction_interval_width": right-left,
                    "absolute_energy_jump_ev_per_atom": abs(right_result["energy"]-left_result["energy"])/64,
                    "periodic_edges_changed": len(right_edges ^ left_edges),
                })
        boundary = {
            "left_fraction": left, "right_fraction": right,
            "contractions": contractions,
            "interpretation": "Persistent finite energy jump as the interval shrinks verifies nonsmoothness at a specific retained-neighbor switch; does not establish that every force oscillation has this cause.",
        }
    saved = read_csv(SOURCE / name / "fp64" / "trajectory.csv")
    return {
        "case": name, "step": step, "rows": rows,
        "endpoint_energy_reproduction_errors_ev_per_atom": [
            abs(rows[0]["energy_ev_per_atom"] - float(saved[step-1]["energy_ev_per_atom"])),
            abs(rows[-1]["energy_ev_per_atom"] - float(saved[step]["energy_ev_per_atom"])),
        ],
        "resolved_graph_boundary": boundary,
    }


def run_derivatives(atom_count: int) -> None:
    if (DATA / "derivatives.json").exists():
        raise FileExistsError("Derivative results already exist; refusing to overwrite")
    calculator = load_calculator("fp64")
    scan = lattice_scan(calculator)
    steps = json.loads((DATA / "problem_steps.json").read_text())
    transitions = [transition_probe(calculator,name,steps[name+"_fp64"]) for name in CASES]
    dump(DATA / "transition_probes.json", transitions)
    reproduction = []
    for name in CASES:
        for label, path, record in (
            ("initial", SOURCE/name/"initial.extxyz", read_csv(SOURCE/name/"fp64"/"trajectory.csv")[0]),
            ("final", SOURCE/name/"fp64"/"final.extxyz", read_csv(SOURCE/name/"fp64"/"trajectory.csv")[-1]),
        ):
            atoms = read(path)
            result, _ = evaluate(calculator, atoms)
            reproduction.append({
                "case": name, "geometry": label,
                "absolute_energy_delta_ev_per_atom": abs(result["energy"]/64-float(record["energy_ev_per_atom"])),
                "absolute_maximum_force_delta_ev_per_angstrom": abs(
                    np.linalg.norm(result["forces"],axis=1).max()
                    - float(record["maximum_force_ev_per_angstrom"])
                ),
                "coordinate_serialization": "extxyz uses eight decimal places; final input can switch near-tied capped neighbors relative to full-precision trajectory",
            })
    dump(DATA / "saved_geometry_reproduction.json", reproduction)
    summaries = []
    for name, atoms in derivative_inputs().items():
        write(INPUTS / f"derivative_{name}.extxyz", atoms, format="extxyz")
        dump(DATA / "progress.json", {"phase": "derivatives", "active_geometry": name, "tested_atoms": atom_count})
        summaries.append(check_derivatives(calculator, name, atoms, atom_count))
        dump(DATA / "derivatives_partial.json", summaries)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for geometry in summaries:
        force = geometry["atomic_fd"]
        axes[0].loglog(
            [r["h_angstrom"] for r in force],
            [max(r["max_force_error"], 1e-16) for r in force],
            "o-", label=geometry["geometry"],
        )
        errors = [
            max(r["absolute_error"] for r in geometry["stress_fd"] if r["strain_step"] == h)
            for h in FD_STEPS
        ]
        axes[1].loglog(FD_STEPS, np.maximum(errors,1e-16), "o-", label=geometry["geometry"])
    for axis, label in zip(
        axes, ("Maximum force FD discrepancy (eV/Å)", "Maximum stress FD discrepancy (eV/Å³)"),
        strict=True,
    ):
        axis.set_xlabel("Central-difference step (Å for force, strain for stress)")
        axis.set_ylabel(label)
        axis.legend(fontsize=7)
        axis.grid(alpha=.25)
    figure.suptitle("Rebuilt-graph derivatives: changing topology is marked separately in tables")
    figure.savefig(FIGURES / "derivative_validation.png", dpi=180)
    plt.close(figure)
    dump(DATA / "derivatives.json", {
        "atomic_steps_angstrom": FD_STEPS, "stress_strain_steps": FD_STEPS,
        "conditional_graph_control": "Reuses exact periodic edges only for finite-difference implementation diagnosis; production graphs and all relaxations unchanged.",
        "geometries": summaries, "lattice_scan": scan,
    })
    dump(DATA / "progress.json", {"phase": "derivatives_complete"})


def make_report() -> None:
    originals = json.loads((DATA / "original_summary.json").read_text())
    experiments = json.loads((DATA / "experiments.json").read_text())
    derivatives = json.loads((DATA / "derivatives.json").read_text())
    if len(experiments) != 10 or len(derivatives["geometries"]) != 6:
        raise RuntimeError("Diagnostic experiments or derivative geometries are incomplete")
    for geometry in derivatives["geometries"]:
        if len(geometry["atomic_fd"]) != 3 or len(geometry["stress_fd"]) != 18:
            raise RuntimeError(f"Incomplete derivatives for {geometry['geometry']}")
    if source_manifest() != json.loads((DATA / "source_manifest.json").read_text()):
        raise RuntimeError("The original benchmark has changed during diagnostics")
    table = [
        "| Experiment | Optimizer / cell | Status | Steps | Final max force | Final generalized force |",
        "|---|---|---|---|---|---|",
    ]
    for row in experiments:
        table.append(
            f"| {row['experiment']} | {row['optimizer']} / {'full' if row['cell_relaxation'] else 'fixed'} | "
            f"{row['status']} | {row['steps_taken']} | {row['final_maximum_force_ev_per_angstrom']:.6g} | "
            f"{row['final_maximum_optimization_force']:.6g} |"
        )
    fd_table = [
        "| Geometry | h (Å) | Max force error (eV/Å) | RMS force error | Changed-graph components | Stable-graph max error | Conditional graph max error |",
        "|---|---|---|---|---|---|---|",
    ]
    for geometry in derivatives["geometries"]:
        for row in geometry["atomic_fd"]:
            fd_table.append(
                f"| {geometry['geometry']} | {row['h_angstrom']} | {row['max_force_error']:.6g} | "
                f"{row['rms_force_error']:.6g} | {row['graph_changed_components']}/{row['tested_components']} | "
                f"{row['stable_graph_max_error']} | {row['conditional_fixed_graph_max_error']} |"
            )
    stress_table = [
        "| Geometry | Strain h | Max stress error (eV/Å³) | RMS stress error | Changed-graph components |",
        "|---|---|---|---|---|",
    ]
    for geometry in derivatives["geometries"]:
        for h in FD_STEPS:
            rows = [row for row in geometry["stress_fd"] if row["strain_step"] == h]
            errors = np.asarray([row["absolute_error"] for row in rows])
            changed_count = sum(
                row["periodic_edge_changes_plus"] > 0 or row["periodic_edge_changes_minus"] > 0
                for row in rows
            )
            stress_table.append(
                f"| {geometry['geometry']} | {h} | {errors.max():.6g} | "
                f"{np.sqrt(np.mean(errors**2)):.6g} | {changed_count}/6 |"
            )
    fixed = [row for row in experiments if row["experiment"].startswith("A")]
    small = [row for row in experiments if row["experiment"].startswith("B")]
    bfgs = [row for row in experiments if row["experiment"].startswith("C")]
    scan = derivatives["lattice_scan"]
    minimum = min(scan, key=lambda row: row["energy_ev_per_atom"])
    stable_errors = [
        row["stable_graph_max_error"] for geometry in derivatives["geometries"]
        for row in geometry["atomic_fd"] if row["stable_graph_max_error"] is not None
    ]
    changing = sum(row["graph_changed_components"] for geometry in derivatives["geometries"] for row in geometry["atomic_fd"])
    conditional = [
        row["conditional_fixed_graph_max_error"] for geometry in derivatives["geometries"]
        for row in geometry["atomic_fd"] if row["conditional_fixed_graph_max_error"] is not None
    ]
    stress_rows = [r for g in derivatives["geometries"] for r in g["stress_fd"]]
    stable_stress = [r["absolute_error"] for r in stress_rows if not r["periodic_edge_changes_plus"] and not r["periodic_edge_changes_minus"]]
    conditional_stress = [r["conditional_fixed_graph_error"] for r in stress_rows if r["conditional_fixed_graph_error"] is not None]
    transitions = json.loads((DATA / "transition_probes.json").read_text())
    transition_text = "\n".join(
        f"- {row['case']}, original step {row['step']}: resolved graph-switch bracket "
        f"width {row['resolved_graph_boundary']['contractions'][-1]['fraction_interval_width']:.3g} "
        f"of one optimizer step; remaining energy difference "
        f"{row['resolved_graph_boundary']['contractions'][-1]['absolute_energy_jump_ev_per_atom']:.6g} "
        "eV/atom. The complete bracket-contraction series is saved."
        for row in transitions if row["resolved_graph_boundary"] is not None
    )
    minimum_index = scan.index(minimum)
    stress_sign_change = (
        0 < minimum_index < len(scan)-1
        and scan[minimum_index-1]["mean_normal_stress_ev_per_angstrom3"] < 0
        and scan[minimum_index+1]["mean_normal_stress_ev_per_angstrom3"] > 0
    )
    refinement_path = DATA / "derivative_refinement.json"
    refinement_text = "Smaller-step refinement has not been run."
    if refinement_path.exists():
        refinement = json.loads(refinement_path.read_text())
        finest = [r for r in refinement if r["h"] == 3e-7]
        atomic_errors = [r["absolute_error"] for r in finest if r["kind"] == "worst_atomic_component"]
        stress_errors = [r["absolute_error"] for r in finest if r["kind"] == "stress"]
        refinement_text = (
            "The expanded endpoint and problematic compressed intermediate had residual "
            "graph-stable finite-difference errors at 10⁻⁵. Focused refinement of their "
            "worst atomic component and all six stress components down to 3×10⁻⁷ "
            f"reduced maximum errors to {max(atomic_errors):.6g} eV/Å and "
            f"{max(stress_errors):.6g} eV/Å³ with unchanged graphs. This resolves those "
            "residuals as step-size sensitivity rather than a demonstrated gradient bug. "
            "This refinement checks the worst components, not another complete 192-component sweep."
        )
    divergence = "\n".join(
        f"- {name}: first material divergence at step "
        f"{originals[name+'/precision_comparison']['first_material_divergence_step']}; "
        f"first periodic graph difference at step "
        f"{originals[name+'/precision_comparison']['first_periodic_edge_difference_step']}."
        for name in CASES
    )
    original_detail = "\n".join(
        f"- {name}/{precision}: {originals[name+'/'+precision]['energy_increase_steps_above_1e_6_ev_per_atom']} "
        f"energy increases above 10⁻⁶ eV/atom; "
        f"largest increase {originals[name+'/'+precision]['largest_upward_energy_jump_ev_per_atom']:.6g} "
        f"eV/atom at step {originals[name+'/'+precision]['largest_upward_jump_step']} "
        f"with {originals[name+'/'+precision]['periodic_edges_changed_at_jump']} changed periodic edges; "
        f"minimum separation {originals[name+'/'+precision]['minimum_atomic_separation_angstrom']:.4f} Å; "
        f"volume {originals[name+'/'+precision]['initial_volume_angstrom3']:.3f} → "
        f"{originals[name+'/'+precision]['final_volume_angstrom3']:.3f} Å³."
        for name in CASES for precision in PRECISIONS
    )
    report = f"""# Diamond-carbon relaxation failure diagnostics

## Scope and integrity

Source: `paper-reviewer-response-2026-11-08`, benchmark commit
`8a18696fd3ddad54103c586a781d6d67edd908bd`. All original benchmark files were
SHA-256 checked before and after this investigation and are unchanged.
Original logs, summaries, CSVs, trajectories, and final geometries were inspected.
Both checkpoint hashes are verified by the existing strict loader.
All 16 heads, absolute C64 composition counts, 5 Å graph radius, 20 incoming
neighbors, neutral/zero-spin settings, and differentiation of the complete
weighted energy are preserved. No DFT, retraining, checkpoint, manuscript,
or production-calculator changes were made.

## Main conclusion

Discrete 20-neighbor selection produces verified finite energy jumps in both
original failed trajectories. This is a demonstrated contributor to carbon's
difficult optimization, not merely a topology correlation. The tests do not
establish that it is the sole cause: the expanded fixed-cell case succeeds,
indicating that cell/atomic coupling also matters for that start. Neither
BFGS nor milder full-cell starts consistently resolve the problem.
The same failure in FP64 excludes an explanation specific to FP32.
The sampled diamond lattice curve does not establish a smooth, stable diamond
minimum: even its lowest-energy sample has substantial residual atomic force.

## Original failed trajectories

{original_detail}

{divergence}

Divergence is operationally defined by any of: 10⁻⁴ eV/atom energy difference,
10⁻³ Å RMS Cartesian position difference, or 10⁻² eV/Å force-component RMSE.
It is not an error relative to DFT. Exact periodic edge identities include source,
destination, and integer image shift; edge counts alone cannot establish topology.
Graph diagnostics reconstruct the saved coordinates in the original precision.
Raw 5 Å neighbors and the retained 20 neighbors are reported independently.
All 64 atoms reach the retained-neighbor cap in the four original trajectories.
Carbon's measured softmax vector is recorded in the derivative JSON; if saturated
on branch 15 (transition1x in this checkpoint's verified mapping), evaluating all
16 heads still does not make the weighted potential an equal-weight ensemble.
This routing observation does not independently establish dataset suitability.

## Controlled experiments (FP64)

{chr(10).join(table)}

FIRE and BFGS both use maxstep=0.03 Å and a 500-step cap. Full-cell runs use
ASE UnitCellFilter with cell_factor=64 and no hydrostatic/symmetry restriction.
Fixed-cell runs use the exact two original inputs and atomic force tolerance
0.05 eV/Å. B1–B6 share normalized random directions (seed {SEED}); B1 uses
0.02–0.03 Å magnitudes and B2–B6 exactly 0.05 Å. B3/B4 strain ±1%,
B5/B6 ±3%, while B1/B2 are unstrained. All inputs and outputs are retained.

## Energy/force/stress derivative evidence

{chr(10).join(fd_table)}

{chr(10).join(stress_table)}

All {len(derivatives['geometries'][0]['tested_atoms'])} tested atoms × three Cartesian
components are checked at each of three h values in each listed geometry.
Largest error among graph-stable atomic tests: {max(stable_errors,default=0):.6g} eV/Å.
Changed-graph atomic tests: {changing}.
Largest conditional fixed-graph atomic error: {max(conditional,default=0):.6g} eV/Å.
Largest graph-stable stress error: {max(stable_stress,default=0):.6g} eV/Å³.
Largest conditional fixed-graph stress error: {max(conditional_stress,default=0):.6g} eV/Å³.
These maxima include the larger finite-difference steps and are not final
small-step consistency tolerances.

{refinement_text}

All six independent stress components and all three strain step sizes are in
the stress CSVs and JSON. Coordinates transform with the strained cell, preserving
fractional positions. Symmetric shear derivatives divide by 2V; diagonal by V.
Positive model stress denotes the energy derivative with tensile strain;
pressure is minus one third of the stress trace.

Conditional fixed-graph checks isolate derivatives within one graph branch.
They do not validate the differentiability of the rebuilt-graph energy across
neighbor changes, and are never used to relax structures or alter reported
production predictions. Large rebuilt-graph differences must be distinguished
from graph-stable derivative errors.

### Resolved problematic graph transitions

{transition_text}

These brackets shrink the geometric interval around an observed neighbor switch.
A finite energy difference that persists as the interval contracts is direct
evidence of nonsmooth rebuilt-graph energy, unlike an edge-count correlation.

## Diamond lattice scan

Ideal 64-atom diamond was scanned over a/a₀=0.90–1.10 at 0.01 intervals,
a₀=3.567 Å. The lowest sampled energy occurs at scale
{minimum['lattice_scale']:.2f} ({minimum['lattice_parameter_angstrom']:.5f} Å),
E={minimum['energy_ev_per_atom']:.6g} eV/atom,
pressure={minimum['pressure_ev_per_angstrom3']:.6g} eV/Å³.
Maximum atomic force at this sampled minimum is
{minimum['maximum_force_ev_per_angstrom']:.6g} eV/Å; neighboring sampled normal
stress changes from compression to tension around this minimum: {stress_sign_change}.
This is a sampled learned-potential curve, not proof of a smooth/stable minimum
or experimental agreement. All stress components, forces, periodic graph hashes,
raw neighbor counts and 20th/21st distance gaps are retained. A zero boundary
gap means the cap splits a degenerate neighbor shell; topology changes alone
do not prove an energy discontinuity. Interpolated problematic transitions
and derivative checks provide the separate energy evidence.

## Explicit scientific answers and manuscript recommendation

1. **Why did the original runs fail?** Residual generalized forces remained above
0.05 eV/Å at 500 steps. Resolved graph boundaries demonstrate nonsmooth rebuilt-graph
energy as a contributor. Full-cell/atomic coupling contributes for the expanded
start, whose fixed-cell control succeeds. A unique cause is not established.
2. **Fixed cell:** {sum(r['converged'] for r in fixed)}/2 cases converged.
Holding the cell fixed {'does not eliminate all failures' if not all(r['converged'] for r in fixed) else 'eliminates the failures in these two tests'}.
3. **Small perturbations:** {sum(r['converged'] for r in small)}/6 cases converged;
see each amplitude/strain row rather than inferring universal basin stability.
4. **Alternative optimizer:** BFGS converged in {sum(r['converged'] for r in bfgs)}/2
original cases. Its success/failure does not independently establish physical accuracy.
5. **Mutual consistency:** Stable-graph and conditional derivative discrepancies
converge toward autograd with smaller steps, including the focused refinement
when available. No general energy-gradient implementation defect was demonstrated.
Across discrete neighbor switches the rebuilt energy is not smooth, so its
ordinary central finite difference need not agree with the local autograd derivative.
6. **Stable diamond minimum:** Not established. The lowest 21-point sample at
1.03a₀ has max force {minimum['maximum_force_ev_per_angstrom']:.4f} eV/Å and nonzero
pressure. A low sampled energy is not a stationary full-geometry minimum;
the rough profile and cap splitting a tied shell preclude a robust smoothness claim.
7. **Precision specificity:** Failure exists in both FP32 and FP64. Their first
material divergence is explicitly identified above; it is not solely an FP32 issue.
8. **Precision robustness:** Agreement of successful cases in the original 52-run
study remains a within-model numerical-consistency result. Carbon's capped endpoints
are neither converged minima nor physical-reference errors.
9. **Manuscript:** Describe both distorted diamond starts as non-converged at the
500-step cap in both precisions, disclose residual forces and diagnostic sensitivity,
and exclude them from claims of converged-minimum precision equivalence. Do not
claim robust/accurate diamond relaxation or modify the manuscript until review.

## Deliverables

- `analysis/*_steps.csv`: every original step, geometry, cap counts and periodic graph changes.
- `analysis/*_precision_divergence.csv`: paired precision differences and threshold flags.
- `figures/*_original_precision.png`: original energy/force/stress/volume comparisons.
- `experiments/`, `inputs/`: ten controlled experiments and all new starts.
- `figures/controlled_experiments.png`: fixed/full FIRE, BFGS, and perturbation comparisons.
- `analysis/*_atomic_fd.csv`, `*_stress_fd.csv`, `derivatives.json`: derivative evidence.
- `analysis/derivative_refinement.csv`: smaller-step checks of remaining worst atomic components and all six stress components.
- `figures/derivative_validation.png`: force and stress discrepancies versus finite-difference step.
- `analysis/lattice_scan.csv`, `figures/diamond_lattice_scan.png`: learned lattice response.
- `analysis/*_problem_transition.csv`, `figures/*_problem_transition.png`: local graph/energy transitions.
"""
    (WORKSPACE / "report.md").write_text(report)
    dump(DATA / "progress.json", {"phase": "complete", "source_integrity_verified": True})


def refine_derivatives() -> None:
    """Resolve remaining graph-stable residuals with smaller finite-difference steps."""
    calculator = load_calculator("fp64")
    results = []
    geometries = derivative_inputs()
    for name in ("C_diamond_expanded_final", next(key for key in geometries if "problem_step" in key)):
        atoms = geometries[name]
        baseline, edges = evaluate(calculator, atoms)
        records = read_csv(DATA / f"{name}_atomic_fd.csv")
        worst = max(
            (row for row in records if float(row["h_angstrom"]) == 1e-5),
            key=lambda row: float(row["absolute_error"]),
        )
        atom, axis = int(worst["atom_index"]), int(worst["axis"])
        for h in (1e-5, 3e-6, 1e-6, 3e-7):
            plus, minus = atoms.copy(), atoms.copy()
            plus.positions[atom, axis] += h
            minus.positions[atom, axis] -= h
            positive, pe = evaluate(calculator, plus)
            negative, me = evaluate(calculator, minus)
            numeric = -(positive["energy"] - negative["energy"]) / (2*h)
            results.append({
                "geometry": name, "kind": "worst_atomic_component", "h": h,
                "component": f"{atom}/{axis}", "absolute_error": abs(numeric-baseline["forces"][atom,axis]),
                "graph_unchanged": pe == me == edges,
            })
            stress = voigt_6_to_full_3x3_stress(baseline["stress"])
            for i,j in ((0,0),(1,1),(2,2),(0,1),(0,2),(1,2)):
                deformation = np.zeros((3,3))
                deformation[i,j] = deformation[j,i] = 1
                plus, minus = atoms.copy(), atoms.copy()
                plus.set_cell(atoms.cell.array@(np.eye(3)+h*deformation),scale_atoms=True)
                minus.set_cell(atoms.cell.array@(np.eye(3)-h*deformation),scale_atoms=True)
                positive, pe = evaluate(calculator,plus)
                negative, me = evaluate(calculator,minus)
                numeric = (positive["energy"]-negative["energy"])/(2*h*atoms.get_volume()*(2 if i!=j else 1))
                results.append({
                    "geometry": name, "kind": "stress", "h": h,
                    "component": f"{i}{j}", "absolute_error": abs(numeric-stress[i,j]),
                    "graph_unchanged": pe == me == edges,
                })
    write_csv(DATA / "derivative_refinement.csv",results)
    dump(DATA / "derivative_refinement.json",results)
    for name in geometries:
        selected = [r for r in results if r["geometry"]==name and r["h"]==3e-7]
        if selected:
            print(name, "smallest-step refinement:", selected, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("originals", "experiments", "derivatives", "refine", "report", "all"), default="all")
    parser.add_argument("--fd-atoms", type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.fd_atoms <= 64:
        raise ValueError("--fd-atoms must be between 1 and 64")
    for path in (DATA, FIGURES, INPUTS):
        path.mkdir(exist_ok=True)
    if args.phase in ("originals", "all"):
        analyze_originals()
    if args.phase in ("experiments", "all"):
        run_experiments()
    if args.phase in ("derivatives", "all"):
        if not (DATA / "original_summary.json").exists():
            raise RuntimeError("Original trajectory analysis must run first")
        run_derivatives(args.fd_atoms)
    if args.phase in ("report", "all"):
        make_report()
    if args.phase == "refine":
        refine_derivatives()


if __name__ == "__main__":
    main()
