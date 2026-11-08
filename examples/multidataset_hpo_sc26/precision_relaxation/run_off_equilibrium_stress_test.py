#!/usr/bin/env python3
"""Stress-test FP32/FP64 cell relaxation from deliberately distorted crystals."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.geometry import find_mic
from ase.io import read, write

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_benchmark import _load_calculator, _resolve_single_head_map, _run_relaxation
from run_periodic_structure_set import (
    ELEMENT_COLORS,
    FORCE_TOLERANCE,
    OPTIMIZER_MAXSTEP,
    PRECISIONS,
    STRUCTURE_BUILDERS,
    _render_displacement_plot,
    _render_ovito,
)

SUPER_CELL_REPETITIONS = (2, 2, 2)
ATOMIC_DISPLACEMENT_RANGE_ANGSTROM = (0.10, 0.15)
CELL_SCALES = {"Si_diamond": 1.05, "MgO_rocksalt": 0.95}
BUILDERS = {
    "Si_diamond": lambda: bulk("Si", "diamond", a=5.430, cubic=True),
    "MgO_rocksalt": lambda: bulk("MgO", "rocksalt", a=4.212, cubic=True),
}


def _make_initial_structures(seed: int, *, broad: bool = False) -> dict[str, Atoms]:
    rng = np.random.default_rng(seed)
    structures: dict[str, Atoms] = {}
    builders = STRUCTURE_BUILDERS if broad else BUILDERS
    for name, builder in builders.items():
        atoms = builder()
        atoms *= SUPER_CELL_REPETITIONS
        directions = rng.normal(size=(len(atoms), 3))
        lengths = np.linalg.norm(directions, axis=1)
        if np.any(lengths == 0):
            raise ValueError(f"Generated a zero-length perturbation for {name}")
        magnitudes = rng.uniform(
            *ATOMIC_DISPLACEMENT_RANGE_ANGSTROM, size=len(atoms)
        )
        perturbations = directions / lengths[:, None] * magnitudes[:, None]
        if not np.all(atoms.pbc):
            raise ValueError(f"{name} must be periodic in all three directions")
        scales = (0.95, 1.05) if broad else (CELL_SCALES[name],)
        for scale in scales:
            initial = atoms.copy()
            initial.set_cell(atoms.cell.array * scale, scale_atoms=True)
            initial.positions += perturbations
            case_name = (
                f"{name}_{'compressed' if scale < 1 else 'expanded'}"
                if broad
                else name
            )
            initial.info["prototype"] = name
            initial.info["cell_scale_from_prototype"] = scale
            structures[case_name] = initial
    return structures


def _geometry_metrics(initial: Atoms, final: Atoms) -> dict[str, float]:
    initial_fractional = np.linalg.solve(
        initial.cell.array.T, initial.positions.T
    ).T
    final_fractional = np.linalg.solve(final.cell.array.T, final.positions.T).T
    _, magnitudes = find_mic(
        (final_fractional - initial_fractional) @ final.cell.array,
        final.cell,
        pbc=final.pbc,
    )
    return {
        "rms_atomic_displacement_angstrom": float(
            np.sqrt(np.mean(magnitudes**2))
        ),
        "maximum_atomic_displacement_angstrom": float(magnitudes.max()),
        "cell_linear_change_percent": float(
            100
            * (
                np.mean(final.cell.lengths())
                / np.mean(initial.cell.lengths())
                - 1
            )
        ),
        "cell_volume_change_percent": float(
            100 * (final.get_volume() / initial.get_volume() - 1)
        ),
    }


def _read_trajectory_summary(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"Relaxation produced no trajectory data: {path}")
    return rows[0], rows[-1]


def _paired_structure_plot(
    initial: Atoms,
    final: Atoms,
    output_path: Path,
    *,
    name: str,
    final_status: str = "converged",
) -> None:
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(13, 6.2), constrained_layout=True)
    axes = [
        figure.add_subplot(1, 2, index + 1, projection="3d")
        for index in range(2)
    ]
    all_points: list[np.ndarray] = []
    stages = (
        ("Initial, deliberately distorted", initial),
        (f"FP64 endpoint ({final_status})", final),
    )
    for axis, (label, atoms) in zip(axes, stages, strict=True):
        fractional = np.linalg.solve(atoms.cell.array.T, atoms.positions.T).T
        positions = np.mod(fractional, 1.0) @ atoms.cell.array
        symbols = atoms.get_chemical_symbols()
        numbers = atoms.get_atomic_numbers()
        all_points.append(positions)
        for symbol in sorted(
            set(symbols),
            key=lambda item: int(atoms[atoms.symbols == item][0].number),
        ):
            mask = np.asarray([atom_symbol == symbol for atom_symbol in symbols])
            axis.scatter(
                positions[mask, 0],
                positions[mask, 1],
                positions[mask, 2],
                s=(0.25 * numbers[mask]) ** 2,
                c=[ELEMENT_COLORS[symbol]],
                edgecolors="#202832",
                linewidths=0.35,
                depthshade=True,
                alpha=0.95,
                label=f"{symbol} (Z={int(numbers[mask][0])})",
            )
        cell = atoms.cell.array
        corners = np.asarray(
            [
                i * cell[0] + j * cell[1] + k * cell[2]
                for i, j, k in np.ndindex(2, 2, 2)
            ]
        )
        all_points.append(corners)
        for i, j, k in np.ndindex(2, 2, 2):
            corner = i * cell[0] + j * cell[1] + k * cell[2]
            for axis_index, vector in enumerate(cell):
                if (i, j, k)[axis_index] == 0:
                    endpoint = corner + vector
                    axis.plot(
                        [corner[0], endpoint[0]],
                        [corner[1], endpoint[1]],
                        [corner[2], endpoint[2]],
                        color="#525f6d",
                        linewidth=0.8,
                        alpha=0.7,
                    )
        lengths = atoms.cell.lengths()
        axis.set_title(
            f"{label}\ncell {lengths[0]:.3f} × {lengths[1]:.3f} × "
            f"{lengths[2]:.3f} Å\n"
            f"angles {', '.join(f'{angle:.2f}' for angle in atoms.cell.angles())}°",
            fontsize=10,
        )
        axis.set_xlabel("x (Å)")
        axis.set_ylabel("y (Å)")
        axis.set_zlabel("z (Å)")
        axis.view_init(elev=24, azim=35)
        axis.legend(loc="upper left", fontsize=9)

    bounds = np.concatenate(all_points, axis=0)
    padding = 0.3
    for axis in axes:
        for setter, component in zip(
            (axis.set_xlim, axis.set_ylim, axis.set_zlim),
            bounds.T,
            strict=True,
        ):
            setter(float(component.min() - padding), float(component.max() + padding))
        axis.set_box_aspect(np.ptp(bounds, axis=0))
    figure.suptitle(
        f"{name}: off-equilibrium FP64 relaxation (same coordinate scale)",
        fontsize=15,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _optimization_plot(structure_dir: Path, output_path: Path, *, name: str) -> None:
    import matplotlib.pyplot as plt
    from ase.cell import Cell

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for precision in PRECISIONS:
        with (structure_dir / precision / "trajectory.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            rows = list(csv.DictReader(stream))
        steps = [int(row["step"]) for row in rows]
        energies = np.asarray([float(row["energy_ev_per_atom"]) for row in rows])
        forces = [float(row["maximum_optimization_force"]) for row in rows]
        cells = [Cell(json.loads(row["cell_angstrom_json"])) for row in rows]
        axes[0, 0].plot(steps, energies - energies[0], label=precision)
        axes[0, 1].semilogy(steps, forces, label=precision)
        axes[1, 0].plot(steps, [cell.volume for cell in cells], label=precision)
        angles = np.asarray([cell.angles() for cell in cells])
        for index, angle in enumerate(("alpha", "beta", "gamma")):
            axes[1, 1].plot(
                steps, angles[:, index], label=f"{precision} {angle}"
            )
    axes[0, 1].axhline(
        FORCE_TOLERANCE, color="black", linestyle="--", label="stop threshold"
    )
    for axis, ylabel in zip(
        axes.flat,
        ("Energy change (eV/atom)", "Max generalized force (eV/Å)",
         "Cell volume (Å³)", "Cell angles (degrees)"),
        strict=True,
    ):
        axis.set_xlabel("FIRE step")
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)
    figure.suptitle(f"{name}: full relaxation history")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _precision_comparisons(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault(row["structure"], {})[row["precision"]] = row
    comparisons = []
    for name, pair in grouped.items():
        first, second = pair["fp32"], pair["fp64"]
        comparisons.append({
            "structure": name,
            "both_converged": first["converged"] and second["converged"],
            "absolute_energy_delta_ev_per_atom": abs(
                first["final_energy_ev_per_atom"] - second["final_energy_ev_per_atom"]
            ),
            "maximum_cell_length_delta_angstrom": float(np.max(np.abs(
                np.asarray(first["final_cell_lengths_angstrom"])
                - np.asarray(second["final_cell_lengths_angstrom"])
            ))),
            "maximum_cell_angle_delta_degrees": float(np.max(np.abs(
                np.asarray(first["final_cell_angles_degrees"])
                - np.asarray(second["final_cell_angles_degrees"])
            ))),
        })
    return comparisons


def _starting_point_comparisons(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[float, dict[str, Any]]] = {}
    for row in results:
        key = (row["prototype"], row["precision"])
        grouped.setdefault(key, {})[row["cell_scale_from_prototype"]] = row
    comparisons = []
    for (prototype, precision), starts in grouped.items():
        if 0.95 not in starts or 1.05 not in starts:
            continue
        compressed, expanded = starts[0.95], starts[1.05]
        comparisons.append({
            "prototype": prototype,
            "precision": precision,
            "both_converged": compressed["converged"] and expanded["converged"],
            "absolute_final_energy_delta_ev_per_atom": abs(
                compressed["final_energy_ev_per_atom"]
                - expanded["final_energy_ev_per_atom"]
            ),
            "maximum_final_cell_length_delta_angstrom": float(np.max(np.abs(
                np.asarray(compressed["final_cell_lengths_angstrom"])
                - np.asarray(expanded["final_cell_lengths_angstrom"])
            ))),
            "maximum_final_cell_angle_delta_degrees": float(np.max(np.abs(
                np.asarray(compressed["final_cell_angles_degrees"])
                - np.asarray(expanded["final_cell_angles_degrees"])
            ))),
        })
    return comparisons


def _write_report(
    output_dir: Path,
    results: list[dict[str, Any]],
    seed: int,
    maximum_steps: int,
    *,
    broad: bool = False,
    cell_mode: str = "hydrostatic",
    complete: bool = True,
) -> None:
    table = [
        "| Structure | Precision | Status | Steps | ΔE (eV/atom) | "
        "Initial→final max force (eV/Å) | RMS atom shift (Å) | "
        "Max atom shift (Å) | Δcell (%) | ΔV (%) | Final generalized force (eV/Å) | "
        "Final max stress (eV/Å³) |",
        "|" + "|".join("---" for _ in range(12)) + "|",
    ]
    for row in results:
        energy_change = row["final_energy_ev_per_atom"] - row["initial_energy_ev_per_atom"]
        table.append(
            f"| {row['structure']} | {row['precision']} | {row['status']} | "
            f"{row['steps']} | {energy_change:+.6f} | "
            f"{row['initial_maximum_force_ev_per_angstrom']:.4f} → "
            f"{row['final_maximum_force_ev_per_angstrom']:.4f} | "
            f"{row['rms_atomic_displacement_angstrom']:.4f} | "
            f"{row['maximum_atomic_displacement_angstrom']:.4f} | "
            f"{row['cell_linear_change_percent']:+.3f} | "
            f"{row['cell_volume_change_percent']:+.3f} | "
            f"{row['final_maximum_generalized_force']:.5f} | "
            f"{row['final_maximum_absolute_stress_ev_per_angstrom3']:.6g} |"
        )
    setup = (
        "All 13 periodic crystal prototypes, each compressed and expanded by 5%."
        if broad
        else "Diamond Si expanded by 5% and rocksalt MgO compressed by 5%."
    )
    comparisons = ""
    if complete and results:
        converged = [row for row in results if row["converged"]]
        capped = [row["structure"] + "/" + row["precision"] for row in results if not row["converged"]]
        unchanged = [
            row["structure"] + "/" + row["precision"]
            for row in results if row["steps"] == 0
        ]
        paired = _precision_comparisons(results)
        converged_pairs = [row for row in paired if row["both_converged"]]
        max_energy_delta = max(
            (row["absolute_energy_delta_ev_per_atom"] for row in converged_pairs),
            default=0.0,
        )
        max_length_delta = max(
            (row["maximum_cell_length_delta_angstrom"] for row in converged_pairs),
            default=0.0,
        )
        assessment = (
            f"\n## Assessment\n\n{len(converged)}/{len(results)} runs met the "
            f"generalized-force threshold. Non-converged cases: {', '.join(capped) or 'none'}.\n\n"
            f"Zero-step cases: {', '.join(unchanged) or 'none'}. These met the stopping "
            "criterion initially; no structural improvement occurred.\n\n"
            f"Among converged precision pairs, the largest final energy difference is "
            f"{max_energy_delta:.6g} eV/atom and the largest cell-length difference "
            f"is {max_length_delta:.6g} Å. Numerical agreement is distinct from "
            "physical accuracy or independence from the initial geometry.\n\n"
            "Compare compressed and expanded endpoints below before claiming "
            "starting-point-independent lattice recovery. Different endpoints "
            "may reflect stopping tolerance or different basins; this study alone "
            "does not distinguish them.\n"
        )
        precision_rows = [
            "| Case | Both converged | Energy delta (eV/atom) | Max length delta (Å) | Max angle delta (degrees) |",
            "|---|---|---|---|---|",
        ]
        for row in _precision_comparisons(results):
            precision_rows.append(
                f"| {row['structure']} | {row['both_converged']} | "
                f"{row['absolute_energy_delta_ev_per_atom']:.6g} | "
                f"{row['maximum_cell_length_delta_angstrom']:.6g} | "
                f"{row['maximum_cell_angle_delta_degrees']:.6g} |"
            )
        start_rows = [
            "| Prototype | Precision | Both converged | Final energy delta (eV/atom) | Max final length delta (Å) | Max final angle delta (degrees) |",
            "|---|---|---|---|---|---|",
        ]
        for row in _starting_point_comparisons(results):
            start_rows.append(
                f"| {row['prototype']} | {row['precision']} | {row['both_converged']} | "
                f"{row['absolute_final_energy_delta_ev_per_atom']:.6g} | "
                f"{row['maximum_final_cell_length_delta_angstrom']:.6g} | "
                f"{row['maximum_final_cell_angle_delta_degrees']:.6g} |"
            )
        comparisons = (
            assessment + "\n## FP32 versus FP64 endpoints\n\n" + "\n".join(precision_rows)
            + "\n\n## Compressed versus expanded endpoints\n\n" + "\n".join(start_rows)
        )
    report = f"""# Off-equilibrium crystal relaxation stress test

Study status: {'complete' if complete else 'in progress'}; {len(results)} recorded runs.

## Setup

- Structures: 2×2×2 supercells. {setup}
- All atoms receive seeded, random-direction displacements of 0.10–0.15 Å; compressed and expanded cases use the same Cartesian perturbation vectors.
- Paired FP32/FP64 FIRE relaxations from identical per-case starting coordinates, relaxing atomic positions and {'all cell lengths and shape' if cell_mode == 'full' else 'hydrostatic cell volume'} together.
- Maximum optimizer step: {OPTIMIZER_MAXSTEP:g} Å; generalized-force threshold: {FORCE_TOLERANCE:g} eV/Å; step cap: {maximum_steps}.
- Distortion seed: {seed}.

## Results

{chr(10).join(table)}

{comparisons}

Energy change and force reduction are predictions of the frozen learned potential.
This is a robustness/stress test, not evidence that the final geometry agrees
with DFT or experiment. The initial distortion may be outside the model's
training distribution. A non-converged endpoint is not a relaxed structure.

## Figures

- `figures/<structure>_paired.png`: initial and FP64 final geometries rendered at the same coordinate scale, with cell outlines and element legends.
- `figures/<structure>_initial.png` and `figures/<structure>_final.png`: shaded OVITO renders.
- `figures/<structure>_displacements.png`: internal atomic motions with explicitly labeled arrow magnification.
- `figures/<structure>_optimization.png`: paired precision energy, force, volume, and cell-angle histories.
- `<structure>/<precision>/`: ASE trajectories, per-step force/energy/cell records, logs, summaries, and final structures.

Internal displacement removes affine cell deformation using fractional-coordinate
matching and the final-cell minimum image. Δcell is the change in mean cell-vector
length, not a measure of shear; full vectors and angles are in `comparison.json`.
No symmetry constraints or external pressure are imposed. Reaching the force
threshold does not establish recovery of the prototype or the same minimum
from compressed and expanded starts.
ASE scales the cell degrees of freedom by atom count: the generalized-force
threshold is not a direct requirement that every stress component vanish.
The JSON also records minimum interatomic distance and initial/final internal
offset from the ideal prototype as structural diagnostics, not physical-reference
validation.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results" / "off_equilibrium_stress_test",
    )
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--broad", action="store_true",
        help="Run all 13 prototypes from both -5%% and +5%% cell strain.",
    )
    parser.add_argument(
        "--cell-mode", choices=("hydrostatic", "full"), default="hydrostatic",
        help="Full mode relaxes all cell lengths and shape, without symmetry constraints.",
    )
    args = parser.parse_args()
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    structures = _make_initial_structures(args.seed, broad=args.broad)
    results: list[dict[str, Any]] = []
    payload = {
        "model": "Frozen HydraGNN/BranchWeightMLP fused 16-head checkpoint",
        "precision_pair": list(PRECISIONS),
        "optimizer": "ASE FIRE",
        "optimizer_maxstep_angstrom": OPTIMIZER_MAXSTEP,
        "force_tolerance_ev_per_angstrom": FORCE_TOLERANCE,
        "maximum_steps": args.max_steps,
        "seed": args.seed,
        "atomic_displacement_range_angstrom": list(ATOMIC_DISPLACEMENT_RANGE_ANGSTROM),
        "broad": args.broad,
        "cell_mode": args.cell_mode,
        "expected_runs": len(structures) * len(PRECISIONS),
        "study_status": "running",
        "cell_relaxation": (
            "full cell lengths and shape, with positions relaxed"
            if args.cell_mode == "full"
            else "hydrostatic volume-only strain, with positions relaxed"
        ),
        "results": results,
        "initial_cases": {
            name: {
                "prototype": initial.info["prototype"],
                "cell_scale_from_prototype": initial.info["cell_scale_from_prototype"],
                "initial_structure": f"{name}/initial.extxyz",
            }
            for name, initial in structures.items()
        },
        "scientific_limit": (
            "Predictions from the frozen learned potential only; no independent "
            "DFT or experimental reference validation was performed."
        ),
    }

    def save_results(*, complete: bool = False) -> None:
        payload["study_status"] = "complete" if complete else "running"
        (output_dir / "comparison.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        _write_report(
            output_dir, results, args.seed, args.max_steps,
            broad=args.broad, cell_mode=args.cell_mode, complete=complete,
        )

    save_results()
    for name, initial in structures.items():
        structure_dir = output_dir / name
        structure_dir.mkdir()
        write(structure_dir / "initial.extxyz", initial, format="extxyz")
    calculators = {
        precision: _load_calculator(
            precision=precision,
            head_mode="fused",
            system="Si",
            single_head_map=_resolve_single_head_map([]),
        )[0]
        for precision in PRECISIONS
    }
    for name, initial in structures.items():
        structure_dir = output_dir / name
        initial_path = structure_dir / "initial.extxyz"
        element_numbers = {
            symbol: int(number)
            for symbol, number in zip(
                initial.get_chemical_symbols(),
                initial.get_atomic_numbers(),
                strict=True,
            )
        }
        precision_finals: dict[str, Atoms] = {}
        for precision in PRECISIONS:
            payload["active_case"] = f"{name}/{precision}"
            save_results()
            print(f"Starting {name}/{precision} ({args.cell_mode} cell)", flush=True)
            starting_atoms = read(initial_path)
            run_dir = structure_dir / precision
            summary = _run_relaxation(
                starting_atoms,
                calculators[precision],
                run_dir,
                max_steps=args.max_steps,
                force_tolerance=FORCE_TOLERANCE,
                relax_cell=True,
                optimizer_name="fire",
                optimizer_maxstep=OPTIMIZER_MAXSTEP,
                hydrostatic_cell=args.cell_mode == "hydrostatic",
            )
            final_atoms = read(run_dir / "final.extxyz")
            precision_finals[precision] = final_atoms
            initial_row, final_row = _read_trajectory_summary(
                run_dir / "trajectory.csv"
            )
            metrics = _geometry_metrics(initial, final_atoms)
            prototype = STRUCTURE_BUILDERS[initial.info["prototype"]]()
            prototype *= SUPER_CELL_REPETITIONS
            distances = final_atoms.get_all_distances(mic=True)
            np.fill_diagonal(distances, np.inf)
            row = {
                "structure": name,
                "precision": precision,
                "status": summary["status"],
                "converged": summary["converged"],
                "steps": summary["steps_taken"],
                "prototype": initial.info["prototype"],
                "cell_scale_from_prototype": initial.info["cell_scale_from_prototype"],
                "initial_energy_ev_per_atom": float(
                    initial_row["energy_ev_per_atom"]
                ),
                "final_energy_ev_per_atom": float(final_row["energy_ev_per_atom"]),
                "initial_maximum_force_ev_per_angstrom": float(
                    initial_row["maximum_force_ev_per_angstrom"]
                ),
                "final_maximum_force_ev_per_angstrom": float(
                    final_row["maximum_force_ev_per_angstrom"]
                ),
                "final_maximum_generalized_force": float(
                    final_row["maximum_optimization_force"]
                ),
                "initial_maximum_absolute_stress_ev_per_angstrom3": float(
                    initial_row["maximum_absolute_stress_ev_per_angstrom3"]
                ),
                "final_maximum_absolute_stress_ev_per_angstrom3": float(
                    final_row["maximum_absolute_stress_ev_per_angstrom3"]
                ),
                "initial_cell_lengths_angstrom": initial.cell.lengths().tolist(),
                "final_cell_lengths_angstrom": final_atoms.cell.lengths().tolist(),
                "initial_cell_vectors_angstrom": initial.cell.array.tolist(),
                "final_cell_vectors_angstrom": final_atoms.cell.array.tolist(),
                "initial_cell_angles_degrees": initial.cell.angles().tolist(),
                "final_cell_angles_degrees": final_atoms.cell.angles().tolist(),
                "final_minimum_interatomic_distance_angstrom": float(distances.min()),
                "initial_rms_internal_offset_from_prototype_angstrom": (
                    _geometry_metrics(prototype, initial)["rms_atomic_displacement_angstrom"]
                ),
                "final_rms_internal_offset_from_prototype_angstrom": (
                    _geometry_metrics(prototype, final_atoms)["rms_atomic_displacement_angstrom"]
                ),
                **metrics,
                "final_structure": f"{name}/{precision}/final.extxyz",
                "error": summary["error"],
            }
            results.append(row)
            save_results()
            print(
                f"{name}/{precision}: {summary['status']} after "
                f"{summary['steps_taken']} steps; "
                f"ΔE/atom={row['final_energy_ev_per_atom'] - row['initial_energy_ev_per_atom']:+.5f} eV, "
                f"RMS shift={metrics['rms_atomic_displacement_angstrom']:.3f} Å",
                flush=True,
            )
        for stage, atoms in (
            ("initial", initial),
            ("final", precision_finals["fp64"]),
        ):
            _render_ovito(
                (
                    initial_path
                    if stage == "initial"
                    else structure_dir / "fp64" / "final.extxyz"
                ),
                output_dir / "figures" / f"{name}_{stage}.png",
                title=(
                    f"{name} - {stage.upper()} FP64"
                    f"{' (' + summary['status'] + ')' if stage == 'final' else ''} "
                    f"| {atoms.get_chemical_formula()} "
                    f"| {len(atoms)} atoms | cell "
                    f"{atoms.cell.lengths()[0]:.3f} x "
                    f"{atoms.cell.lengths()[1]:.3f} x "
                    f"{atoms.cell.lengths()[2]:.3f} A"
                ),
                element_numbers=element_numbers,
                framing_paths=(initial_path, structure_dir / "fp64" / "final.extxyz"),
            )
        _paired_structure_plot(
            initial,
            precision_finals["fp64"],
            output_dir / "figures" / f"{name}_paired.png",
            name=name,
            final_status=summary["status"],
        )
        _render_displacement_plot(
            initial, precision_finals["fp64"],
            output_dir / "figures" / f"{name}_displacements.png",
            structure_name=f"{name} ({summary['status']})",
        )
        _optimization_plot(
            structure_dir, output_dir / "figures" / f"{name}_optimization.png",
            name=name,
        )
    payload["precision_comparisons"] = _precision_comparisons(results)
    payload["starting_point_comparisons"] = _starting_point_comparisons(results)
    payload["active_case"] = None
    save_results(complete=True)
    print(f"Stress-test results written to {output_dir}")


if __name__ == "__main__":
    main()
