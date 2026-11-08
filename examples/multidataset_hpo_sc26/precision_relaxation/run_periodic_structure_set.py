#!/usr/bin/env python3
"""Relax representative periodic crystals in FP32/FP64 and render FP64 endpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.geometry import find_mic
from ase.io import read, write
from ase.spacegroup import crystal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_benchmark import _load_calculator, _resolve_single_head_map, _run_relaxation


STRUCTURE_BUILDERS = {
    "C_diamond": lambda: bulk("C", "diamond", a=3.567, cubic=True),
    "Si_diamond": lambda: bulk("Si", "diamond", a=5.430, cubic=True),
    "Ge_diamond": lambda: bulk("Ge", "diamond", a=5.658, cubic=True),
    "Al_fcc": lambda: bulk("Al", "fcc", a=4.050, cubic=True),
    "Cu_fcc": lambda: bulk("Cu", "fcc", a=3.615, cubic=True),
    "Ni_fcc": lambda: bulk("Ni", "fcc", a=3.524, cubic=True),
    "Fe_bcc": lambda: bulk("Fe", "bcc", a=2.866, cubic=True),
    "W_bcc": lambda: bulk("W", "bcc", a=3.165, cubic=True),
    "MgO_rocksalt": lambda: bulk("MgO", "rocksalt", a=4.212, cubic=True),
    "NaCl_rocksalt": lambda: bulk("NaCl", "rocksalt", a=5.640, cubic=True),
    "GaAs_zincblende": lambda: bulk("GaAs", "zincblende", a=5.653, cubic=True),
    "ZnO_wurtzite": lambda: bulk("ZnO", "wurtzite", a=3.250, c=5.200),
    "SrTiO3_perovskite": lambda: crystal(
        ["Sr", "Ti", "O"],
        basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
        spacegroup=221,
        cellpar=[3.905, 3.905, 3.905, 90, 90, 90],
    ),
}

ELEMENT_COLORS = {
    "C": (0.18, 0.20, 0.22),
    "Si": (0.91, 0.66, 0.13),
    "Ge": (0.55, 0.36, 0.61),
    "Al": (0.62, 0.68, 0.72),
    "Cu": (0.72, 0.37, 0.16),
    "Ni": (0.25, 0.47, 0.70),
    "Fe": (0.82, 0.25, 0.22),
    "W": (0.28, 0.39, 0.50),
    "Mg": (0.30, 0.68, 0.40),
    "O": (0.85, 0.17, 0.17),
    "Na": (0.92, 0.72, 0.15),
    "Cl": (0.20, 0.62, 0.34),
    "Ga": (0.12, 0.61, 0.63),
    "As": (0.57, 0.37, 0.25),
    "Zn": (0.30, 0.48, 0.79),
    "Sr": (0.91, 0.46, 0.16),
    "Ti": (0.47, 0.51, 0.56),
}

PRECISIONS = ("fp32", "fp64")
ATOMIC_RADIUS_PER_Z_ANGSTROM = 0.015
MAXIMUM_STEPS = 500
FORCE_TOLERANCE = 0.05
OPTIMIZER_MAXSTEP = 0.03
SUPERCELL_REPETITIONS = (2, 2, 2)


def _structures(seed: int) -> dict[str, Atoms]:
    rng = np.random.default_rng(seed)
    result: dict[str, Atoms] = {}
    for name, builder in STRUCTURE_BUILDERS.items():
        atoms = builder()
        atoms *= SUPERCELL_REPETITIONS
        directions = rng.normal(size=(len(atoms), 3))
        lengths = np.linalg.norm(directions, axis=1)
        if np.any(lengths == 0):
            raise ValueError(f"Generated a zero-length perturbation for {name}")
        magnitudes = rng.uniform(0.02, 0.05, size=len(atoms))
        atoms.positions += directions / lengths[:, None] * magnitudes[:, None]
        if not np.all(atoms.pbc):
            raise ValueError(f"{name} must be periodic in all three directions")
        result[name] = atoms
    return result


def _render_ovito(
    structure_path: Path,
    output_path: Path,
    *,
    title: str,
    element_numbers: dict[str, int],
    framing_paths: tuple[Path, ...] = (),
) -> None:
    try:
        from ovito.io import import_file
        from ovito.modifiers import PythonScriptModifier
        from ovito.vis import Material, OSPRayRenderer, Viewport
        from ovito.nonpublic import ParticleRenderingQuality
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:
        raise RuntimeError(
            "OVITO and Pillow are required to render the requested structure PNGs."
        ) from error

    type_colors = {
        symbol: ELEMENT_COLORS[symbol] for symbol in element_numbers
    }

    def apply_atom_style(_frame: int, data: Any) -> None:
        particle_types = data.particles_.particle_types_
        type_ids = np.asarray(data.particles["Particle Type"], dtype=np.int64)
        names_by_id = {
            int(type_id): particle_types.type_by_id(int(type_id)).name
            for type_id in np.unique(type_ids)
        }
        numbers_by_id = {
            type_id: element_numbers[symbol]
            for type_id, symbol in names_by_id.items()
        }
        atomic_numbers = np.asarray(
            [numbers_by_id[int(type_id)] for type_id in type_ids],
            dtype=np.float64,
        )
        data.particles_.create_property(
            "Radius",
            data=ATOMIC_RADIUS_PER_Z_ANGSTROM * atomic_numbers,
        )
        particle_vis = data.particles_.vis
        particle_vis.shape = particle_vis.Shape.Sphere
        particle_vis.rendering_quality = ParticleRenderingQuality.HighQuality
        particle_vis.material = Material(
            shading_model=Material.ShadingModel.PhysicallyBased,
            roughness=0.28,
            specular=0.65,
        )
        for type_id, symbol in names_by_id.items():
            particle_types.type_by_id_(type_id).color = type_colors[symbol]

    pipeline = import_file(str(structure_path))
    pipeline.modifiers.append(PythonScriptModifier(function=apply_atom_style))
    pipeline.add_to_scene()
    try:
        viewport = Viewport(type=Viewport.Type.Ortho)
        viewport.camera_dir = (1.0, -1.6, 1.1)
        if framing_paths:
            framing_pipelines = []
            try:
                for path in framing_paths:
                    framing_pipeline = import_file(str(path))
                    framing_pipeline.modifiers.append(
                        PythonScriptModifier(function=apply_atom_style)
                    )
                    framing_pipeline.add_to_scene()
                    framing_pipelines.append(framing_pipeline)
                viewport.zoom_all(size=(1600, 1200))
            finally:
                for framing_pipeline in framing_pipelines:
                    framing_pipeline.remove_from_scene()
        else:
            viewport.zoom_all(size=(1400, 900))
        renderer = OSPRayRenderer()
        renderer.ambient_light_enabled = True
        renderer.ambient_brightness = 0.55
        renderer.direct_light_enabled = True
        renderer.direct_light_intensity = 1.25
        renderer.sky_light_enabled = True
        renderer.samples_per_pixel = 24
        image = viewport.render_image(
            size=(1600, 1200),
            background=(0.985, 0.989, 0.995),
            renderer=renderer,
        )
        if image is None:
            raise RuntimeError(f"OVITO returned no rendered image for {structure_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not image.save(str(output_path)):
            raise OSError(f"OVITO failed to save image: {output_path}")
    finally:
        pipeline.remove_from_scene()

    with Image.open(output_path) as rendered:
        canvas = rendered.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=30)
    small_font = ImageFont.load_default(size=24)
    title_lines = [""]
    for word in title.split():
        candidate = f"{title_lines[-1]} {word}".strip()
        if draw.textlength(candidate, font=font) > canvas.width - 84:
            title_lines.append(word)
        else:
            title_lines[-1] = candidate
    draw.rounded_rectangle(
        (24, 20, canvas.width - 24, 54 + 38 * len(title_lines)),
        radius=12,
        fill=(255, 255, 255),
    )
    for index, line in enumerate(title_lines):
        draw.text((42, 37 + index * 38), line, fill=(25, 32, 40), font=font)
    legend_y = canvas.height - 70
    legend_x = 34
    for symbol in sorted(element_numbers, key=lambda item: element_numbers[item]):
        color = tuple(int(255 * channel) for channel in type_colors[symbol])
        draw.ellipse(
            (legend_x, legend_y, legend_x + 28, legend_y + 28),
            fill=color,
            outline=(35, 38, 42),
            width=2,
        )
        label = f"{symbol} (Z={element_numbers[symbol]})"
        draw.text(
            (legend_x + 36, legend_y + 1),
            label,
            fill=(25, 32, 40),
            font=small_font,
        )
        legend_x += int(draw.textlength(label, font=small_font)) + 78
    canvas.save(output_path, format="PNG", optimize=True)


def _summary_row(
    name: str,
    precision: str,
    initial: Atoms,
    final: Atoms,
    summary: dict[str, Any],
) -> dict[str, Any]:
    initial_fractional = np.linalg.solve(
        initial.cell.array.T, initial.positions.T
    ).T
    final_fractional = np.linalg.solve(final.cell.array.T, final.positions.T).T
    _, displacement_magnitudes = find_mic(
        (final_fractional - initial_fractional) @ final.cell.array,
        final.cell,
        pbc=final.pbc,
    )
    return {
        "structure": name,
        "formula": initial.get_chemical_formula(),
        "atom_count": len(initial),
        "precision": precision,
        "status": summary["status"],
        "converged": summary["converged"],
        "steps": summary["steps_taken"],
        "initial_cell_lengths_angstrom": initial.cell.lengths().tolist(),
        "final_cell_lengths_angstrom": final.cell.lengths().tolist(),
        "rms_internal_atomic_displacement_angstrom": float(
            np.sqrt(np.mean(displacement_magnitudes**2))
        ),
        "maximum_internal_atomic_displacement_angstrom": float(
            displacement_magnitudes.max()
        ),
        "relative_cell_volume_change_percent": float(
            100 * (final.get_volume() / initial.get_volume() - 1)
        ),
        "initial_energy_ev_per_atom": (
            None
            if summary["initial_energy_ev"] is None
            else summary["initial_energy_ev"] / len(initial)
        ),
        "final_energy_ev_per_atom": (
            None
            if summary["final_energy_ev"] is None
            else summary["final_energy_ev"] / len(final)
        ),
        "final_maximum_atomic_force_ev_per_angstrom": (
            summary["final_maximum_force_ev_per_angstrom"]
        ),
        "final_maximum_generalized_force": (
            summary["final_maximum_optimization_force"]
        ),
        "final_maximum_absolute_stress_ev_per_angstrom3": (
            summary["final_maximum_absolute_stress_ev_per_angstrom3"]
        ),
        "final_structure": f"{name}/{precision}/final.extxyz",
        "trajectory": f"{name}/{precision}/trajectory.traj",
        "error": summary["error"],
    }


def _render_displacement_plot(
    initial: Atoms,
    final: Atoms,
    output_path: Path,
    *,
    structure_name: str,
    arrow_scale: float | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    initial_fractional = np.linalg.solve(
        initial.cell.array.T, initial.positions.T
    ).T
    final_fractional = np.linalg.solve(final.cell.array.T, final.positions.T).T
    displacement, displacement_magnitudes = find_mic(
        (final_fractional - initial_fractional) @ final.cell.array,
        final.cell,
        pbc=final.pbc,
    )
    rms_displacement = float(np.sqrt(np.mean(displacement_magnitudes**2)))
    if arrow_scale is None:
        arrow_scale = (
            min(50.0, max(10.0, 0.5 / rms_displacement))
            if rms_displacement
            else 10.0
        )
    initial_in_final_cell = np.mod(initial_fractional, 1.0) @ final.cell.array
    final_positions = np.mod(final_fractional, 1.0) @ final.cell.array
    symbols = final.get_chemical_symbols()
    atomic_numbers = final.get_atomic_numbers()

    figure = plt.figure(figsize=(10.5, 8.5), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    numbers_by_symbol = {
        symbol: int(number)
        for symbol, number in zip(symbols, atomic_numbers, strict=True)
    }
    for symbol in sorted(numbers_by_symbol, key=lambda item: numbers_by_symbol[item]):
        mask = np.asarray([atom_symbol == symbol for atom_symbol in symbols])
        axis.scatter(
            final_positions[mask, 0],
            final_positions[mask, 1],
            final_positions[mask, 2],
            s=(0.25 * atomic_numbers[mask]) ** 2,
            c=[ELEMENT_COLORS[symbol]],
            edgecolors="#202832",
            linewidths=0.35,
            depthshade=True,
            alpha=0.95,
        )

    visible = displacement_magnitudes > 1e-5
    if np.any(visible):
        axis.quiver(
            initial_in_final_cell[visible, 0],
            initial_in_final_cell[visible, 1],
            initial_in_final_cell[visible, 2],
            displacement[visible, 0] * arrow_scale,
            displacement[visible, 1] * arrow_scale,
            displacement[visible, 2] * arrow_scale,
            color="#1f2937",
            linewidth=0.9,
            arrow_length_ratio=0.25,
            normalize=False,
        )

    cell = final.cell.array
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
                    alpha=0.65,
                )
    corners = np.asarray(
        [
            i * cell[0] + j * cell[1] + k * cell[2]
            for i, j, k in np.ndindex(2, 2, 2)
        ]
    )
    arrow_ends = initial_in_final_cell[visible] + arrow_scale * displacement[visible]
    bounds = np.concatenate((corners, final_positions, arrow_ends), axis=0)
    padding = 0.25
    for setter, component in zip(
        (axis.set_xlim, axis.set_ylim, axis.set_zlim), bounds.T, strict=True
    ):
        setter(float(component.min() - padding), float(component.max() + padding))
    axis.set_box_aspect(np.ptp(bounds, axis=0))
    axis.set_xlabel("x (Å)")
    axis.set_ylabel("y (Å)")
    axis.set_zlabel("z (Å)")
    maximum_displacement = float(displacement_magnitudes.max())
    axis.set_title(
        f"{structure_name}: FP64 atomic displacements\n"
        f"RMS = {rms_displacement:.4f} Å; max = {maximum_displacement:.4f} Å; "
        f"arrows ×{arrow_scale:.1f} (atom positions are unscaled)"
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=ELEMENT_COLORS[symbol],
            markeredgecolor="#202832",
            markersize=7,
            label=f"{symbol} (Z={number})",
        )
        for symbol, number in sorted(numbers_by_symbol.items(), key=lambda item: item[1])
    ]
    axis.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _add_precision_differences(rows: list[dict[str, Any]]) -> None:
    rows_by_name: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        rows_by_name.setdefault(row["structure"], {})[row["precision"]] = row
    for pair in rows_by_name.values():
        fp32 = pair["fp32"]
        fp64 = pair["fp64"]
        fp32_cell = np.asarray(fp32["final_cell_lengths_angstrom"], dtype=float)
        fp64_cell = np.asarray(fp64["final_cell_lengths_angstrom"], dtype=float)
        fp32["maximum_absolute_cell_length_difference_fp32_fp64_angstrom"] = float(
            np.max(np.abs(fp32_cell - fp64_cell))
        )
        fp32["absolute_final_energy_difference_fp32_fp64_ev_per_atom"] = (
            None
            if fp32["final_energy_ev_per_atom"] is None
            or fp64["final_energy_ev_per_atom"] is None
            else abs(
                fp32["final_energy_ev_per_atom"]
                - fp64["final_energy_ev_per_atom"]
            )
        )
        fp32_force = fp32["final_maximum_atomic_force_ev_per_angstrom"]
        fp64_force = fp64["final_maximum_atomic_force_ev_per_angstrom"]
        fp32["absolute_final_maximum_force_difference_fp32_fp64_ev_per_angstrom"] = (
            None
            if fp32_force is None or fp64_force is None
            else abs(fp32_force - fp64_force)
        )


def _write_report(
    output_dir: Path,
    rows: list[dict[str, Any]],
    seed: int,
    maximum_steps: int,
) -> None:
    header = (
        "| Structure | Formula | Atoms | FP32 status | FP32 steps | "
        "FP64 status | FP64 steps | Final cell lengths FP64 (Å) | "
        "Max cell delta (Å) | Energy delta (eV/atom) | "
        "Max-force delta (eV/Å) | RMS atom shift (Å) | "
        "Max atom shift (Å) | ΔV/V (FP64, %) | "
        "Final energy FP64 (eV/atom) | Max stress FP64 (eV/Å³) |"
    )
    table = [
        header,
        "|" + "|".join("---" for _ in range(16)) + "|",
    ]
    grouped = {name: {} for name in STRUCTURE_BUILDERS}
    for row in rows:
        grouped[row["structure"]][row["precision"]] = row
    for name, precision_results in grouped.items():
        fp32 = precision_results["fp32"]
        fp64 = precision_results["fp64"]
        lengths = ", ".join(f"{value:.3f}" for value in fp64["final_cell_lengths_angstrom"])
        energy = fp64["final_energy_ev_per_atom"]
        stress = fp64["final_maximum_absolute_stress_ev_per_angstrom3"]
        cell_difference = fp32[
            "maximum_absolute_cell_length_difference_fp32_fp64_angstrom"
        ]
        energy_difference = fp32[
            "absolute_final_energy_difference_fp32_fp64_ev_per_atom"
        ]
        force_difference = fp32[
            "absolute_final_maximum_force_difference_fp32_fp64_ev_per_angstrom"
        ]
        displacement_rms = fp64["rms_internal_atomic_displacement_angstrom"]
        displacement_max = fp64["maximum_internal_atomic_displacement_angstrom"]
        volume_change = fp64["relative_cell_volume_change_percent"]
        table.append(
            f"| {name} | {fp64['formula']} | {fp64['atom_count']} | "
            f"{fp32['status']} | {fp32['steps']} | {fp64['status']} | "
            f"{fp64['steps']} | {lengths} | {cell_difference:.3g} | "
            f"{'n/a' if energy_difference is None else f'{energy_difference:.3g}'} | "
            f"{'n/a' if force_difference is None else f'{force_difference:.3g}'} | "
            f"{displacement_rms:.4f} | {displacement_max:.4f} | "
            f"{volume_change:+.3f} | "
            f"{'n/a' if energy is None else f'{energy:.6f}'} | "
            f"{'n/a' if stress is None else f'{stress:.5g}'} |"
        )
    grouped_results = {
        name: {
            row["precision"]: row
            for row in rows
            if row["structure"] == name
        }
        for name in STRUCTURE_BUILDERS
    }
    converged_pairs = sum(
        pair["fp32"]["converged"] and pair["fp64"]["converged"]
        for pair in grouped_results.values()
    )
    maximum_energy_delta = max(
        (
            pair["fp32"]["absolute_final_energy_difference_fp32_fp64_ev_per_atom"]
            for pair in grouped_results.values()
            if pair["fp32"]["absolute_final_energy_difference_fp32_fp64_ev_per_atom"]
            is not None
        ),
        default=float("nan"),
    )
    maximum_cell_delta = max(
        pair["fp32"]["maximum_absolute_cell_length_difference_fp32_fp64_angstrom"]
        for pair in grouped_results.values()
    )
    maximum_force_delta = max(
        (
            pair["fp32"][
                "absolute_final_maximum_force_difference_fp32_fp64_ev_per_angstrom"
            ]
            for pair in grouped_results.values()
            if pair["fp32"][
                "absolute_final_maximum_force_difference_fp32_fp64_ev_per_angstrom"
            ]
            is not None
        ),
        default=float("nan"),
    )
    nonconverged_details = [
        f"- **{name} / {precision}:** {pair[precision]['status']} after "
        f"{pair[precision]['steps']} steps; final maximum atomic force "
        f"{pair[precision]['final_maximum_atomic_force_ev_per_angstrom']:.4f} "
        "eV/Å."
        for name, pair in grouped_results.items()
        for precision in PRECISIONS
        if not pair[precision]["converged"]
    ]
    nonconverged_section = (
        "\n".join(nonconverged_details)
        if nonconverged_details
        else "All paired runs converged."
    )
    report = f"""# Periodic structure relaxation: FP32 versus FP64

## Protocol

- Structures: {len(STRUCTURE_BUILDERS)} representative periodic crystal prototypes spanning elemental, binary, and ternary compositions.
- Initial structures: standard ASE crystal prototypes, each expanded to a 2×2×2 supercell and given deterministic random atomic displacements (seed {seed}, 0.02–0.05 Å).
- Relaxation: fused 16-branch HydraGNN/BranchWeightMLP, FP32 and FP64, ASE FIRE, 0.03 Å maximum step, hydrostatic variable-cell relaxation, 0.05 eV/Å generalized-force tolerance, up to {maximum_steps} steps.
- All three cell lengths change together under hydrostatic strain; atomic positions relax simultaneously. The graph uses the frozen checkpoint's configured radius and 20-neighbor cap.
- Energies, forces, and stresses are predictions from the frozen learned potential, not DFT or experimental validation.
- The FP64 initial and final structure images are rendered with OVITO's OSPRay renderer using physically based sphere shading and direct/ambient lighting. Atom radii use a common linear scale of atomic number (r = {ATOMIC_RADIUS_PER_Z_ANGSTROM:g} Å × Z); colors are element-specific and identified in each image legend.

## Paired results

{chr(10).join(table)}

Both precisions converged for {converged_pairs}/{len(STRUCTURE_BUILDERS)} structures.
The largest paired final-energy difference is {maximum_energy_delta:.3g} eV/atom,
the largest final cell-length difference is {maximum_cell_delta:.3g} Å, and
the largest difference in maximum atomic force is {maximum_force_delta:.3g} eV/Å.
These compare the two precisions of this same model and are numerical-consistency
metrics, not errors relative to a physical reference.

Non-converged cases remain in the table and retain their trajectories and final
coordinates; a capped result must not be described as a relaxed structure.
Generalized force is the ASE cell-filter convergence quantity. In some cases,
the final maximum atomic force can therefore be slightly above the threshold.
Structures requiring zero optimizer steps already met the generalized-force
criterion at their perturbed starting geometry. For any capped case, the
reported final force is a residual at the step limit, not a converged force.

Non-converged runs:

{nonconverged_section}

## FP64 structure images

Each structure has an OVITO PNG for its perturbed starting geometry and its
FP64 endpoint under `figures/`. A final image is a converged learned-potential
result only when the corresponding FP64 status above is `converged`.
The companion `*_displacements.png` plots show the FP64 endpoint atoms and
initial-to-final displacement arrows, scaled for visibility and labeled with
their scale. Their titles report the actual RMS and maximum atomic shift;
fractional-coordinate matching separates internal atomic motion from
hydrostatic cell expansion.

## Outputs

- `comparison.json`: complete paired metrics and run locations.
- `<structure>/fp32/` and `<structure>/fp64/`: ASE trajectories, per-step CSVs, logs, summaries, and final structures.
- `figures/`: paired FP64 initial and final OVITO renderings, plus displacement plots.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results" / "periodic_structure_set",
    )
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--max-steps", type=int, default=MAXIMUM_STEPS)
    args = parser.parse_args()
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    structures = _structures(args.seed)
    structure_root = output_dir / "structures"
    structure_root.mkdir()
    calculators = {
        precision: _load_calculator(
            precision,
            head_mode="fused",
            system="Si",
            single_head_map=_resolve_single_head_map([]),
        )[0]
        for precision in PRECISIONS
    }
    results: list[dict[str, Any]] = []
    for name, initial in structures.items():
        structure_dir = structure_root / name
        structure_dir.mkdir()
        initial_path = structure_dir / "initial.extxyz"
        write(initial_path, initial, format="extxyz")
        for precision in PRECISIONS:
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
                hydrostatic_cell=True,
            )
            final_atoms = read(run_dir / "final.extxyz")
            results.append(
                _summary_row(name, precision, initial, final_atoms, summary)
            )
            print(
                f"{name}/{precision}: {summary['status']} after "
                f"{summary['steps_taken']} steps",
                flush=True,
            )
        fp64_dir = structure_dir / "fp64"
        initial_lattice = initial.cell.lengths()
        final_fp64 = read(fp64_dir / "final.extxyz")
        numbers = initial.get_atomic_numbers()
        elements = initial.get_chemical_symbols()
        atomic_numbers = {
            symbol: int(number) for symbol, number in zip(elements, numbers, strict=True)
        }
        for stage, atoms, lattice in (
            ("initial", initial, initial_lattice),
            ("final", final_fp64, final_fp64.cell.lengths()),
        ):
            image_title = (
                f"{name} - {stage.upper()} FP64 | {atoms.get_chemical_formula()} "
                f"| {len(atoms)} atoms | cell {lattice[0]:.3f} x "
                f"{lattice[1]:.3f} x {lattice[2]:.3f} A"
            )
            _render_ovito(
                initial_path if stage == "initial" else fp64_dir / "final.extxyz",
                output_dir / "figures" / f"{name}_{stage}.png",
                title=image_title,
                element_numbers=atomic_numbers,
            )
        _render_displacement_plot(
            initial,
            final_fp64,
            output_dir / "figures" / f"{name}_displacements.png",
            structure_name=name,
        )

    comparison = {
        "model": "Frozen HydraGNN/BranchWeightMLP fused 16-head checkpoint",
        "precision_pair": list(PRECISIONS),
        "optimizer": "ASE FIRE",
        "optimizer_maxstep_angstrom": OPTIMIZER_MAXSTEP,
        "maximum_steps": args.max_steps,
        "force_tolerance_ev_per_angstrom": FORCE_TOLERANCE,
        "cell_relaxation": "hydrostatic volume-only strain, with positions relaxed",
        "supercell_repetitions": list(SUPERCELL_REPETITIONS),
        "displacement_seed": args.seed,
        "displacement_range_angstrom": [0.02, 0.05],
        "structures": results,
        "scientific_limit": (
            "Energies, forces, and stresses are predictions from the frozen "
            "learned potential; no DFT or experimental energy/force/stress "
            "reference calculations were performed."
        ),
    }
    _add_precision_differences(results)
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir, results, args.seed, args.max_steps)
    print(f"Results and OVITO PNGs written to {output_dir}")


if __name__ == "__main__":
    main()
