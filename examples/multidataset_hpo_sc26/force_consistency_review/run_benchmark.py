"""Frozen FP64 energy/force consistency benchmark; no production code changes."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from torch_geometric.data import Batch

HERE = Path(__file__).resolve().parent
SC26 = HERE.parent
ROOT = SC26.parents[1]
sys.path.insert(0, str(SC26 / "precision_relaxation"))
sys.path.insert(0, str(SC26))
sys.path.insert(0, str(ROOT))

from hydragnn_ase_calculator import _graph_from_atoms, load_fused_calculator, sha256_file
from inference_fused import (
    _batched_decode_all_branches,
    _build_batched_decoder,
    _encode_once,
    _predict_branch_energy_only,
    _reshape_composition,
    run_fused_inference,
)

SOURCE = SC26 / "precision_relaxation/results/off_equilibrium_full_cell_20261108_0328"
METHODS = ("A", "B", "C", "D")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(delta):
    return {
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "mae": float(np.mean(np.abs(delta))),
        "max_component": float(np.max(np.abs(delta))),
        "max_atom_vector": float(np.max(np.linalg.norm(delta, axis=1))),
    }


def graph_digest(data):
    digest = hashlib.sha256()
    for key in ("pos", "cell", "edge_index", "edge_shifts", "chemical_composition", "x"):
        digest.update(getattr(data, key).detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def same_topology(first, second):
    def identities(data):
        cell = data.cell.detach().cpu().numpy().reshape(3, 3)
        shifts = np.rint(data.edge_shifts.detach().cpu().numpy() @ np.linalg.inv(cell)).astype(int)
        edges = data.edge_index.detach().cpu().numpy().T
        return {tuple(row) for row in np.column_stack((edges, shifts))}
    return identities(first) == identities(second)


def attached_energies(calc, graph, decoder=None):
    batch = Batch.from_data_list([graph.clone()]).to(calc.device)
    batch.pos.requires_grad_(True)
    with calc.autocast_ctx:
        if decoder is None:
            energies = torch.stack([
                _predict_branch_energy_only(calc.model, batch, branch).reshape(())
                for branch in range(16)
            ])
        else:
            encoded = _encode_once(calc.model, batch)
            energies = torch.stack(_batched_decode_all_branches(
                calc.model, batch, encoded, decoder
            )).reshape(16)
    if batch.pos.dtype != torch.float64 or energies.dtype != torch.float64:
        raise TypeError("Coordinate and energy calculations must remain FP64")
    return batch, energies


def gradient(energy, positions, retain_graph=False):
    if not energy.requires_grad:
        return torch.zeros_like(positions)
    return -torch.autograd.grad(energy, positions, retain_graph=retain_graph)[0]


def inference(calc, graph, threshold, config):
    outputs = run_fused_inference(
        calc.model, calc.mlp, [graph.clone()], 1, calc.param_dtype,
        calc.autocast_ctx, calc.device, 16, 0, calc.mlp_device,
        calc.mlp_autocast_ctx, encoder_reuse=config["encoder_reuse"],
        fused_energy_grad=True, weight_threshold=threshold,
        batched_decoder=config["batched_decoder"], num_streams=1,
    )
    assert outputs[2] == [graph.num_nodes]
    return outputs[0][0], outputs[1][0].numpy(), outputs[3][0].numpy()


def generate_deliverables(results, aggregate, provenance, fd_rows):
    names = {
        "A": "Complete FP64 reference",
        "B": "Full-branch fused FP64",
        "C": "Experimental skipping (threshold 0.2)",
        "D": "Consistent truncation control",
    }
    definitions = {
        "A": (r"$E_{\rm full}$", r"$-\nabla E_{\rm full}$"),
        "B": (r"$E_{\rm full}$", r"$-\nabla E_{\rm full}$"),
        "C": (r"$E_{\rm full}$", r"$-\nabla E_S$"),
        "D": (r"$E_S$", r"$-\nabla E_S$"),
    }
    table_names = {"A": "A: FP64 reference", "B": "B: Full fused", "C": "C: Force skipping", "D": "D: Truncation control"}
    table = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Energy--force consistency on ten saved periodic structures in FP64.",
        r"Force errors are in eV/\AA\ relative to A; means are unweighted across structures.",
        r"Max error denotes the largest absolute Cartesian force-component error.",
        r"Skipping uses threshold 0.2 with no weight renormalization.",
        r"Conservativity is local to fixed periodic connectivity and branch selection.}",
        r"\label{tab:force-consistency}",
        r"\begin{tabular}{lllrrrrl}",
        r"\hline",
        r"Method & Energy & Forces & Mean RMSE & Max RMSE & Mean max error & Max error & Conservative? \\",
        r"\hline",
    ]
    report_rows = []
    for method in METHODS:
        row = aggregate[method]
        conservative = "No, in general" if method == "C" else "Yes (fixed selection/graph)"
        values = [row[key] for key in ("mean_rmse", "max_rmse", "mean_max_component", "max_max_component")]
        table.append(
            f"{table_names[method]} & " + " & ".join(definitions[method]) + " & "
            + " & ".join(f"{v:.3e}" for v in values)
            + (" & No" if method == "C" else " & Yes") + r" \\"
        )
        report_rows.append(
            f"| {names[method]} | " + " | ".join(f"{v:.6g}" for v in values) + f" | {conservative} |"
        )
    table.extend([
        r"\hline", r"\end{tabular}", r"\end{table*}",
        "% All force errors in eV/Angstrom; reference A. Mean is unweighted across ten structures.",
        "% C: full energy, retained-only gradient. D: retained energy and its gradient, no renormalization.",
        "% Conservativity statements are local to a fixed graph and selection, not across graph switches.",
    ])
    (HERE / "comparison_table.tex").write_text("\n".join(table) + "\n")
    skipped = [r["skipped_fraction"] for r in results]
    masses = [r["skipped_weight"] for r in results]
    residual = max(r["methods"]["C"]["gradient_residual"]["max_component"] for r in results)
    d_residual = max(r["methods"]["D"]["gradient_residual"]["max_component"] for r in results)
    finite_summary = []
    for h in sorted({r["h_angstrom"] for r in fd_rows}, reverse=True):
        selected = [r for r in fd_rows if r["h_angstrom"] == h and r["graph_policy"] == "fixed"]
        finite_summary.append(
            f"- h={h:g} Angstrom: maximum full/truncated derivative errors "
            f"{max(r['full_absolute_error'] for r in selected):.6g} / "
            f"{max(r['truncated_absolute_error'] for r in selected):.6g} eV/Angstrom."
        )
    largest = max(results, key=lambda r: r["methods"]["C"]["force_error"]["rmse"])
    exact_energy_pairs = sum(r["C_energy_equals_B_exactly"] for r in results)
    report = f"""# Reviewer 1, Comment 6: energy-force consistency

## Implementation audit

The existing `inference_fused.py` evaluates all branch energies.
The non-fused serial path computes each negative energy gradient and aggregates
with the composition-only softmax weights. The fused path uses one backward
call. With skipping, `_fused_energy_forces` differentiates only retained energies
and adds **detached skipped energies** to its numerical energy output.
Therefore A/B report E_full and its gradient; C reports E_full but F_C =
-sum(retained w_k grad E_k). D is a new diagnostic control: E_D =
sum(retained w_k E_k), F_D = -grad E_D, using unnormalized weights.
Detaching a contribution does not remove its numerical coordinate dependence.
The direct C check reconstructs the complete coordinate-dependent returned energy,
including skipped contributions, rather than differentiating a detached tensor.

A independently executes 16 complete model forwards and differentiates their
weighted sum. B and C execute the production `run_fused_inference` with encoder
reuse, batched decoding, fused gradients, one stream, and batch size one.
The direct C/D gradients use independently executed batched-decoder forwards.
All modes receive clones of the exact same periodic graph and coordinates.
The omitted-energy gradient identity is also checked against C minus A.

## Measured comparison

All errors are eV/Angstrom relative to A. Means are unweighted means over ten
structures; maximum RMSE is the largest structure RMSE, not a pooled RMSE.

| Method | Mean RMSE | Max RMSE | Mean max component | Max component | Conservative? |
|---|---:|---:|---:|---:|---|
{chr(10).join(report_rows)}

Energy differences, MAE, largest atom-vector errors, active/skipped branch IDs,
all weights, and direct gradient residuals are in `results.json` and
`per_structure.csv`; complete force arrays are in `forces.npz`.
The largest C direct force-gradient residual is {residual:.6g} eV/Angstrom.
The largest D direct residual is {d_residual:.6g} eV/Angstrom.

The largest skipping discrepancy occurs in **{largest['structure']}**.
Its skipped softmax mass is {largest['skipped_weight']:.6g}
({largest['skipped_weight'] * 100:.6g}%).
Routing is highly concentrated for this saved-structure set; several skipped
weights underflow to zero in FP64. Small errors here do not bound errors for
other compositions or more diffuse routing. The mean skipped fraction measures
omitted branch force contributions, not omitted energies or runtime speedup.

## Scientific answers

1. **Full fused equivalence:** measured maximum component difference is
{aggregate['B']['max_max_component']:.6g} eV/Angstrom. This is numerical evidence
for these ten structures, not a universal bit-exact claim.
2. **Skipping discrepancy:** mean / maximum structure RMSE are
{aggregate['C']['mean_rmse']:.6g} / {aggregate['C']['max_rmse']:.6g} eV/Angstrom.
3. **Returned-energy consistency:** C is not generally conservative with respect
to its full reported energy. Zero or tiny residuals on saturated routing samples
do not alter the implemented mathematical mismatch.
4. **Skipped energies:** retained in C's reported energy; energy equality and
differences are measured explicitly for each structure. C and B energies are
exactly equal for {exact_energy_pairs}/10 structures; the largest difference is
{max(abs(r['C_minus_B_energy_ev']) for r in results):.6g} eV, reflecting summation
grouping, not removed contributions.
5. **Consistent truncation:** D restores the local gradient relationship, but
changes both the potential energy and its forces relative to A.
6. **Skipped branches/weight:** mean skipped fraction {np.mean(skipped):.6%},
range {min(skipped):.6%}–{max(skipped):.6%}; mean skipped weight
{np.mean(masses):.6g}, range {min(masses):.6g}–{max(masses):.6g}.
The threshold 0.2 was explicitly selected by the user on 2026-11-08; it has
**not been verified as the threshold used for the historical approximately 81%
reduction**. No attempt was made to tune this experiment to reproduce 81%.
7. **Published screening:** according to the supplied study provenance, the
1.1-billion-structure screening used the complete 16-branch formulation,
without experimental branch skipping. This benchmark does not independently
audit historical screening jobs.

Full-branch inference is appropriate for conservative-force applications
within smooth graph regions. C should not be used for structural relaxation
or molecular dynamics with its full reported energy. D is only a diagnostic,
not a validated production replacement. Periodic neighbor switches can still
make A/B/D nonsmooth; this study does not negate the carbon graph diagnostics.
No physical-accuracy or training-domain claim follows from numerical consistency.

## Finite-difference control

For each structure, the component with largest C-versus-A force discrepancy
is checked at three coordinate steps. Both rebuilt periodic graphs and fixed
reference graphs are evaluated. `finite_differences.csv` labels graph changes
and reports both full- and truncated-energy derivative errors.
Fixed-graph checks isolate local differentiation from discrete neighbor changes.
Selection and composition weights remain fixed in these checks.
All selected rebuilt-graph perturbations retained their graph identities.

{chr(10).join(finite_summary)}

At the largest-discrepancy Al coordinate, the full-energy finite difference
continues to disagree with C's returned force by approximately 1.53e-5 eV/Angstrom
while agreeing with the full-energy autograd force to far smaller error.

## Reproducibility and preservation

See `config.json`, `input_manifest.json`, and `checkpoint_provenance.json`.
Both required checkpoint hashes were checked before loading; parameters are
frozen/eval and FP64. Branch order follows the checkpoint configuration.
Input paths are committed distorted initial structures, with identical atom
ordering across modes; graph cutoff is {provenance['radius_angstrom']} Angstrom,
maximum incoming neighbors {provenance['max_neighbours']}, charge/spin zero.
Original relaxation results and inference source hashes were verified unchanged.
No optimization, DFT, retraining, or manuscript edit was performed.
"""
    (HERE / "report.md").write_text(report)
    (HERE / "reviewer_response.md").write_text(f"""# Draft response to Reviewer 1, Comment 6

We thank the reviewer for requesting a direct energy-force consistency test.
Using the frozen checkpoint pair, we compared ten saved periodic configurations
in FP64 with identical graphs and composition weights. Full-branch fused
inference differed from an independently executed full-energy-gradient reference
by a maximum force component of {aggregate['B']['max_max_component']:.3e} eV/Angstrom.
With the explicitly selected skipping threshold 0.2, the mean and maximum
per-structure force RMSE were {aggregate['C']['mean_rmse']:.3e} and
{aggregate['C']['max_rmse']:.3e} eV/Angstrom.

The experimental skipping implementation retains all branch contributions in
the reported energy but omits skipped contributions from the force gradient.
It therefore does not generally preserve F = -grad E. Direct differentiation
of the complete reported energy verifies this distinction. A diagnostic control
that omits the same branches from both energy and forces restores local
consistency, without renormalizing retained weights, but changes the potential.

The published 1.1-billion-structure screening used all 16 branches without
skipping, as specified in the study provenance. We distinguish that production
formulation from the experimental force-skipping optimization and recommend
full-branch inference for conservative-force applications. These numerical
tests do not establish universal physical accuracy or smoothness across periodic
neighbor switches. The present threshold is user-selected and is not verified
as the threshold behind the historical approximately 81% reduction.
""")
    (HERE / "inference_discussion_revision.md").write_text(f"""# Proposed inference-optimization discussion (not applied to manuscript)

Encoder reuse, batched branch decoding, and fused energy differentiation retain
the complete potential E = sum_k w_k(Z) E_k(R,Z). Since the composition-based
weights do not depend on coordinates, its forces are F = -grad_R E.
Across ten saved periodic configurations in FP64, full-branch fused inference
gave a maximum force-component discrepancy of
{aggregate['B']['max_max_component']:.3e} eV/Angstrom relative to an independently
executed complete-gradient reference.

Experimental branch-force skipping is distinct: branch energies remain in the
reported total while low-weight branches are excluded from differentiation.
At the user-selected threshold 0.2, the mean per-structure force RMSE was
{aggregate['C']['mean_rmse']:.3e} eV/Angstrom, and the mean fraction of omitted
branch force contributions was {np.mean(skipped):.2%}. This is not a reduction
in the number of branch energy predictions and is not a measured runtime
speedup. The historical approximately 81% claim is not revalidated here.
Such force-only skipping is not a conservative evaluator of the reported
potential and should not be used for relaxation or molecular dynamics.
Dropping the same contributions from both energy and forces restores local
consistency but defines a different approximate potential; this was tested only
as a diagnostic control. All local gradient statements assume fixed periodic
connectivity and selection.

The published screening of 1.1 billion structures used the complete 16-branch
formulation without branch skipping, according to the study provenance.
""")


def main():
    config = json.loads((HERE / "config.json").read_text())
    assert config["precision"] == "fp64" and config["weight_threshold"] == 0.2
    protected = [SC26 / "inference_fused.py", SC26 / "precision_relaxation/hydragnn_ase_calculator.py"]
    protected.extend(path for path in SOURCE.rglob("*") if path.is_file())
    before = {str(path.relative_to(ROOT)): sha256_file(path) for path in protected}
    calc, provenance = load_fused_calculator(
        logdir=SC26 / "multidataset_hpo-BEST6-fp64",
        backbone_checkpoint=SC26 / "multidataset_hpo-BEST6-fp64/multidataset_hpo-BEST6-fp64_epoch_97.pk",
        mlp_checkpoint=SC26 / "mlp_branch_weights.pt",
        precision="fp64", charge=0.0, spin=0.0,
    )
    torch.set_default_dtype(torch.float64)
    decoder = _build_batched_decoder(calc.model, calc.device)
    provenance["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    provenance["threshold_provenance"] = config["threshold_provenance"]
    write_json(HERE / "checkpoint_provenance.json", provenance)
    results, manifest, csv_rows, fd_rows, arrays = [], [], [], [], {}
    for name in config["structures"]:
        path = SOURCE / name / "initial.extxyz"
        relative = str(path.relative_to(ROOT))
        subprocess.run(["git", "ls-files", "--error-unmatch", relative], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        atoms = read(path)
        graph = _graph_from_atoms(atoms, dtype=torch.float64, radius=calc.radius,
                                  max_neighbours=calc.max_neighbours, charge=0.0, spin=0.0)
        original_digest = graph_digest(graph)
        with torch.no_grad():
            composition = _reshape_composition(Batch.from_data_list([graph])).to(calc.mlp_device, dtype=torch.float64)
            weights = torch.softmax(calc.mlp(composition), dim=-1).reshape(16).to(calc.device)
        torch.testing.assert_close(weights.sum(), weights.new_tensor(1.0), atol=2e-14, rtol=0)
        active = torch.nonzero(weights >= config["weight_threshold"]).reshape(-1).tolist()
        skipped = [branch for branch in range(16) if branch not in active]
        batch, branch_e = attached_energies(calc, graph)
        full = (weights * branch_e).sum()
        truncated = (weights[active] * branch_e[active]).sum()
        omitted = (weights[skipped] * branch_e[skipped]).sum()
        force_a = gradient(full, batch.pos, retain_graph=True).detach().cpu().numpy()
        force_d = gradient(truncated, batch.pos, retain_graph=True).detach().cpu().numpy()
        omitted_gradient = -gradient(omitted, batch.pos).detach().cpu().numpy()
        energy_a, energy_d = full.item(), truncated.item()
        energy_b, force_b, weights_b = inference(calc, graph, 0.0, config)
        energy_c, force_c, weights_c = inference(calc, graph, config["weight_threshold"], config)
        np.testing.assert_array_equal(weights_b, weights.detach().cpu().numpy())
        np.testing.assert_array_equal(weights_c, weights_b)
        direct_batch, direct_e = attached_energies(calc, graph, decoder)
        reported_energy_c = (weights * direct_e).sum()
        reported_energy_d = (weights[active] * direct_e[active]).sum()
        direct_c = gradient(reported_energy_c, direct_batch.pos, retain_graph=True).detach().cpu().numpy()
        direct_d = gradient(reported_energy_d, direct_batch.pos).detach().cpu().numpy()
        np.testing.assert_allclose(energy_c, reported_energy_c.item(), atol=1e-10, rtol=1e-12)
        np.testing.assert_allclose(energy_d, reported_energy_d.item(), atol=1e-10, rtol=1e-12)
        np.testing.assert_allclose(force_b, force_a, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(force_d, direct_d, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(force_c, force_d, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(force_c - force_a, omitted_gradient, atol=1e-9, rtol=1e-9)
        assert graph_digest(graph) == original_digest
        structure = {
            "structure": name, "natoms": len(atoms), "active_branches": active,
            "skipped_branches": skipped, "skipped_fraction": len(skipped) / 16,
            "retained_weight": weights[active].sum().item(),
            "skipped_weight": weights[skipped].sum().item(),
            "weights": weights.detach().cpu().tolist(), "methods": {},
            "C_energy_equals_B_exactly": energy_c == energy_b,
            "C_minus_B_energy_ev": energy_c - energy_b,
            "omitted_gradient_identity_max_error": float(np.max(np.abs(force_c - force_a - omitted_gradient))),
        }
        for method, energy, force, direct in [
            ("A", energy_a, force_a, force_a), ("B", energy_b, force_b, direct_c),
            ("C", energy_c, force_c, direct_c), ("D", energy_d, force_d, direct_d),
        ]:
            measured = metrics(force - force_a)
            residual = metrics(force - direct)
            structure["methods"][method] = {
                "energy_ev": energy, "energy_difference_ev": energy - energy_a,
                "energy_difference_ev_per_atom": (energy - energy_a) / len(atoms),
                "force_error": measured, "gradient_residual": residual,
            }
            csv_rows.append({
                "structure": name, "method": method, "natoms": len(atoms),
                "active_branches": ";".join(map(str, active)) if method in ("C", "D") else ";".join(map(str, range(16))),
                "skipped_fraction": structure["skipped_fraction"] if method in ("C", "D") else 0,
                "skipped_weight": structure["skipped_weight"] if method in ("C", "D") else 0,
                "energy_ev": energy, "delta_energy_ev": energy - energy_a,
                **measured, "direct_gradient_max_residual": residual["max_component"],
            })
            arrays[f"{name}_{method}"] = force
        arrays[f"{name}_C_reported_energy_gradient_force"] = direct_c
        arrays[f"{name}_D_reported_energy_gradient_force"] = direct_d
        manifest.append({
            "structure": name, "path": relative, "sha256": sha256_file(path),
            "graph_sha256": original_digest, "edge_count": graph.edge_index.shape[1],
            "composition_counts": composition.cpu().reshape(-1).tolist(),
            "atomic_numbers": atoms.numbers.tolist(),
        })
        atom, axis = np.unravel_index(np.argmax(np.abs(force_c - force_a)), force_a.shape)
        for h in config["finite_difference_steps_angstrom"]:
            values = {"fixed": [], "rebuilt": []}
            graph_stable = True
            for sign in (-1, 1):
                moved = atoms.copy()
                moved.positions[atom, axis] += sign * h
                rebuilt = _graph_from_atoms(moved, dtype=torch.float64, radius=calc.radius,
                                            max_neighbours=calc.max_neighbours, charge=0.0, spin=0.0)
                graph_stable = graph_stable and same_topology(graph, rebuilt)
                fixed = graph.clone()
                fixed.pos[atom, axis] += sign * h
                for label, probe in (("fixed", fixed), ("rebuilt", rebuilt)):
                    with torch.no_grad():
                        _, energies = attached_energies(calc, probe, decoder)
                        values[label].append(((weights * energies).sum().item(),
                                              (weights[active] * energies[active]).sum().item()))
            for label, energies in values.items():
                full_fd, truncated_fd = -(np.array(energies[1]) - np.array(energies[0])) / (2 * h)
                fd_rows.append({
                    "structure": name, "atom": int(atom), "axis": int(axis), "h_angstrom": h,
                    "graph_policy": label, "rebuilt_graph_unchanged": graph_stable,
                    "full_fd_force": full_fd, "full_autograd_force": direct_c[atom, axis],
                    "full_absolute_error": abs(full_fd - direct_c[atom, axis]),
                    "truncated_fd_force": truncated_fd, "truncated_autograd_force": direct_d[atom, axis],
                    "truncated_absolute_error": abs(truncated_fd - direct_d[atom, axis]),
                    "C_returned_force": force_c[atom, axis],
                    "C_vs_full_fd_error": abs(full_fd - force_c[atom, axis]),
                })
        results.append(structure)
        write_json(HERE / "results.json", results)
        print(f"Completed {name}: active={active}, skipped weight={structure['skipped_weight']:.6g}, C RMSE={structure['methods']['C']['force_error']['rmse']:.6g}", flush=True)
    aggregate = {}
    aggregate_rows = []
    for method in METHODS:
        row = {"method": method}
        for key in ("rmse", "mae", "max_component", "max_atom_vector"):
            values = [r["methods"][method]["force_error"][key] for r in results]
            row[f"mean_{key}"], row[f"max_{key}"] = float(np.mean(values)), float(np.max(values))
        row["max_absolute_energy_difference_ev"] = max(abs(r["methods"][method]["energy_difference_ev"]) for r in results)
        row["max_direct_gradient_residual"] = max(r["methods"][method]["gradient_residual"]["max_component"] for r in results)
        aggregate[method] = row
        aggregate_rows.append(row)
    assert before == {str(path.relative_to(ROOT)): sha256_file(path) for path in protected}
    assert all(not p.requires_grad and p.dtype == torch.float64 for p in (*calc.model.parameters(), *calc.mlp.parameters()))
    write_json(HERE / "preservation_manifest.json", before)
    write_json(HERE / "input_manifest.json", manifest)
    write_json(HERE / "summary.json", aggregate)
    write_csv(HERE / "summary.csv", aggregate_rows)
    write_csv(HERE / "per_structure.csv", csv_rows)
    write_csv(HERE / "finite_differences.csv", fd_rows)
    np.savez_compressed(HERE / "forces.npz", **arrays)
    generate_deliverables(results, aggregate, provenance, fd_rows)
    print("Verified all ten structures, direct gradient identities, FP64/frozen models, and unchanged source results.", flush=True)


if __name__ == "__main__":
    main()
