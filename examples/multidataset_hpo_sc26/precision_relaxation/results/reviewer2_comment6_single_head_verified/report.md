# FP32 vs FP64 HydraGNN structural-relaxation benchmark

## Scope and provenance

This benchmark evaluates only numerical precision robustness of the frozen
HydraGNN inference stack. It does not evaluate DFT accuracy, physical accuracy,
or molecular-dynamics stability. For each system, the selected branch is fixed by the configured single-head selection rule; only that decoder head is evaluated during each relaxation force call. The BranchWeightMLP is not loaded or invoked.

- HydraGNN checkpoint SHA-256: `8532b03296a8ab4cfc3b37beb6d57b2091a50807b155fac93cb1882253d32867`
- BranchWeightMLP: not loaded or used in single-head mode.
- HydraGNN parameter dtypes: FP32 `torch.float32`, FP64 `torch.float64`
- Devices: FP32 `cpu`, FP64 `cpu`
- Branch order: `['branch-0', 'branch-1', 'branch-2', 'branch-3', 'branch-4', 'branch-5', 'branch-6', 'branch-7', 'branch-8', 'branch-9', 'branch-10', 'branch-11', 'branch-12', 'branch-13', 'branch-14', 'branch-15']`
- Graph radius: 5.0 Å; maximum neighbors: 20
- Energy and force units follow the existing inference output convention: eV and eV/Å.
- Initial structures: standard ASE H₂O/CH₄ molecules, diamond Si conventional cell,
  rocksalt MgO conventional cell, and FCC Al conventional cell. Each was
  perturbed once using seed 42, then the saved identical coordinates were used
  for both precisions. Cells and PBC were fixed.
- Optimizer: ASE BFGS, maximum 200 steps, convergence if maximum atomic force is
  below 0.05 eV/Å. Failed and capped runs are retained in their trajectory folders.

## Single-head selection

The selected heads below were fixed before relaxation using the highest-weight head in the already-validated FP64 fused single-point results. This uses the MLP only as an offline selection diagnostic; the single-head relaxation run does not load or call it. Branch/dataset IDs follow the training launch's `--multi_model_list` order.

| System | Branch | Dataset |
|---|---|---|
| H2O | branch-8 | OMol25 |
| CH4 | branch-8 | OMol25 |
| Si | branch-7 | OMat24 |
| MgO | branch-2 | MPTrj |
| Al | branch-7 | OMat24 |

## Initial single-point precision comparison

| System | ΔE (eV) | ΔE/N (eV/atom) | Force MAE (eV/Å) | Force RMSE (eV/Å) | Max component ΔF (eV/Å) | MLP weight difference |
|---|---:|---:|---:|---:|---:|---:|
| H2O | 4.96565e-06 | 1.65522e-06 | 1.12017e-06 | 1.4735e-06 | 2.48978e-06 | n/a |
| CH4 | 2.8725e-06 | 5.74501e-07 | 1.82859e-06 | 2.41181e-06 | 5.07758e-06 | n/a |
| Si | 4.36118e-06 | 5.45148e-07 | 1.31016e-06 | 1.79308e-06 | 5.15431e-06 | n/a |
| MgO | 4.36972e-08 | 5.46215e-09 | 2.60635e-07 | 3.24667e-07 | 7.72257e-07 | n/a |
| Al | 7.3106e-06 | 1.82765e-06 | 1.19767e-06 | 1.35333e-06 | 2.30236e-06 | n/a |

## Structural relaxation summary

| System | Head | FP32 converged? | FP64 converged? | FP32 steps | FP64 steps | RMS displacement (Å) | Max cross-precision force (eV/Å) |
|---|---|---:|---:|---:|---:|---:|---:|
| H2O | branch-8 (OMol25) | True | True | 3 | 3 | 1.154608430135675e-08 | 0.03644788122522961 |
| CH4 | branch-8 (OMol25) | True | True | 11 | 11 | 1.1345270205710299e-05 | 0.036115781398387385 |
| Si | branch-7 (OMat24) | False | False | 200 | 200 | 5.94663770551577e-07 | 0.25391265561495596 |
| MgO | branch-2 (MPTrj) | True | True | 8 | 8 | 1.3416407863893358e-07 | 0.0400316444079769 |
| Al | branch-7 (OMat24) | True | True | 8 | 8 | 1.315294642981621e-07 | 0.03719883668284823 |

Unsuccessful runs:

- **Si/fp32:** max_steps_without_convergence; steps=200; final max force=0.25391047914162773 eV/Å; error=None.
- **Si/fp64:** max_steps_without_convergence; steps=200; final max force=0.253911189937239 eV/Å; error=None.

## Interpretation and limitations

Energy and force differences at identical geometries are reported above and in
`single_point_comparison.csv`. Final-geometry displacement is calculated after
translation/rotation alignment for molecules and minimum-image mapping in the
fixed cell for crystals. Both final geometries are also evaluated at a common
FP64 precision; cross-precision maximum residual forces are reported in the
summary table. Every row reports each precision's independent convergence
status and step count; the 0.05 eV/Å threshold is unchanged.

Across the five initial geometries, the largest absolute FP32–FP64 energy
difference was 7.3106e-06 eV and the largest force-component
difference was 5.15431e-06 eV/Å. Both precisions
converged for 4/5 systems. The nonconverged runs were
Si/fp32, Si/fp64; capped runs
must not be described as relaxed structures. The largest aligned final-geometry
RMS difference was 1.13453e-05 Å, and the largest
cross-precision maximum force was 0.253913 eV/Å.
Branch-weight differences are not applicable in single-head mode; the one-hot branch selection mask is not an MLP output.
At least one Si precision did not meet the force criterion. Agreement therefore supports numerical
consistency of this frozen model on these probes, not structural validity or
physical accuracy. No reference calculations were run, and the selected
compositions are not validated against the training distribution.

## Manuscript-ready wording

We compared FP32 and FP64 structural relaxations using the same frozen
HydraGNN checkpoint and identical perturbed starting geometries. Each structure was evaluated using its fixed, selected source-specific head; the MLP was not loaded or invoked in the relaxation.
Across the
five initial geometries, the largest absolute energy difference was
7.31e-06 eV and the largest force-component difference was
5.15e-06 eV/Å (maximum force RMSE
2.41e-06 eV/Å). Both precisions converged for
4/5 systems; Si did not reach the 0.05 eV/Å force criterion within 200 BFGS steps in either precision (final maximum forces 0.254 and 0.254 eV/Å, respectively). The maximum aligned FP32–FP64 final-geometry RMS difference was
1.13e-05 Å. These results characterize numerical precision
sensitivity of the pretrained potential on the tested structures; they do not
establish DFT-level accuracy or structural validity.

## Draft response to Reviewer 2, Comment 6

We thank the reviewer for requesting a direct FP32/FP64 relaxation comparison.
We performed paired relaxations of H₂O, CH₄, Si, MgO, and Al using identical
starting coordinates and the same frozen HydraGNN checkpoint; we selected one source-specific head per composition using the highest-weight FP64 fused single-point route, then ran the relaxation with that head alone and without loading or invoking the MLP.
ASE BFGS used a 0.05 eV/Å maximum-force criterion.
We additionally compared initial energy and force predictions, checked the
energy-gradient force implementation by finite differences, and cross-evaluated
the final structures at the opposite precision. Both precisions converged for
4/5 systems. Si did not reach the 0.05 eV/Å force criterion within 200 BFGS steps in either precision (final maximum forces 0.254 and 0.254 eV/Å, respectively).
The largest initial FP32–FP64 energy difference was 7.31e-06
eV, the largest force-component difference was
5.15e-06 eV/Å (maximum force RMSE
2.41e-06 eV/Å). The largest aligned final geometry RMS
difference was 1.13e-05 Å. We interpret these data as a
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
