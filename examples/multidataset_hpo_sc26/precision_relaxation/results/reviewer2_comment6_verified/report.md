# FP32 vs FP64 HydraGNN structural-relaxation benchmark

## Scope and provenance

This benchmark evaluates only numerical precision robustness of the frozen
HydraGNN + BranchWeightMLP inference stack. It does not evaluate DFT accuracy,
physical accuracy, or molecular-dynamics stability. All 16 source-specific
prediction heads were included at every force step; branch skipping was
disabled. The composition-dependent 16-way softmax weights are fixed during a
relaxation because each system's element counts are unchanged.

- HydraGNN checkpoint SHA-256: `8532b03296a8ab4cfc3b37beb6d57b2091a50807b155fac93cb1882253d32867`
- BranchWeightMLP SHA-256: `c984b338ee947534bbdedf1fc4dd2aea92bb846e3302638d6719b1e3f5ca7ca8`
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

## Initial single-point precision comparison

| System | ΔE (eV) | ΔE/N (eV/atom) | Force MAE (eV/Å) | Force RMSE (eV/Å) | Max component ΔF (eV/Å) | Max Δweight |
|---|---:|---:|---:|---:|---:|---:|
| H2O | 4.96565e-06 | 1.65522e-06 | 1.12017e-06 | 1.4735e-06 | 2.48978e-06 | 1.18687e-29 |
| CH4 | 2.86094e-06 | 5.72188e-07 | 1.67466e-06 | 2.49281e-06 | 7.71938e-06 | 1.37861e-08 |
| Si | 4.43753e-06 | 5.54691e-07 | 1.59102e-06 | 2.09559e-06 | 6.21569e-06 | 4.41725e-08 |
| MgO | 4.36972e-08 | 5.46215e-09 | 2.60635e-07 | 3.24667e-07 | 7.72257e-07 | 8.83591e-126 |
| Al | 7.33678e-06 | 1.8342e-06 | 1.10733e-06 | 1.28252e-06 | 2.45494e-06 | 2.82096e-08 |

## Structural relaxation summary

| System | FP32 converged? | FP64 converged? | FP32 steps | FP64 steps | RMS displacement (Å) | Max cross-precision force (eV/Å) |
|---|---:|---:|---:|---:|---:|---:|
| H2O | True | True | 3 | 3 | 1.154608430135675e-08 | 0.03644788122522961 |
| CH4 | True | True | 11 | 11 | 1.1000104597585498e-05 | 0.03606221316469073 |
| Si | False | False | 200 | 200 | 5.053835178308751e-07 | 0.22812918805614835 |
| MgO | True | True | 8 | 8 | 1.3416407863893358e-07 | 0.0400316444079769 |
| Al | True | True | 8 | 8 | 1.2369316868240412e-07 | 0.037198916308851936 |

Unsuccessful runs:

- **Si/fp32:** max_steps_without_convergence; steps=200; final max force=0.22812408403446494 eV/Å; error=None.
- **Si/fp64:** max_steps_without_convergence; steps=200; final max force=0.2281249972533527 eV/Å; error=None.

## Interpretation and limitations

Energy and force differences at identical geometries are reported above and in
`single_point_comparison.csv`. Final-geometry displacement is calculated after
translation/rotation alignment for molecules and minimum-image mapping in the
fixed cell for crystals. Both final geometries are also evaluated at a common
FP64 precision; cross-precision maximum residual forces are reported in the
summary table. Every row reports each precision's independent convergence
status and step count; the 0.05 eV/Å threshold is unchanged.

Across the five initial geometries, the largest absolute FP32–FP64 energy
difference was 7.33678e-06 eV and the largest force-component
difference was 7.71938e-06 eV/Å. Both precisions
converged for 4/5 systems. The nonconverged runs were
Si/fp32, Si/fp64; capped runs
must not be described as relaxed structures. The largest aligned final-geometry
RMS difference was 1.10001e-05 Å, and the largest
cross-precision maximum force was 0.228129 eV/Å.
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
7.34e-06 eV and the largest force-component difference was
7.72e-06 eV/Å (maximum force RMSE
2.49e-06 eV/Å); the maximum branch-weight difference was
4.42e-08. Both precisions converged for
4/5 systems; Si did not reach the 0.05 eV/Å force
criterion within 200 BFGS steps in either precision (final maximum forces
0.228 and
0.228 eV/Å,
respectively). The maximum aligned FP32–FP64 final-geometry RMS difference was
1.1e-05 Å. These results characterize numerical precision
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
4/5 systems; Si did not meet the force criterion within 200 steps
in either precision, with final maximum forces of
0.228 and
0.228 eV/Å.
The largest initial FP32–FP64 energy difference was 7.34e-06
eV, the largest force-component difference was
7.72e-06 eV/Å (maximum force RMSE
2.49e-06 eV/Å); the maximum branch-weight difference was
4.42e-08. The largest aligned final geometry RMS
difference was 1.1e-05 Å. We interpret these data as a
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
