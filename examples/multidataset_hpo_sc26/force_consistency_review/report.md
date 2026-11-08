# Reviewer 1, Comment 6: energy-force consistency

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
| Complete FP64 reference | 0 | 0 | 0 | 0 | Yes (fixed selection/graph) |
| Full-branch fused FP64 | 1.78796e-16 | 1.00405e-15 | 5.10703e-16 | 2.77556e-15 | Yes (fixed selection/graph) |
| Experimental skipping (threshold 0.2) | 5.98508e-07 | 5.98507e-06 | 1.52731e-06 | 1.52731e-05 | No, in general |
| Consistent truncation control | 5.98508e-07 | 5.98507e-06 | 1.52731e-06 | 1.52731e-05 | Yes (fixed selection/graph) |

Energy differences, MAE, largest atom-vector errors, active/skipped branch IDs,
all weights, and direct gradient residuals are in `results.json` and
`per_structure.csv`; complete force arrays are in `forces.npz`.
The largest C direct force-gradient residual is 1.52731e-05 eV/Angstrom.
The largest D direct residual is 3.88578e-15 eV/Angstrom.

The largest skipping discrepancy occurs in **Al_fcc_compressed**.
Its skipped softmax mass is 0.000491291
(0.0491291%).
Routing is highly concentrated for this saved-structure set; several skipped
weights underflow to zero in FP64. Small errors here do not bound errors for
other compositions or more diffuse routing. The mean skipped fraction measures
omitted branch force contributions, not omitted energies or runtime speedup.

## Scientific answers

1. **Full fused equivalence:** measured maximum component difference is
2.77556e-15 eV/Angstrom. This is numerical evidence
for these ten structures, not a universal bit-exact claim.
2. **Skipping discrepancy:** mean / maximum structure RMSE are
5.98508e-07 / 5.98507e-06 eV/Angstrom.
3. **Returned-energy consistency:** C is not generally conservative with respect
to its full reported energy. Zero or tiny residuals on saturated routing samples
do not alter the implemented mathematical mismatch.
4. **Skipped energies:** retained in C's reported energy; energy equality and
differences are measured explicitly for each structure. C and B energies are
exactly equal for 9/10 structures; the largest difference is
2.22045e-16 eV, reflecting summation
grouping, not removed contributions.
5. **Consistent truncation:** D restores the local gradient relationship, but
changes both the potential energy and its forces relative to A.
6. **Skipped branches/weight:** mean skipped fraction 92.500000%,
range 87.500000%–93.750000%; mean skipped weight
4.91291e-05, range 0–0.000491291.
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

- h=0.0001 Angstrom: maximum full/truncated derivative errors 2.52951e-07 / 2.52951e-07 eV/Angstrom.
- h=1e-05 Angstrom: maximum full/truncated derivative errors 5.06939e-09 / 5.06939e-09 eV/Angstrom.
- h=1e-06 Angstrom: maximum full/truncated derivative errors 2.15007e-08 / 2.15007e-08 eV/Angstrom.

At the largest-discrepancy Al coordinate, the full-energy finite difference
continues to disagree with C's returned force by approximately 1.53e-5 eV/Angstrom
while agreeing with the full-energy autograd force to far smaller error.

## Reproducibility and preservation

See `config.json`, `input_manifest.json`, and `checkpoint_provenance.json`.
Both required checkpoint hashes were checked before loading; parameters are
frozen/eval and FP64. Branch order follows the checkpoint configuration.
Input paths are committed distorted initial structures, with identical atom
ordering across modes; graph cutoff is 5.0 Angstrom,
maximum incoming neighbors 20, charge/spin zero.
Original relaxation results and inference source hashes were verified unchanged.
No optimization, DFT, retraining, or manuscript edit was performed.
