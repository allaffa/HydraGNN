# Diamond-carbon relaxation failure diagnostics

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

- C_diamond_compressed/fp32: 194 energy increases above 10⁻⁶ eV/atom; largest increase 0.0119497 eV/atom at step 173 with 10 changed periodic edges; minimum separation 1.1850 Å; volume 311.294 → 301.355 Å³.
- C_diamond_compressed/fp64: 196 energy increases above 10⁻⁶ eV/atom; largest increase 0.00818247 eV/atom at step 94 with 12 changed periodic edges; minimum separation 1.2140 Å; volume 311.294 → 301.712 Å³.
- C_diamond_expanded/fp32: 196 energy increases above 10⁻⁶ eV/atom; largest increase 0.00142932 eV/atom at step 103 with 58 changed periodic edges; minimum separation 1.3930 Å; volume 420.308 → 399.338 Å³.
- C_diamond_expanded/fp64: 204 energy increases above 10⁻⁶ eV/atom; largest increase 0.00163218 eV/atom at step 48 with 54 changed periodic edges; minimum separation 1.3930 Å; volume 420.308 → 399.269 Å³.

- C_diamond_compressed: first material divergence at step 45; first periodic graph difference at step 50.
- C_diamond_expanded: first material divergence at step 38; first periodic graph difference at step 38.

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

| Experiment | Optimizer / cell | Status | Steps | Final max force | Final generalized force |
|---|---|---|---|---|---|
| A_C_diamond_compressed_fixed_FIRE | fire / fixed | max_steps_without_convergence | 500 | 2.92594 | 2.92594 |
| C_C_diamond_compressed_full_BFGS | bfgs / full | max_steps_without_convergence | 500 | 3.58132 | 3.42233 |
| A_C_diamond_expanded_fixed_FIRE | fire / fixed | converged | 209 | 0.048197 | 0.048197 |
| C_C_diamond_expanded_full_BFGS | bfgs / full | max_steps_without_convergence | 500 | 0.134365 | 0.132049 |
| B1_full_FIRE | fire / full | max_steps_without_convergence | 500 | 0.147025 | 0.151733 |
| B2_full_FIRE | fire / full | max_steps_without_convergence | 500 | 0.136508 | 0.140909 |
| B3_full_FIRE | fire / full | max_steps_without_convergence | 500 | 0.125787 | 0.131147 |
| B4_full_FIRE | fire / full | max_steps_without_convergence | 500 | 0.124205 | 0.126924 |
| B5_full_FIRE | fire / full | max_steps_without_convergence | 500 | 2.02481 | 1.96789 |
| B6_full_FIRE | fire / full | max_steps_without_convergence | 500 | 0.124645 | 0.124901 |

FIRE and BFGS both use maxstep=0.03 Å and a 500-step cap. Full-cell runs use
ASE UnitCellFilter with cell_factor=64 and no hydrostatic/symmetry restriction.
Fixed-cell runs use the exact two original inputs and atomic force tolerance
0.05 eV/Å. B1–B6 share normalized random directions (seed 20261108); B1 uses
0.02–0.03 Å magnitudes and B2–B6 exactly 0.05 Å. B3/B4 strain ±1%,
B5/B6 ±3%, while B1/B2 are unstrained. All inputs and outputs are retained.

## Energy/force/stress derivative evidence

| Geometry | h (Å) | Max force error (eV/Å) | RMS force error | Changed-graph components | Stable-graph max error | Conditional graph max error |
|---|---|---|---|---|---|---|
| ideal | 0.001 | 247.808 | 98.8756 | 192/192 | None | 0.011617182799560283 |
| ideal | 0.0001 | 2479.53 | 990.069 | 192/192 | None | 0.0001333413597037758 |
| ideal | 1e-05 | 24796.7 | 8820.58 | 192/192 | None | 1.3581886449398617e-06 |
| C_diamond_compressed_initial | 0.001 | 117.97 | 8.51375 | 1/192 | 0.10561716088053719 | 0.005801438745074439 |
| C_diamond_compressed_initial | 0.0001 | 0.0333247 | 0.00456398 | 0/192 | 0.033324738216467864 | None |
| C_diamond_compressed_initial | 1e-05 | 3.85327e-08 | 6.4835e-09 | 0/192 | 3.8532666835067175e-08 | None |
| C_diamond_compressed_final | 0.001 | 8.68142 | 0.626535 | 1/192 | 0.016440363278903414 | 9.523264472799386e-05 |
| C_diamond_compressed_final | 0.0001 | 0.00299432 | 0.000342497 | 0/192 | 0.0029943231629092626 | None |
| C_diamond_compressed_final | 1e-05 | 3.37903e-07 | 3.63541e-08 | 0/192 | 3.379029446803905e-07 | None |
| C_diamond_expanded_initial | 0.001 | 8.34423 | 0.859219 | 3/192 | 0.006040111226104261 | 9.871683752793459e-05 |
| C_diamond_expanded_initial | 0.0001 | 1.39285e-06 | 2.68925e-07 | 0/192 | 1.39285423202129e-06 | None |
| C_diamond_expanded_initial | 1e-05 | 1.465e-08 | 2.79607e-09 | 0/192 | 1.4650014607681117e-08 | None |
| C_diamond_expanded_final | 0.001 | 11.2335 | 2.23835 | 67/192 | 0.019314236568822852 | 0.018135535639815983 |
| C_diamond_expanded_final | 0.0001 | 14.8138 | 1.5119 | 2/192 | 0.0059751474442052074 | 3.1791326140090392e-06 |
| C_diamond_expanded_final | 1e-05 | 1.35945e-06 | 2.9308e-07 | 0/192 | 1.3594460812477172e-06 | None |
| compressed_problem_step_94 | 0.001 | 112.714 | 23.2787 | 15/192 | 0.047119449492711796 | 0.01942018343640417 |
| compressed_problem_step_94 | 0.0001 | 0.0297133 | 0.0041374 | 0/192 | 0.029713261211767816 | None |
| compressed_problem_step_94 | 1e-05 | 0.00318062 | 0.000229541 | 0/192 | 0.0031806200562994325 | None |

| Geometry | Strain h | Max stress error (eV/Å³) | RMS stress error | Changed-graph components |
|---|---|---|---|---|
| ideal | 0.001 | 0.442156 | 0.185903 | 6/6 |
| ideal | 0.0001 | 10.9291 | 6.53839 | 6/6 |
| ideal | 1e-05 | 109.056 | 48.7682 | 6/6 |
| C_diamond_compressed_initial | 0.001 | 0.228818 | 0.150742 | 3/6 |
| C_diamond_compressed_initial | 0.0001 | 5.35777e-05 | 3.20171e-05 | 0/6 |
| C_diamond_compressed_initial | 1e-05 | 2.07868e-09 | 1.38742e-09 | 0/6 |
| C_diamond_compressed_final | 0.001 | 0.0397765 | 0.0193904 | 3/6 |
| C_diamond_compressed_final | 0.0001 | 3.40846e-05 | 1.86656e-05 | 0/6 |
| C_diamond_compressed_final | 1e-05 | 1.0437e-09 | 6.31415e-10 | 0/6 |
| C_diamond_expanded_initial | 0.001 | 0.0201223 | 0.0129559 | 5/6 |
| C_diamond_expanded_initial | 0.0001 | 1.99652e-06 | 9.32637e-07 | 0/6 |
| C_diamond_expanded_initial | 1e-05 | 6.28162e-10 | 3.48902e-10 | 0/6 |
| C_diamond_expanded_final | 0.001 | 0.0405759 | 0.0254154 | 6/6 |
| C_diamond_expanded_final | 0.0001 | 0.0626033 | 0.0325159 | 5/6 |
| C_diamond_expanded_final | 1e-05 | 1.55334e-05 | 6.66946e-06 | 0/6 |
| compressed_problem_step_94 | 0.001 | 0.69488 | 0.380717 | 6/6 |
| compressed_problem_step_94 | 0.0001 | 1.82166 | 0.743688 | 1/6 |
| compressed_problem_step_94 | 1e-05 | 8.02507e-05 | 3.3057e-05 | 0/6 |

All 64 tested atoms × three Cartesian
components are checked at each of three h values in each listed geometry.
Largest error among graph-stable atomic tests: 0.105617 eV/Å.
Changed-graph atomic tests: 665.
Largest conditional fixed-graph atomic error: 0.0194202 eV/Å.
Largest graph-stable stress error: 0.00152772 eV/Å³.
Largest conditional fixed-graph stress error: 0.00178335 eV/Å³.
These maxima include the larger finite-difference steps and are not final
small-step consistency tolerances.

The expanded endpoint and problematic compressed intermediate had residual graph-stable finite-difference errors at 10⁻⁵. Focused refinement of their worst atomic component and all six stress components down to 3×10⁻⁷ reduced maximum errors to 3.67116e-08 eV/Å and 1.2284e-10 eV/Å³ with unchanged graphs. This resolves those residuals as step-size sensitivity rather than a demonstrated gradient bug. This refinement checks the worst components, not another complete 192-component sweep.

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

- C_diamond_compressed, original step 94: resolved graph-switch bracket width 2.98e-09 of one optimizer step; remaining energy difference 0.00350871 eV/atom. The complete bracket-contraction series is saved.
- C_diamond_expanded, original step 48: resolved graph-switch bracket width 2.98e-09 of one optimizer step; remaining energy difference 9.33444e-05 eV/atom. The complete bracket-contraction series is saved.

These brackets shrink the geometric interval around an observed neighbor switch.
A finite energy difference that persists as the interval contracts is direct
evidence of nonsmooth rebuilt-graph energy, unlike an edge-count correlation.

## Diamond lattice scan

Ideal 64-atom diamond was scanned over a/a₀=0.90–1.10 at 0.01 intervals,
a₀=3.567 Å. The lowest sampled energy occurs at scale
1.03 (3.67401 Å),
E=-0.426096 eV/atom,
pressure=0.0166575 eV/Å³.
Maximum atomic force at this sampled minimum is
0.217121 eV/Å; neighboring sampled normal
stress changes from compression to tension around this minimum: True.
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
2. **Fixed cell:** 1/2 cases converged.
Holding the cell fixed does not eliminate all failures.
3. **Small perturbations:** 0/6 cases converged;
see each amplitude/strain row rather than inferring universal basin stability.
4. **Alternative optimizer:** BFGS converged in 0/2
original cases. Its success/failure does not independently establish physical accuracy.
5. **Mutual consistency:** Stable-graph and conditional derivative discrepancies
converge toward autograd with smaller steps, including the focused refinement
when available. No general energy-gradient implementation defect was demonstrated.
Across discrete neighbor switches the rebuilt energy is not smooth, so its
ordinary central finite difference need not agree with the local autograd derivative.
6. **Stable diamond minimum:** Not established. The lowest 21-point sample at
1.03a₀ has max force 0.2171 eV/Å and nonzero
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
