# Diamond-carbon diagnostics

This workspace investigates the carbon failures from
`off_equilibrium_full_cell_20261108_0328` at commit
`8a18696fd3ddad54103c586a781d6d67edd908bd`. It does not modify that benchmark,
the frozen checkpoints, the production calculator, or the manuscript.

From the repository root, using the configured HydraGNN environment:

```bash
python examples/multidataset_hpo_sc26/precision_relaxation/results/c_diamond_diagnostics/run_diagnostics.py --phase originals
python examples/multidataset_hpo_sc26/precision_relaxation/results/c_diamond_diagnostics/run_diagnostics.py --phase experiments
python examples/multidataset_hpo_sc26/precision_relaxation/results/c_diamond_diagnostics/run_diagnostics.py --phase derivatives
python examples/multidataset_hpo_sc26/precision_relaxation/results/c_diamond_diagnostics/run_diagnostics.py --phase refine
python examples/multidataset_hpo_sc26/precision_relaxation/results/c_diamond_diagnostics/run_diagnostics.py --phase report
```

Alternatively, `--phase all` runs the four phases sequentially in a fresh
workspace. Existing original-analysis, experiment, and completed derivative
outputs are not overwritten. Do not run independent phases concurrently:
the progress and provenance files are shared. Progress is recorded in
`analysis/progress.json`. The default checks all 64 atoms and all three
Cartesian force components at three finite-difference step sizes.
`--fd-atoms` explicitly limits that scope if necessary.

The 10 FP64 experiments comprise two fixed-cell FIRE runs, two full-cell BFGS
runs, and six full-cell FIRE runs with smaller perturbations. Both optimizers
use a conservative 0.03 Å maximum step and a 500-step limit. Full-cell
`UnitCellFilter` uses `cell_factor=64`, without hydrostatic or symmetry
constraints, and the 0.05 eV/Å generalized-force criterion. Fixed-cell runs
instead use the atomic-force criterion. Initial B-series random directions
are shared and normalized, seed 20261108.

Graph diagnostics retain periodic source/destination/image-shift identities,
not just total edge counts. The raw 5 Å neighbor count and cap-boundary distance
gap are recorded separately from the retained 20 incoming neighbors. Near
degenerate shells, the cap can change topology without changing edge count.
Selected original energy-rise transitions are interpolated and their graph
switch boundaries resolved by bisection. Attribution of nonsmooth energy
requires measured energy changes, not topology changes alone.

Finite differences rebuild the original graph as usual. Additional
**conditional fixed-graph derivative controls** isolate the autograd
implementation from discrete neighbor selection. These use the original
checkpoint and exact reference edge identities but hold that graph fixed
only during diagnostic finite differences; they are never used for relaxation
or the lattice scan. No neighbor-count or composition settings are changed.
The optional `refine` phase checks remaining graph-stable residuals on the
expanded endpoint and problematic intermediate with steps down to 3e-7:
the worst atomic component and all six stress components, not another full
force sweep. Run it before `report` to include the measured refinement.

Source SHA-256 fingerprints cover every original benchmark file; the report
phase verifies they are unchanged. Input/checkpoint hashes and model
provenance are retained. CSV/trajectory comparisons use full-precision
saved frames. A final `extxyz` endpoint has rounded coordinates, which can
change near-tied capped neighbors; the saved-geometry reproduction check
reports that sensitivity rather than silently treating it as exact equality.

These are learned-potential diagnostics, not DFT or experimental validation.
Convergence and a minimum in a sampled lattice curve do not establish
physical accuracy, symmetry preservation, or smoothness.
