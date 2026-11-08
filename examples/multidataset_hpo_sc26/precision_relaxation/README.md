# FP32 vs FP64 structural relaxation

This isolated benchmark compares structural relaxations from the frozen
HydraGNN checkpoint using either its BranchWeightMLP-weighted 16-head ensemble
or one selected decoder head. It does not retrain the model, run DFT, or modify
the production inference scripts.

## Staged execution

Use the configured HydraGNN Python environment. Stage A creates a unique run
directory, writes all five reference and perturbed initial structures, and
validates one H2O system in FP32 and FP64 (single points, finite-difference
forces, graph rebuilding, and at most ten BFGS steps):

```bash
python examples/multidataset_hpo_sc26/precision_relaxation/run_benchmark.py \
  --stage pilot
```

Stage A prints the run directory. Only after its `stage_A/stage_a_status.json`
reports `"status": "passed"` should Stage B run, using that same directory:

```bash
python examples/multidataset_hpo_sc26/precision_relaxation/run_benchmark.py \
  --stage full \
  --output-dir examples/multidataset_hpo_sc26/precision_relaxation/results/run_<timestamp>
```

Stage B refuses to start without a passing Stage A result and refuses to
overwrite existing stage outputs. Supply `--output-dir` to Stage A as well if
you want to choose the run path yourself.

## Single-head mode

To omit the MLP and optimize with one decoder head only, use the same staged
workflow with `--head-mode single` on both commands:

```bash
python examples/multidataset_hpo_sc26/precision_relaxation/run_benchmark.py \
  --stage pilot --head-mode single \
  --output-dir examples/multidataset_hpo_sc26/precision_relaxation/results/single_head_<timestamp>

python examples/multidataset_hpo_sc26/precision_relaxation/run_benchmark.py \
  --stage full --head-mode single \
  --output-dir examples/multidataset_hpo_sc26/precision_relaxation/results/single_head_<timestamp>
```

The configured selections are H₂O/CH₄ → branch-8 (OMol25), Si/Al → branch-7
(OMat24), and MgO → branch-2 (MPTrj). The selection was made once from the
highest FP64 MLP weight on each system's validated fused single-point
composition. That offline result chooses which head to retain; the single-head
benchmark itself does not load or invoke the MLP. Override a choice with a
zero-based branch ID, for example `--head-override MgO=4`. The report records
the chosen branch and dataset. Do not interpret one-head results as equivalent
to the learned 16-head weighted potential.

## Variable-cell relaxation

For fully periodic structures, pass `--relax-cell` to both stages to optimize
the lattice vectors together with atomic positions. Molecular systems remain
fixed-cell. Stage A switches its pilot from H2O to periodic Si and checks the
analytic model stress against a finite-strain energy derivative before running
a short cell-relaxation smoke test:

```bash
python examples/multidataset_hpo_sc26/precision_relaxation/run_benchmark.py \
  --stage pilot --relax-cell --hydrostatic-cell \
  --optimizer fire --optimizer-maxstep 0.03 \
  --output-dir examples/multidataset_hpo_sc26/precision_relaxation/results/cell_<timestamp>

python examples/multidataset_hpo_sc26/precision_relaxation/run_benchmark.py \
  --stage full --relax-cell --hydrostatic-cell \
  --optimizer fire --optimizer-maxstep 0.03 \
  --output-dir examples/multidataset_hpo_sc26/precision_relaxation/results/cell_<timestamp>
```

The calculator differentiates the predicted energy with respect to both atomic
coordinates and the periodic cell. ASE `UnitCellFilter` then relaxes cell and
positions. `--hydrostatic-cell` constrains the variable cell to volume-only
strain; omit it for unrestricted cell-shape relaxation. `--optimizer` selects
BFGS or FIRE, and `--optimizer-maxstep` controls its maximum step. In the
tested Si diagnostic, FIRE with 0.03 Å maximum step converged both the fixed
cell and hydrostatic-cell cases using the original 20-neighbor graph; the
hydrostatic-cell case also converged in FP32 and FP64. The optimizer's
0.05 eV/Å criterion applies to the combined generalized forces when relaxing
the cell, with cell degrees of freedom scaled by atom count. Stress tensors
(eV/Å³), cell vectors, and generalized-force maxima are recorded per step.
This changes the benchmark protocol and should be treated as a separate run
from fixed-cell results. Variable-cell relaxation is only implemented for
fully periodic structures; it does not add a cell to molecular systems or
establish physical accuracy of the learned stress.

Head-to-dataset mapping follows the ordered `--multi_model_list` recorded in
the paired SC26 training run, corroborated by the training code's assignment
of each list index to `dataset_name` and matching `branch-<index>` output
heads. The explicit mapping is preserved in `benchmark_config.json`.

## Method

Reference structures come from ASE's standard H2O and CH4 molecule geometries
and conventional diamond Si, rocksalt MgO, and FCC Al cells. A single
deterministic per-atom displacement (seed 42, magnitudes 0.03–0.05 Å) is saved
and shared by both precisions. PBC flags are fixed; cells are fixed unless
`--relax-cell` is enabled. Each energy/force/stress call reconstructs the radius
graph from the current coordinates and cell. Fused mode feeds the exact
118-element atomic-number count vector to the MLP, evaluates all 16 heads, and
differentiates the complete softmax-weighted energy. Single-head mode evaluates
only its selected branch and differentiates that energy; it does not load the
MLP.

The benchmark reads its system definitions and protocol thresholds from
`benchmark_config.json` and copies that config into each run directory. The
calculator verifies the known checkpoint hashes, branch count, parameter
dtypes, evaluation mode, and frozen parameters before running. The graph
radius and neighbor cap are read from the paired training configuration.
Energy and force units follow the existing inference output convention (eV and
eV/Å); the benchmark does not establish physical accuracy of those predictions.

## Broad periodic structure set and OVITO images

To run paired FP32/FP64 hydrostatic cell relaxations on 13 elemental, binary,
and ternary crystal prototypes, then render the perturbed starting structure
and FP64 endpoint for each with OVITO:

```bash
python -m examples.multidataset_hpo_sc26.precision_relaxation.run_periodic_structure_set \
  --output-dir examples/multidataset_hpo_sc26/precision_relaxation/results/periodic_structure_set
```

The structures are standard ASE crystal prototypes expanded to 2×2×2
supercells, with identical seeded position perturbations across precisions.
Each pair uses FIRE, 0.03 Å maximum step, hydrostatic volume-only cell strain,
and a 0.05 eV/Å generalized-force threshold. OVITO and Pillow must be
installed in the benchmark Python environment for image generation. PNGs are
saved as `figures/<structure>_initial.png` and
`figures/<structure>_final.png`. The element legend is consistent across
structures, and particle radii use one common linear scaling with atomic
number. OVITO's OSPRay rendering backend applies physically based sphere
shading with ambient and direct lighting. Per-run structures, trajectories,
logs, paired metrics, and a summary report are saved alongside the images.
Each `figures/<structure>_displacements.png` companion shows FP64 final atom
positions with initial-to-final arrows scaled for visibility (the scale is
labeled in each image), while reporting the true RMS and maximum internal
displacement. The report and JSON separate internal atomic motion from cell
expansion by matching periodic fractional coordinates in the final cell;
they also record the cell-volume change.
These are representative generated
prototypes, not structures sampled from archived training datasets, and the
relaxation metrics are predictions of the learned potential rather than
independent DFT validation.

## Off-equilibrium relaxation stress test

To test whether cell relaxation and atomic optimization produce visible,
measurable changes from deliberately distorted starting structures, run:

```bash
python -m examples.multidataset_hpo_sc26.precision_relaxation.run_off_equilibrium_stress_test \
  --output-dir examples/multidataset_hpo_sc26/precision_relaxation/results/off_equilibrium_stress_test
```

The test uses 2×2×2 Si-diamond and MgO-rocksalt cells. Si starts with a 5%
isotropic expansion and MgO with a 5% compression; both receive identical
seeded random-direction atomic displacements (0.10–0.15 Å) for their FP32
and FP64 runs. FIRE relaxes positions and hydrostatic cell volume together.
The paired-geometry PNGs use the same coordinate scale for initial and final
views. Results include initial/final energy, force, stress, cell, convergence,
and atomic displacement metrics. These deliberately distorted inputs are a
robustness stress test and may be outside the model's training distribution;
predictions are not independently validated against DFT or experiment.

For the broad study across all 13 prototypes, using both compressed and
expanded starts and **full cell-length/shape relaxation**, run:

```bash
python -m examples.multidataset_hpo_sc26.precision_relaxation.run_off_equilibrium_stress_test \
  --broad --cell-mode full \
  --output-dir examples/multidataset_hpo_sc26/precision_relaxation/results/off_equilibrium_full_cell
```

This executes 52 relaxations (13 prototypes × two starting strains × two
precisions), without symmetry constraints or applied external pressure.
The same Cartesian atomic perturbations are used for both strain directions
and precisions. Full cell vectors and angles, residual stress, displacement
metrics, precision differences, and compressed-versus-expanded endpoint
differences are recorded. OVITO before/after renders use shared framing;
paired-geometry plots keep identical coordinate scales. Energy/force/volume/
angle histories and labeled displacement plots are also saved. JSON and
the report are updated after every run, and indicate whether the study is
still running or complete. Capped endpoints are explicitly labeled.
The legacy two-material hydrostatic command above remains unchanged.

## Outputs

`results/run_*/` contains the exact reference and initial geometries, input and
checkpoint manifests, Stage A results, and Stage B outputs. For each
system/precision, Stage B preserves an ASE trajectory, per-step CSV including
coordinates and edge counts, BFGS log, final structure, runtime, convergence
status, and any failure information. It also writes initial precision
comparisons, common-FP64 and cross-precision final evaluations, two figures,
and a report with a scientific assessment and draft reviewer response.

Failed runs and runs reaching the 200-step cap are reported as unsuccessful;
the convergence threshold is fixed at 0.05 eV/Å.
