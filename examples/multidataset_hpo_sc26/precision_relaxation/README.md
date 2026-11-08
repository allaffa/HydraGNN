# FP32 vs FP64 structural relaxation

This isolated benchmark compares structural relaxations from the frozen
HydraGNN and BranchWeightMLP checkpoints. It does not retrain the model, run
DFT, or modify the production inference scripts.

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

## Method

Reference structures come from ASE's standard H2O and CH4 molecule geometries
and conventional diamond Si, rocksalt MgO, and FCC Al cells. A single
deterministic per-atom displacement (seed 42, magnitudes 0.03–0.05 Å) is saved
and shared by both precisions. Periodic cells/PBC are kept fixed. Each energy
and force call reconstructs the radius graph from current coordinates, feeds
the exact 118-element atomic-number count vector to the MLP, evaluates all 16
heads, and differentiates the complete softmax-weighted energy. The MLP weights
are fixed during each relaxation because composition does not change.

The benchmark reads its system definitions and protocol thresholds from
`benchmark_config.json` and copies that config into each run directory. The
calculator verifies the known checkpoint hashes, branch count, parameter
dtypes, evaluation mode, and frozen parameters before running. The graph
radius and neighbor cap are read from the paired training configuration.
Energy and force units follow the existing inference output convention (eV and
eV/Å); the benchmark does not establish physical accuracy of those predictions.

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
