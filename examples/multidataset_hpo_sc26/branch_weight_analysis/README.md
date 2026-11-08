# BranchWeightMLP composition-routing analysis

This directory contains a reproducible **synthetic composition-probe** analysis.
It is not the full manuscript analysis: no archived evaluation structures or
training-composition distribution were available in the local checkout. The
report therefore leaves domain classification unknown and does not claim
chemical specialization from these synthetic probes.

## Run

From the HydraGNN repository root, using an environment with PyTorch, ASE,
NumPy, pandas, SciPy, and Matplotlib installed:

```bash
python examples/multidataset_hpo_sc26/branch_weight_analysis/analyze.py
```

The script loads `mlp_branch_weights.pt` with `weights_only=True`, reconstructs
the MLP from its saved tensors, and runs only the BranchWeightMLP forward pass.
It does not load or execute the HydraGNN checkpoint. The HydraGNN checkpoint is
hashed and listed for provenance; the MLP-to-backbone pairing was confirmed by
the user.

Generated ASE structures contain 100 atoms arranged on a deterministic simple
cubic carrier. The MLP only receives the 118-element raw count histogram; the
positions are not used. These are not relaxed, DFT-evaluated, or asserted to be
stable molecules/materials. All interpolated path points are composition-space
probes and may contain fractional counts.

## Outputs

- `checkpoint_manifest.json`: checkpoint paths, hashes, architecture, dtype, and
  repository commit.
- `composition_encoding_audit.json`: verified code-level count encoding and
  explicit status of the unavailable archived-structure comparison.
- `branch_mapping.csv`: the 16 branches in the local SC26 launch order.
- `composition_manifest.csv`: generated formula, family, count/fraction
  provenance, and unknown training-domain labels.
- `branch_weights.csv`: the primary result—the complete 16-dimensional softmax
  distribution for every composition, plus secondary per-composition summaries.
- `routing_metrics.csv`: entropy, effective support, dominant branch, maximum
  weight, top-three weight, and 90/95% support sizes.
- `pairwise_similarity.csv`: same-composition arrangement, within-family, and
  between-family distances.
- `composition_paths.csv`: three 21-point fixed-total interpolation paths.
- `summary_statistics.json` and `report.md`: measured results and scientific
  limitations.
- `figures/`: PDF and 300-DPI PNG versions of the heatmaps and plots.

Interpret the weights as learned preferences among source-specific prediction
heads. The analysis does not classify branches as activated/inactive using a
binary per-composition threshold, and the weights are not measures of physical
fidelity or uncertainty.

For production-data conclusions, rerun only after the archived structures and
training composition distribution are available, and complete the three-record
stored-`chemical_composition` audit before labeling domain coverage.
