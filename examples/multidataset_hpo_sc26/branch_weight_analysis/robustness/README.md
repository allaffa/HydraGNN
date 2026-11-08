# Second-round robustness analysis

Run from the repository root with the same Python environment used for the
first-round analysis:

```bash
python examples/multidataset_hpo_sc26/branch_weight_analysis/robustness/analyze_robustness.py
```

The script loads the existing MLP checkpoint only and never runs the HydraGNN
backbone. It validates expected checkpoint hashes and architecture before
inference. It writes all new results under this `robustness/` directory and
does not overwrite first-round outputs.

The ADIOS data files cited by training logs are not present or mounted in this
local environment. The report therefore provides an exact path audit, omits
`domain_proximity.csv`, and assigns no training-domain labels. Synthetic
probes are not evidence of training distribution coverage.

Expected outputs include:

- `report.md`
- `size_sensitivity.csv`
- `logit_diagnostics.csv`
- `real_composition_audit.json`
- `substitution_paths.csv`
- `matched_similarity.csv`
- `checkpoint_manifest.json`
- `branch_mapping_provenance.json`
- `summary_statistics.json`
- `figures/fig_size_sensitivity.{pdf,png}`
- `figures/fig_size_sensitivity_js.{pdf,png}`
- `figures/fig_logit_saturation.{pdf,png}`
- `figures/fig_substitution_routing.{pdf,png}`
- `figures/fig_similarity_controls.{pdf,png}`
