# Frozen PaiNN architecture and parameter audit

Run from the repository root, using the existing HydraGNN environment:

```bash
python examples/multidataset_hpo_sc26/architecture_review/inspect_checkpoint.py
```

The script uses the same existing frozen HydraGNN and BranchWeightMLP files
as the relaxation and force-consistency studies. It verifies both hashes,
reconstructs from the saved configuration, strictly restores every tensor,
counts unique parameter objects, and matches the saved optimizer's parameter
order and moment shapes. No forward pass, training, optimization, or model
architecture change is performed.

Outputs:

- `inspection.json`: exact counts, checkpoint/source provenance, layer dimensions,
  optimizer coverage, and supporting source locations.
- `parameters.csv` and `module_hierarchy.json`: tensor/component breakdown.
  Recursive module counts are diagnostic and should not themselves be summed.
- `architecture_table.tex` and `parameter_breakdown.tex`: proposed SI tables.
- `report.md`: architectural interpretation and count-discrepancy evidence.
- `reviewer_response.md` and `supporting_information_description.md`: unapplied drafts.

The report distinguishes complete parameters, requires-grad eligibility under
the saved configuration, saved optimizer coverage, and inference-frozen counts.
Neither the manuscript nor Supporting Information source is available in this
workspace; the manuscript correction checklist remains provisional.
