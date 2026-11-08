# Energy-force consistency reviewer benchmark

Run from the repository root using an environment with the existing HydraGNN,
PyTorch, PyG, ASE, and periodic-neighbor dependencies:

```bash
python examples/multidataset_hpo_sc26/force_consistency_review/run_benchmark.py
```

The two frozen checkpoint files must be available at the existing SC26 paths.
Their hashes are verified by the strict relaxation-model loader before execution.

- A: independent complete-model forwards; gradient of the full weighted energy.
- B: production full fused inference, encoder reuse and batched decoding.
- C: same production path, with user-authorized threshold 0.2.
- D: diagnostic truncation of both energy and forces, no weight renormalization.

The threshold is not independently verified as the setting for the historical
approximately 81% branch-force reduction. Each structure is evaluated alone,
so the production batch-mean selection rule acts on its own composition weights.
All branch energy predictions remain evaluated in B/C.

Outputs include manifests, complete force arrays, individual and aggregate
CSV/JSON results, finite-difference checks, a LaTeX table, scientific report,
and unapplied reviewer-response/manuscript-discussion drafts.
No geometry optimization, DFT, retraining, or production inference modification
is performed. The script checks hashes of all source benchmark files and the
inference/calculator implementations before and after execution.
