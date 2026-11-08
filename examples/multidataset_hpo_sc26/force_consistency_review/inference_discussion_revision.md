# Proposed inference-optimization discussion (not applied to manuscript)

Encoder reuse, batched branch decoding, and fused energy differentiation retain
the complete potential E = sum_k w_k(Z) E_k(R,Z). Since the composition-based
weights do not depend on coordinates, its forces are F = -grad_R E.
Across ten saved periodic configurations in FP64, full-branch fused inference
gave a maximum force-component discrepancy of
2.776e-15 eV/Angstrom relative to an independently
executed complete-gradient reference.

Experimental branch-force skipping is distinct: branch energies remain in the
reported total while low-weight branches are excluded from differentiation.
At the user-selected threshold 0.2, the mean per-structure force RMSE was
5.985e-07 eV/Angstrom, and the mean fraction of omitted
branch force contributions was 92.50%. This is not a reduction
in the number of branch energy predictions and is not a measured runtime
speedup. The historical approximately 81% claim is not revalidated here.
Such force-only skipping is not a conservative evaluator of the reported
potential and should not be used for relaxation or molecular dynamics.
Dropping the same contributions from both energy and forces restores local
consistency but defines a different approximate potential; this was tested only
as a diagnostic control. All local gradient statements assume fixed periodic
connectivity and selection.

The published screening of 1.1 billion structures used the complete 16-branch
formulation without branch skipping, according to the study provenance.
