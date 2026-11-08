# Draft response to Reviewer 1, Comment 6

We thank the reviewer for requesting a direct energy-force consistency test.
Using the frozen checkpoint pair, we compared ten saved periodic configurations
in FP64 with identical graphs and composition weights. Full-branch fused
inference differed from an independently executed full-energy-gradient reference
by a maximum force component of 2.776e-15 eV/Angstrom.
With the explicitly selected skipping threshold 0.2, the mean and maximum
per-structure force RMSE were 5.985e-07 and
5.985e-06 eV/Angstrom.

The experimental skipping implementation retains all branch contributions in
the reported energy but omits skipped contributions from the force gradient.
It therefore does not generally preserve F = -grad E. Direct differentiation
of the complete reported energy verifies this distinction. A diagnostic control
that omits the same branches from both energy and forces restores local
consistency, without renormalizing retained weights, but changes the potential.

The published 1.1-billion-structure screening used all 16 branches without
skipping, as specified in the study provenance. We distinguish that production
formulation from the experimental force-skipping optimization and recommend
full-branch inference for conservative-force applications. These numerical
tests do not establish universal physical accuracy or smoothness across periodic
neighbor switches. The present threshold is user-selected and is not verified
as the threshold behind the historical approximately 81% reduction.
