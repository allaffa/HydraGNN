# Proposed response to Reviewer 1, Comment 5

We thank the reviewer for requesting an architecture and parameter-count audit.
Strict reconstruction and loading of the frozen checkpoint verified a complete
sixteen-branch HydraGNN model containing **12,205,540 parameters**. The separate
BranchWeightMLP contains **24,528 parameters**. The shared PaiNN blocks
contain 1,486,880 parameters, shared graph
conditioning contains 114,580, and all branch-local prediction
networks contain 10,604,080 (662,755 each).

The previously reported approximately 12.1M value matches the exact
12,090,960 printed in the training logs before the lazy
graph-conditioning projector was instantiated. The checkpoint contains that
additional 114,580-parameter layer, bringing its complete size to
approximately 12.2M. The saved optimizer also excludes this layer: although
all 12,205,540 parameters are requires-grad eligible under the reconstructed
training configuration, only 12,090,960 have saved optimizer coverage.
We report these definitions separately rather than counting an inference-frozen
model as having zero originally trainable parameters.

The verified model uses two PaiNN blocks, hidden width 337, four fixed radial
functions with a cosine cutoff at 5.0 Angstrom, at most twenty incoming
neighbors, sum pooling, and sixteen distinct scalar-energy branches.
Each branch follows 337 -> 50 -> 50 -> 776 -> 776 -> 1.
The unembedded atomic-number input has one channel in the first message/update
block, followed by expansion to 337; the second block uses 337 channels.
The generic config value num_filters=126 is consumed by SchNet, not PaiNN.
PaiNN's actual radial-filter maps are 4 -> 3 and 4 -> 1011, so 126 should
not be listed as an independent PaiNN architectural hyperparameter.

The accompanying proposed SI tables document these verified definitions.
The inspection does not alter the distinction between validation-loss HPO
selection and the resource-aware PaiNN choice for downstream training under
the fixed production budget. Manuscript/SI corrections remain proposed until
the document sources and inspection results have been reviewed.
