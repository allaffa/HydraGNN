# Proposed Supporting Information architecture description

The frozen HydraGNN checkpoint comprises a shared PaiNN encoder and sixteen
dataset-specific graph-energy prediction networks, with 12,205,540 parameters
in total. The input is the scalar atomic number without a separate learned
embedding table. The first of two message/update blocks operates on one input
channel and includes scalar adaptation 1 -> 337 -> 337 and vector-channel
adaptation 1 -> 337; the second block operates on 337 scalar and vector channels.
Vectors are stored with a Cartesian axis, N x 3 x channels, and channel-linear
maps reuse their weights along that axis. Four fixed radial functions
sin(n pi r / 5)/r, n=1,...,4, feed block-specific affine filter maps producing
three coefficients per input channel. The filters are multiplied by the
cosine cutoff, zero for distances at or above 5.0 Angstrom. Periodic graph
construction retains at most twenty incoming neighbors per atom.

Each block includes additive message/update residuals, SiLU message/update
MLPs, Tanh scalar adaptation, and subsequent ReLU activation. A shared
339 -> 337 affine conditioning layer concatenates two graph attributes with
the node scalars after each block. Sum pooling feeds sixteen independent
prediction networks, each 337 -> 50 -> 50 -> 776 -> 776 -> 1, with ReLU hidden
activations and one scalar energy output. These networks contribute
10,604,080 parameters; the shared encoder including conditioning
contributes 1,601,460. No separate PaiNN interaction-filter width of 126 is
used. The separate composition-routing MLP is 118 -> 128 -> 64 -> 16 and
contains 24,528 parameters.

The logged pre-lazy-construction count and saved optimizer coverage are
12,090,960; the complete checkpoint includes the additional
114,580-parameter conditioning layer. All 12,205,540 parameters
are requires-grad eligible in the reconstructed training configuration, but
the two conditioning tensors are absent from the saved optimizer. Report this
training-coverage distinction alongside the total architectural count.
