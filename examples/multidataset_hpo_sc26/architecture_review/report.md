# Reviewer 1, Comment 5: frozen PaiNN checkpoint audit

This is a self-contained architectural inspection report, suitable for copying
into another chat. No manuscript or Supporting Information document has been
edited. No model training, forward inference experiment, or geometry
optimization was performed for this audit.

## Checkpoint identity and scope

- Repository: allaffa/HydraGNN, local branch
  `paper-reviewer-response-2026-11-08`.
- Backbone file: `multidataset_hpo-BEST6-fp64_epoch_97.pk`.
- Backbone SHA-256: `8532b03296a8ab4cfc3b37beb6d57b2091a50807b155fac93cb1882253d32867`.
- Separate routing file: `mlp_branch_weights.pt`.
- Routing SHA-256: `c984b338ee947534bbdedf1fc4dd2aea92bb846e3302638d6719b1e3f5ca7ca8`.
- Both hashes were measured and matched before checkpoint use.
- These are the same frozen checkpoints used in the preceding relaxation
  and energy-force consistency studies.
- The backbone checkpoint contains `model_state_dict` and
  `optimizer_state_dict`; it does not embed a full architecture configuration
  or original requires-grad flags. The neighboring saved `config.json` is
  used to reconstruct the model, and its compatibility is established by
  exact checkpoint-key, shape, and tensor-value agreement.

## Exact counts and definitions

| Quantity | Unique parameter elements |
|---|---:|
| Complete HydraGNN, all 16 dataset branches | 12,205,540 |
| Requires-grad eligible in reconstructed training configuration | 12,205,540 |
| Saved optimizer-covered parameters, all with optimizer state | 12,090,960 |
| After explicit inference freezing | 0 |
| Shared PaiNN blocks (including internal scalar/vector adaptations) | 1,486,880 |
| Shared graph-conditioning projector | 114,580 |
| Entire shared encoder, including conditioning | 1,601,460 |
| All 16 branch-local prediction networks | 10,604,080 |
| Separate learned atomic input embedding | 0 |
| Other parameters | 0 |
| Separate BranchWeightMLP | 24,528 |

Each dataset branch contains 19,450
parameters in its own `graph_shared` stack and
643,305 in its own output MLP.
Despite the `graph_shared` name, these layers are **not shared across datasets**:
each branch has distinct parameter objects. Their dimensions are
337 -> 50 -> 50 -> 776 -> 776 -> 1, with four hidden linear layers and
one scalar output layer. The entire encoder is shared across branches.
The separate BranchWeightMLP has dimensions 118 -> 128 -> 64 -> 16 with
ReLU hidden activations; the subsequent softmax has no parameters.
The backbone count excludes the separate routing MLP, optimizer moment tensors,
gradients, input tensors, and activation storage. Counting FP64 instead of
FP32 changes memory per element, not the number of parameters.

The following subtotals reconcile exactly:

- Shared blocks: 115,290 +
  1,371,590 =
  1,486,880.
- Shared encoder including conditioning:
  1,486,880 + 114,580 = 1,601,460.
- Dataset-specific prediction networks:
  16 x 662,755 = 10,604,080.
- Complete HydraGNN:
  1,601,460 + 10,604,080 = **12,205,540**.
- HydraGNN plus the separate routing MLP:
  12,205,540 + 24,528 = **12,230,068**, if a combined stack
  count is desired; this is not the HydraGNN-alone count.

## Block-level trainable-component breakdown

| Block | Submodule | Parameters |
|---|---|---:|
| 1 | module_0: PainnMessage | 23 |
| 1 | module_1: PainnUpdate | 13 |
| 1 | module_2: Sequential | 114,580 |
| 1 | module_3: Linear | 674 |
| 2 | module_0: PainnMessage | 460,679 |
| 2 | module_1: PainnUpdate | 683,099 |
| 2 | module_2: Sequential | 227,812 |

All linear layers in these blocks include biases. Parameter-free SiLU, Tanh,
ReLU, Identity operations, geometric norms/inner products, aggregation,
radial functions and cutoff functions contribute no parameter elements.

## Representative checkpoint tensor shapes

These measured shapes expose the first-block channel width, filter sizes,
head dimensions, and lazy layer underlying the count discrepancy.

| Checkpoint parameter (normalized name) | Shape | Elements |
|---|---|---:|
| model.graph_convs.0.module_0.filter_layer.weight | 3 x 4 | 12 |
| model.graph_convs.0.module_2.0.weight | 337 x 1 | 337 |
| model.graph_convs.0.module_2.2.weight | 337 x 337 | 113,569 |
| model.graph_convs.0.module_3.weight | 337 x 1 | 337 |
| model.graph_convs.1.module_0.filter_layer.weight | 1011 x 4 | 4,044 |
| model.graph_shared.branch-0.0.weight | 50 x 337 | 16,850 |
| model.graph_shared.branch-0.2.weight | 50 x 50 | 2,500 |
| model.heads_NN.0.branch-0.0.weight | 776 x 50 | 38,800 |
| model.heads_NN.0.branch-0.2.weight | 776 x 776 | 602,176 |
| model.heads_NN.0.branch-0.4.weight | 1 x 776 | 776 |
| model.graph_concat_projector.weight | 337 x 339 | 114,243 |
| model.graph_concat_projector.bias | 337 | 337 |

| Zero-based dataset branch | Parameters including all branch-local layers |
|---|---:|
| branch-0 | 662,755 |
| branch-1 | 662,755 |
| branch-2 | 662,755 |
| branch-3 | 662,755 |
| branch-4 | 662,755 |
| branch-5 | 662,755 |
| branch-6 | 662,755 |
| branch-7 | 662,755 |
| branch-8 | 662,755 |
| branch-9 | 662,755 |
| branch-10 | 662,755 |
| branch-11 | 662,755 |
| branch-12 | 662,755 |
| branch-13 | 662,755 |
| branch-14 | 662,755 |
| branch-15 | 662,755 |

The saved generic branch names do not themselves establish dataset-label
semantics. The training command records the dataset ordering separately.

## Why the historical 12.1M count differs

The local training logs explicitly print **12,090,960**, which rounds
to 12.1M. The same number is obtained immediately after construction, before
checkpoint restoration materializes the lazy projector, and from the saved
optimizer's moment tensors. The frozen checkpoint itself contains **12,205,540**
(12.205540M, approximately **12.2M** at one decimal place).
The difference is exactly **114,580**, the projector weight of shape
337 x 339 plus its 337 biases. This is not a one-head versus sixteen-head
discrepancy: both counts include all sixteen branches.
A one-branch subset with the same complete shared encoder would have
2,264,215 parameters, while the complete shared encoder
alone has 1,601,460.

The saved training configuration sets `freeze_conv_layers=false`.
All 12,205,540 parameter elements have requires-grad enabled in the reconstructed
training model; checkpoint state dictionaries do not encode original
requires-grad flags. However, the saved optimizer contains only
12,090,960 elements in 198 parameter tensors, versus 200 tensors in the
model. Matching the pre-lazy-construction parameter order against optimizer
IDs and both Adam moment shapes identifies the two omitted tensors as
`model.graph_concat_projector.weight` and `.bias`. All optimizer-covered
tensors have positive step counters.

This distinction is important: **requires-grad eligible is not the same as
actually optimizer-updated**. Under the inspected standard AdamW training
path the projector is not updated by that optimizer. The checkpoint does not
establish whether an unrecorded/manual update ever occurred. No such update
mechanism was found in the inspected path; no training bug is corrected here.
The training script creates the optimizer and prints the model before the
training forward passes that lazily create this layer. This explains the
logged count and the checkpoint/optimizer mismatch.

Training-log evidence:

- [examples/multidataset_hpo_sc26/multidataset_hpo-BEST6-fp64/run-4200350.log](../../../examples/multidataset_hpo_sc26/multidataset_hpo-BEST6-fp64/run-4200350.log) (line 4514): 12,090,960.
- [examples/multidataset_hpo_sc26/multidataset_hpo-BEST6-fp64/run-4251572.log](../../../examples/multidataset_hpo_sc26/multidataset_hpo-BEST6-fp64/run-4251572.log) (line 33186): 12,090,960.
- [examples/multidataset_hpo_sc26/multidataset_hpo-BEST6-fp64/run-4251574.log](../../../examples/multidataset_hpo_sc26/multidataset_hpo-BEST6-fp64/run-4251574.log) (line 33186): 12,090,960.
- [examples/multidataset_hpo_sc26/multidataset_hpo-BEST6-fp64/run-4284444.log](../../../examples/multidataset_hpo_sc26/multidataset_hpo-BEST6-fp64/run-4284444.log) (line 33186): 12,090,960.

## Verified architecture and hyperparameter meanings

- Model: `EnhancedModelWrapper` containing `PAINNStack`; the wrapper adds
  loss handling but no trainable parameter tensors.
- Two blocks, configured hidden width 337, four radial functions, cutoff
  5.0 Angstrom, maximum incoming neighbors 20, and sixteen dataset branches
  are confirmed from the saved config, instantiated modules, and state shapes.
  The neighbor limit is a preprocessing setting, not a PaiNN parameter tensor.
- **Hidden-width qualification:** global attention and separate input encoding
  are disabled. Atomic number enters as one scalar channel, with no learned
  lookup embedding. The first `PainnMessage` and `PainnUpdate` therefore use
  **one** channel, then scalar adaptation 1 -> 337 -> 337 and vector
  adaptation 1 -> 337 produce the hidden representation. The second block
  uses **337** scalar/vector channels. Thus the supplied statement that every
  message block has 337 input channels is not accurate for this implementation.
- Vector storage starts at N x 3 x 1 and becomes N x 3 x 337 between blocks.
  The Cartesian axis stores vector components; linear channel maps are applied
  with the same weights along that axis, not three independently parameterized
  Cartesian networks. This is an implementation/storage statement, not a new
  numerical test or blanket guarantee of rotational equivariance.
- `sinc_expansion` uses fixed functions sin(n pi r / cutoff) / r for n=1..4.
  Each block's affine `filter_layer` maps these four features to three
  coefficients per **input** channel: 3 outputs in block 1, 1011 in block 2.
  The resulting coefficients multiply a cosine cutoff
  0.5[cos(pi r / cutoff)+1] below the cutoff, zero above it.
  The saved `radial_type=bessel` and `envelope_exponent=5` generic fields do
  not switch the hard-coded PaiNN radial/cutoff functions.
- Message blocks use scalar MLPs with SiLU, coefficient splitting into two
  vector gates and a scalar message, sum aggregation, and additive scalar/
  vector residuals. The directional term uses the supplied normalized edge
  vector divided again by edge distance, as written in this implementation;
  the audit does not silently replace it with a textbook variant.
- Updates apply channel-linear U/V maps to vector features, their Cartesian
  norms and inner products, a SiLU scalar MLP, and additive residuals.
  The last block updates only scalar output, leaving the vector output unused
  by the energy heads. Extra scalar adaptations use Tanh; a ReLU follows each
  block after conditioning. The first block also adapts vector channels.
- A shared lazy 339 -> 337 affine map concatenates two graph attributes to
  scalar node features after each block. It is the **same projector object**
  reused after both blocks. Graph pooling is a sum.
- No BatchNorm, LayerNorm, global-attention modules, trainable radial bases,
  separate learned atom embeddings, or active dropout layers are instantiated.
  Vector norms in updates are geometric operations, not learned normalization.

## The erroneous interaction-filter value 126

**126 does appear** as `num_filters` in the saved generic architecture config.
However, `create_model` passes `num_filters` to the SchNet `SCFStack` branch
and **does not pass it to `PAINNStack`**. Neither `PainnMessage` nor `PainnUpdate`
accepts a separate filter-width argument. Actual PaiNN filter sizes are
determined by the four radial features and three gates per block input channel,
as confirmed by checkpoint weights of shapes 3 x 4 and 1011 x 4.
Consequently 126 does not describe an independent dimension of this checkpoint's
PaiNN architecture. This is an audit finding only; no document has been changed.

## Strict loading, provenance and quality control

Both checkpoint SHA-256 values match the supplied frozen pair. Reconstruction
uses the original saved config and the repository `create_model_config`.
The reused strict restoration helper removes repeated leading `module.`
prefixes, creates the checkpoint-declared lazy concat projector from its shape,
checks exact state-key sets, and loads with `strict=True`.
Every loaded tensor is additionally compared **exactly** with its normalized
checkpoint tensor. All 200 state entries are parameters in this model; no
buffers inflate the state-element count. All sixteen branch-local networks
are present. Unique parameter IDs form disjoint component sets, and their sum
equals the full checkpoint count. The inspection performs no forward pass,
training, geometry optimization, or accuracy experiment.
The legacy inference loader freezes parameters and uses non-strict loading;
this audit instead constructs the training model directly and applies the
strict restoration helper before counting. It does not use the frozen loader
to infer original trainability.

## Supporting implementation sources

- `PAINNStack`: [hydragnn/models/PAINNStack.py](../../../hydragnn/models/PAINNStack.py) (line 27).
- `PainnMessage`: [hydragnn/models/PAINNStack.py](../../../hydragnn/models/PAINNStack.py) (line 194).
- `PainnUpdate`: [hydragnn/models/PAINNStack.py](../../../hydragnn/models/PAINNStack.py) (line 275).
- `sinc_expansion`: [hydragnn/models/PAINNStack.py](../../../hydragnn/models/PAINNStack.py) (line 331).
- `cosine_cutoff`: [hydragnn/models/PAINNStack.py](../../../hydragnn/models/PAINNStack.py) (line 341).
- `create_model_config`: [hydragnn/models/create.py](../../../hydragnn/models/create.py) (line 41).
- `head_construction`: [hydragnn/models/Base.py](../../../hydragnn/models/Base.py) (line 590).
- `graph_conditioning`: [hydragnn/models/Base.py](../../../hydragnn/models/Base.py) (line 299).
- `lazy_projector_creation`: [hydragnn/models/Base.py](../../../hydragnn/models/Base.py) (line 261).
- `strict_checkpoint_restoration`: [examples/multidataset_hpo_sc26/precision_relaxation/hydragnn_ase_calculator.py](../../../examples/multidataset_hpo_sc26/precision_relaxation/hydragnn_ase_calculator.py) (line 47).
- `BranchWeightMLP`: [examples/multidataset_hpo_sc26/inference_fused.py](../../../examples/multidataset_hpo_sc26/inference_fused.py) (line 83).
- `PaiNN_model_factory`: [hydragnn/models/create.py](../../../hydragnn/models/create.py) (line 483).
- `SchNet_num_filters`: [hydragnn/models/create.py](../../../hydragnn/models/create.py) (line 375).
- `input_embedding_and_vector_storage`: [hydragnn/models/PAINNStack.py](../../../hydragnn/models/PAINNStack.py) (line 149).
- `convolution_freezing`: [hydragnn/models/Base.py](../../../hydragnn/models/Base.py) (line 495).
- `training_optimizer_before_forward`: [examples/multidataset_hpo_sc26/gfm_mlip_all_mpnn.py](../../../examples/multidataset_hpo_sc26/gfm_mlip_all_mpnn.py) (line 899).
- `legacy_parameter_log`: [hydragnn/utils/model/model.py](../../../hydragnn/utils/model/model.py) (line 451).
- `periodic_neighbor_cap`: [hydragnn/preprocess/graph_samples_checks_and_updates.py](../../../hydragnn/preprocess/graph_samples_checks_and_updates.py) (line 266).

See `inspection.json`, `parameters.csv`, and `module_hierarchy.json` for
tensor-level shapes, counts, optimizer membership, source hashes and provenance.
The configuration is copied to `saved_configuration.json` for reproducibility.

## Manuscript comparison: scope limitation, not document edits

No manuscript or Supporting Information source was located in this workspace.
The user confirmed the same checkpoint pair but did not supply manuscript/SI
paths. Thus exact wording, all occurrences, and page/line references are
**not verified**. The supplied task identifies Sections 3.4 and 5.2, Table 4,
and the SI as relevant locations, but their actual text was not inspected.
The verified findings available for the user's independent manuscript review
are the logged-versus-complete count distinction, actual optimizer coverage,
non-use of num_filters=126 by PaiNN, first-block one-channel adaptation,
and branch-local versus encoder-shared parameters.

This audit neither modifies nor reassesses architecture selection based on
validation loss during HPO or the separate resource-aware PaiNN choice under
the fixed downstream production-training budget. It supplies no new accuracy
comparison, timing experiment, or evidence for a different architecture choice.

## Bottom line

The complete saved all-sixteen-branch HydraGNN has **12,205,540 parameters**;
the separate routing MLP has **24,528**. The historical approximately
12.1M value matches **12,090,960**, the logged pre-lazy count and saved
optimizer coverage, not the complete checkpoint. The difference is exactly the
**114,580**-parameter shared graph-conditioning projector.
All complete-model parameters are requires-grad eligible in the reconstructed
configuration, but that projector is absent from the saved optimizer.
The first PaiNN message/update block uses one input channel before expanding
to 337; the second uses 337. The actual four-feature filter maps are 4 -> 3
and 4 -> 1011, with no independent PaiNN filter width 126.
These are checkpoint/source inspection results, not manuscript changes.
