"""Inspect the frozen PaiNN checkpoint without forwarding or updating the model."""

from __future__ import annotations

import csv
import inspect
import json
import sys
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
SC26 = HERE.parent
ROOT = SC26.parents[1]
sys.path.insert(0, str(SC26 / "precision_relaxation"))
sys.path.insert(0, str(SC26))
sys.path.insert(0, str(ROOT))

from hydragnn_ase_calculator import (
    BACKBONE_SHA256,
    MLP_SHA256,
    _restore_verified_state,
    sha256_file,
)
from hydragnn.models import create_model_config
from hydragnn.models.PAINNStack import PainnMessage, PainnUpdate, cosine_cutoff, sinc_expansion
from inference_fused import _reconstruct_mlp_from_state_dict


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def unique_parameters(module: nn.Module) -> dict[int, nn.Parameter]:
    return {id(parameter): parameter for parameter in module.parameters()}


def count(parameters: dict[int, nn.Parameter]) -> int:
    return sum(parameter.numel() for parameter in parameters.values())


def source_reference(symbol: object) -> dict[str, object]:
    filename = inspect.getsourcefile(symbol)
    if filename is None:
        raise ValueError(f"No source file for {symbol}")
    return {
        "path": str(Path(filename).resolve().relative_to(ROOT)),
        "line": inspect.getsourcelines(symbol)[1],
    }


def linear_layers(module: nn.Module) -> list[dict[str, object]]:
    return [
        {
            "path": name,
            "in_features": child.in_features,
            "out_features": child.out_features,
            "bias": child.bias is not None,
        }
        for name, child in module.named_modules()
        if isinstance(child, nn.Linear)
    ]


def normalize_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = {}
    for name, tensor in state.items():
        while name.startswith("module."):
            name = name[len("module."):]
        if name in normalized:
            raise ValueError(f"Duplicate normalized checkpoint name: {name}")
        normalized[name] = tensor
    return normalized


def write_documents(result: dict) -> None:
    total = result["total_parameters"]
    optimizer_count = result["optimizer_parameter_elements"]
    projector_count = result["components"]["Graph-attribute concat projector"]
    head_counts = result["head_parameter_counts"]
    shared = result["shared_parameters_including_conditioning"]
    mlp_count = result["branch_weight_mlp_parameters"]
    architecture_rows = [
        ("Backbone class", "PAINNStack, wrapped by EnhancedModelWrapper"),
        ("Message-passing blocks", str(result["number_of_blocks"])),
        ("Hidden dimension", str(result["hidden_dim"])),
        ("Input scalar channels", str(result["input_dim"])),
        ("Block 1 / block 2 message channels", "1 / 337"),
        ("Vector tensor dimensions", "N x 3 x 1 initially; N x 3 x 337 between blocks"),
        ("Radial expansion", "4 fixed sin(n pi r / cutoff) / r functions"),
        ("Filter maps, blocks 1 / 2", "4 to 3 / 4 to 1011 (three gates per channel)"),
        ("Cutoff", "5.0 Angstrom, cosine cutoff"),
        ("Maximum incoming neighbors", str(result["max_neighbours"])),
        ("Input embedding", "No separate learned atomic embedding table"),
        ("Scalar adaptation within block 1", "1 to 337 to 337; Tanh between linear layers"),
        ("Pooling", result["pooling"]),
        ("Graph conditioning", "Two attributes; shared 339 to 337 concat projector after each block"),
        ("Dataset-specific branches", str(result["number_of_branches"])),
        ("Per-branch dense dimensions", "337 to 50 to 50 to 776 to 776 to 1"),
        ("Per-branch depth", "4 hidden linear layers plus 1 scalar output layer"),
        ("Head activation", "ReLU between hidden linear layers"),
        ("Total HydraGNN parameters", f"{total:,}"),
        ("requires_grad eligible before inference freezing", f"{result['training_configuration_requires_grad_parameters']:,}"),
        ("Parameters in saved training optimizer", f"{optimizer_count:,}"),
        ("BranchWeightMLP (separate)", f"118 to 128 to 64 to 16; {mlp_count:,} parameters"),
    ]
    def latex_escape(value: str) -> str:
        return value.replace("_", r"\_").replace("&", r"\&")

    def latex_table(rows: list[tuple[str, str]], caption: str, label: str) -> str:
        lines = [
            r"\begin{table*}[t]", r"\centering", r"\small",
            r"\caption{" + caption + "}", r"\label{" + label + "}",
            r"\begin{tabular}{p{0.38\textwidth}p{0.55\textwidth}}", r"\hline",
        ]
        lines.extend(f"{latex_escape(key)} & {latex_escape(value)}" + r" \\" for key, value in rows)
        lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}"])
        return "\n".join(lines) + "\n"

    (HERE / "architecture_table.tex").write_text(latex_table(
        architecture_rows,
        "Verified architecture of the frozen PaiNN checkpoint. The first block "
        "operates on the unembedded one-channel input; hidden width 337 is not "
        "the input width of every message block.",
        "tab:painn-verified-architecture",
    ))
    component_rows = [
        ("Atomic input embedding (separate module)", "0; not instantiated"),
        ("PaiNN block 1, including input/output adaptations", f"{result['block_parameter_counts'][0]:,}; shared"),
        ("PaiNN block 2", f"{result['block_parameter_counts'][1]:,}; shared"),
        ("Graph-attribute concat projector", f"{projector_count:,}; shared; absent from saved optimizer"),
        ("Dataset-specific branch 0", f"{head_counts['branch-0']:,}; not shared"),
        ("Dataset-specific branches 1--15", f"{sum(v for k, v in head_counts.items() if k != 'branch-0'):,}; not shared"),
        ("Other components", f"{result['unclassified_parameters']:,}"),
        ("Total HydraGNN (all 16 branches)", f"{total:,}"),
        ("Original optimizer-covered parameters", f"{optimizer_count:,}"),
        ("BranchWeightMLP (separate model)", f"{mlp_count:,}"),
    ]
    (HERE / "parameter_breakdown.tex").write_text(latex_table(
        component_rows,
        "Unique-parameter breakdown. Branch indices are zero-based. All branch "
        "counts include both branch-local graph-shared layers and the output "
        "MLP. The conditioning projector has requires-grad enabled in the "
        "reconstructed training configuration but is missing from the saved "
        "optimizer, unlike the other parameters.",
        "tab:painn-parameter-breakdown",
    ))
    per_head = "\n".join(
        f"| {name} | {value:,} |"
        for name, value in head_counts.items()
    )
    source_rows = "\n".join(
        f"- `{name}`: [{ref['path']}](../../../{ref['path']}) (line {ref['line']})."
        for name, ref in result["sources"].items()
    )
    log_rows = "\n".join(
        f"- [{entry['path']}](../../../{entry['path']}) (line {entry['line']}): "
        f"{entry['printed_total']:,}."
        for entry in result["training_log_totals"]
    )
    block_rows = "\n".join(
        f"| {row['block'] + 1} | {row['module']} | {row['parameters']:,} |"
        for row in result["block_submodule_counts"]
    )
    layer_rows = "\n".join(
        f"| {name} | {shape} | {elements:,} |"
        for name, shape, elements in result["representative_parameter_shapes"]
    )
    report = f"""# Reviewer 1, Comment 5: frozen PaiNN checkpoint audit

This is a self-contained architectural inspection report, suitable for copying
into another chat. No manuscript or Supporting Information document has been
edited. No model training, forward inference experiment, or geometry
optimization was performed for this audit.

## Checkpoint identity and scope

- Repository: allaffa/HydraGNN, local branch
  `paper-reviewer-response-2026-11-08`.
- Backbone file: `multidataset_hpo-BEST6-fp64_epoch_97.pk`.
- Backbone SHA-256: `{BACKBONE_SHA256}`.
- Separate routing file: `mlp_branch_weights.pt`.
- Routing SHA-256: `{MLP_SHA256}`.
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
| Complete HydraGNN, all 16 dataset branches | {total:,} |
| Requires-grad eligible in reconstructed training configuration | {result['training_configuration_requires_grad_parameters']:,} |
| Saved optimizer-covered parameters, all with optimizer state | {optimizer_count:,} |
| After explicit inference freezing | {result['frozen_requires_grad_parameters']:,} |
| Shared PaiNN blocks (including internal scalar/vector adaptations) | {sum(result['block_parameter_counts']):,} |
| Shared graph-conditioning projector | {projector_count:,} |
| Entire shared encoder, including conditioning | {shared:,} |
| All 16 branch-local prediction networks | {sum(head_counts.values()):,} |
| Separate learned atomic input embedding | 0 |
| Other parameters | {result['unclassified_parameters']:,} |
| Separate BranchWeightMLP | {mlp_count:,} |

Each dataset branch contains {result['per_head_graph_shared_parameters']:,}
parameters in its own `graph_shared` stack and
{result['per_head_output_mlp_parameters']:,} in its own output MLP.
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

- Shared blocks: {result['block_parameter_counts'][0]:,} +
  {result['block_parameter_counts'][1]:,} =
  {sum(result['block_parameter_counts']):,}.
- Shared encoder including conditioning:
  {sum(result['block_parameter_counts']):,} + {projector_count:,} = {shared:,}.
- Dataset-specific prediction networks:
  16 x {head_counts['branch-0']:,} = {sum(head_counts.values()):,}.
- Complete HydraGNN:
  {shared:,} + {sum(head_counts.values()):,} = **{total:,}**.
- HydraGNN plus the separate routing MLP:
  {total:,} + {mlp_count:,} = **{total + mlp_count:,}**, if a combined stack
  count is desired; this is not the HydraGNN-alone count.

## Block-level trainable-component breakdown

| Block | Submodule | Parameters |
|---|---|---:|
{block_rows}

All linear layers in these blocks include biases. Parameter-free SiLU, Tanh,
ReLU, Identity operations, geometric norms/inner products, aggregation,
radial functions and cutoff functions contribute no parameter elements.

## Representative checkpoint tensor shapes

These measured shapes expose the first-block channel width, filter sizes,
head dimensions, and lazy layer underlying the count discrepancy.

| Checkpoint parameter (normalized name) | Shape | Elements |
|---|---|---:|
{layer_rows}

| Zero-based dataset branch | Parameters including all branch-local layers |
|---|---:|
{per_head}

The saved generic branch names do not themselves establish dataset-label
semantics. The training command records the dataset ordering separately.

## Why the historical 12.1M count differs

The local training logs explicitly print **{optimizer_count:,}**, which rounds
to 12.1M. The same number is obtained immediately after construction, before
checkpoint restoration materializes the lazy projector, and from the saved
optimizer's moment tensors. The frozen checkpoint itself contains **{total:,}**
(12.205540M, approximately **12.2M** at one decimal place).
The difference is exactly **{projector_count:,}**, the projector weight of shape
337 x 339 plus its 337 biases. This is not a one-head versus sixteen-head
discrepancy: both counts include all sixteen branches.
A one-branch subset with the same complete shared encoder would have
{shared + head_counts['branch-0']:,} parameters, while the complete shared encoder
alone has {shared:,}.

The saved training configuration sets `freeze_conv_layers=false`.
All {total:,} parameter elements have requires-grad enabled in the reconstructed
training model; checkpoint state dictionaries do not encode original
requires-grad flags. However, the saved optimizer contains only
{optimizer_count:,} elements in 198 parameter tensors, versus 200 tensors in the
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

{log_rows}

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

{source_rows}

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

The complete saved all-sixteen-branch HydraGNN has **{total:,} parameters**;
the separate routing MLP has **{mlp_count:,}**. The historical approximately
12.1M value matches **{optimizer_count:,}**, the logged pre-lazy count and saved
optimizer coverage, not the complete checkpoint. The difference is exactly the
**{projector_count:,}**-parameter shared graph-conditioning projector.
All complete-model parameters are requires-grad eligible in the reconstructed
configuration, but that projector is absent from the saved optimizer.
The first PaiNN message/update block uses one input channel before expanding
to 337; the second uses 337. The actual four-feature filter maps are 4 -> 3
and 4 -> 1011, with no independent PaiNN filter width 126.
These are checkpoint/source inspection results, not manuscript changes.
"""
    (HERE / "report.md").write_text(report)
    (HERE / "reviewer_response.md").write_text(f"""# Proposed response to Reviewer 1, Comment 5

We thank the reviewer for requesting an architecture and parameter-count audit.
Strict reconstruction and loading of the frozen checkpoint verified a complete
sixteen-branch HydraGNN model containing **{total:,} parameters**. The separate
BranchWeightMLP contains **{mlp_count:,} parameters**. The shared PaiNN blocks
contain {sum(result['block_parameter_counts']):,} parameters, shared graph
conditioning contains {projector_count:,}, and all branch-local prediction
networks contain {sum(head_counts.values()):,} ({head_counts['branch-0']:,} each).

The previously reported approximately 12.1M value matches the exact
{optimizer_count:,} printed in the training logs before the lazy
graph-conditioning projector was instantiated. The checkpoint contains that
additional {projector_count:,}-parameter layer, bringing its complete size to
approximately 12.2M. The saved optimizer also excludes this layer: although
all {total:,} parameters are requires-grad eligible under the reconstructed
training configuration, only {optimizer_count:,} have saved optimizer coverage.
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
""")
    (HERE / "supporting_information_description.md").write_text(f"""# Proposed Supporting Information architecture description

The frozen HydraGNN checkpoint comprises a shared PaiNN encoder and sixteen
dataset-specific graph-energy prediction networks, with {total:,} parameters
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
{sum(head_counts.values()):,} parameters; the shared encoder including conditioning
contributes {shared:,}. No separate PaiNN interaction-filter width of 126 is
used. The separate composition-routing MLP is 118 -> 128 -> 64 -> 16 and
contains {mlp_count:,} parameters.

The logged pre-lazy-construction count and saved optimizer coverage are
{optimizer_count:,}; the complete checkpoint includes the additional
{projector_count:,}-parameter conditioning layer. All {total:,} parameters
are requires-grad eligible in the reconstructed training configuration, but
the two conditioning tensors are absent from the saved optimizer. Report this
training-coverage distinction alongside the total architectural count.
""")


def main() -> None:
    logdir = SC26 / "multidataset_hpo-BEST6-fp64"
    backbone_path = logdir / "multidataset_hpo-BEST6-fp64_epoch_97.pk"
    mlp_path = SC26 / "mlp_branch_weights.pt"
    protected = [
        logdir / "config.json", backbone_path, mlp_path,
        ROOT / "hydragnn/models/PAINNStack.py",
        ROOT / "hydragnn/models/Base.py", ROOT / "hydragnn/models/create.py",
        SC26 / "precision_relaxation/hydragnn_ase_calculator.py",
        SC26 / "gfm_mlip_all_mpnn.py",
        ROOT / "hydragnn/utils/model/model.py",
    ]
    protected.extend(sorted(logdir.glob("*.log")))
    hashes_before = {str(path.relative_to(ROOT)): sha256_file(path) for path in protected}
    for path, expected in ((backbone_path, BACKBONE_SHA256), (mlp_path, MLP_SHA256)):
        if sha256_file(path) != expected:
            raise ValueError(f"Checkpoint hash mismatch: {path}")
    config = json.loads((logdir / "config.json").read_text())
    torch.set_default_dtype(torch.float64)
    model = create_model_config(config["NeuralNetwork"], verbosity=0, use_gpu=False).cpu()
    before_named = list(model.named_parameters())
    before_count = sum(parameter.numel() for _, parameter in before_named)
    state = torch.load(backbone_path, map_location="cpu", weights_only=True)
    _restore_verified_state(model, state["model_state_dict"], label="HydraGNN")
    normalized = normalize_state(state["model_state_dict"])
    restored = model.state_dict()
    if restored.keys() != normalized.keys():
        raise ValueError("Strictly loaded state keys differ")
    for name, tensor in restored.items():
        if not torch.equal(tensor, normalized[name]):
            raise ValueError(f"Checkpoint value mismatch: {name}")
    core = model.model
    all_parameters = unique_parameters(model)
    components: dict[str, dict[int, nn.Parameter]] = {
        f"PaiNN block {index + 1}": unique_parameters(block)
        for index, block in enumerate(core.graph_convs)
    }
    components["Graph-attribute concat projector"] = unique_parameters(core.graph_concat_projector)
    head_counts = {}
    for branch in core.graph_shared:
        parameters = unique_parameters(core.graph_shared[branch])
        for head in core.heads_NN:
            parameters.update(unique_parameters(head[branch]))
        components[f"Dataset {branch}"] = parameters
        head_counts[branch] = count(parameters)
    seen: set[int] = set()
    membership = {}
    for component, parameters in components.items():
        if seen.intersection(parameters):
            raise ValueError(f"Parameter double counting in {component}")
        seen.update(parameters)
        membership.update({identifier: component for identifier in parameters})
    other = {identifier: parameter for identifier, parameter in all_parameters.items() if identifier not in seen}
    components["Other"] = other
    membership.update({identifier: "Other" for identifier in other})
    component_counts = {name: count(parameters) for name, parameters in components.items()}
    if sum(component_counts.values()) != count(all_parameters):
        raise ValueError("Component counts do not sum to total")
    current_named = dict(model.named_parameters())
    if set(current_named) != set(normalized):
        raise ValueError("State dictionary contains nonparameter entries or parameter aliases")
    optimizer = state["optimizer_state_dict"]
    optimizer_ids = [identifier for group in optimizer["param_groups"] for identifier in group["params"]]
    if len(optimizer_ids) != len(before_named) or len(set(optimizer_ids)) != len(optimizer_ids):
        raise ValueError("Cannot uniquely align saved optimizer with original parameter order")
    if [name for name, _ in before_named] != [name for name in current_named if name in dict(before_named)]:
        raise ValueError("Original construction parameter order changed")
    optimizer_names = set()
    optimizer_steps = {}
    for identifier, (name, parameter) in zip(optimizer_ids, before_named):
        moment = optimizer["state"][identifier]
        if any(tuple(moment[key].shape) != tuple(parameter.shape) for key in ("exp_avg", "exp_avg_sq")):
            raise ValueError(f"Optimizer shape mismatch for {name}")
        optimizer_names.add(name)
        optimizer_steps[name] = float(moment["step"])
    if not all(step > 0 for step in optimizer_steps.values()):
        raise ValueError("Optimizer-covered parameter lacks a positive update counter")
    total = count(all_parameters)
    eligible = sum(parameter.numel() for parameter in all_parameters.values() if parameter.requires_grad)
    parameter_rows = [
        {
            "name": name, "shape": "x".join(map(str, parameter.shape)),
            "numel": parameter.numel(), "component": membership[id(parameter)],
            "training_configuration_requires_grad": parameter.requires_grad,
            "saved_optimizer_covered": name in optimizer_names,
            "saved_optimizer_steps": optimizer_steps.get(name, ""),
        }
        for name, parameter in current_named.items()
    ]
    frozen_parameters = list(model.parameters())
    for parameter in frozen_parameters:
        parameter.requires_grad_(False)
    model.eval()
    frozen_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    mlp_state = torch.load(mlp_path, map_location="cpu", weights_only=True)
    mlp = _reconstruct_mlp_from_state_dict(mlp_state["mlp_state_dict"]).to(dtype=torch.float64)
    _restore_verified_state(mlp, mlp_state["mlp_state_dict"], label="BranchWeightMLP")
    for name, parameter in mlp.state_dict().items():
        if not torch.equal(parameter, mlp_state["mlp_state_dict"][name]):
            raise ValueError(f"MLP tensor mismatch: {name}")
    blocks = []
    block_submodules = []
    for index, block in enumerate(core.graph_convs):
        message = next(child for child in block.modules() if isinstance(child, PainnMessage))
        update = next(child for child in block.modules() if isinstance(child, PainnUpdate))
        blocks.append({
            "index": index, "message_channels": message.node_size,
            "update_channels": update.update_U.in_features,
            "num_radial": message.num_radial, "cutoff": message.cutoff,
            "filter_shape": list(message.filter_layer.weight.shape),
            "last_layer": update.last_layer, "linear_layers": linear_layers(block),
        })
        for name, module in block.named_children():
            module_count = count(unique_parameters(module))
            if module_count:
                block_submodules.append({
                    "block": index, "module": f"{name}: {type(module).__name__}",
                    "parameters": module_count,
                })
    representative_names = [
        "model.graph_convs.0.module_0.filter_layer.weight",
        "model.graph_convs.0.module_2.0.weight",
        "model.graph_convs.0.module_2.2.weight",
        "model.graph_convs.0.module_3.weight",
        "model.graph_convs.1.module_0.filter_layer.weight",
        "model.graph_shared.branch-0.0.weight",
        "model.graph_shared.branch-0.2.weight",
        "model.heads_NN.0.branch-0.0.weight",
        "model.heads_NN.0.branch-0.2.weight",
        "model.heads_NN.0.branch-0.4.weight",
        "model.graph_concat_projector.weight",
        "model.graph_concat_projector.bias",
    ]
    sources = {
        "PAINNStack": source_reference(type(core)),
        "PainnMessage": source_reference(PainnMessage),
        "PainnUpdate": source_reference(PainnUpdate),
        "sinc_expansion": source_reference(sinc_expansion),
        "cosine_cutoff": source_reference(cosine_cutoff),
        "create_model_config": source_reference(create_model_config),
        "head_construction": source_reference(core._multihead),
        "graph_conditioning": source_reference(core._apply_graph_conditioning),
        "lazy_projector_creation": source_reference(core._ensure_graph_concat_projector),
        "strict_checkpoint_restoration": source_reference(_restore_verified_state),
        "BranchWeightMLP": source_reference(type(mlp)),
    }
    sources.update({
        "PaiNN_model_factory": {"path": "hydragnn/models/create.py", "line": 483},
        "SchNet_num_filters": {"path": "hydragnn/models/create.py", "line": 375},
        "input_embedding_and_vector_storage": {"path": "hydragnn/models/PAINNStack.py", "line": 149},
        "convolution_freezing": {"path": "hydragnn/models/Base.py", "line": 495},
        "training_optimizer_before_forward": {"path": "examples/multidataset_hpo_sc26/gfm_mlip_all_mpnn.py", "line": 899},
        "legacy_parameter_log": {"path": "hydragnn/utils/model/model.py", "line": 451},
        "periodic_neighbor_cap": {"path": "hydragnn/preprocess/graph_samples_checks_and_updates.py", "line": 266},
    })
    log_totals = []
    for path in sorted(logdir.glob("*.log")):
        with path.open(errors="strict") as stream:
            for line_number, line in enumerate(stream, start=1):
                fields = line.split()
                if "Total" in fields and fields[-1].isdigit():
                    log_totals.append({
                        "path": str(path.relative_to(ROOT)), "line": line_number,
                        "printed_total": int(fields[-1]),
                    })
    result = {
        "checkpoint_sha256": {"HydraGNN": BACKBONE_SHA256, "BranchWeightMLP": MLP_SHA256},
        "source_sha256": hashes_before, "number_of_branches": len(head_counts),
        "number_of_blocks": len(blocks), "hidden_dim": core.hidden_dim,
        "input_dim": core.input_dim, "pooling": core.graph_pooling,
        "max_neighbours": config["NeuralNetwork"]["Architecture"]["max_neighbours"],
        "total_parameters": total, "unique_parameter_tensors": len(all_parameters),
        "state_dictionary_elements": sum(tensor.numel() for tensor in normalized.values()),
        "training_configuration_requires_grad_parameters": eligible,
        "original_constructor_parameters": before_count,
        "optimizer_parameter_elements": sum(current_named[name].numel() for name in optimizer_names),
        "optimizer_parameter_tensors": len(optimizer_names),
        "optimizer_omitted_parameter_names": sorted(set(current_named) - optimizer_names),
        "optimizer_alignment": "Original constructor parameter order and both Adam moment shapes matched; all positive step counters",
        "frozen_requires_grad_parameters": frozen_count,
        "components": component_counts, "block_parameter_counts": [count(unique_parameters(block)) for block in core.graph_convs],
        "head_parameter_counts": head_counts,
        "shared_parameters_including_conditioning": sum(value for key, value in component_counts.items() if not key.startswith("Dataset ")),
        "per_head_graph_shared_parameters": count(unique_parameters(core.graph_shared["branch-0"])),
        "per_head_output_mlp_parameters": count(unique_parameters(core.heads_NN[0]["branch-0"])),
        "unclassified_parameters": count(other), "branch_weight_mlp_parameters": count(unique_parameters(mlp)),
        "blocks": blocks, "head_linear_layers": linear_layers(core.graph_shared["branch-0"]) + linear_layers(core.heads_NN[0]["branch-0"]),
        "block_submodule_counts": block_submodules,
        "representative_parameter_shapes": [
            [name, " x ".join(map(str, current_named[name].shape)), current_named[name].numel()]
            for name in representative_names
        ],
        "sources": sources, "training_log_totals": log_totals,
        "strict_tensor_equality_verified": True,
        "manuscript_available": False,
    }
    if result["state_dictionary_elements"] != total or len(head_counts) != 16:
        raise ValueError("Checkpoint parameter total or sixteen-branch count failed")
    expected_head_dims = [(337, 50), (50, 50), (50, 776), (776, 776), (776, 1)]
    for branch in core.graph_shared:
        layers = linear_layers(core.graph_shared[branch]) + linear_layers(core.heads_NN[0][branch])
        if [(layer["in_features"], layer["out_features"]) for layer in layers] != expected_head_dims:
            raise ValueError(f"Report architecture dimensions do not match {branch}")
    if (
        [(block["message_channels"], block["update_channels"], block["num_radial"], block["cutoff"]) for block in blocks]
        != [(1, 1, 4, 5.0), (337, 337, 4, 5.0)]
        or core.hidden_dim != 337 or core.use_global_attn or core.use_encodings
        or core.graph_concat_projector.in_features != 339
        or core.graph_concat_projector.out_features != 337
        or result["max_neighbours"] != 20 or result["pooling"] != "add"
        or result["unclassified_parameters"] != 0
    ):
        raise ValueError("Verified model differs from the architecture described in this report")
    if any(isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.LayerNorm, nn.Dropout, nn.Embedding)) for module in model.modules()):
        raise ValueError("Unexpected normalization, dropout or embedding module")
    save_json(HERE / "inspection.json", result)
    save_json(HERE / "saved_configuration.json", config)
    save_json(HERE / "module_hierarchy.json", [
        {"path": name, "class": type(module).__name__, "parameters_recursive": count(unique_parameters(module))}
        for name, module in model.named_modules()
    ])
    with (HERE / "parameters.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(parameter_rows[0]))
        writer.writeheader()
        writer.writerows(parameter_rows)
    write_documents(result)
    if hashes_before != {str(path.relative_to(ROOT)): sha256_file(path) for path in protected}:
        raise ValueError("Source artifact changed during inspection")
    print(json.dumps({
        "total_parameters": total, "training_configuration_requires_grad": eligible,
        "saved_optimizer_parameters": result["optimizer_parameter_elements"],
        "frozen_trainable": frozen_count, "components": component_counts,
        "BranchWeightMLP": result["branch_weight_mlp_parameters"],
        "optimizer_omissions": result["optimizer_omitted_parameter_names"],
    }, indent=2))


if __name__ == "__main__":
    main()
