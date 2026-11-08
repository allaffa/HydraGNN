# BranchWeightMLP robustness analysis (second round)

## Executive scientific assessment

1. **Atom-count sensitivity:** proportional rescaling of raw count vectors
   changes routing in the specified families; the maximum per-family JS and TV
   changes are reported below. Therefore the input representation is not
   scale-invariant. This is a characterization, not a claim that scale
   invariance was intended.
2. **Softmax saturation:** concentration and underflow are reproducible for the
   supplied checkpoint. Across first-round and size probes, 2588
   of 3472 FP64 values
   were exactly zero after softmax. This is numerical underflow, not explicit
   branch pruning. Size trends are summarized in the logit diagnostics and
   figure; training atom-count distributions were unavailable, so “realistic”
   training sizes cannot be verified.
3. **Chemistry–routing after count control:** all first-round compositions have
   exactly 100 atoms. Their matched-pair and complete-vector permutation
   results therefore isolate composition better than the first-round
   unadjusted comparison. Observed Spearman rho was
   0.2956, composition-level bootstrap 95%
   interval [0.2218,
   0.3977], versus the
   within-size-stratum permutation null (mean
   -0.0002, 95% interval
   [-0.0272,
   0.0308], two-sided p
   0.002). This remains synthetic-only,
   with overlapping pairwise comparisons and constructed motif labels.
4. **Element substitutions:** measured path changes are listed below. A head
   change or sharp weight movement describes the learned preference response;
   it is not a phase transition or proof of specialization.
5. **Branch–chemistry associations:** the mapping is independently confirmed
   as zero-based using the config head list, launcher and training log
   `multi_model_list`, dataset-name enumeration in training code, and branch
   IDs in the backbone checkpoint. Even so, synthetic motif-to-head association
   is not empirical dataset specialization.
6. **Manuscript evidence:** these robustness results do not yet justify a
   manuscript figure about real chemical specialization. A synthetic
   supplementary/diagnostic figure may be useful if clearly labeled. Archived
   compositions and training-distribution statistics remain necessary for
   scientific claims about the target evaluation domain.

## Scope, checkpoint, and model

The trained MLP weights were not modified; no training, HydraGNN backbone
inference, DFT, or HPC jobs were run. The first-round results remain unchanged.

- MLP SHA-256: `c984b338ee947534bbdedf1fc4dd2aea92bb846e3302638d6719b1e3f5ca7ca8`
- HydraGNN checkpoint SHA-256: `8532b03296a8ab4cfc3b37beb6d57b2091a50807b155fac93cb1882253d32867`
- Base MLP: 118 -> 128 -> 64 -> 16, float64; Git commit
  `590fcb910e4d88b8ed2ce1c7838a159b09365d54`.
- Pairing is user-confirmed. The MLP checkpoint does not itself record the
  backbone checkpoint hash.
- Full branch mapping is in `branch_mapping_provenance.json`; all branch IDs are
  zero-based.
- No discrepancy was found in the checkpoint hashes or model dimensions.

## Test 1 — Proportional atom-count scaling

**Hypothesis:** with raw-count inputs, identical elemental fractions at
different total counts may yield different full weight distributions.

**Construction:** exact integer series for Nb-Ta, TiO2, CH4, LiFePO4, and a
Nb-Ta-V-Hf-Zr alloy. Inputs remain unnormalized. Each row, all 16 weights,
dominant branch, top-three, effective support, rank changes, JS, and TV are in
`size_sensitivity.csv`.

- **Nb-Ta binary alloy:** N=[10, 20, 50, 100, 200]; max JS vs smallest 0.9577 bits, max TV 0.9894; dominant branches branch-7:OMat24, branch-2:MPTrj, branch-2:MPTrj, branch-15:transition1x, branch-15:transition1x; effective-support range 1.092–1.996.
- **TiO2:** N=[6, 30, 60, 120]; max JS vs smallest 1.0000 bits, max TV 1.0000; dominant branches branch-2:MPTrj, branch-4:OC2022, branch-4:OC2022, branch-4:OC2022; effective-support range 1.000–1.000.
- **CH4:** N=[5, 20, 50, 100]; max JS vs smallest 0.0000 bits, max TV 0.0000; dominant branches branch-8:OMol25, branch-8:OMol25, branch-8:OMol25, branch-8:OMol25; effective-support range 1.000–1.000.
- **LiFePO4:** N=[7, 14, 35, 70]; max JS vs smallest 1.0000 bits, max TV 1.0000; dominant branches branch-7:OMat24, branch-2:MPTrj, branch-2:MPTrj, branch-2:MPTrj; effective-support range 1.000–1.000.
- **Nb-Ta-V-Hf-Zr multicomponent alloy:** N=[10, 20, 50, 100, 200]; max JS vs smallest 0.9994 bits, max TV 0.9999; dominant branches branch-7:OMat24, branch-4:OC2022, branch-2:MPTrj, branch-7:OMat24, branch-7:OMat24; effective-support range 1.000–2.214.

**Conclusion:** this model should not be assumed invariant to system size:
proportional count scaling is an actual change in input magnitude. Whether
these sizes occur in training cannot be established because atom-count records
are unavailable.

## Test 2 — Logit spread and numerical saturation

`logit_diagnostics.csv` records FP64 and FP32 logits, FP64 log probabilities,
full weights, logit ranges, largest-minus-second-largest logit gap, exact zeros,
counts below 1e-12/1e-8/1e-4/1e-2, entropy from `log_softmax`, and effective
branch count. Entropy is computed as
`-sum(exp(log_softmax(logits)) * log_softmax(logits))`.

- Logits finite in FP64: True;
  FP32: True.
- Maximum absolute difference between FP64 softmax and `exp(log_softmax)`:
  1.110e-16.
- Maximum FP64-vs-FP32 logit difference:
  0.0542305; weight
  difference: 0.00088165.
- Exact zero FP64 weights: 2588;
  FP32: 3012.
- Temperature scaling was not used. No logits were clamped.

**Conclusion:** the zero weights are finite-precision softmax underflow, not a
runtime branch-activation decision. The plot contrasts 100-atom records with
the specified variable-size probes. Training structure sizes are missing, so
we cannot classify the observed saturation as typical or atypical for training.

## Test 3 — Archived data and composition encoding

The local analysis checkout and nearby HydraGNN checkout contain no dataset
directory. The ADIOS dataset paths referenced by the Frontier run log were
checked for all 16 `*-v2.bp` files; the `/lustre` root is not mounted here and
the paths are unavailable. The logs show that ADIOS records contain
`trainset/chemical_composition`, `valset/chemical_composition`, and
`testset/chemical_composition` variables, but logs do not provide per-record
vectors. No three-structure stored-field comparison could be conducted.
`real_composition_audit.json` records each exact path and its status.

The repository code supports the code-level conclusion that `chemical_composition`
is a raw 118-element atomic-number count histogram, feature index Z-1 is
element Z, and no normalization is applied before the MLP. This still needs
three archived-record comparisons for independent validation. Public curated
composition records could be obtained without running calculations, but none
were acquired here; if added later they must be labeled real-composition
probes, not training-domain validation. No training/validation split or domain
labels were invented.

## Test 4 — Controlled elemental substitutions

Seven exact-count paths include Nb-Ta, Mo-W, Ti-Zr dioxide, Fe-Co monoxide,
and C/H, C/H/N, C/H/O composition-space probes. The organic paths are explicitly
abstract fixed-size elemental substitutions, not claimed molecular series.
The oxide paths are composition probes; no structural stability is asserted.
Full weights and adjacent JS/TV values are in `substitution_paths.csv`.

- **Nb-Ta alloy (fixed 100 atoms):** 11 points, dominant branch branch-15:transition1x → branch-15:transition1x, 0 dominant changes; max adjacent JS 0.0201 bits, max adjacent TV 0.1295. binary alloy composition path
- **Mo-W alloy (fixed 100 atoms):** 11 points, dominant branch branch-2:MPTrj → branch-2:MPTrj, 0 dominant changes; max adjacent JS 0.0173 bits, max adjacent TV 0.1031. binary alloy composition path
- **Ti-Zr dioxide substitution (99 atoms):** 12 points, dominant branch branch-4:OC2022 → branch-4:OC2022, 0 dominant changes; max adjacent JS 0.0000 bits, max adjacent TV 0.0000. 33 cation sites exchanged; fixed O66; composition-space alloyed oxide probe
- **Fe-Co monoxide substitution (100 atoms):** 11 points, dominant branch branch-4:OC2022 → branch-4:OC2022, 0 dominant changes; max adjacent JS 0.0000 bits, max adjacent TV 0.0000. fixed-size composition-space oxide probe; mixed Fe/Co phase not asserted
- **Organic C-H substitution probe (100 atoms):** 11 points, dominant branch branch-8:OMol25 → branch-8:OMol25, 0 dominant changes; max adjacent JS 0.0000 bits, max adjacent TV 0.0000. abstract fixed-size C/H composition path; not a molecular series
- **Organic C-H-N substitution probe (100 atoms):** 11 points, dominant branch branch-8:OMol25 → branch-8:OMol25, 0 dominant changes; max adjacent JS 0.0000 bits, max adjacent TV 0.0000. abstract fixed-size C/H/N composition path; not a molecular series
- **Organic C-H-O substitution probe (100 atoms):** 11 points, dominant branch branch-15:transition1x → branch-8:OMol25, 1 dominant changes; max adjacent JS 1.0000 bits, max adjacent TV 1.0000. abstract fixed-size C/H/O composition path; not a molecular series

No branch transition is interpreted as a phase transition. “Gradual” versus
“abrupt” refers only to the per-step measured weight distance on the chosen
discretized path.

## Test 5 — Similarity controls

All first-round compositions are fixed at 100 atoms. `matched_similarity.csv`
contains pair-level atomic-fraction L1, JS divergence, total variation, and
dominant-branch agreement. It separates same-size small/large composition
distances and provides matched within-/between-family pairs in common
0.1-wide L1 bins. Same-stoichiometry, different-size pairs from Test 1 are
also recorded.

- Same-size small-distance pairs (L1 <= 0.5): within-family
  n=237, mean JS
  0.2428 bits, mean TV
  0.2461, dominant-branch agreement
  75.5%; between-family
  n=165, mean JS
  0.1273 bits, mean TV
  0.1278, dominant-branch agreement
  87.3%.
- Same-size large-distance pairs (L1 >= 1.5): within-family
  n=380, mean JS
  0.3834 bits, mean TV
  0.4678, dominant-branch agreement
  43.9%; between-family
  n=14478, mean JS
  0.8158 bits, mean TV
  0.8485, dominant-branch agreement
  14.7%.
- Distance-matched within-/between-family comparisons:
  1261 matched pair sets across bins
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]; mean within-minus-between JS
  -0.15940284828615336
  bits, composition-bootstrap 95% CI
  [-0.24079681224596663, -0.08274973695877501].
- Same stoichiometry but different counts: 38
  pairs, mean JS 0.4994 bits.
- Permutation shuffles complete 16-weight vectors among fixed-size composition
  records; bootstrap resamples composition records, not pair rows.

If matched common support is small, do not interpret within-/between-family
differences as controlled chemical-family effects. Constructed motif families
remain synthetic descriptors.

## Test 6 — Training-domain proximity

`domain_proximity.csv` was not generated because no training composition
reference vectors are accessible. Exact-composition occurrence, element-set
coverage, nearest-training-composition distance, and in-/out-of-domain labels
are therefore unavailable.

## Branch mapping provenance

The ordered config output heads are `branch-0` through `branch-15`. The launcher
defines the ordered dataset aliases; the run log records the corresponding
16-name `multi_model_list`; training code enumerates that list and maps each
name to `torch.tensor([[i]])`; and the loaded HydraGNN checkpoint has branch
parameter keys 0 through 15. The full evidence appears in
`branch_mapping_provenance.json`. This verifies the structural zero-based
mapping, but does not establish that each synthetic chemical family is
specialized to its named dataset.

## Conclusions and unresolved blockers

- **Atom count at fixed stoichiometry:** changes the input and, as reported in
  the series, may materially change the full routing distribution.
- **Softmax saturation at realistic sizes:** underflow is measurable; its
  prevalence in the training regime cannot be decided without archived atom
  counts.
- **Chemistry after count control:** the 100-atom permutation and matched-pair
  results are the strongest available synthetic evidence; still no real
  composition validation.
- **Substitutions:** path-level responses are measured; abruptness is defined
  by adjacent JS/TV, not physical transitions.
- **Consistent branch–chemistry associations:** not established for real
  datasets because synthetic family labels are not source-dataset labels.
- **Manuscript figure:** no figure is yet supported as evidence for real
  chemical specialization. A clearly labeled full-weight synthetic heatmap
  could serve only as a model-behavior illustration.
- **Remaining blockers:** archived structures and training composition
  distributions; three-record encoding audit; evidence that tested count sizes
  overlap the training range; real-record matched composition analysis.

Weights throughout are the MLP's learned preferences among source-specific
prediction heads. They are not indicators of physical fidelity, predictive
uncertainty, dataset-membership probabilities, or branch activation.
