# BranchWeightMLP composition-routing probe

## Scope and limitation

This is a reproducible **synthetic composition-probe analysis**, not the requested
production-data analysis. The user confirmed that the supplied MLP pairs with the
local HydraGNN epoch-97 checkpoint and asked to generate structures. There are no
archived validation/held-out structures or training composition distributions in
the local checkout. Consequently, the required three-archive encoding comparison,
empirical in-domain/novel/distant labels, dataset-specialization claims, and
manuscript-ready generalizations cannot be established here. All 195 generated
100-atom ASE structures are composition carriers with deterministic simple-cubic
positions; no relaxation, DFT, HydraGNN backbone inference, or training was run.
They should not be mistaken for stable molecules, crystals, or sampled training
records.

## Model and checkpoint provenance

- MLP checkpoint: `../mlp_branch_weights.pt`
- MLP SHA-256: `c984b338ee947534bbdedf1fc4dd2aea92bb846e3302638d6719b1e3f5ca7ca8`
- MLP checkpoint epoch field: 207 (this is the checkpoint's saved
  epoch, not independent evidence of best validation performance).
- HydraGNN checkpoint: `../multidataset_hpo-BEST6-fp64/multidataset_hpo-BEST6-fp64_epoch_97.pk`
- HydraGNN SHA-256: `8532b03296a8ab4cfc3b37beb6d57b2091a50807b155fac93cb1882253d32867`
- Git commit: `590fcb910e4d88b8ed2ce1c7838a159b09365d54`
- Pairing: user-confirmed; the MLP file itself contains no HydraGNN checkpoint
  identifier.
- MLP dimensions: 118 -> 128 ->
  64 -> 16; parameter dtype
  torch.float64; softmax output.
- Branch order is recorded in `branch_mapping.csv` from the local SC26 launch
  metadata: Alexandria, ANI1x, MPTrj, OC2020, OC2022, OC25, ODAC23, OMat24, OMol25, OMol25-neutral, OMol25-non-neutral, OPoly2026, Nabla2DFT, QCML, QM7X, transition1x.

## Composition encoding

The adapters construct `chemical_composition` as an 118-bin histogram of atomic
numbers 1 through 118, then cast it to float32. Feature index `Z-1` is the count
of element Z; absent elements have zero; counts are not normalized. Thus the
vector depends on the number of atoms in a structure. The inference reshape
changes layout to `[graphs, 118]` but does not normalize or otherwise transform
the counts before MLP inference. The generated inputs use the same integer-count
histogram over ASE atomic numbers, then the original checkpoint dtype (float64).

Archived-field validation on three saved structures was **not performed**:
there are no local archived datasets. `composition_encoding_audit.json` records
this explicitly; generated vectors are an implementation exercise, not an
independent archive audit.

## Evaluation construction and validation

- 195 unique generated compositions, 13
  balanced synthetic motif families, 15 samples per family.
- Every probe has 100 atoms so the raw-count feature scale is held fixed
  across this synthetic suite.
- All records use source `synthetic_generated`; training combination coverage and
  domain class are `unknown`.
- All pairwise composition distances use L1 distance between atomic-fraction
  vectors. Routing total variation is one half the L1 distance between weight
  vectors; Jensen-Shannon divergence is computed with base-2 logarithms and is
  reported in bits. “Related” means members of the same generated template
  family, not a measured chemical-similarity label.
- A five-composition pilot passed shape, finite/nonnegative weight, normalization,
  and repeatability checks. Maximum softmax-sum error was
  0.000e+00; maximum repeat difference was
  0.000e+00.
- A composition-only arrangement check covered 18
  same-composition pairs. Maximum observed JS divergence was 0.000e+00. This is
  a check of the composition-only implementation, not geometric transferability.
- Inference loaded only BranchWeightMLP and did not invoke the HydraGNN backbone.

## Quantitative routing results

The primary observation is the complete 16-dimensional softmax vector for each
input, saved in `branch_weights.csv` and visualized in the heatmaps. Read these
values as learned preferences among source-specific prediction heads. The
dominant-branch, top-three, entropy, and cumulative-support summaries are
secondary descriptions of that distribution; no branch is labeled activated or
inactive using a hard per-composition threshold.

- Overall mean effective support: 1.281
  of 16 branches (range 1.000 to
  2.612).
- Mean maximum branch weight: 0.8984;
  mean top-three cumulative weight: 1.0000.
- Mean support size: 1.32 branches
  for 90% cumulative weight and 1.34
  branches for 95%.
- 2380 of
  3120 computed float64 weights were
  represented as exact zero by softmax underflow. These are finite-precision
  zeros, not branch skipping or a per-composition activation rule; mathematical
  softmax weights are positive.
- Related-family pairs: n=1365, mean JS divergence
  0.2953 bits, median 0.0063; family-cluster
  descriptive 95% bootstrap interval for the mean
  [0.2015, 0.3868]; mean total-variation
  distance 0.3356, dominant-branch agreement
  62.9%.
- Different-family pairs: n=17550, mean JS divergence
  0.7505 bits, median 1.0000; family-cluster
  descriptive 95% bootstrap interval for the mean
  [0.7153, 0.7856]; mean
  total-variation distance 0.7792,
  dominant-branch agreement 21.6%.
- Same-composition/different-arrangement pairs: n=18,
  maximum JS divergence 0.000e+00 bits.
- Spearman association between normalized-composition L1 distance and routing JS
  divergence over related and distinct composition pairs:
  rho=0.3257. This is descriptive;
  pairs share compositions and the statistic is not an independent-pair
  significance test.
- Confidence intervals resample family-level mean routing distances, not
  individual pair rows; with generated families they are descriptive summaries,
  not population-inference intervals.

### Motif-family dominant branches

- **synthetic C-H motifs:** most frequent dominant branch branch-8:OMol25 (14/15); mean weight on that branch 0.9333; mean effective support 1.00.
- **synthetic C-H-N-O motifs:** most frequent dominant branch branch-8:OMol25 (12/15); mean weight on that branch 0.8000; mean effective support 1.00.
- **synthetic C-H-N-O-S-P motifs:** most frequent dominant branch branch-8:OMol25 (13/15); mean weight on that branch 0.8667; mean effective support 1.00.
- **synthetic transition-metal oxides:** most frequent dominant branch branch-4:OC2022 (15/15); mean weight on that branch 1.0000; mean effective support 1.00.
- **synthetic mixed-metal oxides:** most frequent dominant branch branch-4:OC2022 (14/15); mean weight on that branch 0.9333; mean effective support 1.00.
- **synthetic silicate/oxide motifs:** most frequent dominant branch branch-4:OC2022 (10/15); mean weight on that branch 0.6667; mean effective support 1.00.
- **synthetic transition-metal chalcogenides:** most frequent dominant branch branch-2:MPTrj (10/15); mean weight on that branch 0.5580; mean effective support 1.79.
- **synthetic binary refractory alloys:** most frequent dominant branch branch-2:MPTrj (11/15); mean weight on that branch 0.5335; mean effective support 1.84.
- **synthetic ternary refractory alloys:** most frequent dominant branch branch-2:MPTrj (7/15); mean weight on that branch 0.3353; mean effective support 1.55.
- **synthetic multicomponent alloys:** most frequent dominant branch branch-2:MPTrj (14/15); mean weight on that branch 0.7615; mean effective support 1.66.
- **synthetic semiconductor motifs:** most frequent dominant branch branch-2:MPTrj (6/15); mean weight on that branch 0.4061; mean effective support 1.81.
- **synthetic ionic/halide motifs:** most frequent dominant branch branch-2:MPTrj (9/15); mean weight on that branch 0.6000; mean effective support 1.00.
- **synthetic rare-earth oxides:** most frequent dominant branch branch-4:OC2022 (11/15); mean weight on that branch 0.7360; mean effective support 1.01.

### Continuous composition-space probes

Each path contains 21 points with a constant 100-count vector scale. Interpolated
fractional counts are composition-space probes, not discrete atomistic structures
or asserted phase transitions.

- **Nb-Ta binary alloy probe:** dominant branch branch-15:transition1x at t=0 and branch-15:transition1x at t=1; 0 changes between adjacent sampled points; the five plotted branches are branch-15:transition1x, branch-2:MPTrj, branch-7:OMat24, branch-4:OC2022, branch-8:OMol25.
- **Ti-Fe oxide substitution probe:** dominant branch branch-4:OC2022 at t=0 and branch-4:OC2022 at t=1; 0 changes between adjacent sampled points; the five plotted branches are branch-4:OC2022, branch-15:transition1x, branch-14:QM7X, branch-13:QCML, branch-12:Nabla2DFT.
- **Nb-Ta-V ternary alloy probe:** dominant branch branch-7:OMat24 at t=0 and branch-15:transition1x at t=1; 1 changes between adjacent sampled points; the five plotted branches are branch-7:OMat24, branch-15:transition1x, branch-2:MPTrj, branch-8:OMol25, branch-4:OC2022.

## Figure interpretation

- **Figure 1:** heatmap of 26 representative generated compositions, ordered by
  synthetic motif family. It visualizes this checkpoint's response only; it is
  not evidence of routing on archived chemical records.
- **Supplementary full heatmap:** all 195 generated compositions.
- **Figure 2:** five highest-mean branches along each of three fixed-total
  interpolation paths; remaining weight is shown as “Other.”
- **Figure 3:** same-composition, within-motif, and between-motif routing
  distances. Pairwise rows overlap by construction, so boxes are descriptive.
- **Figure 4:** concentration metrics by generated motif family. It does not
  compare empirical domain classes.
- **Figure 5:** family-balanced mean branch weights, with 15 compositions per
  generated motif family.

## Scientific assessment

- **Are similar compositions routed similarly?** In this synthetic suite, the
  measured relationship is the reported Spearman rho and the within-/between-
  family JS summaries above. These generated labels do not establish the result
  for manuscript data.
- **Which branches dominate for each chemical family?** The measured synthetic
  motif-family results are listed above and in `summary_statistics.json`. They
  are not dataset specialization evidence.
- **Does routing remain stable under small composition changes?** Figure 2 and
  `composition_paths.csv` give measured weight changes on three chosen paths.
  They establish behavior only along those constructed paths.
- **Are compositionally novel inputs routed differently from familiar ones?**
  This cannot be answered: training compositions are unavailable, so every
  generated record has unknown domain status.
- **Is weighting concentrated or diffuse?** The synthetic-suite effective
  support, maximum weights, top-three sums, and support sizes are in
  `routing_metrics.csv` and `summary_statistics.json`.
- **Established versus hypothesized:** established here are the checkpoint's
  deterministic softmax outputs for the exact generated count vectors and the
  arithmetic summaries. Dataset specialization, physical confidence, prediction
  accuracy, and behavior on in-domain or out-of-domain production data remain
  untested.

Softmax weights are learned preferences among source-specific prediction heads,
not probabilities that a composition belongs to a dataset, measures of physical
fidelity, uncertainty estimates, or indicators of DFT accuracy. A high or low
entropy must not be interpreted as confidence without independent calibration
evidence.

## Manuscript text and reviewer response (provisional only)

**Section 5.5 draft:** “The BranchWeightMLP maps an 118-component elemental-count
vector to a 16-branch softmax distribution. In a synthetic, fixed-size
composition-probe check, branch weights varied across tested compositions and
along selected interpolation paths; because archived structures and training
composition distributions were unavailable for this check, these results do not
establish dataset specialization or behavior on in-domain and novel materials.”

**Reviewer 2, Comment 5:** “We performed a composition-only checkpoint probe and
report the complete branch-weight distributions, concentration measures, and
composition-path responses. The probe used generated structures and therefore
does not substitute for analysis of the manuscript's archived evaluation data.
We will report claims about chemical specialization and domain coverage only
after validating the exact composition vectors against archived records and
comparing them with the training composition distribution.”

**Figure recommendation:** Do not use the present synthetic-only figures as
manuscript evidence. After rerunning with audited archived records, use the
representative branch-weight heatmap as the likely main-text figure and place
composition-path and pairwise-similarity plots in Supporting Information.
