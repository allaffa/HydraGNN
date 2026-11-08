# Si hydrostatic relaxation: starting-cell sensitivity

## Protocol

To test dependence on the initial volume, the same perturbed fractional
coordinates of the eight-atom diamond-Si cell were used at initial cubic
lattice parameters from 4.80 to 6.00 Å. This spans initial cells about 12%
smaller to 10% larger than the original 5.43 Å cell. Both FP32 and FP64 used
the same frozen fused HydraGNN/BranchWeightMLP checkpoints, 5 Å graph radius,
and 20-neighbor cap.

Each run used ASE FIRE with 0.03 Å maximum step, hydrostatic `UnitCellFilter`,
and `fmax=0.05 eV/Å` for up to 300 steps. Since the FP64 run starting at
4.80 Å had not converged at that limit, it was repeated from the same initial
geometry with a 700-step limit; it converged in 367 steps. ASE's filter
criterion is the maximum generalized optimization force, so the atomic-force
maximum can be slightly above 0.05 eV/Å in an otherwise converged cell run.

## Results

| Initial lattice (Å) | FP32 final lattice (Å) | FP32 steps | FP64 final lattice (Å) | FP64 steps |
|---:|---:|---:|---:|---:|
| 4.80 | 5.51645 | 238 | 5.51697 | 367* |
| 5.00 | 5.50299 | 83 | 5.50310 | 83 |
| 5.43 | 5.50563 | 49 | 5.50566 | 68 |
| 5.80 | 5.50733 | 54 | 5.50733 | 54 |
| 6.00 | 5.50718 | 56 | 5.50730 | 79 |

All five starting sizes converged in both precisions after the extended
4.80 Å FP64 run. Across these starts, the converged lattice parameters lie
between 5.5030 and 5.5170 Å, a total range of about 0.014 Å (0.25% of the
mean). The model therefore returns a very similar cell over this tested range
of initial volumes, although the smaller 4.80 Å start takes much longer.

The final model-predicted energies are about 0.1551–0.1554 eV/atom, and
maximum predicted absolute stresses are approximately 0.0018–0.0052 eV/Å³.
These are HydraGNN outputs, not independent DFT values.

## Interpretation and limits

This is good evidence that the hydrostatic relaxation is not simply preserving
its starting lattice: substantially compressed and expanded starts approach
the same model-preferred volume. In that numerical sense, the cell relaxation
behaves consistently.

The final lattice remains around 1.3–1.6% larger than the NIST CODATA
experimental lattice parameter of 5.431020511 Å. The repeated convergence
supports a stable minimum of this learned potential, not physical accuracy of
that minimum. The experiment also tests one perturbed atomic geometry and a
finite range of cubic starting cells; it does not establish a global minimum
or rule out other basins, distortions, or pressure dependence.

See [`comparison.json`](./comparison.json) and the per-run trajectories and
summaries in this directory for complete results.
