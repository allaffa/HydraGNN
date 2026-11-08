# Off-equilibrium crystal relaxation stress test

## Setup

- Structures: 2×2×2 diamond Si and rocksalt MgO supercells.
- Deliberate starting distortion: Si cell expanded by 5%; MgO cell compressed by 5%; all atoms receive seeded, random-direction displacements of 0.10–0.15 Å.
- Paired FP32/FP64 FIRE relaxations from identical per-structure starting coordinates, relaxing atomic positions and hydrostatic cell volume together.
- Maximum optimizer step: 0.03 Å; generalized-force threshold: 0.05 eV/Å; step cap: 500.
- Distortion seed: 20261109.

## Results

| Structure | Precision | Status | Steps | ΔE (eV/atom) | Initial→final max force (eV/Å) | RMS atom shift (Å) | Max atom shift (Å) | Δcell (%) | ΔV (%) |
|---|---|---|---|---|---|---|---|---|---|
| Si_diamond | fp32 | converged | 181 | -0.026906 | 0.3827 → 0.0362 | 0.1165 | 0.1489 | -4.125 | -11.871 |
| Si_diamond | fp64 | converged | 181 | -0.026923 | 0.3827 → 0.0329 | 0.1165 | 0.1495 | -4.125 | -11.871 |
| MgO_rocksalt | fp32 | converged | 84 | -0.022983 | 0.8327 → 0.0271 | 0.1271 | 0.1621 | +1.502 | +4.574 |
| MgO_rocksalt | fp64 | converged | 84 | -0.022983 | 0.8327 → 0.0271 | 0.1271 | 0.1621 | +1.502 | +4.574 |

Energy change and force reduction are predictions of the frozen learned potential.
This is a robustness/stress test, not evidence that the final geometry agrees
with DFT or experiment. The initial distortion may be outside the model's
training distribution. A non-converged endpoint is not a relaxed structure.

## Figures

- `figures/<structure>_paired.png`: initial and FP64 final geometries rendered at the same coordinate scale, with cell outlines and element legends.
- `figures/<structure>_initial.png` and `figures/<structure>_final.png`: shaded OVITO renders.
- `<structure>/<precision>/`: ASE trajectories, per-step force/energy/cell records, logs, summaries, and final structures.
