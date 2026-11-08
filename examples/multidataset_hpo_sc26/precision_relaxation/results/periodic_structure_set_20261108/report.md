# Periodic structure relaxation: FP32 versus FP64

## Protocol

- Structures: 13 representative periodic crystal prototypes spanning elemental, binary, and ternary compositions.
- Initial structures: standard ASE crystal prototypes, each expanded to a 2×2×2 supercell and given deterministic random atomic displacements (seed 20261108, 0.02–0.05 Å).
- Relaxation: fused 16-branch HydraGNN/BranchWeightMLP, FP32 and FP64, ASE FIRE, 0.03 Å maximum step, hydrostatic variable-cell relaxation, 0.05 eV/Å generalized-force tolerance, up to 500 steps.
- All three cell lengths change together under hydrostatic strain; atomic positions relax simultaneously. The graph uses the frozen checkpoint's configured radius and 20-neighbor cap.
- Energies, forces, and stresses are predictions from the frozen learned potential, not DFT or experimental validation.
- The FP64 initial and final structure images are rendered with OVITO's OSPRay renderer using physically based sphere shading and direct/ambient lighting. Atom radii use a common linear scale of atomic number (r = 0.015 Å × Z); colors are element-specific and identified in each image legend.

## Paired results

| Structure | Formula | Atoms | FP32 status | FP32 steps | FP64 status | FP64 steps | Final cell lengths FP64 (Å) | Max cell delta (Å) | Energy delta (eV/atom) | Max-force delta (eV/Å) | RMS atom shift (Å) | Max atom shift (Å) | ΔV/V (FP64, %) | Final energy FP64 (eV/atom) | Max stress FP64 (eV/Å³) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C_diamond | C64 | 64 | max_steps_without_convergence | 500 | max_steps_without_convergence | 500 | 7.361, 7.361, 7.361 | 0.000334 | 0.000126 | 0.125 | 0.0409 | 0.0617 | +9.870 | -0.426401 | 0.0015435 |
| Si_diamond | Si64 | 64 | converged | 8 | converged | 8 | 10.858, 10.858, 10.858 | 4.46e-08 | 2.73e-06 | 1.82e-06 | 0.0244 | 0.0391 | -0.065 | 0.724009 | 0.0021598 |
| Ge_diamond | Ge64 | 64 | converged | 125 | converged | 125 | 11.681, 11.681, 11.681 | 5.54e-06 | 4.31e-06 | 0.00137 | 0.0365 | 0.0518 | +9.991 | -0.103100 | 0.0020834 |
| Al_fcc | Al32 | 32 | converged | 42 | converged | 42 | 7.944, 7.944, 7.944 | 3.44e-08 | 1.88e-08 | 6.76e-07 | 0.0329 | 0.0551 | -5.657 | 1.212177 | 0.0031475 |
| Cu_fcc | Cu32 | 32 | converged | 5 | converged | 5 | 7.229, 7.229, 7.229 | 5.59e-09 | 3.91e-07 | 2.53e-07 | 0.0063 | 0.0104 | -0.032 | -0.330976 | 0.0019342 |
| Ni_fcc | Ni32 | 32 | converged | 17 | converged | 17 | 7.042, 7.042, 7.042 | 8.74e-08 | 2.62e-07 | 2.5e-06 | 0.0307 | 0.0442 | -0.271 | -0.789667 | 0.0048602 |
| Fe_bcc | Fe16 | 16 | converged | 17 | converged | 17 | 5.672, 5.672, 5.672 | 2.23e-07 | 3.98e-08 | 1.65e-07 | 0.0467 | 0.0675 | -3.116 | -0.241883 | 0.0027143 |
| W_bcc | W16 | 16 | converged | 30 | converged | 30 | 6.260, 6.260, 6.260 | 1.1e-06 | 3.16e-07 | 2.31e-07 | 0.1559 | 0.2278 | -3.279 | 0.273042 | 0.0021287 |
| MgO_rocksalt | Mg32O32 | 64 | converged | 5 | converged | 5 | 8.423, 8.423, 8.423 | 9.34e-08 | 9.67e-08 | 7.01e-08 | 0.0041 | 0.0079 | -0.020 | -0.533653 | 0.0030832 |
| NaCl_rocksalt | Cl32Na32 | 64 | converged | 6 | converged | 6 | 11.280, 11.280, 11.280 | 9.61e-09 | 2.01e-07 | 2.52e-07 | 0.0070 | 0.0127 | -0.010 | 0.212124 | 0.00051899 |
| GaAs_zincblende | As32Ga32 | 64 | converged | 139 | converged | 139 | 11.718, 11.718, 11.718 | 5.87e-06 | 2.16e-06 | 6.04e-05 | 0.0388 | 0.0585 | +11.326 | -0.155099 | 0.0019922 |
| ZnO_wurtzite | O16Zn16 | 32 | converged | 0 | converged | 0 | 6.500, 6.500, 10.400 | 0 | 7.7e-08 | 4.41e-08 | 0.0000 | 0.0000 | +0.000 | 0.010937 | 0.00083896 |
| SrTiO3_perovskite | O24Sr8Ti8 | 40 | converged | 0 | converged | 0 | 7.810, 7.810, 7.810 | 0 | 1.46e-07 | 9.58e-08 | 0.0000 | 0.0000 | +0.000 | -0.034873 | 0.0011918 |

Both precisions converged for 12/13 structures.
The largest paired final-energy difference is 0.000126 eV/atom,
the largest final cell-length difference is 0.000334 Å, and
the largest difference in maximum atomic force is 0.125 eV/Å.
These compare the two precisions of this same model and are numerical-consistency
metrics, not errors relative to a physical reference.

Non-converged cases remain in the table and retain their trajectories and final
coordinates; a capped result must not be described as a relaxed structure.
Generalized force is the ASE cell-filter convergence quantity. In some cases,
the final maximum atomic force can therefore be slightly above the threshold.
Structures requiring zero optimizer steps already met the generalized-force
criterion at their perturbed starting geometry. For any capped case, the
reported final force is a residual at the step limit, not a converged force.

Non-converged runs:

- **C_diamond / fp32:** max_steps_without_convergence after 500 steps; final maximum atomic force 0.0778 eV/Å.
- **C_diamond / fp64:** max_steps_without_convergence after 500 steps; final maximum atomic force 0.2026 eV/Å.

## FP64 structure images

Each structure has an OVITO PNG for its perturbed starting geometry and its
FP64 endpoint under `figures/`. A final image is a converged learned-potential
result only when the corresponding FP64 status above is `converged`.
The companion `*_displacements.png` plots show the FP64 endpoint atoms and
initial-to-final displacement arrows, scaled for visibility and labeled with
their scale. Their titles report the actual RMS and maximum atomic shift;
fractional-coordinate matching separates internal atomic motion from
hydrostatic cell expansion.

## Outputs

- `comparison.json`: complete paired metrics and run locations.
- `<structure>/fp32/` and `<structure>/fp64/`: ASE trajectories, per-step CSVs, logs, summaries, and final structures.
- `figures/`: paired FP64 initial and final OVITO renderings, plus displacement plots.
