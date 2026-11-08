# Off-equilibrium crystal relaxation stress test

Study status: complete; 52 recorded runs.

## Setup

- Structures: 2×2×2 supercells. All 13 periodic crystal prototypes, each compressed and expanded by 5%.
- All atoms receive seeded, random-direction displacements of 0.10–0.15 Å; compressed and expanded cases use the same Cartesian perturbation vectors.
- Paired FP32/FP64 FIRE relaxations from identical per-case starting coordinates, relaxing atomic positions and all cell lengths and shape together.
- Maximum optimizer step: 0.03 Å; generalized-force threshold: 0.05 eV/Å; step cap: 500.
- Distortion seed: 20261109.

## Results

| Structure | Precision | Status | Steps | ΔE (eV/atom) | Initial→final max force (eV/Å) | RMS atom shift (Å) | Max atom shift (Å) | Δcell (%) | ΔV (%) | Final generalized force (eV/Å) | Final max stress (eV/Å³) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C_diamond_compressed | fp32 | max_steps_without_convergence | 500 | -0.171819 | 5.3156 → 2.3560 | 0.5473 | 0.9276 | -1.075 | -3.193 | 2.33306 | 0.0381164 |
| C_diamond_compressed | fp64 | max_steps_without_convergence | 500 | -0.196659 | 5.3156 → 1.9419 | 0.5755 | 1.1580 | -1.036 | -3.078 | 1.92607 | 0.0287197 |
| C_diamond_expanded | fp32 | max_steps_without_convergence | 500 | -0.097944 | 1.8434 → 0.1263 | 0.1216 | 0.1639 | -1.691 | -4.989 | 0.12417 | 0.00392534 |
| C_diamond_expanded | fp64 | max_steps_without_convergence | 500 | -0.097286 | 1.8434 → 0.1253 | 0.1223 | 0.1676 | -1.697 | -5.006 | 0.12316 | 0.00237111 |
| Si_diamond_compressed | fp32 | converged | 142 | -0.033358 | 1.0005 → 0.0271 | 0.1477 | 0.2932 | +2.995 | +9.257 | 0.04999 | 0.00274666 |
| Si_diamond_compressed | fp64 | converged | 142 | -0.033358 | 1.0005 → 0.0270 | 0.1477 | 0.2931 | +2.995 | +9.257 | 0.04995 | 0.00274413 |
| Si_diamond_expanded | fp32 | converged | 183 | -0.027310 | 0.3839 → 0.0317 | 0.1190 | 0.1494 | -4.162 | -11.974 | 0.04991 | 0.00230479 |
| Si_diamond_expanded | fp64 | converged | 183 | -0.027299 | 0.3839 → 0.0316 | 0.1193 | 0.1485 | -4.162 | -11.974 | 0.04943 | 0.00229931 |
| Ge_diamond_compressed | fp32 | converged | 339 | -0.114381 | 1.4158 → 0.0230 | 0.1403 | 0.1646 | +8.623 | +28.165 | 0.04886 | 0.00213224 |
| Ge_diamond_compressed | fp64 | converged | 339 | -0.114384 | 1.4158 → 0.0220 | 0.1403 | 0.1645 | +8.623 | +28.165 | 0.04896 | 0.00213647 |
| Ge_diamond_expanded | fp32 | converged | 28 | -0.009517 | 0.3689 → 0.0472 | 0.1002 | 0.1488 | -0.010 | -0.030 | 0.04719 | 0.000676134 |
| Ge_diamond_expanded | fp64 | converged | 28 | -0.009517 | 0.3689 → 0.0472 | 0.1002 | 0.1488 | -0.010 | -0.030 | 0.04719 | 0.000676143 |
| Al_fcc_compressed | fp32 | converged | 50 | -0.016528 | 0.3455 → 0.0225 | 0.1186 | 0.1643 | -1.722 | -5.077 | 0.03919 | 0.00284789 |
| Al_fcc_compressed | fp64 | converged | 50 | -0.016528 | 0.3455 → 0.0225 | 0.1186 | 0.1643 | -1.722 | -5.077 | 0.03918 | 0.00284771 |
| Al_fcc_expanded | fp32 | converged | 132 | -0.039143 | 0.1140 → 0.0180 | 0.1152 | 0.1637 | -6.668 | -18.700 | 0.04696 | 0.0027953 |
| Al_fcc_expanded | fp64 | converged | 132 | -0.039142 | 0.1140 → 0.0183 | 0.1152 | 0.1636 | -6.668 | -18.700 | 0.04652 | 0.0027692 |
| Cu_fcc_compressed | fp32 | converged | 70 | -0.027124 | 0.4517 → 0.0148 | 0.1150 | 0.1383 | +3.077 | +9.516 | 0.04888 | 0.00454332 |
| Cu_fcc_compressed | fp64 | converged | 70 | -0.027124 | 0.4517 → 0.0148 | 0.1150 | 0.1383 | +3.077 | +9.516 | 0.04888 | 0.00454335 |
| Cu_fcc_expanded | fp32 | converged | 86 | -0.017394 | 0.1382 → 0.0047 | 0.1155 | 0.1383 | -3.947 | -11.380 | 0.04988 | 0.00395045 |
| Cu_fcc_expanded | fp64 | converged | 86 | -0.017394 | 0.1382 → 0.0047 | 0.1155 | 0.1383 | -3.947 | -11.380 | 0.04988 | 0.00395045 |
| Ni_fcc_compressed | fp32 | converged | 107 | -0.220590 | 3.0388 → 0.0423 | 0.1213 | 0.1609 | +5.023 | +15.836 | 0.04445 | 0.00207822 |
| Ni_fcc_compressed | fp64 | converged | 107 | -0.220591 | 3.0388 → 0.0423 | 0.1213 | 0.1609 | +5.023 | +15.836 | 0.04445 | 0.00207822 |
| Ni_fcc_expanded | fp32 | converged | 100 | -0.136341 | 0.8335 → 0.0451 | 0.1244 | 0.1473 | -4.922 | -14.052 | 0.04286 | 0.00338832 |
| Ni_fcc_expanded | fp64 | converged | 100 | -0.136340 | 0.8335 → 0.0451 | 0.1244 | 0.1473 | -4.922 | -14.052 | 0.04286 | 0.00338823 |
| Fe_bcc_compressed | fp32 | converged | 52 | -0.053672 | 0.3418 → 0.0520 | 0.0990 | 0.1663 | -4.690 | -13.421 | 0.04956 | 0.00121954 |
| Fe_bcc_compressed | fp64 | converged | 52 | -0.053672 | 0.3418 → 0.0520 | 0.0990 | 0.1663 | -4.690 | -13.421 | 0.04956 | 0.00121977 |
| Fe_bcc_expanded | fp32 | converged | 61 | -0.020457 | 0.1124 → 0.0414 | 0.1159 | 0.2398 | -5.545 | -15.730 | 0.03912 | 0.00230285 |
| Fe_bcc_expanded | fp64 | converged | 61 | -0.020457 | 0.1124 → 0.0414 | 0.1159 | 0.2398 | -5.545 | -15.730 | 0.03912 | 0.00230274 |
| W_bcc_compressed | fp32 | converged | 30 | -0.010596 | 0.2396 → 0.0456 | 0.1374 | 0.1812 | -1.127 | -3.345 | 0.04509 | 0.00111145 |
| W_bcc_compressed | fp64 | converged | 30 | -0.010597 | 0.2396 → 0.0456 | 0.1374 | 0.1812 | -1.127 | -3.345 | 0.04509 | 0.00111123 |
| W_bcc_expanded | fp32 | converged | 29 | -0.006779 | 0.0702 → 0.0485 | 0.1597 | 0.2686 | -0.261 | -0.780 | 0.04848 | 0.00169379 |
| W_bcc_expanded | fp64 | converged | 29 | -0.006779 | 0.0702 → 0.0485 | 0.1597 | 0.2686 | -0.261 | -0.780 | 0.04848 | 0.00169383 |
| MgO_rocksalt_compressed | fp32 | converged | 83 | -0.023295 | 0.7326 → 0.0375 | 0.1249 | 0.1608 | +1.483 | +4.514 | 0.04730 | 0.00573214 |
| MgO_rocksalt_compressed | fp64 | converged | 83 | -0.023295 | 0.7326 → 0.0375 | 0.1249 | 0.1608 | +1.483 | +4.514 | 0.04730 | 0.00573216 |
| MgO_rocksalt_expanded | fp32 | converged | 176 | -0.018383 | 0.1718 → 0.0354 | 0.1183 | 0.1497 | -4.061 | -11.695 | 0.04977 | 0.00500037 |
| MgO_rocksalt_expanded | fp64 | converged | 183 | -0.018545 | 0.1718 → 0.0324 | 0.1182 | 0.1499 | -4.249 | -12.212 | 0.04994 | 0.00503744 |
| NaCl_rocksalt_compressed | fp32 | converged | 170 | -0.036121 | 0.3703 → 0.0095 | 0.1306 | 0.1810 | +4.137 | +12.932 | 0.04993 | 0.00239388 |
| NaCl_rocksalt_compressed | fp64 | converged | 170 | -0.036121 | 0.3703 → 0.0095 | 0.1306 | 0.1810 | +4.137 | +12.932 | 0.04993 | 0.00239389 |
| NaCl_rocksalt_expanded | fp32 | converged | 166 | -0.012654 | 0.0860 → 0.0122 | 0.1165 | 0.1563 | -3.932 | -11.338 | 0.04948 | 0.00206482 |
| NaCl_rocksalt_expanded | fp64 | converged | 166 | -0.012654 | 0.0860 → 0.0126 | 0.1165 | 0.1564 | -3.932 | -11.338 | 0.04947 | 0.00206466 |
| GaAs_zincblende_compressed | fp32 | converged | 352 | -0.175251 | 1.6797 → 0.0368 | 0.1349 | 0.1788 | +9.057 | +29.704 | 0.04976 | 0.00215825 |
| GaAs_zincblende_compressed | fp64 | converged | 352 | -0.175252 | 1.6797 → 0.0372 | 0.1350 | 0.1793 | +9.057 | +29.704 | 0.04986 | 0.00216227 |
| GaAs_zincblende_expanded | fp32 | converged | 37 | -0.009858 | 0.3637 → 0.0433 | 0.1186 | 0.1730 | -0.179 | -0.537 | 0.04843 | 0.00185929 |
| GaAs_zincblende_expanded | fp64 | converged | 37 | -0.009858 | 0.3637 → 0.0433 | 0.1186 | 0.1730 | -0.179 | -0.537 | 0.04843 | 0.00185928 |
| ZnO_wurtzite_compressed | fp32 | converged | 51 | -0.018709 | 1.0726 → 0.0127 | 0.1087 | 0.1663 | +1.878 | +5.862 | 0.04931 | 0.0046624 |
| ZnO_wurtzite_compressed | fp64 | converged | 51 | -0.018709 | 1.0726 → 0.0127 | 0.1087 | 0.1663 | +1.878 | +5.862 | 0.04931 | 0.0046624 |
| ZnO_wurtzite_expanded | fp32 | converged | 0 | +0.000000 | 0.0315 → 0.0315 | 0.0000 | 0.0000 | +0.000 | +0.000 | 0.03152 | 0.00205706 |
| ZnO_wurtzite_expanded | fp64 | converged | 0 | +0.000000 | 0.0315 → 0.0315 | 0.0000 | 0.0000 | +0.000 | +0.000 | 0.03152 | 0.00205706 |
| SrTiO3_perovskite_compressed | fp32 | converged | 39 | -0.027789 | 5.1761 → 0.0292 | 0.0879 | 0.1354 | +0.840 | +2.542 | 0.04950 | 0.00476606 |
| SrTiO3_perovskite_compressed | fp64 | converged | 39 | -0.027789 | 5.1761 → 0.0292 | 0.0879 | 0.1354 | +0.840 | +2.542 | 0.04950 | 0.00476608 |
| SrTiO3_perovskite_expanded | fp32 | converged | 0 | +0.000000 | 0.0318 → 0.0318 | 0.0000 | 0.0000 | +0.000 | +0.000 | 0.03180 | 7.67311e-05 |
| SrTiO3_perovskite_expanded | fp64 | converged | 0 | +0.000000 | 0.0318 → 0.0318 | 0.0000 | 0.0000 | +0.000 | +0.000 | 0.03180 | 7.67274e-05 |


## Assessment

48/52 runs met the generalized-force threshold. Non-converged cases: C_diamond_compressed/fp32, C_diamond_compressed/fp64, C_diamond_expanded/fp32, C_diamond_expanded/fp64.

Zero-step cases: ZnO_wurtzite_expanded/fp32, ZnO_wurtzite_expanded/fp64, SrTiO3_perovskite_expanded/fp32, SrTiO3_perovskite_expanded/fp64. These met the stopping criterion initially; no structural improvement occurred.

Among converged precision pairs, the largest final energy difference is 0.00016232 eV/atom and the largest cell-length difference is 0.0166178 Å. Numerical agreement is distinct from physical accuracy or independence from the initial geometry.

Compare compressed and expanded endpoints below before claiming starting-point-independent lattice recovery. Different endpoints may reflect stopping tolerance or different basins; this study alone does not distinguish them.

## FP32 versus FP64 endpoints

| Case | Both converged | Energy delta (eV/atom) | Max length delta (Å) | Max angle delta (degrees) |
|---|---|---|---|---|
| C_diamond_compressed | False | 0.0248405 | 0.0232942 | 0.288952 |
| C_diamond_expanded | False | 0.000658711 | 0.00125494 | 0.00506505 |
| Si_diamond_compressed | True | 2.74725e-06 | 8.62996e-06 | 5.06949e-05 |
| Si_diamond_expanded | True | 1.45279e-05 | 0.000116624 | 0.00551523 |
| Ge_diamond_compressed | True | 5.87858e-07 | 1.28628e-05 | 0.000225848 |
| Ge_diamond_expanded | True | 1.00603e-06 | 2.78171e-07 | 1.3494e-06 |
| Al_fcc_compressed | True | 3.16685e-07 | 2.63615e-08 | 5.65311e-08 |
| Al_fcc_expanded | True | 1.26648e-06 | 1.0639e-05 | 0.000162953 |
| Cu_fcc_compressed | True | 1.16275e-07 | 2.03545e-08 | 1.10443e-07 |
| Cu_fcc_expanded | True | 5.91191e-07 | 3.14411e-06 | 1.50861e-05 |
| Ni_fcc_compressed | True | 8.21405e-07 | 1.34894e-07 | 6.18744e-08 |
| Ni_fcc_expanded | True | 1.16039e-06 | 2.50015e-08 | 1.23158e-07 |
| Fe_bcc_compressed | True | 4.2059e-07 | 5.98454e-08 | 1.19562e-07 |
| Fe_bcc_expanded | True | 4.37354e-07 | 1.24194e-07 | 5.32924e-07 |
| W_bcc_compressed | True | 2.62249e-07 | 4.23794e-07 | 2.62962e-07 |
| W_bcc_expanded | True | 4.34009e-07 | 1.35779e-06 | 4.10672e-07 |
| MgO_rocksalt_compressed | True | 1.6496e-08 | 7.0419e-08 | 2.73542e-08 |
| MgO_rocksalt_expanded | True | 0.00016232 | 0.0166178 | 0.00105859 |
| NaCl_rocksalt_compressed | True | 5.32246e-08 | 3.2426e-08 | 3.27002e-07 |
| NaCl_rocksalt_expanded | True | 6.15963e-08 | 1.74069e-07 | 8.261e-06 |
| GaAs_zincblende_compressed | True | 1.0388e-06 | 9.37254e-05 | 0.000416409 |
| GaAs_zincblende_expanded | True | 2.2343e-08 | 3.77413e-08 | 1.65691e-07 |
| ZnO_wurtzite_compressed | True | 5.57778e-08 | 6.42257e-08 | 5.43459e-08 |
| ZnO_wurtzite_expanded | True | 2.40169e-07 | 0 | 0 |
| SrTiO3_perovskite_compressed | True | 2.66563e-07 | 3.48608e-08 | 2.53028e-08 |
| SrTiO3_perovskite_expanded | True | 7.58709e-08 | 0 | 0 |

## Compressed versus expanded endpoints

| Prototype | Precision | Both converged | Final energy delta (eV/atom) | Max final length delta (Å) | Max final angle delta (degrees) |
|---|---|---|---|---|---|
| C_diamond | fp32 | False | 0.0185803 | 0.663386 | 0.3429 |
| C_diamond | fp64 | False | 0.00691894 | 0.666407 | 0.135855 |
| Si_diamond | fp32 | True | 7.27177e-05 | 0.304266 | 0.0498517 |
| Si_diamond | fp64 | True | 6.0937e-05 | 0.30415 | 0.055359 |
| Ge_diamond | fp32 | True | 1.63019e-05 | 0.204458 | 0.0164939 |
| Ge_diamond | fp64 | True | 1.58837e-05 | 0.204446 | 0.0166164 |
| Al_fcc | fp32 | True | 0.00840759 | 0.376624 | 0.00812917 |
| Al_fcc | fp64 | True | 0.00840918 | 0.376631 | 0.00829214 |
| Cu_fcc | fp32 | True | 0.000154942 | 0.212749 | 0.0509839 |
| Cu_fcc | fp64 | True | 0.000155417 | 0.212749 | 0.0509757 |
| Ni_fcc | fp32 | True | 0.00129133 | 0.00849381 | 0.155965 |
| Ni_fcc | fp64 | True | 0.00129167 | 0.0084937 | 0.155965 |
| Fe_bcc | fp32 | True | 0.0591675 | 0.49978 | 0.023103 |
| Fe_bcc | fp64 | True | 0.0591675 | 0.49978 | 0.023103 |
| W_bcc | fp32 | True | 0.0168922 | 0.689457 | 0.121946 |
| W_bcc | fp64 | True | 0.016892 | 0.689455 | 0.121947 |
| MgO_rocksalt | fp32 | True | 8.41618e-05 | 0.364688 | 0.00624743 |
| MgO_rocksalt | fp64 | True | 0.000246498 | 0.34807 | 0.0057322 |
| NaCl_rocksalt | fp32 | True | 0.000319004 | 0.220145 | 0.018651 |
| NaCl_rocksalt | fp64 | True | 0.000318996 | 0.220145 | 0.0186565 |
| GaAs_zincblende | fp32 | True | 0.000282198 | 0.138573 | 0.156704 |
| GaAs_zincblende | fp64 | True | 0.000283259 | 0.138626 | 0.157036 |
| ZnO_wurtzite | fp32 | True | 0.00193484 | 0.869285 | 0.0250338 |
| ZnO_wurtzite | fp64 | True | 0.00193466 | 0.869285 | 0.0250338 |
| SrTiO3_perovskite | fp32 | True | 0.00221225 | 0.720182 | 0.00310722 |
| SrTiO3_perovskite | fp64 | True | 0.00221244 | 0.720182 | 0.00310721 |

Energy change and force reduction are predictions of the frozen learned potential.
This is a robustness/stress test, not evidence that the final geometry agrees
with DFT or experiment. The initial distortion may be outside the model's
training distribution. A non-converged endpoint is not a relaxed structure.

## Figures

- `figures/<structure>_paired.png`: initial and FP64 final geometries rendered at the same coordinate scale, with cell outlines and element legends.
- `figures/<structure>_initial.png` and `figures/<structure>_final.png`: shaded OVITO renders.
- `figures/<structure>_displacements.png`: internal atomic motions with explicitly labeled arrow magnification.
- `figures/<structure>_optimization.png`: paired precision energy, force, volume, and cell-angle histories.
- `<structure>/<precision>/`: ASE trajectories, per-step force/energy/cell records, logs, summaries, and final structures.

Internal displacement removes affine cell deformation using fractional-coordinate
matching and the final-cell minimum image. Δcell is the change in mean cell-vector
length, not a measure of shear; full vectors and angles are in `comparison.json`.
No symmetry constraints or external pressure are imposed. Reaching the force
threshold does not establish recovery of the prototype or the same minimum
from compressed and expanded starts.
ASE scales the cell degrees of freedom by atom count: the generalized-force
threshold is not a direct requirement that every stress component vanish.
The JSON also records minimum interatomic distance and initial/final internal
offset from the ideal prototype as structural diagnostics, not physical-reference
validation.
