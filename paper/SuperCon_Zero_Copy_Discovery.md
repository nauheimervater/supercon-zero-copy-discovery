# Toward Zero-Copy ML/DFT Materials Screening on Unified-Memory APUs: Pipeline Architecture and First DFT Results (Work in Progress)

**Thomas Nauheimer**  
Independent Researcher · `nauheimer.t@gmail.com`  
November 2026 · Version 1.0.2 (technical note, work in progress)

---

## Status and Erratum

- **Withdrawn (since v1.0.1):** the per-step relaxation latency of 144.7 ms and all speedup figures. In v1.0.0 this value was a fixed default in an analytic model, not a measurement. A same-GPU benchmark (mapped vs. copied buffers on gfx1151) is pending.
- **Withdrawn (since v1.0.1):** the DiffCSP-SC loss-convergence table (no training log published) and the parametric $T_c$ estimate for BaBiTe.
- **Restored in v1.0.2:** the Quantum ESPRESSO SCF results for five generated candidates. v1.0.1 withdrew them by mistake; the raw input and output files are now included in `data/qe_top5/`, and the table below is regenerated from them by `supercon_discovery/qe_parse.py`. Correction: the calculations used **Quantum ESPRESSO v7.5** (v1.0.0 stated v7.3).

---

## Abstract

We describe the architecture of a closed-loop pipeline for superconductor candidate screening on a unified-memory workstation (AMD Ryzen AI Max+ 395 / Strix Halo, Radeon 8060S iGPU, 128 GB shared LPDDR5X, ROCm), and report first-principles SCF calculations for five generated candidate structures. The pipeline couples (1) Magpie-style compositional screening on the UCI SuperCon dataset, (2) a DiffCSP-style crystal-structure generator, (3) a C++/HIP relaxation layer in which coordinate and force buffers are allocated with `hipHostMalloc(..., hipHostMallocMapped)` and wrapped in place with `torch::from_blob`, and (4) Quantum ESPRESSO verification. All five candidates (CeIn$_3$, Sb$_2$Te$_3$, BaBiTe, AuPd$_{11}$, CuIn$_2$K) reached SCF convergence with PBE/PAW at 50/400 Ry. These are static single-point calculations on unrelaxed generated cells; they do not establish thermodynamic or dynamical stability, and the large residual stress of BaBiTe (+189 kbar) indicates that its cell is far from equilibrium.

---

## 1. Motivation

In copy-based hybrid ML/ab-initio loops (e.g. Python/ASE driving a TorchScript potential), coordinates and forces are copied between host and device buffers at every step. On unified-memory APUs such as Strix Halo, CPU and iGPU share one physical memory pool, so explicit copies are in principle avoidable. There is no PCIe link between CPU and iGPU on Strix Halo; any benefit of the zero-copy design comes from avoided copy/allocation/dispatch overhead and must be established by a direct benchmark, which is pending.

## 2. Components

### 2.1 Compositional screening
An 81-feature Magpie-style representation is implemented in `supercon_discovery/screening.py`. In the original pipeline run, a gradient-boosted regressor trained on 21,263 UCI SuperCon entries was logged with a test $R^2 \approx 0.93$; the training script and split are not yet part of this repository, so this number is preliminary.

### 2.2 Crystal-structure generation
A DiffCSP-style generator proposed candidate structures; the five structures evaluated below are included as CIF files in `data/qe_top5/`. Training logs are not yet published.

### 2.3 Zero-copy relaxation layer and core-repulsion filter
`cpp/mace_zero_copy_engine.cpp` and `cpp/verify_zero_copy.cpp` allocate mapped host memory and wrap it as tensors without copies. A periodic minimum-image filter separates atom pairs closer than $d_{\min} = 1.20$ Å before force evaluation (unit-tested).

### 2.4 DFT settings
Quantum ESPRESSO v7.5 `pw.x`, single-point SCF; PBE exchange–correlation; PAW pseudopotentials from pslibrary 1.0.0; $E_{\text{cut}}^{\text{wfc}} = 50$ Ry, $E_{\text{cut}}^{\rho} = 400$ Ry; $4 \times 4 \times 2$ Monkhorst–Pack mesh; Marzari–Vanderbilt cold smearing, 0.02 Ry; convergence threshold $10^{-6}$ Ry; forces and stress printed. Inputs, outputs and the run script are in `data/qe_top5/`.

## 3. DFT Results (regenerated from raw outputs)

| Candidate | Atoms/cell | $E_{\text{tot}}$ (Ry) | SCF iter. | Pressure (kbar) | $E_F$ (eV) | Total force (Ry/Bohr) | Valence e⁻ |
|---|---:|---:|---:|---:|---:|---:|---:|
| CeIn$_3$ | 4 | −1703.79 | 16 | −27.8 | 2.34 | 0.024 | 50 |
| Sb$_2$Te$_3$ | 5 | −1782.81 | 18 | −33.1 | 1.26 | 0.073 | 28 |
| BaBiTe | 3 | −1704.43 | 24 | +189.4 | 2.98 | 0.021 | 31 |
| AuPd$_{11}$ | 12 | −4415.78 | 22 | −20.8 | 6.15 | 0.676 | 121 |
| CuIn$_2$K | 4 | −1147.38 | 15 | −19.7 | 1.16 | 0.036 | 46 |

### 3.1 Interpretation and limitations
1. **Single-point SCF only.** Convergence shows that the electronic problem is well posed for these cells; it does not show that the structures are stable. No `vc-relax`, convex-hull or completed phonon (DFPT) calculation is reported. A DFPT run for BaBiTe was started but did not complete.
2. **Residual stress.** BaBiTe's +189 kbar and the −20 to −33 kbar of the others indicate that the generated cells are not at equilibrium; AuPd$_{11}$ additionally has large residual forces (0.68 Ry/Bohr).
3. **Metallicity.** Calculations used metallic smearing; a Fermi energy is reported for every smeared calculation and is not by itself evidence of metallic character. For BaBiTe, the odd number of valence electrons per cell in a non-spin-polarized calculation implies a partially filled band at the PBE level, but a DOS/band-structure check is pending for all candidates.
4. **Known compositions.** CeIn$_3$ and Sb$_2$Te$_3$ are known compounds (CeIn$_3$ is a known pressure-induced heavy-fermion superconductor); their appearance shows that the generator reproduces known stoichiometries, not that these specific generated cells match the experimental structures.
5. **No $T_c$ claims.** No electron–phonon coupling has been computed; no critical temperature is claimed for any candidate.

## 4. Planned work

1. Same-GPU benchmark on gfx1151: mapped vs. explicitly copied buffers for a MACE relaxation step, with raw timing logs.
2. `vc-relax`, DOS and DFPT for the candidates; convex-hull comparison against competing phases.
3. Publication of the screening training script and generator training logs.

## References

1. A. Merchant et al., "Scaling deep learning for materials discovery," *Nature* 624, 80–85 (2023).
2. R. Jiao et al., "Crystal structure prediction by joint equivariant diffusion," *NeurIPS* (2023).
3. I. Batatia et al., "MACE: Higher order equivariant message passing neural networks for fast and accurate force fields," *NeurIPS* (2022).
4. P. Giannozzi et al., "QUANTUM ESPRESSO," *J. Phys.: Condens. Matter* 21, 395502 (2009); *J. Phys.: Condens. Matter* 29, 465901 (2017).
5. A. Dal Corso, "Pseudopotentials periodic table: From H to Pu," *Comput. Mater. Sci.* 95, 337 (2014).
