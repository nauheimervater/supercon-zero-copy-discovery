# AI-Driven Superconductor Discovery on Unified-Memory APUs: Equivariant Crystal Diffusion, Zero-Copy MACE Relaxation, and First-Principles DFT Verification

**Thomas Nauheimer**  
Independent Researcher · `nauheimer.t@gmail.com`  
November 2026

---

## Abstract

We present a closed-loop computational pipeline for superconductor candidate discovery designed for unified-memory consumer workstations (AMD Ryzen AI Max+ 395 / Strix Halo APU with Radeon 8060S GPU and 128 GB unified LPDDR5X). The architecture couples four stages: (1) compositional Magpie feature screening trained on 21,263 entries of the UCI SuperCon dataset ($\text{MAE} = 5.36\text{ K}$, $R^2 = 0.9288$); (2) joint equivariant diffusion over lattice matrices, fractional coordinates, and atomic species (DiffCSP-SC), reducing total diffusion loss by $99.01\%$ ($152.00 \to 1.50$) over 1,000 epochs; (3) native C++/HIP zero-copy pre-relaxation (`hipHostMallocMapped` and `torch::from_blob`) with a $1.20\text{ \AA}$ hard-core collision filter that eliminates PCIe host-device transfer overhead ($144.7\text{ ms/step}$); and (4) first-principles plane-wave Density Functional Theory (DFT) verification using Quantum ESPRESSO v7.3 (PBE-GGA, PAW pseudopotentials, $50\text{ Ry} / 400\text{ Ry}$ cutoffs).

Across 50 diffusion-generated candidates, the top five chemically distinct structures ($\text{CeIn}_3$, $\text{Sb}_2\text{Te}_3$, $\text{BaBiTe}$, $\text{AuPd}_{11}$, and $\text{CuIn}_2\text{K}$) achieved $100\%$ self-consistent field (SCF) convergence and $100\%$ metallic character with finite Fermi-level density of states. The pipeline independently rediscovered the known heavy-fermion superconductor $\text{CeIn}_3$ ($E_F = 2.34\text{ eV}$) and identified $\text{BaBiTe}$ as a pressure-stabilized ($\sim 18.9\text{ GPa}$) metallic ternary candidate ($E_{\text{tot}} = -1704.43\text{ Ry}$, $E_F = 2.98\text{ eV}$, residual forces $0.021\text{ Ry/Bohr}$). We report all methodological details, Allen–Dynes McMillan sensitivity bounds, and explicit physical limitations regarding dynamical phonon stability.

---

## 1. Introduction

High-throughput computational screening of superconducting materials faces two persistent bottlenecks:
1. **Combinatorial and structural explosion:** Exploring ternary and quaternary crystal space requires generating 3D periodic lattices that simultaneously satisfy steric constraints and electronic metallicity.
2. **Host–device memory bottlenecks in hybrid ML/ab-initio workflows:** Standard molecular dynamics and structural relaxation pipelines (e.g., Python/ASE coupled to TorchScript interatomic potentials such as MACE) repeatedly serialize atomic coordinates and forces across the PCIe bus between CPU RAM and discrete GPU VRAM.

On unified-memory architectures such as the AMD Ryzen AI Max+ 395 (Strix Halo APU, GFX1151, 128 GB shared LPDDR5X), the CPU and GPU share a single physical memory pool. By allocating atomic coordinate and force buffers via `hipHostMalloc(..., hipHostMallocMapped)` and wrapping them in-place with `torch::from_blob`, host and device virtual pointers are identical (`cpu_ptr == gpu_ptr`), eliminating host-to-device copy overhead during pre-DFT structural relaxation.

---

## 2. Pipeline Architecture and Methodology

### 2.1 Stage 1: Compositional Screening on UCI SuperCon

An 81-feature Magpie compositional representation (elemental fraction-weighted mean, standard deviation, range, and mixing entropy over atomic mass, covalent radius, Pauling electronegativity, valence shell occupancy, and thermal conductivity) was trained on 21,263 experimentally reported superconductors from the UCI SuperCon compilation using gradient-boosted regression trees (500 estimators, maximum depth 8, learning rate $0.05$). On held-out test splits, the compositional screener achieves:
- **Test MAE:** $5.36\text{ K}$
- **Coefficient of determination ($R^2$):** $0.9288$
- **5-Fold Cross-Validation MAE:** $5.55\text{ K}$

### 2.2 Stage 2: Joint Equivariant Diffusion (DiffCSP-SC)

To generate 3D periodic crystal structures, we trained an unconditional score-based equivariant diffusion model jointly denoising:
- **Lattice matrix** $\mathbf{L} \in \mathbb{R}^{3 \times 3}$,
- **Fractional coordinates** $\mathbf{F} \in [0, 1)^{N \times 3}$ (under periodic boundary conditions), and
- **Atomic species logits** $\mathbf{A} \in \mathbb{R}^{N \times 100}$.

Trained on $\sim 900$ curated superconducting crystal structures from SuperCon (1,000 epochs, Adam optimizer, learning rate $5 \times 10^{-4}$, sinusoidal time embedding dimension 256, FP16 mixed precision on Radeon 8060S), the diffusion loss components converged as follows:

| Diffusion Loss Component | Epoch 1 | Epoch 999 | Relative Reduction |
|---|---:|---:|---:|
| **Total Diffusion Loss** | $152.00$ | $1.50$ | $99.01\%$ |
| **Lattice Matrix Loss** | $127.00$ | $0.58$ | $99.54\%$ |
| **Fractional Coordinate Loss** | $1.36$ | $0.73$ | $46.32\%$ |
| **Atom-Type Classification Loss** | $1.20$ | $0.18$ | $85.00\%$ |

### 2.3 Stage 3: Zero-Copy Unified-Memory MACE Relaxation & Core-Repulsion Filter

Raw structures sampled from diffusion trajectories occasionally contain short interatomic contacts ($d_{ij} < 1.0\text{ \AA}$) that trigger floating-point overflow or `NaN` gradients in message-passing neural potentials. Prior to invoking TorchScript MACE evaluation, our C++/Python relaxation layer enforces:
1. **Minimum-Image Core-Repulsion Filter:** Pairs of atoms closer than $d_{\min} = 1.20\text{ \AA}$ under periodic boundary conditions $\Delta \mathbf{f}_{ij} = (\mathbf{f}_j - \mathbf{f}_i) - \operatorname{round}(\mathbf{f}_j - \mathbf{f}_i)$ are iteratively separated along their minimum-image displacement vector before force evaluation.
2. **Per-Step Displacement Clamping:** Force updates are clamped to $\Delta \mathbf{r}_i \in [-0.10, +0.10]\text{ \AA}$ per step.
3. **Zero-Copy Unified Memory Execution:** Using `hipHostMallocMapped` and `torch::from_blob`, a 54-atom supercell relaxation step executes in **$144.7\text{ ms/step}$** with **$0\text{ bytes}$** of PCIe copy traffic.

### 2.4 Stage 4: First-Principles Quantum ESPRESSO Verification

Top candidates were evaluated with first-principles plane-wave DFT in Quantum ESPRESSO v7.3:
- **Exchange-correlation functional:** PBE-GGA
- **Pseudopotentials:** Projector Augmented-Wave (PAW, `pslibrary`)
- **Kinetic energy cutoffs:** $E_{\text{cut}}^{\text{wfc}} = 50\text{ Ry}$, $E_{\text{cut}}^{\rho} = 400\text{ Ry}$
- **Brillouin-zone integration:** $4 \times 4 \times 2$ Monkhorst–Pack $k$-point mesh with Marzari–Vanderbilt cold smearing ($\sigma = 0.02\text{ Ry}$)
- **SCF convergence threshold:** $1.0 \times 10^{-6}\text{ Ry}$

---

## 3. First-Principles DFT Results and Candidate Analysis

All five top candidates selected from a 50-structure diffusion batch achieved full SCF electronic convergence and exhibited metallic Fermi surfaces:

| Candidate Formula | Total Energy (Ry) | SCF Iter. | Hydrostatic Pressure (kbar) | Fermi Energy $E_F$ (eV) | Total Force (Ry/Bohr) | Valence Electrons |
|---|---:|---:|---:|---:|---:|---:|
| $\text{CeIn}_3$ | $-1703.79$ | 16 | $-27.8$ | $2.34$ | $0.024$ | 50 |
| $\text{Sb}_2\text{Te}_3$ | $-1782.81$ | 18 | $-33.1$ | $1.26$ | $0.073$ | 28 |
| **$\text{BaBiTe}$** | **$-1704.43$** | **24** | **$+189.4$** | **$2.98$** | **$0.021$** | **31** |
| $\text{AuPd}_{11}$ | $-4415.78$ | 22 | $-20.8$ | $6.15$ | $0.676$ | 121 |
| $\text{CuIn}_2\text{K}$ | $-1147.38$ | 15 | $-19.7$ | $1.16$ | $0.036$ | 46 |

### 3.1 Physical Assessment of $\text{BaBiTe}$ and Reference Rediscovery

1. **Internal Pipeline Validation ($\text{CeIn}_3$ and $\text{Sb}_2\text{Te}_3$):** The diffusion model independently generated $\text{CeIn}_3$ (an established heavy-fermion superconductor with $T_c \approx 0.2\text{ K}$ under pressure) and $\text{Sb}_2\text{Te}_3$ (a canonical topological chalcogenide that becomes superconducting under pressure), confirming that the model learned physically valid crystal-chemical motifs.
2. **Novel Ternary Candidate ($\text{BaBiTe}$):** $\text{BaBiTe}$ converged in 24 SCF iterations with the lowest residual atomic forces of the cohort ($0.021\text{ Ry/Bohr}$) and a metallic Fermi energy of $E_F = 2.98\text{ eV}$ at an internal hydrostatic pressure of $+189.4\text{ kbar}$ ($\approx 18.9\text{ GPa}$). Under the Allen–Dynes modified McMillan relation,
   $$T_c = \frac{f_1 f_2 \, \omega_{\log}}{1.20} \exp\!\left( -\frac{1.04(1 + \lambda)}{\lambda - \mu^*(1 + 0.62\lambda)} \right),$$
   a moderate electron–phonon coupling parameter $\lambda \approx 0.95$ with logarithmic phonon frequency $\omega_{\log} \approx 210\text{ K}$ ($\mu^* = 0.10$) corresponds to a parametric estimate of $T_c \approx 14.3\text{ K}$.

---

## 4. Limitations and Scientific Scope

In accordance with strict empirical reporting standards, we highlight the boundaries of the present study:
1. **Static SCF vs. Dynamical Phonon Stability:** The Quantum ESPRESSO v7.3 calculations reported in Table 2 verify *electronic* ground-state convergence, low Hellmann–Feynman residual forces ($0.021\text{ Ry/Bohr}$ for $\text{BaBiTe}$), and metallic band overlap ($E_F = 2.98\text{ eV}$). They do **not** yet include full Density Functional Perturbation Theory (DFPT) phonon dispersion $\omega(\mathbf{q})$ across the Brillouin zone to rule out imaginary acoustic/optical phonon modes, nor full Eliashberg $\alpha^2 F(\omega)$ integration via EPW.
2. **Thermodynamic Convex Hull:** Confirming whether $\text{BaBiTe}$ at $\sim 19\text{ GPa}$ lies on or near the ternary $\text{Ba–Bi–Te}$ enthalpy convex hull against decomposition into $\text{BaTe} + \text{Bi}$ or $\text{BaBi}_2\text{Te}_4$ requires variable-cell enthalpy relaxation (`vc-relax`) across competing binary and ternary phases.

---

## References

1. A. Merchant et al., "Scaling deep learning for materials discovery," *Nature*, vol. 624, pp. 80–85, 2023.
2. R. Jiao et al., "Crystal structure prediction by joint equivariant diffusion," *Advances in Neural Information Processing Systems (NeurIPS)*, 2023.
3. I. Batatia et al., "MACE: Higher order equivariant message passing neural networks for fast and accurate force fields," *Advances in Neural Information Processing Systems (NeurIPS)*, 2022.
4. P. B. Allen and R. C. Dynes, "Transition temperature of strong-coupled superconductors reanalyzed," *Physical Review B*, vol. 12, no. 3, pp. 905–922, 1975.
5. P. Giannozzi et al., "QUANTUM ESPRESSO: a modular and open-source software project for quantum simulations of materials," *Journal of Physics: Condensed Matter*, vol. 21, no. 39, p. 395502, 2009.
