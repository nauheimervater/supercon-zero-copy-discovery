# AI-Driven Superconductor Discovery on Unified-Memory APUs: Equivariant Crystal Diffusion, Zero-Copy MACE Relaxation, and First-Principles DFT Verification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![C++17 / HIP](https://img.shields.io/badge/C%2B%2B17-ROCm%20%2F%20HIP-ee4c2c.svg)](cpp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper: PDF](https://img.shields.io/badge/Paper-PDF-red.svg)](paper/SuperCon_Zero_Copy_Discovery.pdf)

> **AI-Driven Superconductor Discovery on Unified-Memory APUs: Equivariant Crystal Diffusion, Zero-Copy MACE Relaxation, and First-Principles DFT Verification**  
> Thomas Nauheimer (`nauheimer.t@gmail.com`), November 2026.  
> Paper: [`paper/SuperCon_Zero_Copy_Discovery.pdf`](paper/SuperCon_Zero_Copy_Discovery.pdf) · Source: [`paper/SuperCon_Zero_Copy_Discovery.md`](paper/SuperCon_Zero_Copy_Discovery.md) · LaTeX: [`paper/SuperCon_Zero_Copy_Discovery.tex`](paper/SuperCon_Zero_Copy_Discovery.tex)

---

## Overview

This repository provides the open-source verification library, C++/HIP unified-memory zero-copy kernels, Allen–Dynes modified McMillan $T_c$ models, and Quantum ESPRESSO v7.3 PBE-GGA verification data for superconductor candidate discovery on unified-memory consumer workstations (AMD Ryzen AI Max+ 395 / Strix Halo APU with Radeon 8060S GPU and 128 GB shared LPDDR5X).

### Key Findings & Pipeline Results
1. **Stage 1 — Compositional Screening (UCI SuperCon):** 81 Magpie compositional features trained on 21,263 superconductors achieve **$\text{MAE} = 5.36\text{ K}$** and **$R^2 = 0.9288$** (5-fold CV $\text{MAE} = 5.55\text{ K}$).
2. **Stage 2 — Joint Equivariant Crystal Diffusion (DiffCSP-SC):** Jointly denoises lattice matrices $\mathbf{L}$, fractional coordinates $\mathbf{F}$, and atomic species $\mathbf{A}$ over 1,000 epochs ($99.01\%$ total loss reduction from $152.00 \to 1.50$).
3. **Stage 3 — Zero-Copy Unified-Memory MACE Relaxation (`cpp/`):** Using `hipHostMallocMapped` and `torch::from_blob`, host and device virtual pointers are identical (`cpu_ptr == gpu_ptr`), executing 54-atom TorchScript MACE relaxation steps in **$144.7\text{ ms/step}$** with **$0\text{ bytes}$** of PCIe copy traffic and a pre-MACE **$1.20\text{ \AA}$ minimum-image collision filter**.
4. **Stage 4 — First-Principles Quantum ESPRESSO v7.3 Verification:** All five top candidates ($\text{CeIn}_3$, $\text{Sb}_2\text{Te}_3$, $\text{BaBiTe}$, $\text{AuPd}_{11}$, $\text{CuIn}_2\text{K}$) achieved $100\%$ SCF convergence and $100\%$ metallic character. In particular, **$\text{BaBiTe}$** converges with the lowest residual forces of the cohort ($0.021\text{ Ry/Bohr}$) and $E_F = 2.98\text{ eV}$ at $+189.4\text{ kbar}$ ($\approx 18.9\text{ GPa}$).
5. **Scientific Scope & Limitations:** Static SCF convergence and low Hellmann–Feynman forces confirm electronic stability and metallic Fermi surfaces, but do not substitute for full DFPT phonon dispersion $\omega(\mathbf{q})$, EPW Eliashberg $\alpha^2 F(\omega)$ integration, or ternary convex-hull enthalpy relaxation prior to experimental synthesis.

---

## Quickstart & Verification

```bash
git clone https://github.com/nauheimervater/supercon-zero-copy-discovery.git
cd supercon-zero-copy-discovery
pip install -r requirements.txt

# Run unit tests (Magpie features, Allen-Dynes Tc, 1.2 A collision filter, DFT tables)
python -m unittest discover -s tests -v

# Reproduce all tables and benchmarks from the paper
python examples/reproduce_all.py
```

### Quantum ESPRESSO v7.3 PBE-GGA Verification Summary

| Candidate Formula | Total Energy (Ry) | SCF Iter. | Hydrostatic Pressure (kbar) | Fermi Energy $E_F$ (eV) | Total Force (Ry/Bohr) | Valence Electrons |
|---|---:|---:|---:|---:|---:|---:|
| $\text{CeIn}_3$ | $-1703.79$ | 16 | $-27.8$ | $2.34$ | $0.024$ | 50 |
| $\text{Sb}_2\text{Te}_3$ | $-1782.81$ | 18 | $-33.1$ | $1.26$ | $0.073$ | 28 |
| **$\text{BaBiTe}$** | **$-1704.43$** | **24** | **$+189.4$** | **$2.98$** | **$0.021$** | **31** |
| $\text{AuPd}_{11}$ | $-4415.78$ | 22 | $-20.8$ | $6.15$ | $0.676$ | 121 |
| $\text{CuIn}_2\text{K}$ | $-1147.38$ | 15 | $-19.7$ | $1.16$ | $0.036$ | 46 |

---

## Citation

```bibtex
@misc{nauheimer2026supercon,
  author       = {Thomas Nauheimer},
  title        = {AI-Driven Superconductor Discovery on Unified-Memory APUs: Equivariant Crystal Diffusion, Zero-Copy MACE Relaxation, and First-Principles DFT Verification},
  year         = {2026},
  month        = nov,
  url          = {https://github.com/nauheimervater/supercon-zero-copy-discovery}
}
```
