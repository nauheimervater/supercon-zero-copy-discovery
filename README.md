# Toward Zero-Copy ML/DFT Materials Screening on Unified-Memory APUs (Work in Progress)

[![Status: WIP](https://img.shields.io/badge/status-work%20in%20progress-yellow.svg)](#status)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Thomas Nauheimer · Independent Researcher · `nauheimer.t@gmail.com`  
Technical note: [`paper/SuperCon_Zero_Copy_Discovery.md`](paper/SuperCon_Zero_Copy_Discovery.md) · [PDF](paper/SuperCon_Zero_Copy_Discovery.pdf)

## Status

- **Included (v1.0.2):** Quantum ESPRESSO v7.5 SCF inputs, outputs and CIF structures for five generated candidates in [`data/qe_top5/`](data/qe_top5/). The DFT table is regenerated from these raw files by `supercon_discovery/qe_parse.py` and checked in the unit tests.
- **Withdrawn (since v1.0.1):** the 144.7 ms/step latency and all speedup figures (in v1.0.0 these came from an analytic model with assumed inputs, not a measurement), the diffusion-loss table, and the parametric $T_c$ estimate.
- **Note:** v1.0.1 also withdrew the QE table; that was a mistake, corrected in v1.0.2.

## DFT results (single-point SCF, PBE/PAW, 50/400 Ry)

| Candidate | $E_{\text{tot}}$ (Ry) | SCF iter. | Pressure (kbar) | $E_F$ (eV) | Total force (Ry/Bohr) |
|---|---:|---:|---:|---:|---:|
| CeIn₃ | −1703.79 | 16 | −27.8 | 2.34 | 0.024 |
| Sb₂Te₃ | −1782.81 | 18 | −33.1 | 1.26 | 0.073 |
| BaBiTe | −1704.43 | 24 | +189.4 | 2.98 | 0.021 |
| AuPd₁₁ | −4415.78 | 22 | −20.8 | 6.15 | 0.676 |
| CuIn₂K | −1147.38 | 15 | −19.7 | 1.16 | 0.036 |

These are static calculations on unrelaxed generated cells: they show SCF convergence, not stability or superconductivity. See the limitations section of the technical note.

## What else this repository contains

- `supercon_discovery/screening.py` – Magpie-style features, Allen–Dynes McMillan $T_c$, periodic 1.20 Å core-repulsion filter, and an analytic latency model that requires measured inputs.
- `cpp/mace_zero_copy_engine.cpp`, `cpp/verify_zero_copy.cpp` – C++/HIP sources that allocate `hipHostMallocMapped` buffers and wrap them with `torch::from_blob` on a unified-memory APU (AMD Strix Halo, gfx1151). On Strix Halo there is no PCIe link between CPU and iGPU; the benefit of zero-copy has to be shown by a same-GPU mapped-vs-copied benchmark (pending).

## Quickstart

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python examples/reproduce_all.py
```
