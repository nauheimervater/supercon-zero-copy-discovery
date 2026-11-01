# Toward Zero-Copy ML/DFT Materials Screening on Unified-Memory APUs (Work in Progress)

[![Status: WIP](https://img.shields.io/badge/status-work%20in%20progress-yellow.svg)](#status)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Thomas Nauheimer · Independent Researcher · `nauheimer.t@gmail.com`  
Technical note: [`paper/SuperCon_Zero_Copy_Discovery.md`](paper/SuperCon_Zero_Copy_Discovery.md) · [PDF](paper/SuperCon_Zero_Copy_Discovery.pdf)

## Status

> **Erratum (v1.0.1):** v1.0.0 reported a 144.7 ms/step relaxation latency, a Quantum ESPRESSO results table for five generated candidates (incl. BaBiTe) and a diffusion-loss table. These figures are **not backed by measurement outputs in this repository** and have been **withdrawn**. Measured results will only be added together with their raw logs and input/output files.

## What this repository contains

- `supercon_discovery/screening.py` – Magpie-style compositional features, Allen–Dynes McMillan $T_c$, periodic 1.20 Å core-repulsion filter, and an analytic latency model that requires measured inputs.
- `cpp/mace_zero_copy_engine.cpp`, `cpp/verify_zero_copy.cpp` – C++/HIP sources that allocate `hipHostMallocMapped` buffers and wrap them with `torch::from_blob` (no explicit host↔device copies) on a unified-memory APU (AMD Strix Halo, gfx1151).
- `tests/` – unit tests for the components above.

Note: on Strix Halo there is no PCIe link between CPU and iGPU. Any benefit of the zero-copy design comes from avoided copy/dispatch overhead and has to be shown by a same-GPU mapped-vs-copied benchmark.

## Planned

1. Same-GPU benchmark on gfx1151: mapped vs. copied buffers for a 54-atom MACE relaxation step, with raw timing logs.
2. Quantum ESPRESSO SCF/DFPT runs for generated candidates, with all input and output files.
3. Screening training script and generator training logs.

## Quickstart

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python examples/reproduce_all.py
```
