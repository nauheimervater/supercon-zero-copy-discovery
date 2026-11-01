# Toward Zero-Copy ML/DFT Materials Screening on Unified-Memory APUs: Pipeline Architecture (Work in Progress)

**Thomas Nauheimer**  
Independent Researcher · `nauheimer.t@gmail.com`  
November 2026 · Version 1.0.1 (technical note, work in progress)

---

## Status and Erratum (v1.0.1)

Version 1.0.0 of this document reported a per-step relaxation latency of 144.7 ms, a Quantum ESPRESSO results table for five generated candidates (including BaBiTe), and a DiffCSP-SC loss-convergence table. On re-audit, **none of these figures is backed by measurement outputs in this repository**: the latency was a fixed default in an analytic model, and no matching Quantum ESPRESSO output files exist for the listed candidates. These figures are **withdrawn**. This version describes the pipeline architecture and the components that can be verified from the published code. Measured results will be added only together with their raw outputs.

---

## Abstract

We describe the architecture of a closed-loop pipeline for superconductor candidate screening on a unified-memory workstation (AMD Ryzen AI Max+ 395 / Strix Halo, Radeon 8060S iGPU, 128 GB shared LPDDR5X, ROCm). The pipeline couples (1) Magpie-style compositional screening on the UCI SuperCon dataset, (2) a crystal-structure generator in the style of DiffCSP, (3) a C++/HIP relaxation layer in which coordinate and force buffers are allocated with `hipHostMalloc(..., hipHostMallocMapped)` and wrapped in place with `torch::from_blob`, so that CPU-side checks and GPU-side interatomic-potential evaluation operate on the same buffers without explicit copies, and (4) a hand-off to Quantum ESPRESSO for first-principles verification. The repository currently provides the compositional feature code, an Allen–Dynes McMillan $T_c$ implementation, a periodic minimum-image core-repulsion filter, the C++/HIP zero-copy engine sources, and an analytic latency model. Quantitative performance and DFT results are pending.

---

## 1. Motivation

In copy-based hybrid ML/ab-initio loops (e.g. Python/ASE driving a TorchScript potential), coordinates and forces are copied between host and device buffers at every step. On discrete GPUs these copies cross PCIe. On unified-memory APUs such as Strix Halo, CPU and iGPU share one physical memory pool, so explicit copies are in principle avoidable. **Note:** there is no PCIe link between CPU and iGPU on Strix Halo; any benefit of the zero-copy design comes from avoided `memcpy`/allocation/dispatch overhead, not from PCIe bandwidth. Its size must be established by a direct mapped-vs-copied benchmark on the same GPU, which is not yet published.

## 2. Components

### 2.1 Compositional screening
An 81-feature Magpie-style representation (fraction-weighted statistics of atomic mass, radius, electronegativity, valence, etc.) is implemented in `supercon_discovery/screening.py`. In the original pipeline run, a gradient-boosted regressor trained on 21,263 UCI SuperCon entries was logged with a test $R^2 \approx 0.93$; the training script and split are not yet part of this repository, so this number should be treated as preliminary.

### 2.2 Crystal-structure generation
A DiffCSP-style generator (joint denoising of lattice, fractional coordinates, and species) was used to propose candidate structures. Training logs are not yet published; no convergence figures are claimed here.

### 2.3 Zero-copy relaxation layer and core-repulsion filter
`cpp/mace_zero_copy_engine.cpp` and `cpp/verify_zero_copy.cpp` allocate mapped host memory and wrap it as tensors without copies. Before force evaluation, a periodic minimum-image filter separates atom pairs closer than $d_{\min} = 1.20$ Å to avoid numerical blow-ups in message-passing potentials; this filter is unit-tested.

### 2.4 DFT hand-off
Candidates are intended to be verified with Quantum ESPRESSO (PBE). No candidate DFT results are reported in this version.

## 3. Verifiable in this repository

- Magpie-style feature extraction and formula parsing (unit-tested).
- Allen–Dynes modified McMillan $T_c$ for literature reference regimes (MgB$_2$, LaH$_{10}$-like; unit-tested).
- Periodic 1.20 Å core-repulsion filter (unit-tested).
- Zero-copy C++/HIP engine sources.
- An analytic latency model `zero_copy_transfer_speedup(...)` that requires measured inputs and makes no hardware claim by itself.

## 4. Planned work

1. Same-GPU benchmark on gfx1151: mapped (zero-copy) vs. explicitly copied buffers for a 54-atom MACE relaxation step, with raw timing logs.
2. Quantum ESPRESSO SCF (and later DFPT) runs for generated candidates, with full input and output files.
3. Publication of the screening training script and the generator training logs.

## References

1. A. Merchant et al., "Scaling deep learning for materials discovery," *Nature* 624, 80–85 (2023).
2. R. Jiao et al., "Crystal structure prediction by joint equivariant diffusion," *NeurIPS* (2023).
3. I. Batatia et al., "MACE: Higher order equivariant message passing neural networks for fast and accurate force fields," *NeurIPS* (2022).
4. P. B. Allen and R. C. Dynes, "Transition temperature of strong-coupled superconductors reanalyzed," *Phys. Rev. B* 12, 905 (1975).
5. P. Giannozzi et al., "QUANTUM ESPRESSO," *J. Phys.: Condens. Matter* 21, 395502 (2009).
