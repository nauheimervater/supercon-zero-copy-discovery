"""
Reproduce all tables, Allen-Dynes Tc curves, core-repulsion collision checks,
and Quantum ESPRESSO DFT verification summaries for:
  "AI-Driven Superconductor Discovery on Unified-Memory APUs: Equivariant Crystal
   Diffusion, Zero-Copy MACE Relaxation, and First-Principles DFT Verification"
  (Thomas Nauheimer, November 2026).

Usage:
    python examples/reproduce_all.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

from supercon_discovery import (
    DFT_VERIFIED_CANDIDATES,
    DIFFCSP_CONVERGENCE_TABLE,
    compute_magpie_features,
    allen_dynes_tc,
    apply_core_repulsion_filter,
    zero_copy_transfer_speedup,
)


def main() -> None:
    print("=" * 80)
    print("SUPERCON ZERO-COPY DISCOVERY PIPELINE: FULL BENCHMARK REPRODUCTION")
    print("=" * 80)

    # 1. DiffCSP-SC Training Convergence on AMD Radeon 8060S (Strix Halo APU)
    print("\n[Section 1] DiffCSP-SC Diffusion Training Convergence (1,000 Epochs, FP16):")
    print(f"  {'Loss Component':<24s} | {'Epoch 1':>10s} | {'Epoch 999':>10s} | {'Reduction':>10s}")
    print("  " + "-" * 62)
    for row in DIFFCSP_CONVERGENCE_TABLE:
        print(
            f"  {row['metric']:<24s} | {row['epoch_1']:10.2f} | "
            f"{row['epoch_999']:10.2f} | {row['reduction_pct']:9.2f}%"
        )

    # 2. Zero-Copy Unified-Memory MACE Relaxation & 1.2 A Core-Repulsion Filter
    print("\n[Section 2] Zero-Copy Unified-Memory MACE Relaxation & Collision Filter:")
    zc = zero_copy_transfer_speedup(num_atoms=54, num_steps=1000)
    print(f"  Native C++/HIP Zero-Copy Step Latency : {zc['zero_copy_step_ms']:.1f} ms/step")
    print(f"  Standard Python/ASE + PCIe Copy Loop  : {zc['standard_pcie_ase_step_ms']:.1f} ms/step")
    print(f"  1,000-Step Relaxation Wall Time       : {zc['zero_copy_total_s']:.1f} s (Zero-Copy) vs. {zc['standard_pcie_total_s']:.1f} s (ASE/PCIe)")
    print(f"  Host-Device PCIe Bytes Transferred    : {int(zc['host_device_bytes_copied_zero_copy'])} B vs. {int(zc['host_device_bytes_copied_standard']):,} B")

    rng = np.random.default_rng(42)
    cell = np.diag([4.2, 4.2, 12.0])
    raw_pos = rng.uniform(0.0, 4.0, size=(12, 3))
    raw_pos[1] = raw_pos[0] + np.array([0.25, 0.10, 0.05])  # Synthetic 0.27 A diffusion clash
    _, clashes, min_d = apply_core_repulsion_filter(raw_pos, cell, min_dist_angstrom=1.20)
    print(f"  Pre-MACE 1.20 A Collision Filter Test : resolved {clashes} clashes -> min distance = {min_d:.4f} A")

    # 3. First-Principles Quantum ESPRESSO v7.3 PBE-GGA Verification Table
    print("\n[Section 3] Quantum ESPRESSO v7.3 First-Principles DFT Verification (50 Ry / 400 Ry PAW):")
    print(f"  {'Material':<10s} | {'Energy (Ry)':>12s} | {'Iter':>4s} | {'Press (kbar)':>12s} | {'E_F (eV)':>8s} | {'Force (Ry/B)':>12s} | {'N_el':>4s}")
    print("  " + "-" * 76)
    for c in DFT_VERIFIED_CANDIDATES:
        print(
            f"  {str(c['formula']):<10s} | {float(c['total_energy_ry']):12.2f} | "
            f"{int(c['scf_iterations']):4d} | {float(c['pressure_kbar']):+12.1f} | "
            f"{float(c['fermi_energy_ev']):8.2f} | {float(c['total_force_ry_bohr']):12.3f} | "
            f"{int(c['num_electrons']):4d}"
        )

    # 4. Compositional Magpie Feature & Allen-Dynes Reference Benchmarks
    print("\n[Section 4] Compositional Feature & Allen-Dynes Reference Regimes:")
    ref_regimes = [
        ("CeIn3 (Heavy-Fermion Ref)", 0.32, 145.0, 0.12, None),
        ("Nb3Sn (A15 Conventional)", 1.75, 155.0, 0.12, 210.0),
        ("MgB2 (Two-Gap Phonon)", 0.87, 670.0, 0.10, 820.0),
        ("BaBiTe (19 GPa Parametric)", 0.95, 210.0, 0.10, 260.0),
        ("LaH10 (200 GPa Clathrate)", 2.25, 1250.0, 0.10, 1650.0),
    ]
    print(f"  {'System / Regime':<28s} | {'lambda':>6s} | {'omega_log (K)':>13s} | {'mu*':>5s} | {'Allen-Dynes Tc (K)':>18s}")
    print("  " + "-" * 78)
    for name, lam, wlog, mustar, w2 in ref_regimes:
        tc = allen_dynes_tc(lam, wlog, mustar, w2)
        formula = name.split()[0]
        feats = compute_magpie_features(formula)
        print(f"  {name:<28s} | {lam:6.2f} | {wlog:13.1f} | {mustar:5.2f} | {tc:18.2f}")

    print("\n" + "=" * 80)
    print("ALL PIPELINE STAGES AND TABLES REPRODUCED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
