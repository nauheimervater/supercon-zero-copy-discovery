"""
Reproduce the verifiable components of the SuperCon Zero-Copy Discovery repository
(work in progress, v1.0.1):
  - Allen-Dynes McMillan Tc for literature reference regimes,
  - the pre-MACE 1.20 A core-repulsion filter on a synthetic clash,
  - Magpie-style compositional features.

Measured gfx1151 zero-copy benchmarks and Quantum ESPRESSO outputs are NOT yet part of
this repository and are therefore not reproduced here.

Usage:
    python examples/reproduce_all.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

from supercon_discovery import (
    compute_magpie_features,
    allen_dynes_tc,
    apply_core_repulsion_filter,
)


def main() -> None:
    print("=" * 80)
    print("SUPERCON ZERO-COPY DISCOVERY (WORK IN PROGRESS): VERIFIABLE COMPONENTS")
    print("=" * 80)

    print("\n[1] Pre-MACE 1.20 A core-repulsion filter (synthetic clash):")
    rng = np.random.default_rng(42)
    cell = np.diag([4.2, 4.2, 12.0])
    raw_pos = rng.uniform(0.0, 4.0, size=(12, 3))
    raw_pos[1] = raw_pos[0] + np.array([0.25, 0.10, 0.05])
    _, clashes, min_d = apply_core_repulsion_filter(raw_pos, cell, min_dist_angstrom=1.20)
    print(f"  resolved {clashes} clashes -> min distance = {min_d:.4f} A")

    print("\n[2] Allen-Dynes Tc for literature reference regimes (inputs from literature, not computed here):")
    ref_regimes = [
        ("Nb3Sn (A15 conventional)", 1.75, 155.0, 0.12, 210.0),
        ("MgB2 (two-gap phonon)", 0.87, 670.0, 0.10, 820.0),
        ("LaH10 (~200 GPa clathrate)", 2.25, 1250.0, 0.10, 1650.0),
    ]
    print(f"  {'System':<28s} | {'lambda':>6s} | {'omega_log (K)':>13s} | {'mu*':>5s} | {'Tc (K)':>8s}")
    for name, lam, wlog, mustar, w2 in ref_regimes:
        print(f"  {name:<28s} | {lam:6.2f} | {wlog:13.1f} | {mustar:5.2f} | {allen_dynes_tc(lam, wlog, mustar, w2):8.2f}")

    print("\n[3] Magpie-style compositional features (example):")
    feats = compute_magpie_features("YBa2Cu3O7")
    for k in ("num_elements", "mean_mass", "mixing_entropy"):
        print(f"  YBa2Cu3O7 {k:<16s} = {feats[k]:.4f}")

    print("\nPending (not yet published): measured mapped-vs-copied benchmark on gfx1151,")
    print("Quantum ESPRESSO SCF outputs for generated candidates, DiffCSP-SC training logs.")


if __name__ == "__main__":
    main()
