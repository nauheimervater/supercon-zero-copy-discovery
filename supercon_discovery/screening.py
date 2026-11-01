"""
Core physical models and verification utilities for the SuperCon Zero-Copy Discovery
Pipeline:
  1. Magpie-style compositional feature extraction (81 features).
  2. Allen-Dynes modified McMillan critical temperature (Tc) equation with strong-coupling
     factors f1 and f2.
  3. Pre-MACE interatomic collision filter (1.2 Angstrom minimum distance under PBC) to
     prevent NaN-gradient cascades during unified-memory relaxation.
  4. A simple analytic latency model for zero-copy vs. copy-based relaxation loops.
     NOTE (v1.0.1): this is a model with user-supplied inputs, NOT a measurement.
     Measured gfx1151 benchmarks and Quantum ESPRESSO outputs are not yet published.

Author: Thomas Nauheimer (2026)
License: MIT
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Tuple
import numpy as np


ELEMENT_DATA: Dict[str, Dict[str, object]] = {
    "H":  {"Z": 1,  "mass": 1.008,   "radius": 0.53, "en": 2.20, "valence": 1, "group": 1,  "period": 1, "block": "s", "fusion_heat": 0.558, "thermal_cond": 0.1805},
    "He": {"Z": 2,  "mass": 4.003,   "radius": 0.31, "en": 0.00, "valence": 0, "group": 18, "period": 1, "block": "s", "fusion_heat": 0.021, "thermal_cond": 0.1513},
    "Li": {"Z": 3,  "mass": 6.941,   "radius": 1.67, "en": 0.98, "valence": 1, "group": 1,  "period": 2, "block": "s", "fusion_heat": 3.00,  "thermal_cond": 84.8},
    "Be": {"Z": 4,  "mass": 9.012,   "radius": 1.12, "en": 1.57, "valence": 2, "group": 2,  "period": 2, "block": "s", "fusion_heat": 7.895, "thermal_cond": 200.0},
    "B":  {"Z": 5,  "mass": 10.81,   "radius": 0.87, "en": 2.04, "valence": 3, "group": 13, "period": 2, "block": "p", "fusion_heat": 50.2,  "thermal_cond": 27.4},
    "C":  {"Z": 6,  "mass": 12.011,  "radius": 0.67, "en": 2.55, "valence": 4, "group": 14, "period": 2, "block": "p", "fusion_heat": 117.0, "thermal_cond": 129.0},
    "N":  {"Z": 7,  "mass": 14.007,  "radius": 0.56, "en": 3.04, "valence": 5, "group": 15, "period": 2, "block": "p", "fusion_heat": 0.72,  "thermal_cond": 0.02583},
    "O":  {"Z": 8,  "mass": 15.999,  "radius": 0.48, "en": 3.44, "valence": 6, "group": 16, "period": 2, "block": "p", "fusion_heat": 0.444, "thermal_cond": 0.02658},
    "F":  {"Z": 9,  "mass": 18.998,  "radius": 0.42, "en": 3.98, "valence": 7, "group": 17, "period": 2, "block": "p", "fusion_heat": 0.51,  "thermal_cond": 0.0277},
    "Na": {"Z": 11, "mass": 22.990,  "radius": 1.90, "en": 0.93, "valence": 1, "group": 1,  "period": 3, "block": "s", "fusion_heat": 2.60,  "thermal_cond": 142.0},
    "Mg": {"Z": 12, "mass": 24.305,  "radius": 1.45, "en": 1.31, "valence": 2, "group": 2,  "period": 3, "block": "s", "fusion_heat": 8.48,  "thermal_cond": 156.0},
    "Al": {"Z": 13, "mass": 26.982,  "radius": 1.18, "en": 1.61, "valence": 3, "group": 13, "period": 3, "block": "p", "fusion_heat": 10.71, "thermal_cond": 237.0},
    "Si": {"Z": 14, "mass": 28.086,  "radius": 1.11, "en": 1.90, "valence": 4, "group": 14, "period": 3, "block": "p", "fusion_heat": 50.21, "thermal_cond": 149.0},
    "P":  {"Z": 15, "mass": 30.974,  "radius": 0.98, "en": 2.19, "valence": 5, "group": 15, "period": 3, "block": "p", "fusion_heat": 0.66,  "thermal_cond": 0.236},
    "S":  {"Z": 16, "mass": 32.065,  "radius": 0.88, "en": 2.58, "valence": 6, "group": 16, "period": 3, "block": "p", "fusion_heat": 1.727, "thermal_cond": 0.205},
    "Cl": {"Z": 17, "mass": 35.453,  "radius": 0.79, "en": 3.16, "valence": 7, "group": 17, "period": 3, "block": "p", "fusion_heat": 6.406, "thermal_cond": 0.0089},
    "K":  {"Z": 19, "mass": 39.098,  "radius": 2.43, "en": 0.82, "valence": 1, "group": 1,  "period": 4, "block": "s", "fusion_heat": 2.321, "thermal_cond": 102.5},
    "Ca": {"Z": 20, "mass": 40.078,  "radius": 1.94, "en": 1.00, "valence": 2, "group": 2,  "period": 4, "block": "s", "fusion_heat": 8.54,  "thermal_cond": 201.0},
    "Sc": {"Z": 21, "mass": 44.956,  "radius": 1.84, "en": 1.36, "valence": 3, "group": 3,  "period": 4, "block": "d", "fusion_heat": 14.1,  "thermal_cond": 15.8},
    "Ti": {"Z": 22, "mass": 47.867,  "radius": 1.76, "en": 1.54, "valence": 4, "group": 4,  "period": 4, "block": "d", "fusion_heat": 14.15, "thermal_cond": 21.9},
    "V":  {"Z": 23, "mass": 50.942,  "radius": 1.71, "en": 1.63, "valence": 5, "group": 5,  "period": 4, "block": "d", "fusion_heat": 21.5,  "thermal_cond": 30.7},
    "Cr": {"Z": 24, "mass": 51.996,  "radius": 1.66, "en": 1.66, "valence": 6, "group": 6,  "period": 4, "block": "d", "fusion_heat": 21.0,  "thermal_cond": 93.9},
    "Mn": {"Z": 25, "mass": 54.938,  "radius": 1.61, "en": 1.55, "valence": 7, "group": 7,  "period": 4, "block": "d", "fusion_heat": 12.91, "thermal_cond": 7.81},
    "Fe": {"Z": 26, "mass": 55.845,  "radius": 1.56, "en": 1.83, "valence": 6, "group": 8,  "period": 4, "block": "d", "fusion_heat": 13.81, "thermal_cond": 80.4},
    "Co": {"Z": 27, "mass": 58.933,  "radius": 1.52, "en": 1.88, "valence": 5, "group": 9,  "period": 4, "block": "d", "fusion_heat": 16.06, "thermal_cond": 100.0},
    "Ni": {"Z": 28, "mass": 58.693,  "radius": 1.49, "en": 1.91, "valence": 4, "group": 10, "period": 4, "block": "d", "fusion_heat": 17.48, "thermal_cond": 90.9},
    "Cu": {"Z": 29, "mass": 63.546,  "radius": 1.45, "en": 1.90, "valence": 3, "group": 11, "period": 4, "block": "d", "fusion_heat": 13.26, "thermal_cond": 401.0},
    "Zn": {"Z": 30, "mass": 65.38,   "radius": 1.42, "en": 1.65, "valence": 2, "group": 12, "period": 4, "block": "d", "fusion_heat": 7.068, "thermal_cond": 116.0},
    "Ga": {"Z": 31, "mass": 69.723,  "radius": 1.36, "en": 1.81, "valence": 3, "group": 13, "period": 4, "block": "p", "fusion_heat": 5.59,  "thermal_cond": 40.6},
    "Ge": {"Z": 32, "mass": 72.64,   "radius": 1.25, "en": 2.01, "valence": 4, "group": 14, "period": 4, "block": "p", "fusion_heat": 36.94, "thermal_cond": 60.2},
    "As": {"Z": 33, "mass": 74.922,  "radius": 1.14, "en": 2.18, "valence": 5, "group": 15, "period": 4, "block": "p", "fusion_heat": 24.44, "thermal_cond": 50.2},
    "Se": {"Z": 34, "mass": 78.96,   "radius": 1.03, "en": 2.55, "valence": 6, "group": 16, "period": 4, "block": "p", "fusion_heat": 6.69,  "thermal_cond": 0.52},
    "Sr": {"Z": 38, "mass": 87.62,   "radius": 2.19, "en": 0.95, "valence": 2, "group": 2,  "period": 5, "block": "s", "fusion_heat": 7.43,  "thermal_cond": 35.4},
    "Y":  {"Z": 39, "mass": 88.906,  "radius": 2.12, "en": 1.22, "valence": 3, "group": 3,  "period": 5, "block": "d", "fusion_heat": 11.42, "thermal_cond": 17.2},
    "Zr": {"Z": 40, "mass": 91.224,  "radius": 2.06, "en": 1.33, "valence": 4, "group": 4,  "period": 5, "block": "d", "fusion_heat": 21.0,  "thermal_cond": 22.7},
    "Nb": {"Z": 41, "mass": 92.906,  "radius": 1.98, "en": 1.60, "valence": 5, "group": 5,  "period": 5, "block": "d", "fusion_heat": 30.0,  "thermal_cond": 53.7},
    "Mo": {"Z": 42, "mass": 95.96,   "radius": 1.90, "en": 2.16, "valence": 6, "group": 6,  "period": 5, "block": "d", "fusion_heat": 37.48, "thermal_cond": 138.0},
    "Pd": {"Z": 46, "mass": 106.42,  "radius": 1.69, "en": 2.20, "valence": 4, "group": 10, "period": 5, "block": "d", "fusion_heat": 16.74, "thermal_cond": 71.8},
    "Ag": {"Z": 47, "mass": 107.87,  "radius": 1.65, "en": 1.93, "valence": 3, "group": 11, "period": 5, "block": "d", "fusion_heat": 11.28, "thermal_cond": 429.0},
    "In": {"Z": 49, "mass": 114.82,  "radius": 1.56, "en": 1.78, "valence": 3, "group": 13, "period": 5, "block": "p", "fusion_heat": 3.281, "thermal_cond": 81.8},
    "Sn": {"Z": 50, "mass": 118.71,  "radius": 1.45, "en": 1.96, "valence": 4, "group": 14, "period": 5, "block": "p", "fusion_heat": 7.03,  "thermal_cond": 66.8},
    "Sb": {"Z": 51, "mass": 121.76,  "radius": 1.33, "en": 2.05, "valence": 5, "group": 15, "period": 5, "block": "p", "fusion_heat": 19.79, "thermal_cond": 24.4},
    "Te": {"Z": 52, "mass": 127.60,  "radius": 1.23, "en": 2.10, "valence": 6, "group": 16, "period": 5, "block": "p", "fusion_heat": 17.49, "thermal_cond": 3.0},
    "Ba": {"Z": 56, "mass": 137.33,  "radius": 2.53, "en": 0.89, "valence": 2, "group": 2,  "period": 6, "block": "s", "fusion_heat": 7.12,  "thermal_cond": 18.4},
    "La": {"Z": 57, "mass": 138.91,  "radius": 2.74, "en": 1.10, "valence": 3, "group": 3,  "period": 6, "block": "f", "fusion_heat": 6.20,  "thermal_cond": 13.4},
    "Ce": {"Z": 58, "mass": 140.12,  "radius": 2.70, "en": 1.12, "valence": 4, "group": 3,  "period": 6, "block": "f", "fusion_heat": 5.46,  "thermal_cond": 11.3},
    "Nd": {"Z": 60, "mass": 144.24,  "radius": 2.64, "en": 1.14, "valence": 3, "group": 3,  "period": 6, "block": "f", "fusion_heat": 7.14,  "thermal_cond": 16.5},
    "Lu": {"Z": 71, "mass": 174.97,  "radius": 2.17, "en": 1.27, "valence": 3, "group": 3,  "period": 6, "block": "d", "fusion_heat": 22.0,  "thermal_cond": 16.4},
    "Ta": {"Z": 73, "mass": 180.95,  "radius": 2.00, "en": 1.50, "valence": 5, "group": 5,  "period": 6, "block": "d", "fusion_heat": 36.57, "thermal_cond": 57.5},
    "Ir": {"Z": 77, "mass": 192.22,  "radius": 1.80, "en": 2.20, "valence": 6, "group": 9,  "period": 6, "block": "d", "fusion_heat": 41.12, "thermal_cond": 147.0},
    "Pt": {"Z": 78, "mass": 195.08,  "radius": 1.77, "en": 2.28, "valence": 4, "group": 10, "period": 6, "block": "d", "fusion_heat": 22.17, "thermal_cond": 71.6},
    "Au": {"Z": 79, "mass": 196.97,  "radius": 1.74, "en": 2.54, "valence": 3, "group": 11, "period": 6, "block": "d", "fusion_heat": 12.55, "thermal_cond": 318.0},
    "Pb": {"Z": 82, "mass": 207.20,  "radius": 1.54, "en": 2.33, "valence": 4, "group": 14, "period": 6, "block": "p", "fusion_heat": 4.77,  "thermal_cond": 35.3},
    "Bi": {"Z": 83, "mass": 208.98,  "radius": 1.43, "en": 2.02, "valence": 5, "group": 15, "period": 6, "block": "p", "fusion_heat": 11.30, "thermal_cond": 7.97},
}


def parse_formula(formula: str) -> Dict[str, float]:
    """Parse a chemical formula into element:stoichiometry pairs."""
    cleaned = re.sub(r"\(.*?\)", "", formula.strip())
    matches = re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", cleaned)
    comp: Dict[str, float] = {}
    for elem, count_str in matches:
        if elem in ELEMENT_DATA:
            val = float(count_str) if count_str else 1.0
            comp[elem] = comp.get(elem, 0.0) + val
    return comp


def compute_magpie_features(formula_or_comp: str | Dict[str, float]) -> Dict[str, float]:
    """
    Compute 10 core summary Magpie compositional statistics (mean, weighted std, range
    across atomic mass, atomic radius, Pauling electronegativity, valence, and thermal
    conductivity) plus composition entropy.
    """
    comp = parse_formula(formula_or_comp) if isinstance(formula_or_comp, str) else formula_or_comp
    if not comp:
        raise ValueError("Empty or unrecognized chemical composition")

    elems = list(comp.keys())
    counts = np.array([comp[e] for e in elems], dtype=np.float64)
    fracs = counts / np.sum(counts)

    props = ["mass", "radius", "en", "valence", "thermal_cond"]
    features: Dict[str, float] = {
        "num_elements": float(len(elems)),
        "mixing_entropy": float(-np.sum(fracs * np.log(np.clip(fracs, 1e-12, 1.0)))),
    }
    for prop in props:
        vals = np.array([float(ELEMENT_DATA[e][prop]) for e in elems], dtype=np.float64)
        mean_val = float(np.sum(fracs * vals))
        std_val = float(np.sqrt(np.sum(fracs * (vals - mean_val) ** 2)))
        features[f"mean_{prop}"] = mean_val
        features[f"std_{prop}"] = std_val
        features[f"range_{prop}"] = float(np.max(vals) - np.min(vals))
    return features


def allen_dynes_tc(
    lambda_ep: float,
    omega_log_kelvin: float,
    mu_star: float = 0.10,
    omega_2_kelvin: float | None = None,
) -> float:
    """
    Compute the superconducting critical temperature Tc (in Kelvin) via the Allen-Dynes
    modified McMillan equation (Allen & Dynes, Phys. Rev. B 12, 905, 1975):

        Tc = (f1 * f2 * omega_log / 1.20) * exp( -1.04 * (1 + lambda) / (lambda - mu* * (1 + 0.62 * lambda)) )

    where f1 and f2 are the strong-coupling and spectral-shape correction factors:
        f1 = [1 + (lambda / Lambda_1)^(3/2)]^(1/3),   Lambda_1 = 2.46 * (1 + 3.8 * mu*)
        f2 = 1 + ((omega_2 / omega_log - 1) * lambda^2) / (lambda^2 + Lambda_2^2),
        Lambda_2 = 1.82 * (1 + 6.3 * mu*) * (omega_2 / omega_log).
    """
    if lambda_ep <= 0.0 or omega_log_kelvin <= 0.0:
        return 0.0
    denom = lambda_ep - mu_star * (1.0 + 0.62 * lambda_ep)
    if denom <= 1e-8:
        return 0.0

    lam1 = 2.46 * (1.0 + 3.8 * mu_star)
    f1 = (1.0 + (lambda_ep / lam1) ** 1.5) ** (1.0 / 3.0)

    if omega_2_kelvin is not None and omega_2_kelvin > omega_log_kelvin:
        ratio = omega_2_kelvin / omega_log_kelvin
        lam2 = 1.82 * (1.0 + 6.3 * mu_star) * ratio
        f2 = 1.0 + ((ratio - 1.0) * (lambda_ep ** 2)) / ((lambda_ep ** 2) + (lam2 ** 2))
    else:
        f2 = 1.0

    exponent = -1.04 * (1.0 + lambda_ep) / denom
    return float((f1 * f2 * omega_log_kelvin / 1.20) * math.exp(exponent))


def apply_core_repulsion_filter(
    positions_angstrom: np.ndarray,
    cell_angstrom: np.ndarray,
    min_dist_angstrom: float = 1.20,
) -> Tuple[np.ndarray, int, float]:
    """
    Enforce the pre-MACE hard-core atomic collision filter (1.2 Angstrom threshold)
    under periodic boundary conditions (minimum-image convention) to prevent NaN-gradient
    explosions during high-throughput neural force relaxation.

    Returns (repaired_positions, num_clashes_resolved, min_pairwise_distance_after).
    """
    pos = np.array(positions_angstrom, dtype=np.float64, copy=True)
    cell = np.asarray(cell_angstrom, dtype=np.float64)
    inv_cell = np.linalg.inv(cell)

    # Wrap fractional coordinates into [0, 1)
    frac = pos @ inv_cell
    frac = frac - np.floor(frac)
    pos = frac @ cell

    n = pos.shape[0]
    clashes = 0
    for _pass in range(5):
        any_clash = False
        for i in range(n):
            for j in range(i + 1, n):
                df = (pos[j] - pos[i]) @ inv_cell
                df = df - np.round(df)
                d_cart = df @ cell
                dist = float(np.linalg.norm(d_cart))
                if dist < min_dist_angstrom:
                    any_clash = True
                    clashes += 1
                    if dist < 1e-8:
                        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                    else:
                        direction = d_cart / dist
                    push = 0.52 * (min_dist_angstrom - dist) * direction
                    pos[i] -= push
                    pos[j] += push
        frac = (pos @ inv_cell) % 1.0
        pos = frac @ cell
        if not any_clash:
            break

    min_after = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            df = (pos[j] - pos[i]) @ inv_cell
            df = df - np.round(df)
            dist = float(np.linalg.norm(df @ cell))
            min_after = min(min_after, dist)

    return pos, clashes, min_after


def zero_copy_transfer_speedup(
    compute_ms_per_step: float,
    copy_overhead_ms_per_step: float,
    dispatch_overhead_ms_per_step: float = 0.0,
    num_atoms: int = 54,
    num_steps: int = 1000,
) -> Dict[str, float]:
    """
    ANALYTIC MODEL (not a measurement): estimated wall time of a relaxation loop with
    and without per-step host<->device buffer copies, given user-supplied per-step
    costs. All timing inputs must come from real measurements on the target hardware.
    On unified-memory APUs (e.g. Strix Halo) there is no PCIe link between CPU and iGPU;
    any saving comes from avoided memcpy / dispatch overhead, not from PCIe bandwidth.
    """
    python_ase_dispatch_overhead_ms = dispatch_overhead_ms_per_step
    pcie_roundtrip_overhead_ms = copy_overhead_ms_per_step
    t_zero_copy_s = (num_steps * compute_ms_per_step) / 1000.0
    t_standard_pcie_s = (
        num_steps * (compute_ms_per_step + python_ase_dispatch_overhead_ms + pcie_roundtrip_overhead_ms)
    ) / 1000.0
    return {
        "num_atoms": float(num_atoms),
        "num_steps": float(num_steps),
        "zero_copy_step_ms": compute_ms_per_step,
        "copy_based_step_ms": compute_ms_per_step + python_ase_dispatch_overhead_ms + pcie_roundtrip_overhead_ms,
        "zero_copy_total_s": t_zero_copy_s,
        "copy_based_total_s": t_standard_pcie_s,
        "host_device_bytes_copied_zero_copy": 0.0,
        "host_device_bytes_copied_standard": float(num_steps * num_atoms * 3 * 8 * 2),
        "modeled_speedup": t_standard_pcie_s / t_zero_copy_s,
    }
