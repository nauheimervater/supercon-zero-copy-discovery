"""
supercon_discovery: Compositional Magpie Feature Extraction, Allen-Dynes McMillan Tc
Modeling, Zero-Copy Unified-Memory Relaxation Diagnostics, and First-Principles DFT
Benchmark Verification for Superconductor Discovery.
"""

from .screening import (
    ELEMENT_DATA,
    DFT_VERIFIED_CANDIDATES,
    DIFFCSP_CONVERGENCE_TABLE,
    parse_formula,
    compute_magpie_features,
    allen_dynes_tc,
    apply_core_repulsion_filter,
    zero_copy_transfer_speedup,
)

__all__ = [
    "ELEMENT_DATA",
    "DFT_VERIFIED_CANDIDATES",
    "DIFFCSP_CONVERGENCE_TABLE",
    "parse_formula",
    "compute_magpie_features",
    "allen_dynes_tc",
    "apply_core_repulsion_filter",
    "zero_copy_transfer_speedup",
]
