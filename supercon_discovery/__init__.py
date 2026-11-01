"""
supercon_discovery: Magpie-style compositional features, Allen-Dynes McMillan Tc
modeling, a pre-MACE core-repulsion filter, and an analytic zero-copy latency model.

Status (v1.0.1): work in progress. Measured gfx1151 benchmarks and Quantum ESPRESSO
outputs are not yet published in this repository.
"""

from .screening import (
    ELEMENT_DATA,
    parse_formula,
    compute_magpie_features,
    allen_dynes_tc,
    apply_core_repulsion_filter,
    zero_copy_transfer_speedup,
)

__all__ = [
    "ELEMENT_DATA",
    "parse_formula",
    "compute_magpie_features",
    "allen_dynes_tc",
    "apply_core_repulsion_filter",
    "zero_copy_transfer_speedup",
]
