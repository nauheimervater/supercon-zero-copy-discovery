"""
supercon_discovery: Magpie-style compositional features, Allen-Dynes McMillan Tc
modeling, a pre-MACE core-repulsion filter, an analytic zero-copy latency model, and a
parser for the published Quantum ESPRESSO outputs.

Status (v1.0.2): Quantum ESPRESSO SCF outputs for five generated candidates are included
in data/qe_top5/. Measured gfx1151 zero-copy benchmarks are not yet published.
"""

from .screening import (
    ELEMENT_DATA,
    parse_formula,
    compute_magpie_features,
    allen_dynes_tc,
    apply_core_repulsion_filter,
    zero_copy_transfer_speedup,
)
from .qe_parse import QE_TOP5_FORMULAS, parse_pw_output, load_qe_top5

__all__ = [
    "ELEMENT_DATA",
    "parse_formula",
    "compute_magpie_features",
    "allen_dynes_tc",
    "apply_core_repulsion_filter",
    "zero_copy_transfer_speedup",
    "QE_TOP5_FORMULAS",
    "parse_pw_output",
    "load_qe_top5",
]
