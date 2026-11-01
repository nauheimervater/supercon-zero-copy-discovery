"""
Minimal parser for Quantum ESPRESSO pw.x SCF output files, used to regenerate the
DFT table in the paper directly from the raw outputs in data/qe_top5/.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List

QE_TOP5_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "qe_top5"))
QE_TOP5_FORMULAS = ["CeIn3", "Sb2Te3", "BaBiTe", "AuPd11", "CuIn2K"]


def _last(pattern: str, text: str):
    hits = re.findall(pattern, text)
    return hits[-1] if hits else None


def parse_pw_output(path: str) -> Dict[str, object]:
    """Extract key quantities from a pw.x SCF output file."""
    text = open(path, encoding="utf-8", errors="ignore").read()

    def num(p):
        v = _last(p, text)
        return float(v) if v is not None else None

    iters = _last(r"convergence has been achieved in\s+(\d+)\s+iterations", text)
    return {
        "file": os.path.basename(path),
        "qe_version": _last(r"Program PWSCF (v\.[\d.]+)", text),
        "converged": iters is not None,
        "job_done": "JOB DONE" in text,
        "scf_iterations": int(iters) if iters else None,
        "total_energy_ry": num(r"!\s+total energy\s+=\s+(-?[\d.]+)\s+Ry"),
        "fermi_energy_ev": num(r"the Fermi energy is\s+(-?[\d.]+)\s+ev"),
        "pressure_kbar": num(r"P=\s*(-?[\d.]+)"),
        "total_force_ry_bohr": num(r"Total force =\s+([\d.]+)"),
        "num_electrons": num(r"number of electrons\s+=\s+([\d.]+)"),
        "ecutwfc_ry": num(r"kinetic-energy cutoff\s+=\s+([\d.]+)\s+Ry"),
        "ecutrho_ry": num(r"charge density cutoff\s+=\s+([\d.]+)\s+Ry"),
        "num_atoms": num(r"number of atoms/cell\s+=\s+(\d+)"),
    }


def load_qe_top5(directory: str = QE_TOP5_DIR) -> List[Dict[str, object]]:
    """Parse the five published candidate SCF outputs in a fixed order."""
    return [parse_pw_output(os.path.join(directory, f"{f}_scf.out")) | {"formula": f} for f in QE_TOP5_FORMULAS]
