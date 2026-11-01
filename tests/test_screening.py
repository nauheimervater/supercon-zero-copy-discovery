import unittest
import numpy as np

from supercon_discovery import (
    DFT_VERIFIED_CANDIDATES,
    DIFFCSP_CONVERGENCE_TABLE,
    parse_formula,
    compute_magpie_features,
    allen_dynes_tc,
    apply_core_repulsion_filter,
    zero_copy_transfer_speedup,
)


class TestSuperConDiscovery(unittest.TestCase):
    def test_formula_parsing_and_features(self):
        comp = parse_formula("YBa2Cu3O7")
        self.assertEqual(comp, {"Y": 1.0, "Ba": 2.0, "Cu": 3.0, "O": 7.0})
        feats = compute_magpie_features("BaBiTe")
        self.assertEqual(feats["num_elements"], 3.0)
        self.assertGreater(feats["mean_mass"], 100.0)
        self.assertGreater(feats["mixing_entropy"], 1.0)

    def test_allen_dynes_tc_benchmarks(self):
        # MgB2 approx benchmark (lambda ~ 0.87, omega_log ~ 670 K, mu* = 0.10) -> ~34-40 K
        tc_mgb2 = allen_dynes_tc(lambda_ep=0.87, omega_log_kelvin=670.0, mu_star=0.10)
        self.assertGreater(tc_mgb2, 30.0)
        self.assertLess(tc_mgb2, 45.0)

        # High-pressure hydride regime (LaH10-like: lambda ~ 2.2, omega_log ~ 1250 K, mu* = 0.10)
        tc_hydride = allen_dynes_tc(
            lambda_ep=2.20,
            omega_log_kelvin=1250.0,
            mu_star=0.10,
            omega_2_kelvin=1600.0,
        )
        self.assertGreater(tc_hydride, 200.0)

        # Sub-threshold coupling returns 0 K
        self.assertEqual(allen_dynes_tc(0.05, 500.0, mu_star=0.10), 0.0)

    def test_core_repulsion_filter(self):
        cell = np.diag([5.0, 5.0, 5.0])
        # Place two atoms only 0.3 Angstrom apart
        pos = np.array([
            [1.0, 1.0, 1.0],
            [1.2, 1.0, 1.0],
            [3.5, 3.5, 3.5],
        ])
        repaired, clashes, min_dist = apply_core_repulsion_filter(pos, cell, min_dist_angstrom=1.20)
        self.assertGreaterEqual(clashes, 1)
        self.assertGreaterEqual(min_dist, 1.199)

    def test_dft_and_zero_copy_tables(self):
        self.assertEqual(len(DFT_VERIFIED_CANDIDATES), 5)
        babite = [c for c in DFT_VERIFIED_CANDIDATES if c["formula"] == "BaBiTe"][0]
        self.assertAlmostEqual(babite["fermi_energy_ev"], 2.98, places=2)
        self.assertAlmostEqual(babite["total_energy_ry"], -1704.43, places=2)

        zc = zero_copy_transfer_speedup()
        self.assertEqual(zc["host_device_bytes_copied_zero_copy"], 0.0)
        self.assertGreater(zc["speedup_vs_ase_pcie"], 1.3)


if __name__ == "__main__":
    unittest.main()
