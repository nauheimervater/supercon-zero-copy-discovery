import unittest
import numpy as np

from supercon_discovery import (
    parse_formula,
    compute_magpie_features,
    allen_dynes_tc,
    apply_core_repulsion_filter,
    zero_copy_transfer_speedup,
    QE_TOP5_FORMULAS,
    load_qe_top5,
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
        # Place two atoms only 0.2 Angstrom apart
        pos = np.array([
            [1.0, 1.0, 1.0],
            [1.2, 1.0, 1.0],
            [3.5, 3.5, 3.5],
        ])
        repaired, clashes, min_dist = apply_core_repulsion_filter(pos, cell, min_dist_angstrom=1.20)
        self.assertGreaterEqual(clashes, 1)
        self.assertGreaterEqual(min_dist, 1.199)

    def test_zero_copy_latency_model_arithmetic(self):
        # Pure arithmetic check of the analytic model with arbitrary inputs (not a benchmark).
        zc = zero_copy_transfer_speedup(compute_ms_per_step=100.0, copy_overhead_ms_per_step=10.0,
                                        dispatch_overhead_ms_per_step=0.0, num_steps=10)
        self.assertEqual(zc["host_device_bytes_copied_zero_copy"], 0.0)
        self.assertAlmostEqual(zc["zero_copy_total_s"], 1.0)
        self.assertAlmostEqual(zc["copy_based_total_s"], 1.1)
        self.assertAlmostEqual(zc["modeled_speedup"], 1.1)

    def test_qe_top5_outputs_match_paper_table(self):
        # Parses the raw pw.x outputs shipped in data/qe_top5/ (no hard-coded results).
        rows = {r["formula"]: r for r in load_qe_top5()}
        self.assertEqual(set(rows), set(QE_TOP5_FORMULAS))
        for r in rows.values():
            self.assertTrue(r["converged"] and r["job_done"])
            self.assertEqual(r["qe_version"], "v.7.5")
            self.assertEqual((r["ecutwfc_ry"], r["ecutrho_ry"]), (50.0, 400.0))
        b = rows["BaBiTe"]
        self.assertAlmostEqual(b["total_energy_ry"], -1704.43, places=2)
        self.assertAlmostEqual(b["fermi_energy_ev"], 2.98, places=2)
        self.assertAlmostEqual(b["pressure_kbar"], 189.36, places=2)
        self.assertEqual(b["scf_iterations"], 24)


if __name__ == "__main__":
    unittest.main()
