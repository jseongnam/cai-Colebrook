import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeflow_ext.data import (
    audit_legacy_npz, build_features, dataset_fingerprint, generate_dataset,
    sample_params,
)
from pipeflow_ext.physics import baseline_initializer, minimum_reynolds, newton_refine, solve_reference


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.params = {
            "Q_total": 0.05,
            "D": np.array([0.12, 0.18, 0.22]),
            "eps": np.array([1e-4, 2e-4, 1.5e-4]),
            "L": np.array([180.0, 320.0, 250.0]),
            "rho": 998.0,
            "mu": 1.0e-3,
            "g": 9.81,
        }

    def test_baseline_is_feasible(self):
        q, x = baseline_initializer(self.params)
        self.assertTrue(np.all(q > 0))
        self.assertTrue(np.all(x > 0))
        self.assertAlmostEqual(float(q.sum()), self.params["Q_total"], places=13)

    def test_reference_and_newton_converge(self):
        reference = solve_reference(self.params)
        self.assertTrue(reference.converged)
        self.assertGreater(minimum_reynolds(reference.q, self.params), 4000.0)
        q0, x0 = baseline_initializer(self.params)
        refined = newton_refine(q0, x0, self.params)
        self.assertTrue(refined.converged)
        self.assertLess(refined.residual_inf, 1e-9)

    def test_small_generator(self):
        data = generate_dataset(2, 2, "iid", 123)
        self.assertEqual(data["D"].shape, (2, 2))
        self.assertEqual(data["q_target"].shape, (2, 2))
        dims = {}
        for mode in ["full", "no_state", "physics_only"]:
            branch, glob, target = build_features(data, mode)
            dims[mode] = branch.shape[-1]
            self.assertEqual(glob.shape, (2, 5))
            self.assertEqual(target.shape, (2, 2, 2))
        self.assertGreater(dims["full"], dims["no_state"])
        self.assertGreater(dims["no_state"], dims["physics_only"])

    def test_factorial_targets_are_finite_and_conservative(self):
        data = generate_dataset(3, 2, "iid", 321)
        for target_mode in ["correction", "direct"]:
            for parameterization in ["logit", "raw_ratio"]:
                _, _, target = build_features(
                    data, "full", target_mode, parameterization)
                self.assertEqual(target.shape, (3, 2, 2))
                self.assertTrue(np.all(np.isfinite(target)))
                if parameterization == "raw_ratio" and target_mode == "correction":
                    np.testing.assert_allclose(target[..., 0].sum(axis=1), 0.0, atol=1e-12)

    def test_dataset_fingerprint_detects_content_changes(self):
        data = generate_dataset(2, 2, "iid", 456)
        first = dataset_fingerprint(data)
        copied = {key: value.copy() for key, value in data.items()}
        self.assertEqual(first, dataset_fingerprint(copied))
        copied["D"][0, 0] += 1e-9
        self.assertNotEqual(first, dataset_fingerprint(copied))

    def test_legacy_leakage_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "legacy.npz"
            target = np.ones((4, 3))
            np.savez_compressed(p, center=target.copy(), target=target)
            report = audit_legacy_npz(p)
            self.assertTrue(report["unsafe"])
            self.assertEqual(report["center_equals_target_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
