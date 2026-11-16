"""Small algebra and artifact-contract fixtures; no model or video experiment."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from video_metric_feedback.matrices import (
    bf16_bits_to_float32, export_candidate, float32_to_bf16_rne,
    preflight_manifest, sha256_file,
)


class MatrixExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.width = 3
        self.layers = 2
        self.base_bits = np.full((3, 3), 0x3F80, dtype=np.uint16)  # exact 1.0
        np.save(self.root / "old.npy", self.base_bits)
        self.entries = {
            f"{expert}/{layer}": {
                "path": "old.npy", "sha256": sha256_file(self.root / "old.npy"),
                "shape": [3, 3], "dtype": "BF16_RAW_UINT16",
                "provenance": {"old_site": f"{expert}/{layer}"},
            }
            for expert in ("high", "low") for layer in range(self.layers)
        }
        self.manifest = self.root / "base.json"
        self.manifest.write_text(json.dumps(self.entries))
        # Delta has known values: [[1,0,-1], [0,2,2], [1,2,1]].
        self.a = np.zeros((3, 3), dtype=np.float64)
        self.delta = np.array([[1, 0, -1], [0, 2, 2], [1, 2, 1]], dtype=np.float64)
        self.kit = {
            "A_q1": self.a,
            "P64": np.array([[2, 0], [0, 1], [2, 1]], dtype=np.float64),
            "gamma_L": np.array([2, 1], dtype=np.float64),
            "C_L": np.array([[1, 0], [0, 2]], dtype=np.float64),
            "Q32": np.array([[1, 0], [0, 1], [-1, 1]], dtype=np.float64),
            "A_Q1_BF16": np.zeros((3, 3), dtype=np.uint16),
            "A_L_BF16": np.array([[0x3F80, 0, 0xBF80], [0, 0x4000, 0x4000],
                                      [0x3F80, 0x4000, 0x3F80]], dtype=np.uint16),
        }
        self.kit_path = self.root / "kit.npz"
        self.save_kit()
        self.sites = [{"site": "high/0", "parameter": "high", "kit_path": "kit.npz"}]

    def tearDown(self):
        self.temp.cleanup()

    def save_kit(self):
        np.savez(self.kit_path, **self.kit)

    def export(self, theta, name="candidate", sites=None):
        return export_candidate(
            self.manifest, self.sites if sites is None else sites, theta,
            self.root / name, expected_width=3, expected_layers=self.layers,
        )

    def candidate_bits(self, result):
        entries = json.loads(Path(result["manifest_path"]).read_text())
        return np.load(entries["high/0"]["path"], allow_pickle=False)

    def test_rne_ties_sign_and_float32_conversion(self):
        # Exactly halfway BF16 significands: even low endpoint stays, odd rounds up.
        value = np.array([1 + 1/256, 1 + 3/256, -1 - 1/256, -1 - 3/256], dtype=np.float32)
        expected = np.array([0x3F80, 0x3F82, 0xBF80, 0xBF82], dtype=np.uint16)
        np.testing.assert_array_equal(float32_to_bf16_rne(value), expected)
        np.testing.assert_array_equal(bf16_bits_to_float32(expected), [1., 1.015625, -1., -1.015625])
        # Value is above an exact tie in FP64, but rounds to the tie in FP32 first.
        self.assertEqual(int(float32_to_bf16_rne(np.array([1 + 1/256 + 1e-12]))[0]), 0x3F80)

    def test_exact_archived_endpoints_and_interior_orientation(self):
        zero = self.export({"high": 0}, "zero")
        one = self.export({"high": 1}, "one")
        half = self.export({"high": 0.5}, "half")
        np.testing.assert_array_equal(self.candidate_bits(zero), self.kit["A_Q1_BF16"])
        np.testing.assert_array_equal(self.candidate_bits(one), self.kit["A_L_BF16"])
        np.testing.assert_array_equal(bf16_bits_to_float32(self.candidate_bits(half)), self.delta / 2)
        # This nonsymmetric matrix also catches accidental export transposition.
        self.assertNotEqual(float(self.delta[0, 2]), float(self.delta[2, 0]))

    def test_preserves_full_manifest_and_reports_changes(self):
        result = self.export({"high": 0})
        manifest = json.loads(Path(result["manifest_path"]).read_text())
        metadata = json.loads(Path(result["metadata_path"]).read_text())
        self.assertEqual(set(manifest), set(self.entries))
        for site in ("high/1", "low/0", "low/1"):
            self.assertEqual(manifest[site]["sha256"], self.entries[site]["sha256"])
            self.assertEqual(manifest[site]["path"], str(self.root / "old.npy"))
            self.assertEqual(manifest[site]["provenance"], self.entries[site]["provenance"])
        info = metadata["selected_sites"][0]
        self.assertEqual(info["changed_fraction_from_base"], 1.)
        self.assertEqual(info["changed_fraction_from_current_q1"], 0.)
        self.assertEqual(info["endpoint_check"]["selected_route"], "archived_A_Q1_BF16")
        self.assertIn("not historical full Q1r2", metadata["baseline_identity"])
        self.assertEqual(sha256_file(result["manifest_path"]), result["matrix_manifest_sha256"])
        checked = preflight_manifest(result["manifest_path"], expected_width=3, expected_layers=2)
        self.assertTrue(checked["payload_sha256_and_finite_verified"])
        self.assertEqual(checked["sites"], 4)
        self.assertEqual(checked["distinct_payloads"], 2)

    def test_archived_endpoint_is_authority_when_reconstruction_differs(self):
        self.kit["A_L_BF16"] = self.kit["A_L_BF16"].copy()
        self.kit["A_L_BF16"][0, 0] = 0x3F81
        self.save_kit()
        result = self.export({"high": 1})
        self.assertEqual(int(self.candidate_bits(result)[0, 0]), 0x3F81)
        meta = json.loads(Path(result["metadata_path"]).read_text())
        self.assertEqual(meta["selected_sites"][0]["endpoint_check"]["reconstructed_l_mismatch_count"], 1)

    def test_idempotent_reuse_and_conflicting_candidate_refused(self):
        first = self.export({"high": 0.5})
        before = Path(first["manifest_path"]).read_bytes()
        second = self.export({"high": 0.5})
        self.assertEqual(first, second)
        with self.assertRaises(FileExistsError):
            self.export({"high": 0.75})
        self.assertEqual(before, Path(first["manifest_path"]).read_bytes())

    def test_two_sites_can_share_parameter_and_fingerprint_ignores_output_dir(self):
        sites = self.sites + [{"site": "low/1", "parameter": "high", "kit_path": "kit.npz"}]
        first = self.export({"high": 0.25}, "a", sites)
        second = self.export({"high": 0.25}, "b", sites)
        self.assertEqual(first["fingerprint"], second["fingerprint"])
        self.assertEqual(first["selected_site_count"], 2)

    def test_execution_fingerprint_identifies_bf16_plateaus(self):
        first = self.export({"high": 0.5001}, "plateau_a")
        second = self.export({"high": 0.5002}, "plateau_b")
        changed = self.export({"high": 0.51}, "different_bits")
        np.testing.assert_array_equal(self.candidate_bits(first), self.candidate_bits(second))
        self.assertEqual(first["fingerprint"], second["fingerprint"])
        self.assertNotEqual(first["provenance_fingerprint"], second["provenance_fingerprint"])
        self.assertNotEqual(first["fingerprint"], changed["fingerprint"])

    def test_same_execution_does_not_overwrite_different_provenance(self):
        first = self.export({"high": 0.5001}, "same_directory")
        metadata_before = Path(first["metadata_path"]).read_bytes()
        manifest_before = Path(first["manifest_path"]).read_bytes()
        with self.assertRaises(FileExistsError):
            self.export({"high": 0.5002}, "same_directory")
        self.assertEqual(metadata_before, Path(first["metadata_path"]).read_bytes())
        self.assertEqual(manifest_before, Path(first["manifest_path"]).read_bytes())

    def test_nonfinite_gamma_shape_and_sha_rejected(self):
        for gamma in (0., -1., float("nan")):
            with self.subTest(gamma=gamma):
                self.kit["gamma_L"] = np.array([gamma, 1.])
                self.save_kit()
                with self.assertRaises(ValueError):
                    self.export({"high": 0.5})
        self.kit["gamma_L"] = np.array([2., 1.])
        self.kit["P64"] = np.ones((2, 2))
        self.save_kit()
        with self.assertRaisesRegex(ValueError, "shape"):
            self.export({"high": 0.5})
        bad_sites = [dict(self.sites[0], kit_sha256="0" * 64)]
        with self.assertRaisesRegex(ValueError, "Kit SHA256"):
            self.export({"high": 0.5}, sites=bad_sites)

    def test_rejects_uncontrolled_parameter_dimensions_and_duplicate_sites(self):
        for theta in ({}, {"high": 0., "unused": 0.}, {"high": True}, {"high": float("inf")}):
            with self.subTest(theta=theta), self.assertRaises(ValueError):
                self.export(theta)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.export({"high": 0}, sites=self.sites * 2)

    def test_selected_payload_sha_checked_and_untouched_payload_preflight(self):
        # Tamper the selected old payload: export checks its real SHA.
        np.save(self.root / "old.npy", np.zeros((3, 3), dtype=np.uint16))
        with self.assertRaisesRegex(ValueError, "Matrix SHA256"):
            self.export({"high": 0})
        np.save(self.root / "old.npy", self.base_bits)
        result = self.export({"high": 0})
        # After export, alter only the old payload still used by untouched sites.
        np.save(self.root / "old.npy", np.zeros((3, 3), dtype=np.uint16))
        header = preflight_manifest(result["manifest_path"], expected_width=3,
                                    expected_layers=2, verify_hashes=False)
        self.assertFalse(header["payload_sha256_and_finite_verified"])
        with self.assertRaisesRegex(ValueError, "Matrix SHA256"):
            preflight_manifest(result["manifest_path"], expected_width=3, expected_layers=2)

    def test_production_site_count_with_small_matrix_fixture(self):
        self.entries = {f"{e}/{i}": dict(self.entries["high/0"])
                        for e in ("high", "low") for i in range(40)}
        self.manifest.write_text(json.dumps(self.entries))
        result = export_candidate(self.manifest, self.sites, {"high": 0.5},
                                  self.root / "all80", expected_width=3)
        self.assertEqual(result["total_site_count"], 80)
        self.assertEqual(len(json.loads(Path(result["metadata_path"]).read_text())["untouched_sites"]), 79)


if __name__ == "__main__":
    unittest.main()
