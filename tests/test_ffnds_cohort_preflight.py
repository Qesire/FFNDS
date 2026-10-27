"""CPU-only preflight regressions, with synthetic data only."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from ffnds_cohort_preflight import audit


def dataset(mode="CUSTOM_PAIRED", count=1):
    sources = [{"source_id": "srcA", "image_id": "imgA", "prompt_id": "pA",
                "role": "FROZEN_CONFIRMATION", "exposure_history": {}}]
    samples = []
    for variant in ("dense", "candidate"):
        for i in range(count):
            samples.append({"source_id": "srcA", "variant": variant, "seed": 100 + i,
                "sample_index": i, "run_id": variant + "-run-" + str(i),
                "asset_id": variant + "-asset-" + str(i), "conditioning_id": "conditionA",
                "initial_noise_id": "noise-" + str(i), "scheduler_id": "four-step",
                "frames": 81, "fps": 16, "width": 832, "height": 464})
    return {"schema": "ffnds.cohort.v1", "mode": mode,
            "sources": sources, "samples": samples, "calibration": []}


class CohortTests(unittest.TestCase):
    def test_paired_custom_single_video_passes(self):
        self.assertEqual(audit(dataset())["status"], "PASS")

    def test_five_real_runs_for_official(self):
        self.assertEqual(audit(dataset("OFFICIAL_I2V_5SAMPLE", 5))["status"], "PASS")

    def test_historical_five_aliases_fail(self):
        d = dataset("OFFICIAL_I2V_5SAMPLE", 5)
        for x in d["samples"]:
            x["run_id"] = x["variant"] + "-same-run"
            x["asset_id"] = x["variant"] + "-same-video"
        self.assertEqual(audit(d)["status"], "INVALID")
        self.assertTrue(any("reused run_id" in e for e in audit(d)["errors"]))

    def test_five_samples_are_not_full_official_leaderboard(self):
        result = audit(dataset("OFFICIAL_I2V_5SAMPLE", 5))
        self.assertFalse(result["summary"]["official_comparability_verified"])
        self.assertTrue(any("Image Suite" in x for x in result["warnings"]))

    def test_one_video_with_five_aliases_is_not_official(self):
        d = dataset("OFFICIAL_I2V_5SAMPLE", 1)
        self.assertIn("five distinct", " ".join(audit(d)["errors"]))

    def test_exposed_confirmation_fails(self):
        d = dataset()
        d["sources"][0]["exposure_history"]["prior_evaluation"] = True
        self.assertIn("previously exposed", " ".join(audit(d)["errors"]))

    def test_same_image_across_splits_fails(self):
        d = dataset()
        d["sources"].append({"source_id": "srcB", "image_id": "imgA",
            "prompt_id": "pB", "role": "CALIBRATION"})
        self.assertIn("multiple data roles", " ".join(audit(d)["errors"]))

    def test_pair_noise_mismatch_fails(self):
        d = dataset()
        d["samples"][-1]["initial_noise_id"] = "different"
        self.assertIn("pair mismatch", " ".join(audit(d)["errors"]))

    def test_calibration_row_identity_and_state(self):
        d = dataset()
        d["sources"].append({"source_id": "srcB", "image_id": "imgB",
            "prompt_id": "pB", "role": "CALIBRATION"})
        d["calibration"] = [{"source_id": "srcB", "step": 0, "block": 0,
            "source_state": "student_input", "population_rows": 100,
            "n_rows": 3, "selected_row_ids": [2, 4, 7]}]
        self.assertEqual(audit(d)["status"], "PASS")
        d["calibration"][0]["selected_row_ids"] = [2, 2, 7]
        self.assertIn("sampled row coordinates", " ".join(audit(d)["errors"]))

    def test_byte_identical_copy_fails_when_files_available(self):
        d = dataset("OFFICIAL_I2V_5SAMPLE", 5)
        with tempfile.TemporaryDirectory() as folder:
            for row in d["samples"]:
                path = Path(folder) / (row["asset_id"] + ".mp4")
                path.write_bytes(b"same-video")
                row["video_path"] = str(path)
            self.assertIn("byte-identical", " ".join(audit(d, verify_files=True)["errors"]))


if __name__ == "__main__":
    unittest.main()
