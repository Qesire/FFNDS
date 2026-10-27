"""The evaluation bridge accepts complete original video receipts only."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from ffnds_stage_registered_i2v import valid_name, collect, stage
from ffnds_experiment_plans import plan_video, save_new


class StageTests(unittest.TestCase):
    def test_filename_safety(self):
        with self.assertRaises(ValueError):
            valid_name("../unsafe")
        self.assertEqual(valid_name("camera pans left-0"), "camera pans left-0")

    def test_missing_native_receipts_fail(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)
            bank = {"schema": "ffnds.source.bank.v1", "sources": [
                {"source_id": "s", "image_id": "i", "prompt_id": "p",
                 "conditioning_id": "c", "role": "DEVELOPMENT",
                 "scene_stratum": "i2v_subject", "prompt": "animal",
                 "image_name": "animal.jpg", "image_path": str(p / "animal.jpg"),
                 "exposure_history": {"prior_evaluation": True}}]}
            save_new(p / "bank.json", bank)
            plan = plan_video(p / "bank.json", role="DEVELOPMENT",
                              strata="i2v_subject:1", seed=42, samples=1,
                              variants=["dense", "UB1"], experiment="R1",
                              frames=81, fps=16, width=832, height=464,
                              scheduler="4step", output=p / "plan.json")
            with self.assertRaises(FileNotFoundError):
                collect(plan, p / "runs")

    def test_staging_requires_exact_original_metadata(self):
        complete = {"sources": [{"source_id": "src", "image_id": "img", "prompt": "p",
                                 "image_name": "p.jpg", "image_path": "/not/read.jpg"}],
                    "samples": [{"source_id": "src", "variant": "dense", "sample_index": 0,
                                 "video_path": "/not/read.mp4"}], "mode": "CUSTOM_PAIRED"}
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "pinned original"):
                stage(complete, [{"prompt_en": "OTHER", "image_name": "p.jpg",
                                  "dimension": ["i2v_subject"]}], Path(root) / "stage")


if __name__ == "__main__":
    unittest.main()
