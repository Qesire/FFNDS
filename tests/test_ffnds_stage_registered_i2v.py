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

    def test_custom_image_lookup_matches_official_vbench_prompt_extractor(self):
        import re
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)
            image = p / "original.png"
            image.write_bytes(b"actual input")
            vids = []
            for i in range(2):
                v = p / f"clip{i}.mp4"
                v.write_bytes(f"real separately generated clip{i}".encode())
                vids.append({"source_id": "s", "variant": "dense", "sample_index": i,
                             "video_path": str(v)})
            complete = {"sources": [{"source_id": "s", "image_id": "i",
                       "prompt": "camera pans left", "image_name": "original.png",
                       "image_path": str(image)}],
                        "samples": vids, "mode": "CUSTOM_PAIRED"}
            stage(complete, [{"prompt_en": "camera pans left",
                             "image_name": "original.png",
                             "dimension": ["i2v_subject"]}], p / "staged")
            videos = sorted((p / "staged/dense/i2v_subject/videos").glob("*.mp4"))
            photos = list((p / "staged/dense/i2v_subject/images").glob("*"))
            self.assertEqual(len(videos), 2)
            self.assertEqual(len(photos), 1)
            self.assertEqual(photos[0].stem, "camera pans left")
            for video in videos:
                key = re.sub(r"-\\d+$", "", video.stem)
                self.assertTrue((photos[0].parent / (key + ".png")).is_file())

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
