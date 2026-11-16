"""Exercise official-API orchestration using tiny fake compute functions only."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.video_metric_feedback.vbench_worker import (
    PINNED_REVISION, _verify_checkout, evaluate_vbench, main, validate_videos_manifest,
)


class VBenchWorkerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def sample(self, source, eligible):
        movie = self.root / f"{source}.mp4"
        image = self.root / f"{source}.png"
        movie.touch()
        image.touch()
        row = {
            "source_id": source, "image_id": source + "-image", "seed": 1234,
            "sample_index": 0, "video_path": str(movie), "image_path": str(image),
            "prompt": "The subject moves.", "eligible_metrics": eligible,
            "initial_noise_sha256": "a" * 64,
        }
        return row

    def request(self, rows, objectives, guards=()):
        return {
            "schema_version": 1, "candidate_id": "candidate-1", "mode": "video",
            "evaluation_key": "key", "execution_fingerprint": "execution",
            "cohort": copy.deepcopy(rows),
            "metrics": {
                "objectives": [{"name": name, "direction": "max", "weight": 1.0, "scale": 1.0}
                               for name in objectives],
                "guards": list(guards),
            },
        }

    def generation_manifest(self, request, rows, *, synthetic=False):
        matrix = self.root / "matrix_manifest.json"
        matrix.write_text("{}\n")
        matrix_sha = hashlib.sha256(matrix.read_bytes()).hexdigest()
        request["matrix_manifest_path"] = str(matrix)
        request["matrix_export"] = {"fingerprint": request["execution_fingerprint"],
                                    "matrix_manifest_sha256": matrix_sha}
        return {
            "schema_version": 1, "schema": "ffnds.video_metric_feedback.videos.v1",
            **{field: request[field] for field in (
                "candidate_id", "evaluation_key", "execution_fingerprint", "matrix_manifest_path",
            )},
            "matrix_manifest_sha256": matrix_sha, "synthetic": synthetic,
            "records": copy.deepcopy(rows),
        }

    def test_grouped_dimensions_explicit_i2v_image_and_musiq_units(self):
        rows = [self.sample("s1", ["i2v_subject", "imaging_quality"]),
                self.sample("s2", ["subject_consistency", "imaging_quality"])]
        request = self.request(rows, ["i2v_subject", "subject_consistency", "imaging_quality"])
        calls = []

        def fake_compute(dimension, metadata_path):
            information = json.loads(Path(metadata_path).read_text())
            calls.append((dimension, information))
            details = []
            for item in information:
                self.assertEqual(item["dimension"], [dimension])
                self.assertEqual(len(item["video_list"]), 1)
                detail = {"video_path": item["video_list"][0],
                          "video_results": 80.0 if dimension == "imaging_quality" else 0.9}
                if dimension == "i2v_subject":
                    self.assertEqual(item["custom_image_path"], rows[0]["image_path"])
                    self.assertNotIn("image_name", item)
                    detail["image_path"] = item["custom_image_path"]
                details.append(detail)
            return 0.8, details

        with patch("tools.video_metric_feedback.vbench_worker._official_compute") as heavy:
            result = evaluate_vbench(request, rows, self.root / "work", compute=fake_compute)
            heavy.assert_not_called()
        self.assertEqual([name for name, _ in calls], ["i2v_subject", "imaging_quality", "subject_consistency"])
        self.assertEqual([len(info) for _, info in calls], [1, 2, 1])
        self.assertEqual(result["official_evaluator_calls"], 3)
        self.assertTrue(result["synthetic"])
        self.assertEqual(result["records"][0]["metrics"], {"i2v_subject": 0.9, "imaging_quality": 0.8})
        self.assertEqual(result["records"][1]["metrics"], {"imaging_quality": 0.8, "subject_consistency": 0.9})
        self.assertEqual(result["records"][0]["initial_noise_sha256"], "a" * 64)
        self.assertEqual(result["execution_fingerprint"], "execution")

    def test_camera_stages_explicit_instruction_and_restores_original_path(self):
        row = self.sample("camera", ["camera_motion"])
        # Deliberately no instruction in the free-text prompt or native filename.
        row["camera_instruction"] = "camera pans left"
        request = self.request([row], ["camera_motion"])

        def fake_compute(dimension, metadata_path):
            information = json.loads(Path(metadata_path).read_text())
            staged = Path(information[0]["video_list"][0])
            self.assertTrue(staged.is_symlink())
            self.assertEqual(staged.resolve(), Path(row["video_path"]))
            self.assertIn("camera pans left", staged.name)
            self.assertEqual(information[0]["camera_instruction_source"], "cohort.camera_instruction")
            return (1.0, {"pan_left": 1.0, "pan_right": float("nan")}, [
                {"video_path": str(staged), "video_results": 1.0,
                 "prompt_type": "pan_left", "predict_type": ["pan_left"]}
            ])

        result = evaluate_vbench(request, [row], self.root / "camera_work", compute=fake_compute)
        self.assertEqual(result["records"][0]["video_path"], row["video_path"])
        self.assertEqual(result["records"][0]["metrics"]["camera_motion"], 1.0)
        raw = json.loads((self.root / "camera_work/official_results.json").read_text())
        self.assertIsNone(raw["camera_motion"][1]["pan_right"])

    def test_camera_requires_metadata_even_when_prompt_contains_instruction(self):
        row = self.sample("camera", ["camera_motion"])
        row["prompt"] = "camera pans left"
        request = self.request([row], ["camera_motion"])
        with patch("tools.video_metric_feedback.vbench_worker._official_compute") as heavy:
            with self.assertRaisesRegex(ValueError, "explicit official camera_instruction"):
                evaluate_vbench(request, [row], self.root / "missing_camera")
            heavy.assert_not_called()

    def test_i2v_condition_mismatch_fails_before_model_initialization(self):
        row = self.sample("subject", ["i2v_subject"])
        request = self.request([row], ["i2v_subject"])
        request["cohort"][0]["image_path"] = str(self.root / "different.png")
        with patch("tools.video_metric_feedback.vbench_worker._official_compute") as heavy:
            with self.assertRaisesRegex(ValueError, "condition image does not match"):
                evaluate_vbench(request, [row], self.root / "wrong_image")
            heavy.assert_not_called()

    def test_only_local_diagnostics_never_imports_or_calls_vbench(self):
        row = self.sample("local", ["temporal_delta_rmse", "flow_magnitude_ratio"])
        request = self.request([row], ["temporal_delta_rmse"], [
            {"name": "flow_magnitude_ratio", "direction": "max", "min_value": 0.8}
        ])
        with patch("tools.video_metric_feedback.vbench_worker._official_compute") as heavy:
            result = evaluate_vbench(request, [row], self.root / "local_only")
            heavy.assert_not_called()
        self.assertEqual(result["official_evaluator_calls"], 0)
        self.assertEqual(result["deferred_local_metrics"], ["flow_magnitude_ratio", "temporal_delta_rmse"])
        self.assertEqual(result["records"][0]["metrics"], {})
        self.assertFalse(result["synthetic"])

    def test_missing_video_return_is_rejected_in_single_video_call(self):
        row = self.sample("subject", ["subject_consistency"])
        request = self.request([row], ["subject_consistency"])
        with self.assertRaisesRegex(ValueError, "omitted eligible videos"):
            evaluate_vbench(request, [row], self.root / "incomplete",
                            compute=lambda dimension, info: (0.9, []))

    def test_checkout_revision_and_tracked_modifications_are_checked(self):
        expected_files = ["vbench/__init__.py", "vbench/utils.py", "vbench/subject_consistency.py",
                          "vbench2_beta_i2v/__init__.py", "vbench2_beta_i2v/utils.py"]
        repository = self.root / "repo"
        for name in expected_files:
            path = repository / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# source fixture\n")
        with patch("tools.video_metric_feedback.vbench_worker.subprocess.run", side_effect=[
            SimpleNamespace(stdout=PINNED_REVISION + "\n"), SimpleNamespace(stdout="")
        ]):
            result = _verify_checkout(repository, PINNED_REVISION, ["subject_consistency"])
        self.assertEqual(set(result["source_sha256"]), set(expected_files))
        self.assertTrue(result["tracked_source_clean"])
        with patch("tools.video_metric_feedback.vbench_worker.subprocess.run", side_effect=[
            SimpleNamespace(stdout=PINNED_REVISION + "\n"),
            SimpleNamespace(stdout=" M vbench/subject_consistency.py\n")
        ]), self.assertRaisesRegex(ValueError, "tracked modifications"):
            _verify_checkout(repository, PINNED_REVISION, ["subject_consistency"])
        with patch("tools.video_metric_feedback.vbench_worker.subprocess.run", return_value=
                   SimpleNamespace(stdout="b" * 40)), self.assertRaisesRegex(ValueError, "expected"):
            _verify_checkout(repository, PINNED_REVISION, ["subject_consistency"])

    def test_generation_manifest_request_and_matrix_identities_are_verified(self):
        row = self.sample("local", ["temporal_delta_rmse"])
        request = self.request([row], ["temporal_delta_rmse"])
        manifest = self.generation_manifest(request, [row])
        self.assertEqual(validate_videos_manifest(request, manifest), [row])
        for field in ("candidate_id", "evaluation_key", "execution_fingerprint",
                      "matrix_manifest_sha256", "matrix_manifest_path"):
            bad = {**manifest, field: "wrong"}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                validate_videos_manifest(request, bad)
        bad_request = {**request, "matrix_export": {**request["matrix_export"], "fingerprint": "wrong"}}
        with self.assertRaisesRegex(ValueError, "matrix_export fingerprint"):
            validate_videos_manifest(bad_request, manifest)
        Path(request["matrix_manifest_path"]).write_text('{"changed":true}\n')
        with self.assertRaisesRegex(ValueError, "recorded SHA256"):
            validate_videos_manifest(request, manifest)

    def test_cli_requires_complete_manifest_and_preserves_generated_synthetic(self):
        row = self.sample("local", ["temporal_delta_rmse"])
        request = self.request([row], ["temporal_delta_rmse"])
        request["mode"] = "synthetic_unit_fixture"
        manifest = self.generation_manifest(request, [row], synthetic=True)
        request_path, manifest_path, result_path = [self.root / name for name in (
            "request.json", "videos_manifest.json", "result.json",
        )]
        request_path.write_text(json.dumps(request))
        manifest_path.write_text(json.dumps(manifest))
        arguments = ["--request", str(request_path), "--videos-manifest", str(manifest_path),
                     "--result", str(result_path), "--vbench-repo", str(self.root / "unused")]
        with patch("tools.video_metric_feedback.vbench_worker._official_compute") as heavy:
            main(arguments)
            heavy.assert_not_called()
        result = json.loads(result_path.read_text())
        self.assertTrue(result["synthetic"])
        self.assertEqual(result["generation_identity"]["matrix_manifest_sha256"],
                         manifest["matrix_manifest_sha256"])
        with self.assertRaisesRegex(ValueError, "synthetic_unit_fixture"):
            validate_videos_manifest({**request, "mode": "video"}, manifest)
        manifest_path.write_text(json.dumps({**manifest, "candidate_id": "another-candidate"}))
        with self.assertRaisesRegex(ValueError, "candidate_id does not match"):
            main(arguments)
        manifest_path.write_text(json.dumps(manifest["records"]))
        with self.assertRaisesRegex(ValueError, "complete native generation manifest"):
            main(arguments)


if __name__ == "__main__":
    unittest.main()
