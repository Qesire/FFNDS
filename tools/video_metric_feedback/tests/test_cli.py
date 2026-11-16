"""CLI sealing of measured-record fixtures, without generation or scoring models."""
from __future__ import annotations

import contextlib
import hashlib
import io
import tempfile
from pathlib import Path
import unittest

from tools.video_metric_feedback.__main__ import assemble_result, main
from tools.video_metric_feedback.workflow import read_json, write_json


class AssemblyCLITests(unittest.TestCase):
    def test_local_diagnostic_binds_exact_dense_reference_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            reference = Path(temporary) / "dense.mp4"
            reference.write_bytes(b"synthetic reference fixture, not a video")
            reference_sha = hashlib.sha256(reference.read_bytes()).hexdigest()
            sample = {"source_id": "s", "image_id": "i", "seed": 1,
                      "reference_video_path": str(reference)}
            request = {"mode": "synthetic_unit_fixture", "candidate_id": "c", "evaluation_key": "e",
                       "execution_fingerprint": "f", "cohort": [sample],
                       "metrics": {"objectives": [{"name": "flow_l2", "direction": "min",
                                                    "weight": 1.0, "scale": 1.0}]}}
            video = {"source_id": "s", "image_id": "i", "seed": 1,
                     "video_path": str(Path(temporary) / "candidate.mp4")}
            report = {"schema": "ffnds.video.metrics.v1", "synthetic": True,
                      "identity": {"source_id": "s", "seed": 1},
                      "candidate": {"path": video["video_path"]},
                      "reference": {"path": str(reference), "sha256": reference_sha},
                      "metrics": {"paired_motion": {"downsampled_Farneback_flow_L2_px": {"mean": 0.2}}}}
            self.assertEqual(assemble_result(request, [video], [], [report])["records"][0]["metrics"]["flow_l2"], 0.2)
            report["reference"]["path"] = str(Path(temporary) / "other.mp4")
            with self.assertRaisesRegex(ValueError, "different Dense"):
                assemble_result(request, [video], [], [report])
            report["reference"]["path"] = str(reference)
            reference.write_bytes(b"changed reference after scoring")
            with self.assertRaisesRegex(ValueError, "reference SHA256"):
                assemble_result(request, [video], [], [report])

    def test_complete_manifest_local_only_cli_and_matrix_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix = root / "matrix_manifest.json"
            write_json(matrix, {"fixture": "not a model matrix"})
            matrix_sha = hashlib.sha256(matrix.read_bytes()).hexdigest()
            sample = {"source_id": "source-a", "image_id": "image-a", "seed": 1,
                      "sample_index": 0}
            video = {**sample, "video_path": str(root / "fixture.mp4"),
                     "initial_noise_sha256": "a" * 64}
            request = {
                "schema_version": 1, "mode": "synthetic_unit_fixture",
                "candidate_id": "candidate-a", "evaluation_key": "evaluation-a",
                "execution_fingerprint": "execution-a", "cohort": [sample],
                "matrix_manifest_path": str(matrix),
                "matrix_export": {"fingerprint": "execution-a", "matrix_manifest_sha256": matrix_sha},
                "metrics": {"objectives": [{"name": "spatial_rgb_rmse", "direction": "min",
                                             "weight": 1.0, "scale": 1.0}]},
            }
            manifest = {"schema_version": 1, "schema": "ffnds.video_metric_feedback.videos.v1",
                        "synthetic": True, "records": [video],
                        "matrix_manifest_sha256": matrix_sha,
                        **{key: request[key] for key in ("candidate_id", "evaluation_key",
                                                        "execution_fingerprint", "matrix_manifest_path")}}
            report = {"schema": "ffnds.video.metrics.v1", "synthetic": True,
                      "identity": sample, "candidate": {"path": video["video_path"]},
                      "metrics": {"paired_spatial": {"RGB_RMSE_0to1": {"mean": 0.125}}}}
            for name, data in (("request", request), ("videos", manifest), ("local", report)):
                write_json(root / f"{name}.json", data)
            argv = ["assemble", "--request", str(root / "request.json"), "--videos-manifest",
                    str(root / "videos.json"), "--local-report", str(root / "local.json"),
                    "--output", str(root / "result.json")]
            with contextlib.redirect_stdout(io.StringIO()):
                main(argv)
            result = read_json(root / "result.json")
            self.assertTrue(result["synthetic"])
            self.assertEqual(result["records"][0]["metrics"]["spatial_rgb_rmse"], 0.125)
            matrix.write_text("changed after generation\n")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                main(argv)


if __name__ == "__main__":
    unittest.main()
