"""End-to-end protocol checks with a deterministic analytic argv oracle.

These temporary fixtures generate JSON only; no video, model, or benchmark is
loaded.  They exercise the same external-command/cache route as a real caller.
"""

from __future__ import annotations

import copy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.video_metric_feedback.workflow import digest, load_config, read_json, run_search, write_json


ORACLE_SOURCE = r'''
import hashlib
import json
from pathlib import Path
import sys

request = json.loads(Path(sys.argv[1]).read_text())
parameters = request["parameters"]
high, low = parameters["high"], parameters["low"]
with Path(sys.argv[3]).open("a") as stream:
    stream.write(json.dumps({"parameters": parameters, "evaluation_key": request["evaluation_key"]}) + "\n")
if request["oracle_identity"].get("fail_perturbation") and (high != 1.0 or low != 1.0):
    raise SystemExit(9)
quality = 1.0 - 0.25 * ((high - 0.5) ** 2 + (low - 0.5) ** 2)
motion = 1.0 - 0.05 * (2.0 - high - low)
records = []
for sample in request["cohort"]:
    source, seed, index = sample["source_id"], sample["seed"], sample.get("sample_index", 0)
    identity = repr((source, seed, index))
    if request["oracle_identity"].get("wrong_noise") and (high != 1.0 or low != 1.0):
        identity += repr((high, low))
    records.append({
        "source_id": source, "image_id": sample["image_id"], "seed": seed,
        "sample_index": index,
        "metrics": {"quality": quality - (0.01 if source == "source-b" else 0.0), "motion": motion},
        "initial_noise_sha256": hashlib.sha256(identity.encode()).hexdigest(),
    })
bundle = {
    "schema_version": 1, "candidate_id": request["candidate_id"],
    "evaluation_key": request["evaluation_key"],
    "execution_fingerprint": request["execution_fingerprint"],
    "synthetic": True, "records": records,
}
Path(sys.argv[2]).write_text(json.dumps(bundle))
'''


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.script = self.directory / "analytic_oracle.py"
        self.script.write_text(ORACLE_SOURCE)
        self.counter = self.directory / "oracle_calls.jsonl"
        self.output = self.directory / "run"
        self.path = self.directory / "config.json"
        self.config = {
            "schema_version": 1, "mode": "synthetic_unit_fixture",
            "parameters": [
                {"name": name, "initial": 1.0, "lower": 0.0, "upper": 2.0,
                 "finite_difference_step": 0.1, "trust_radius": 0.25}
                for name in ("high", "low")
            ],
            "cohort": [
                {"source_id": "source-a", "image_id": "image-a", "seed": 1, "sample_index": 0, "split": "CAL"},
                {"source_id": "source-a", "image_id": "image-a", "seed": 2, "sample_index": 0, "split": "CAL"},
                {"source_id": "source-b", "image_id": "image-b", "seed": 1, "sample_index": 0, "split": "CAL"},
            ],
            "metrics": {
                "objectives": [{"name": "quality", "direction": "max", "weight": 1.0, "scale": 1.0}],
                "guards": [{"name": "motion", "direction": "max", "max_drop": 0.05, "level": "cluster"}],
            },
            "optimizer": {"max_iterations": 1, "max_evaluations": 6, "damping": 0.1},
            "oracle": {
                "command": [sys.executable, str(self.script), "{request}", "{result}", str(self.counter)],
                "cwd": str(self.directory), "timeout_seconds": 10,
                "require_initial_noise_identity": True,
                "identity": {"generator": "analytic-json-fixture", "evaluator": "analytic-quality-motion", "recipe": "quadratic-v1"},
            },
        }

    def tearDown(self):
        self.temporary.cleanup()

    def load(self):
        write_json(self.path, self.config)
        return load_config(self.path)

    def calls(self):
        return [] if not self.counter.exists() else [json.loads(line) for line in self.counter.read_text().splitlines()]

    def test_external_argv_update_budget_and_persistent_resume(self):
        config = self.load()
        result = run_search(config, self.output)
        self.assertEqual(result["optimization"]["status"], "accepted")
        self.assertEqual(result["external_calls"], 6)
        self.assertEqual(result["external_attempts_total"], 6)
        self.assertEqual(len(self.calls()), 6)
        self.assertEqual(result["plan"]["max_candidate_videos"], 18)
        self.assertEqual(result["plan"]["independent_image_clusters"], 2)
        self.assertLess(result["parameters"]["high"], 1.0)
        self.assertLess(result["parameters"]["low"], 1.0)
        self.assertEqual(set(result["optimization"]["evaluation"]["unit_losses"]), {"image-a", "image-b"})
        self.assertTrue(all(value >= 0.0 for value in result["optimization"]["evaluation"]["guard_margins"].values()))
        resumed = run_search(config, self.output)
        self.assertEqual(resumed["external_calls"], 0)
        self.assertEqual(resumed["external_attempts_total"], 6)
        self.assertEqual(resumed["parameters"], result["parameters"])
        self.assertEqual(len(self.calls()), 6)

    def test_changed_context_cannot_reuse_output_directory(self):
        run_search(self.load(), self.output)
        self.config["oracle"]["identity"]["recipe"] = "different-recipe"
        with self.assertRaisesRegex(ValueError, "different cohort, recipe or configuration"):
            run_search(self.load(), self.output)
        self.assertEqual(len(self.calls()), 6)

    def test_motion_guard_stays_relative_to_original_reference_across_updates(self):
        self.config["optimizer"].update(max_iterations=2, max_evaluations=11)
        self.config["metrics"]["guards"][0]["max_drop"] = 0.04
        result = run_search(self.load(), self.output)
        self.assertEqual(result["optimization"]["accepted_steps"], 2)
        self.assertLessEqual(len(self.calls()), 11)
        selected = read_json(self.output / "selected_metrics.json")
        self.assertTrue(all(row["metrics"]["motion"] >= 0.96 for row in selected["records"]))
        self.assertAlmostEqual(result["parameters"]["high"], 0.6, places=7)
        self.assertAlmostEqual(result["parameters"]["low"], 0.6, places=7)

    def test_noise_mismatch_is_rejected_before_valid_cache_commit(self):
        self.config["oracle"]["identity"]["wrong_noise"] = True
        with self.assertRaisesRegex(ValueError, "different initial noise"):
            run_search(self.load(), self.output)
        self.assertEqual(len(self.calls()), 2)
        self.assertEqual(len(list((self.output / "evaluations").glob("*/metrics.json"))), 1)

    def test_failed_attempts_survive_restart_and_cannot_exceed_total_budget(self):
        self.config["oracle"]["identity"]["fail_perturbation"] = True
        config = self.load()
        # The first invocation evaluates the center, then fails its first probe.
        # Four restarts reuse the center and each consume one additional attempt.
        for _ in range(5):
            with self.assertRaises(subprocess.CalledProcessError):
                run_search(config, self.output)
        self.assertEqual(len(self.calls()), 6)
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "attempt budget exhausted"):
                run_search(config, self.output)
        self.assertEqual(len(self.calls()), 6)
        attempts = [json.loads(line) for line in (self.output / "oracle_attempts.jsonl").read_text().splitlines()]
        self.assertEqual(sum(event["phase"] == "started" for event in attempts), 6)
        self.assertEqual(sum(event["phase"] == "failed" for event in attempts), 5)
        self.assertEqual(len(list((self.output / "evaluations").glob("*/metrics.json"))), 1)

    def test_dev_is_rejected_before_any_external_call(self):
        self.config["cohort"][0]["split"] = "DEV"
        with self.assertRaisesRegex(ValueError, "CAL"):
            run_search(self.load(), self.output)
        self.assertEqual(self.calls(), [])

    def test_cached_execution_identity_is_rechecked(self):
        config = self.load()
        run_search(config, self.output)
        initial = next(row for row in self.calls() if row["parameters"] == {"high": 1.0, "low": 1.0})
        cached = self.output / "evaluations" / initial["evaluation_key"] / "metrics.json"
        bundle = read_json(cached)
        bundle["execution_fingerprint"] = "0" * 64
        write_json(cached, bundle)
        with self.assertRaisesRegex(ValueError, "execution|fingerprint|matrices"):
            run_search(config, self.output)
        self.assertEqual(len(self.calls()), 6)

    def test_optimizer_and_metric_configuration_fail_before_rendering(self):
        for mutation in ("damping", "unknown_option", "metric_scale"):
            with self.subTest(mutation=mutation):
                config = copy.deepcopy(self.config)
                if mutation == "damping":
                    config["optimizer"]["damping"] = 0.0
                elif mutation == "unknown_option":
                    config["optimizer"]["unsupported_strategy"] = "anything"
                else:
                    config["metrics"]["objectives"][0]["scale"] = 0.0
                with self.assertRaises((ValueError, TypeError)):
                    run_search(config, self.output / mutation)
        self.assertEqual(self.calls(), [])

    def test_plain_callback_also_obeys_the_evaluation_budget(self):
        calls = []
        config = self.load()

        def callback(theta):
            values = {"high": float(theta[0]), "low": float(theta[1])}
            calls.append(values)
            quality = 1.0 - 0.25 * ((theta[0] - 0.5) ** 2 + (theta[1] - 0.5) ** 2)
            motion = 1.0 - 0.05 * (2.0 - theta[0] - theta[1])
            fingerprint = digest(values)
            return {
                "schema_version": 1, "candidate_id": fingerprint,
                "evaluation_key": fingerprint, "execution_fingerprint": fingerprint,
                "synthetic": True,
                "records": [
                    {"source_id": row["source_id"], "image_id": row["image_id"],
                     "seed": row["seed"], "sample_index": row["sample_index"],
                     "metrics": {"quality": float(quality), "motion": float(motion)},
                     "initial_noise_sha256": hashlib.sha256(repr((row["source_id"], row["seed"], row["sample_index"])).encode()).hexdigest()}
                    for row in config["cohort"]
                ],
            }

        result = run_search(config, self.output, oracle=callback)
        self.assertEqual(result["optimization"]["status"], "accepted")
        self.assertEqual(len(calls), 6)
        self.assertEqual(len({digest(values) for values in calls}), 6)

    def assembly_fixture(self):
        row = self.config["cohort"][0]
        request = {
            "schema_version": 1, "mode": "synthetic_unit_fixture", "candidate_id": "fixture-candidate",
            "evaluation_key": "fixture-evaluation", "execution_fingerprint": "fixture-execution",
            "cohort": [dict(row)],
            "metrics": {"objectives": [{"name": "spatial_rgb_rmse", "direction": "min", "scale": 1.0, "weight": 1.0}]},
        }
        video = {key: row[key] for key in ("source_id", "image_id", "seed", "sample_index")}
        video.update(video_path=str(self.directory / "fixture.mp4"), initial_noise_sha256="a" * 64)
        report = {
            "schema": "ffnds.video.metrics.v1", "synthetic": True,
            "identity": {key: row[key] for key in ("source_id", "image_id", "seed", "sample_index")},
            "candidate": {"path": video["video_path"]},
            "metrics": {"paired_spatial": {"RGB_RMSE_0to1": {"mean": 0.1}}},
        }
        return request, video, report

    def test_assemble_local_diagnostics_without_official_bundle(self):
        from tools.video_metric_feedback.__main__ import assemble_result
        request, video, report = self.assembly_fixture()
        result = assemble_result(request, [video], [], [report])
        self.assertTrue(result["synthetic"])
        self.assertEqual(result["candidate_id"], request["candidate_id"])
        self.assertEqual(result["records"][0]["metrics"]["spatial_rgb_rmse"], 0.1)
        self.assertEqual(result["records"][0]["initial_noise_sha256"], "a" * 64)

    def test_assemble_rejects_extra_generation_samples(self):
        from tools.video_metric_feedback.__main__ import assemble_result
        request, video, report = self.assembly_fixture()
        extra = {**video, "source_id": "unrequested-source", "image_id": "unrequested-image"}
        with self.assertRaisesRegex(ValueError, "exactly match"):
            assemble_result(request, [video, extra], [], [report])

    def test_cli_refuses_synthetic_generation_manifest_for_video_request(self):
        from tools.video_metric_feedback.__main__ import main
        request, video, report = self.assembly_fixture()
        request["mode"] = "video"
        report.pop("synthetic")
        request_path, manifest_path, report_path = (self.directory / name for name in ("request.json", "videos.json", "local.json"))
        write_json(request_path, request)
        write_json(manifest_path, {**{key: request[key] for key in ("candidate_id", "evaluation_key", "execution_fingerprint")}, "synthetic": True, "records": [video]})
        write_json(report_path, report)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "synthetic"):
            main(["assemble", "--request", str(request_path), "--videos-manifest", str(manifest_path), "--local-report", str(report_path), "--output", str(self.directory / "assembled.json")])

    def test_import_cli_preserves_generation_synthetic_label(self):
        from tools.video_metric_feedback.__main__ import main
        request, video, _ = self.assembly_fixture()
        manifest_path, scores_path, output_path = (self.directory / name for name in ("videos.json", "official.json", "imported.json"))
        write_json(manifest_path, {"synthetic": True, "records": [video]})
        write_json(scores_path, {"imaging_quality": [0.75, [{"video_path": video["video_path"], "video_results": 75.0}]]})
        with contextlib.redirect_stdout(io.StringIO()):
            main(["import-vbench", "--results", str(scores_path), "--videos-manifest", str(manifest_path), "--candidate-id", request["candidate_id"], "--revision", "test-fixture", "--output", str(output_path)])
        self.assertTrue(read_json(output_path)["synthetic"])


if __name__ == "__main__":
    unittest.main()
