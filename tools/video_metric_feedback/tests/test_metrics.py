"""Unit checks for metric contracts and the independent statistical unit.

These fixtures contain no model outputs and run no video/model experiment.
"""

import copy
import math
import unittest

from tools.video_metric_feedback.metrics import (
    import_local_diagnostics,
    import_vbench_results,
    merge_record_sets,
    score_records,
    validate_metric_specification,
)


def video(source="s1", image="image1", seed=1, index=0, path=None, eligible=None):
    row = {
        "source_id": source, "image_id": image, "seed": seed,
        "sample_index": index, "video_path": path or f"/candidate/{source}-{seed}-{index}.mp4",
    }
    if eligible is not None:
        row["eligible_metrics"] = eligible
    return row


def bundle(rows, metrics, candidate="candidate"):
    records = []
    for row, values in zip(rows, metrics):
        record = copy.deepcopy(row)
        record["metrics"] = values
        records.append(record)
    return {"schema_version": 1, "candidate_id": candidate, "records": records}


def spec(name="quality", direction="max", weight=1.0, scale=1.0, guards=None):
    return {
        "objectives": [{"name": name, "weight": weight, "scale": scale, "direction": direction}],
        "guards": guards or [],
    }


class VBenchImportTest(unittest.TestCase):
    def test_per_video_musiq_scale_and_camera_three_tuple(self):
        row = video()
        data = {
            "imaging_quality": [0.75, [{"video_path": row["video_path"], "video_results": 75.0}]],
            "camera_motion": [1.0, {"pan_left": 1.0, "pan_right": float("nan")}, [
                {"video_path": row["video_path"], "video_results": 1.0,
                 "prompt_type": "pan_left", "predict_type": ["pan_left"]}
            ]],
            "dynamic_degree": [1.0, [{"video_path": row["video_path"], "video_results": True}]],
            "i2v_subject": [0.95, [{"video_path": row["video_path"], "video_results": 0.95,
                                    "image_path": "/images/one.png"}]],
        }
        result = import_vbench_results(data, [row], "candidate", "pinned-sha")
        record = result["records"][0]
        self.assertEqual(record["metrics"]["imaging_quality"], 0.75)
        self.assertEqual(record["raw_metrics"]["imaging_quality"], 75.0)
        self.assertEqual(record["metrics"]["dynamic_degree"], 1.0)
        self.assertEqual(record["metrics"]["camera_motion"], 1.0)
        self.assertEqual(record["image_path"], "/images/one.png")
        self.assertEqual(record["camera_motion_details"]["prompt_type"], "pan_left")
        self.assertEqual(record["metric_origins"]["imaging_quality"]["kind"], "official_vbench")

    def test_explicit_path_mapping_does_not_guess_basename(self):
        rows = [video(path="/a/same.mp4"), video("s2", "image2", path="/b/same.mp4")]
        data = {"subject_consistency": [0.5, [
            {"video_path": "/b/same.mp4", "video_results": 0.8},
            {"video_path": "/a/same.mp4", "video_results": 0.2},
        ]]}
        result = import_vbench_results(data, rows, "candidate", "sha")
        self.assertEqual([x["metrics"]["subject_consistency"] for x in result["records"]], [0.2, 0.8])
        data["subject_consistency"][1][0]["video_path"] = "/unknown/same.mp4"
        with self.assertRaisesRegex(ValueError, "unmapped video_path"):
            import_vbench_results(data, rows, "candidate", "sha")

    def test_partial_dimension_import_keeps_all_manifest_rows(self):
        rows = [video(), video("s2", "image2")]
        result = import_vbench_results({"subject_consistency": [0.9, [
            {"video_path": rows[0]["video_path"], "video_results": 0.9}
        ]]}, rows, "candidate", "sha")
        self.assertEqual(len(result["records"]), 2)
        self.assertEqual(result["records"][1]["metrics"], {})

    def test_initial_noise_identity_survives_both_imports_and_merge(self):
        row = video()
        row["initial_noise_sha256"] = "a" * 64
        official = import_vbench_results({"subject_consistency": [0.9, [
            {"video_path": row["video_path"], "video_results": 0.9}
        ]]}, [row], "candidate", "sha")
        report = {
            "schema": "ffnds.video.metrics.v1",
            "identity": {"source_id": row["source_id"], "seed": row["seed"]},
            "candidate": {"path": row["video_path"]}, "metrics": {},
        }
        local = import_local_diagnostics(report, row)
        merged = merge_record_sets(official, local)
        self.assertEqual(official["records"][0]["initial_noise_sha256"], "a" * 64)
        self.assertEqual(local["initial_noise_sha256"], "a" * 64)
        self.assertEqual(merged["records"][0]["initial_noise_sha256"], "a" * 64)

    def test_duplicate_nonfinite_and_wrong_camera_shape_fail(self):
        row = video()
        score = {"video_path": row["video_path"], "video_results": 0.9}
        cases = [
            {"subject_consistency": [0.9, [score, score]]},
            {"subject_consistency": [0.9, [{**score, "video_results": float("nan")}]]},
            {"camera_motion": [0.9, [score]]},
        ]
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValueError):
                import_vbench_results(data, [row], "candidate", "sha")


class LocalDiagnosticsTest(unittest.TestCase):
    def report(self, row):
        return {
            "schema": "ffnds.video.metrics.v1", "implementation": "fixture",
            "identity": {"source_id": row["source_id"], "seed": row["seed"]},
            "candidate": {"path": row["video_path"]},
            "reference": {"path": "/dense/one.mp4"},
            "metrics": {
                "paired_spatial": {"RGB_RMSE_0to1": {"mean": 0.02}},
                "paired_temporal": {"adjacent_delta_RGB_RMSE_0to1": {"mean": 0.03}},
                "paired_motion": {
                    "candidate_to_reference_flow_magnitude_ratio": {"mean": 0.8},
                    "downsampled_Farneback_flow_L2_px": {"mean": 0.15},
                },
            },
        }

    def test_known_paths_are_labelled_as_local(self):
        row = video()
        record = import_local_diagnostics(self.report(row), row)
        self.assertEqual(record["metrics"], {
            "flow_magnitude_ratio": 0.8, "temporal_delta_rmse": 0.03,
            "spatial_rgb_rmse": 0.02, "flow_l2": 0.15,
        })
        self.assertEqual(record["metric_origins"]["flow_l2"]["kind"], "ffnds_local_diagnostic")
        self.assertNotIn("dynamic_degree", record["metrics"])
        self.assertNotIn("temporal_flickering", record["metrics"])

    def test_wrong_identity_path_and_nonfinite_fail(self):
        row = video()
        reports = [self.report(row) for _ in range(4)]
        reports[0]["identity"]["seed"] += 1
        reports[1]["candidate"]["path"] = "/other/one.mp4"
        reports[2]["schema"] = "unrecognized"
        reports[3]["metrics"]["paired_motion"]["candidate_to_reference_flow_magnitude_ratio"]["mean"] = float("inf")
        for report in reports:
            with self.subTest(report=report), self.assertRaises(ValueError):
                import_local_diagnostics(report, row)

    def test_merge_combines_dimensions_and_diagnostics(self):
        row = video()
        official = import_vbench_results({"subject_consistency": [0.9, [
            {"video_path": row["video_path"], "video_results": 0.9}
        ]]}, [row], "candidate", "sha")
        local = import_local_diagnostics(self.report(row), row)
        merged = merge_record_sets(official, local)
        record = merged["records"][0]
        self.assertEqual(merged["evaluator"]["name"], "combined")
        self.assertEqual(record["metrics"]["subject_consistency"], 0.9)
        self.assertEqual(record["metrics"]["flow_magnitude_ratio"], 0.8)
        self.assertEqual(record["metric_origins"]["flow_magnitude_ratio"]["kind"], "ffnds_local_diagnostic")
        conflict = copy.deepcopy(local)
        conflict["metrics"]["subject_consistency"] = 0.7
        with self.assertRaisesRegex(ValueError, "conflicting metrics"):
            merge_record_sets(official, conflict)

    def test_merge_rejects_candidate_and_source_identity_conflicts(self):
        rows = [video()]
        a = bundle(rows, [{"quality": 0.5}], "a")
        b = bundle(rows, [{"quality": 0.5}], "b")
        with self.assertRaisesRegex(ValueError, "candidate_id"):
            merge_record_sets(a, b)
        b["candidate_id"] = "a"
        b["records"][0]["image_id"] = "other"
        with self.assertRaisesRegex(ValueError, "image_id"):
            merge_record_sets(a, b)

    def test_merge_preserves_request_identities_and_propagates_synthetic(self):
        base = bundle([video()], [{"quality": 0.5}])
        identity = {"execution_fingerprint": "execution", "evaluation_key": "evaluation"}
        marked = {**base, **identity, "synthetic": True}
        real = {**base, **identity, "synthetic": False}
        for parts in ((base, marked, real), (real, marked, base)):
            merged = merge_record_sets(*parts)
            self.assertTrue(merged["synthetic"])
            for field, value in identity.items():
                self.assertEqual(merged[field], value)
        self.assertFalse(merge_record_sets(base, real)["synthetic"])
        self.assertNotIn("synthetic", merge_record_sets(base))

    def test_merge_rejects_conflicting_request_identities_and_invalid_synthetic(self):
        base = bundle([video()], [{"quality": 0.5}])
        for field in ("execution_fingerprint", "evaluation_key"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                merge_record_sets({**base, field: "one"}, {**base, field: "two"})
        with self.assertRaisesRegex(ValueError, "synthetic must be a boolean"):
            merge_record_sets({**base, "synthetic": "false"})

    def test_reference_identity_is_checked_when_recorded(self):
        row = video()
        report = self.report(row)
        report["reference_identity"] = {"source_id": row["source_id"], "seed": 99}
        with self.assertRaisesRegex(ValueError, "reference has mismatched seed"):
            import_local_diagnostics(report, row)


class ClusterScoringTest(unittest.TestCase):
    def test_samples_then_sources_then_images_not_video_mean(self):
        rows = [video("s1", "i1", seed=1), video("s1", "i1", seed=2),
                video("s2", "i1"), video("s3", "i2")]
        candidate = bundle(rows, [{"quality": x} for x in (2, 4, 10, 5)])
        baseline = bundle(rows, [{"quality": x} for x in (1, 1, 0, 1)], "reference")
        for record in baseline["records"]:
            record["video_path"] = record["video_path"].replace("/candidate/", "/reference/")
        result = score_records(candidate, baseline, rows, spec())
        # i1 delta = ((mean(2,4)-1)+(10-0))/2 = 6; i2 delta = 4.
        self.assertEqual(result["loss"], -5.0)
        self.assertEqual(result["unit_losses"], {"i1": -6.0, "i2": -4.0})
        self.assertEqual(result["metrics"]["quality"], 5.75)

    def test_metric_eligibility_and_unit_loss_correction(self):
        rows = [video("s1", "i1", eligible=["quality", "error"]),
                video("s2", "i2", eligible=["quality"])]
        candidate = bundle(rows, [{"quality": 3, "error": 1}, {"quality": 4}])
        baseline = bundle(rows, [{"quality": 1, "error": 2}, {"quality": 1}], "reference")
        specification = spec()
        specification["objectives"].append({"name": "error", "direction": "min", "scale": 2, "weight": 3})
        result = score_records(candidate, baseline, rows, specification)
        self.assertEqual(result["loss"], -4.0)  # -2.5 quality -1.5 error.
        self.assertEqual(result["unit_losses"], {"i1": -5.0, "i2": -3.0})
        self.assertEqual(result["metric_cluster_counts"], {"error": 1, "quality": 2})
        self.assertEqual(sum(result["unit_losses"].values()) / 2, sum(result["objective_terms"].values()))

    def test_cluster_guards_detect_regression_hidden_by_aggregate(self):
        rows = [video("s1", "i1"), video("s2", "i2")]
        candidate = bundle(rows, [{"quality": 0.4}, {"quality": 0.8}])
        baseline = bundle(rows, [{"quality": 0.6}, {"quality": 0.6}], "reference")
        guards = [
            {"name": "quality", "direction": "max", "max_drop": 0.1},
            {"name": "quality", "direction": "max", "max_drop": 0.1, "level": "aggregate"},
        ]
        result = score_records(candidate, baseline, rows, spec(guards=guards))
        margins = result["guard_margins"]
        self.assertEqual(len(margins), 3)
        self.assertEqual(sum(value < 0 for value in margins.values()), 1)
        repeated = score_records(candidate, baseline, rows, spec(guards=guards))
        self.assertEqual(margins, repeated["guard_margins"])

    def test_min_direction_max_drop_and_absolute_bounds(self):
        rows = [video()]
        candidate = bundle(rows, [{"quality": 0.3}])
        baseline = bundle(rows, [{"quality": 0.2}], "reference")
        guards = [{"name": "quality", "direction": "min", "max_drop": 0.05,
                   "min_value": 0.1, "max_value": 0.4}]
        result = score_records(candidate, baseline, rows, spec(direction="min", guards=guards))
        self.assertAlmostEqual(result["loss"], 0.1)
        self.assertEqual(sum(x < 0 for x in result["guard_margins"].values()), 1)

    def test_sample_index_participates_in_key_and_defaults_to_zero(self):
        rows = [video(index=0), video(index=1)]
        candidate = bundle(rows, [{"quality": 0.6}, {"quality": 0.8}])
        baseline = bundle(rows, [{"quality": 0.5}, {"quality": 0.5}], "reference")
        candidate["records"][0].pop("sample_index")
        result = score_records(candidate, baseline, rows, spec())
        self.assertAlmostEqual(result["loss"], -0.2)

    def test_missing_extra_duplicate_and_inconsistent_image_fail(self):
        rows = [video(), video("s2", "image2")]
        reference = bundle(rows, [{"quality": 1}, {"quality": 1}], "reference")
        good = bundle(rows, [{"quality": 1}, {"quality": 1}])
        cases = []
        missing = copy.deepcopy(good); missing["records"].pop(); cases.append(missing)
        extra = copy.deepcopy(good); extra["records"].append({**video("s3", "i3"), "metrics": {"quality": 1}}); cases.append(extra)
        duplicate = copy.deepcopy(good); duplicate["records"].append(copy.deepcopy(duplicate["records"][0])); cases.append(duplicate)
        wrong_image = copy.deepcopy(good); wrong_image["records"][0]["image_id"] = "wrong"; cases.append(wrong_image)
        missing_metric = copy.deepcopy(good); missing_metric["records"][0]["metrics"] = {}; cases.append(missing_metric)
        nonfinite = copy.deepcopy(good); nonfinite["records"][0]["metrics"]["quality"] = float("inf"); cases.append(nonfinite)
        for candidate in cases:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                score_records(candidate, reference, rows, spec())

    def test_same_source_cannot_map_to_two_images(self):
        rows = [video("same", "i1", seed=1), video("same", "i2", seed=2)]
        panel = bundle(rows, [{"quality": 1}, {"quality": 1}])
        with self.assertRaisesRegex(ValueError, "different image_id"):
            score_records(panel, panel, rows, spec())

    def test_invalid_specification_fails(self):
        rows = [video()]
        panel = bundle(rows, [{"quality": 0.5}])
        cases = [spec(direction="up"), spec(scale=0), spec(scale=-1),
                 spec(scale=float("nan")), spec(weight=-1),
                 spec(guards=[{"name": "quality", "direction": "max", "max_drop": -0.1}]),
                 spec(guards=[{"name": "quality", "direction": "max", "min_value": 0.5, "max_value": 0.4}]),
                 spec(guards=[{"name": "quality", "direction": "max", "min_value": 0, "level": "sample"}])]
        for specification in cases:
            with self.subTest(specification=specification), self.assertRaises(ValueError):
                score_records(panel, panel, rows, specification)
            with self.subTest(early_validation=specification), self.assertRaises(ValueError):
                validate_metric_specification(specification)
        self.assertIsNone(validate_metric_specification(spec()))

    def test_empty_metric_eligibility_is_not_a_zero_score(self):
        rows = [video(eligible=[])]
        panel = bundle(rows, [{}])
        with self.assertRaisesRegex(ValueError, "no eligible image clusters"):
            score_records(panel, panel, rows, spec())


if __name__ == "__main__":
    unittest.main()
