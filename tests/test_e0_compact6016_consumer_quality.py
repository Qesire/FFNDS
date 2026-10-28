"""Synthetic CPU controls: provenance, bounded dispatch and paired statistics.

No real video decoding, metric model, Torch/NumPy import or GPU is involved.
The generation fixture replaces the separately tested upstream core validator
and collection implementation, so its tests isolate this driver's SHA graph.
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import e0_compact6016_consumer_quality as q


class Fixture:
    def __init__(self, root):
        self.repo, self.hf = root / "repo", root / "hf"
        self.repo.mkdir(); self.hf.mkdir()
        self.items, self.gens, self.regs, self.cores, self.phases = {}, {}, {}, {}, {}
        self.images = []
        self.cases = []
        for source in q.SOURCES:
            image = self.raw(f"images/{source}.jpg", f"image{source}".encode())
            self.images.append({"source": source, "name": f"image{source}.jpg", "binding": image})
            case = {"index": source, "source_id": f"source{source}", "image_sha256": image["sha256"],
                    "prompt": f"prompt{source}", "prompt_sha256": q.hashlib.sha256(f"prompt{source}".encode()).hexdigest(), "seed": 42}
            self.cases.append(case)
            for mode in q.MODES:
                key = source, mode
                folder = f"producer/{source}_{mode}"
                video = self.raw(folder + "/output.mp4", f"video{source}_{mode}".encode())
                noise = self.raw(folder + "/noise.safetensors", f"sameBF16noise{source}".encode())
                raw = self.raw(folder + "/raw.safetensors", f"sameFP32noise{source}".encode())
                item = {"source": source, "mode": mode, **{k: case[k] for k in ("source_id", "image_sha256", "prompt", "prompt_sha256", "seed")},
                        "image_name": f"image{source}.jpg", "video": video, "initial_noise": noise, "raw_noise": raw}
                self.items[key] = item
                self.gens[key] = {"source_index": source, "mode": mode, "source_id": case["source_id"], "image_sha256": case["image_sha256"],
                                  "prompt_sha256": case["prompt_sha256"], "seed": 42, "run_id": f"real{source}_{mode}",
                                  "status": "PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE", "completed_native_flow_steps": 4,
                                  "video": {**video, "path": str(self.hf / video["path"]), "frames": 81, "fps": 16, "size": [832, 464]},
                                  "initial_noise": noise, "raw_FP32_noise": raw}
                self.regs[key] = {"status": "COMPLETE_NATIVE_GENERATION_ONLY", "job": {"run_id": f"real{source}_{mode}"},
                                  "native_ffn_sites": 160, "scheduler_generator_reset": True, "initial_noise": noise,
                                  "video_path": str(self.hf / video["path"]), "decoded_frames": 81, "decoded_fps": 16, "decoded_size": [832, 464]}
                self.cores[key] = {"candidate": {"sha256": video["sha256"]},
                                   "identity": {"source_id": case["image_sha256"], "seed": 42, "expected_frames": 81,
                                                "input_prompt_digest": case["prompt_sha256"]}}
                if mode != "dense":
                    self.cores[key]["reference"] = {"sha256": self.items[source, "dense"]["video"]["sha256"]}
                    self.cores[key]["reference_identity"] = {}
                self.phases[key] = {**copy.deepcopy(self.gens[key]),
                                    "historical_comparison": {"passed": True},
                                    "actual_initial_noise_files_byte_equal_to_fresh_dense": True,
                                    "actual_raw_FP32_noise_byte_equal_to_fresh_dense": True}
            case["historical_dense"] = {"video": self.items[source, "dense"]["video"]}
        self.parent = {"id": q.PARENT_ID, "cases": self.cases, "limits": {"video_generations": 6}}
        self.parent_binding = self.doc(q.PARENT_PROTOCOL, self.parent, self.repo)
        for key in self.gens:
            self.gens[key]["protocol_sha256"] = self.parent_binding["sha256"]
            self.phases[key]["protocol_sha256"] = self.parent_binding["sha256"]
            self.cores[key]["identity"]["protocol_sha256"] = self.parent_binding["sha256"]
        sources = [{"source_id": c["source_id"], "image_id": c["image_sha256"], "image_name": f"image{c['index']}.jpg",
                    "image_path": str(self.hf / next(x["binding"]["path"] for x in self.images if x["source"] == c["index"]))} for c in self.cases]
        self.verified = {"phase": "COMPLETE_REGISTERED_GENERATION_VERIFIED", "sources": sources}
        self.plan = {"videos": list(self.items.values()), "images": self.images,
                     "parent": {"protocol": self.parent_binding,
                                "cohort_plan": self.doc("producer/plan.json", {"sources": sources}, self.hf),
                                "verified_cohort": self.doc("producer/VERIFIED_GENERATIONS.json", self.verified, self.hf)}}
        self.run = {"id": q.PARENT_ID, "status": q.PARENT_STATUS, "protocol_sha256": self.parent_binding["sha256"],
                    "automatic_retry": False, "operation_ledger": {p: {"video_generations": 6} for p in ("predebited", "observed")},
                    "assigned_GPU_resources": {"violations": []},
                    "verified_generation_cohort": {"sha256": self.plan["parent"]["verified_cohort"]["sha256"]}}
        self.resign()

    def raw(self, relative, value):
        p = self.hf / relative; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(value)
        return q.binding(p, self.hf)

    def doc(self, relative, value, base):
        p = base / relative; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(value))
        return q.binding(p, base)

    def resign(self):
        for source in q.SOURCES:
            for mode in q.MODES:
                key = source, mode; folder = f"producer/{source}_{mode}"
                item = self.items[key]
                item["generation"] = self.doc(folder + "/GENERATION.json", self.gens[key], self.hf)
                self.regs[key]["immutable_generation_receipt"] = item["generation"]
                item["registered"] = self.doc(folder + "/RUN.json", self.regs[key], self.hf)
                self.cores[key]["identity"]["run_sha256"] = item["generation"]["sha256"]
                if mode != "dense":
                    self.cores[key]["reference_identity"]["run_sha256"] = self.items[source, "dense"]["generation"]["sha256"]
                item["core"] = self.doc(folder + "/CORE.json", self.cores[key], self.hf)
                self.phases[key].update(generation_receipt=item["generation"], registered_receipt=item["registered"],
                                        core_metrics={"sha256": item["core"]["sha256"], "status": "ALL_81_FRAMES_CORE_METRICS_PASS_QUALITY_PENDING"})
        self.run["phases"] = list(self.phases.values())
        self.plan["parent"]["RUN"] = self.doc("producer/RUN.json", self.run, self.hf)
        exit = {"id": q.PARENT_ID, "protocol_sha256": self.parent_binding["sha256"], "child_returncode": 0,
                "process_exited": True, "timeout": False, "automatic_retry": False, "run_sha256": self.plan["parent"]["RUN"]["sha256"]}
        self.plan["parent"]["exit"] = self.doc("producer/EXIT.json", exit, self.hf)

    def check(self):
        with mock.patch("ffnds_video_metrics.validate", return_value=[]), \
                mock.patch("ffnds_stage_registered_i2v.collect", return_value=self.verified):
            return q.verify_generation(self.plan, self.hf, repo=self.repo)

    def legacy(self):
        env = {k: k for k in q.ENVIRONMENT_KEYS}
        videos = []
        original_cases, original_sources = [], []
        for c in self.cases:
            s = c["index"]; item = copy.deepcopy(self.items[s, "dense"])
            item["source_id"] = c["image_sha256"]
            historical = copy.deepcopy(self.gens[s, "dense"]); historical["source_id"] = c["image_sha256"]
            item["generation"] = self.doc(f"historical/{s}/GENERATION.json", historical, self.hf)
            c["historical_dense"]["generation"] = item["generation"]
            videos.append(item)
            original_cases.append({"index": s, "source_id": c["image_sha256"], "semantic_source_id": c["source_id"],
                                   "image": {"sha256": c["image_sha256"]}, "dimension": "i2v_subject" if s == 0 else "i2v_background"})
            original_sources.append({"source_id": c["source_id"], "legacy_image_fingerprint": c["image_sha256"],
                                     "prompt": c["prompt"], "image_name": item["image_name"]})
        gb = self.doc("config/original_generation.json", {"cases": original_cases, "source_bank": {"sources": original_sources}}, self.repo)
        old = {"id": q.LEGACY_ID, "dimensions": list(q.DIMENSIONS), "videos": videos, "hf_bindings": [], "generation_protocol": gb, **env}
        oldbinding = self.doc(q.LEGACY_PROTOCOL, old, self.repo)
        published = {"protocol_sha256": q.LEGACY_PROTOCOL_SHA, "artifact_sha256": {}}
        parts = []
        for part, source in enumerate(q.SOURCES):
            video = next(x for x in videos if x["source"] == source)
            path = str(self.hf / video["video"]["path"])
            completed = []
            for dim in list(q.COMMON) + ["i2v_subject" if source == 0 else "i2v_background"]:
                inp = [{"prompt_en": video["prompt"], "dimension": [dim], "image_name": video["image_name"], "video_list": [path]}]
                ib = self.doc(f"legacy/part{part}/{dim}_INPUT.json", inp, self.hf)
                entry = {"dimension": dim, "input_sha256": ib["sha256"], "result": [0.5, [{"video_path": path, "video_results": False if dim == "dynamic_degree" else .5}]]}
                if dim == "dynamic_degree":
                    entry["RAFT_observations"] = [raft(path, 4.)]
                completed.append(entry)
            run = {"id": q.LEGACY_ID, "status": "COMPLETE_SEVEN_PART_DIMENSIONS", "part": part,
                   "protocol_sha256": q.LEGACY_PROTOCOL_SHA, "completed": completed,
                   "runtime_precision": {"matmul_TF32": False, "float32_matmul_precision": "highest", "bf16_reduced_precision_reduction": True}}
            rb = self.doc(f"legacy/part{part}/RUN.json", run, self.hf)
            eb = self.doc(f"legacy/part{part}/EXIT.json", {"id": q.LEGACY_ID, "protocol_sha256": q.LEGACY_PROTOCOL_SHA,
                          "part": part, "launcher_returncode": 0, "process_exited": True, "automatic_retry": False}, self.hf)
            parts.append({"part": part, "RUN": rb, "exit": eb})
            published["artifact_sha256"][f"part{part}_run"] = rb["sha256"]
            published["artifact_sha256"][f"part{part}_launch"] = eb["sha256"]
        self.plan.update(environment=env, hf_bindings=[], legacy={"protocol": oldbinding, "parts": parts,
                                                               "published_results": self.doc(q.LEGACY_REPORT, published, self.repo)})
        return self.plan


def raft(path, value):
    return {"video_path": path, "scores": [value] * 40, "params": {"thres": 10.875, "count_num": 10}, "official_binary": value > 10.875}


class GenerationControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.f = Fixture(Path(self.temp.name))

    def test_complete_byte_graph_accepts_no_tensor_loads(self):
        _, r, cores = self.f.check(); self.assertEqual(len(r["phases"]), 6); self.assertEqual(len(cores), 6)

    def test_parent_incomplete_after_valid_resign(self):
        self.f.run["status"] = "RUNNING"; self.f.resign()
        with self.assertRaisesRegex(ValueError, "producer incomplete"):
            self.f.check()

    def test_parent_actual_exit_and_RUN_SHA_are_required(self):
        for field, value in [("child_returncode", 1), ("process_exited", False), ("run_sha256", "f" * 64), ("timeout", True)]:
            self.f.resign(); b = self.f.plan["parent"]["exit"]; receipt = q.read(self.f.hf / b["path"]); receipt[field] = value
            self.f.plan["parent"]["exit"] = self.f.doc(b["path"], receipt, self.f.hf)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Actual exit0/exact RUN SHA"):
                self.f.check()

    def test_parent_counts_not_just_complete_status(self):
        self.f.run["operation_ledger"]["observed"]["video_generations"] = 5; self.f.resign()
        with self.assertRaisesRegex(ValueError, "operation ledger incomplete"):
            self.f.check()

    def test_half_cohort_and_duplicate_phase_rejected(self):
        self.f.run["phases"] = self.f.run["phases"][:-1]
        self.f.plan["parent"]["RUN"] = self.f.doc("producer/RUN.json", self.f.run, self.f.hf)
        exit = q.read(self.f.hf / self.f.plan["parent"]["exit"]["path"]); exit["run_sha256"] = self.f.plan["parent"]["RUN"]["sha256"]
        self.f.plan["parent"]["exit"] = self.f.doc("producer/EXIT.json", exit, self.f.hf)
        with self.assertRaisesRegex(ValueError, "Six unique completed"):
            self.f.check()

    def test_registered_run_and_native_depth_gate(self):
        self.f.regs[0, "energy94"]["native_ffn_sites"] = 159; self.f.resign()
        with self.assertRaisesRegex(ValueError, "Registered generation incomplete"):
            self.f.check()

    def test_prompt_identity_not_a_stale_file_sha(self):
        self.f.items[0, "energy94"]["prompt"] = "different valid string"; self.f.resign()
        with self.assertRaisesRegex(ValueError, "Source/prompt/seed"):
            self.f.check()

    def test_core_target_and_paired_reference_identity(self):
        self.f.cores[0, "energy94"]["reference"]["sha256"] = "f" * 64; self.f.resign()
        with self.assertRaisesRegex(ValueError, "Core paired Dense reference"):
            self.f.check()

    def test_core_wrong_source_with_valid_new_binding(self):
        self.f.cores[0, "dense"]["identity"]["source_id"] = "wrong-image"; self.f.resign()
        with self.assertRaisesRegex(ValueError, "core report/source identity"):
            self.f.check()

    def test_actual_raw_noise_bytes_checked_after_all_sha_rebinding(self):
        key = 0, "energy94"; item = self.f.items[key]
        nb = self.f.raw(item["raw_noise"]["path"], b"different-but-validly-bound-FP32-noise")
        item["raw_noise"] = nb; self.f.gens[key]["raw_FP32_noise"] = nb; self.f.phases[key]["raw_FP32_noise"] = nb
        self.f.resign()
        with self.assertRaisesRegex(ValueError, "Actual Dense/candidate noise bytes differ"):
            self.f.check()

    def test_reused_dense_requires_historical_match(self):
        self.f.phases[0, "dense"]["historical_comparison"]["passed"] = False; self.f.resign()
        with self.assertRaisesRegex(ValueError, "Dense historical/core identity"):
            self.f.check()

    def test_legacy_full_scores_accept_published_sha_and_real_exit(self):
        self.f.legacy(); scores, flows, _ = q.verify_legacy(self.f.plan, self.f.parent, self.f.hf, repo=self.f.repo)
        self.assertEqual(sum(map(len, scores.values())), 14); self.assertEqual(len(flows), 2)

    def test_legacy_unpublished_RUN_is_not_accepted(self):
        self.f.legacy(); p = self.f.plan["legacy"]["parts"][0]; r = q.read(self.f.hf / p["RUN"]["path"])
        r["completed"][0]["result"][-1][0]["video_results"] = .123
        p["RUN"] = self.f.doc(p["RUN"]["path"], r, self.f.hf)
        with self.assertRaisesRegex(ValueError, "published RUN/exit SHA"):
            q.verify_legacy(self.f.plan, self.f.parent, self.f.hf, repo=self.f.repo)

    def test_legacy_bad_exit_after_rebinding_publication_fails_exit_gate(self):
        self.f.legacy(); p = self.f.plan["legacy"]["parts"][0]; e = q.read(self.f.hf / p["exit"]["path"]); e["launcher_returncode"] = 1
        p["exit"] = self.f.doc(p["exit"]["path"], e, self.f.hf)
        b = self.f.plan["legacy"]["published_results"]; published = q.read(self.f.repo / b["path"])
        published["artifact_sha256"]["part0_launch"] = p["exit"]["sha256"]
        self.f.plan["legacy"]["published_results"] = self.f.doc(b["path"], published, self.f.repo)
        with self.assertRaisesRegex(ValueError, "Legacy actual exit0"):
            q.verify_legacy(self.f.plan, self.f.parent, self.f.hf, repo=self.f.repo)

    def test_changed_evaluator_assets_disallow_Dense_reuse(self):
        self.f.legacy(); self.f.plan["hf_bindings"] = [{"path": "new-model-version"}]
        with self.assertRaisesRegex(ValueError, "Evaluator source/model/environment"):
            q.verify_legacy(self.f.plan, self.f.parent, self.f.hf, repo=self.f.repo)

    def test_legacy_fingerprint_semantic_mapping_is_explicit(self):
        self.f.legacy(); old_binding = self.f.plan["legacy"]["protocol"]; old = q.read(self.f.repo / old_binding["path"])
        gb = old["generation_protocol"]; generation = q.read(self.f.repo / gb["path"])
        generation["cases"][0]["semantic_source_id"] = "another-source"
        generation["source_bank"]["sources"][0]["source_id"] = "another-source"
        old["generation_protocol"] = self.f.doc(gb["path"], generation, self.f.repo)
        self.f.plan["legacy"]["protocol"] = self.f.doc(old_binding["path"], old, self.f.repo)
        with self.assertRaisesRegex(ValueError, "fingerprint/semantic source-bank mapping"):
            q.verify_legacy(self.f.plan, self.f.parent, self.f.hf, repo=self.f.repo)


class MathAndDispatchControls(unittest.TestCase):
    def test_six_same_environment_and_original_image_group_dispatch(self):
        clips = [{"source": s, "mode": m} for s in q.SOURCES for m in q.MODES]
        for d in q.COMMON:
            rows = q.rows_for_dimension(clips, d); self.assertEqual(len(rows), 6); self.assertEqual({r["mode"] for r in rows}, set(q.MODES))
        self.assertEqual({r["source"] for r in q.rows_for_dimension(clips, "i2v_subject")}, {0})
        self.assertEqual({r["source"] for r in q.rows_for_dimension(clips, "i2v_background")}, {12})

    def test_binary_separate_from_continuous_motion_and_fixed_windows(self):
        c, d = raft("candidate", 2.), raft("dense", 4.)
        out = q.motion_comparison(c, d)
        self.assertEqual(out["official_binary_separate"], {"candidate": False, "Dense": False})
        self.assertEqual(out["candidate_to_Dense_mean_ratio"]["value"], .5)
        self.assertEqual([v["value"] for v in out["fixed_window_candidate_to_Dense_ratios"]], [.5] * 5)
        windows = out["candidate"]["five_fixed_8pair_windows"]
        self.assertEqual([(w["pair_first"], w["pair_last"], w["frame_first"], w["frame_last"]) for w in windows],
                         [(0, 7, 0, 16), (8, 15, 16, 32), (16, 23, 32, 48), (24, 31, 48, 64), (32, 39, 64, 80)])

    def test_zero_dense_motion_is_missing_ratio_not_fake_improvement(self):
        x = q.motion_comparison(raft("c", 1.), raft("d", 0.))
        self.assertEqual(x["candidate_to_Dense_mean_ratio"], {"value": None, "status": "ZERO_DENSE_MOTION"})

    def test_raft_count_nonfinite_negative_and_binary_mismatch(self):
        for scores in ([1.] * 39, [1.] * 41, [float("nan")] * 40, [-1.] * 40):
            with self.subTest(scores=len(scores)), self.assertRaisesRegex(ValueError, "Exactly40 finite nonnegative"):
                q.raft_summary(scores)
        bad = raft("x", 20.); bad["official_binary"] = False
        with self.assertRaisesRegex(ValueError, "Official binary"):
            q.validate_raft(bad)

    def test_actual_fake_raft_240_call_dispatch_and_restore(self):
        report = {}; ledger = q.Ledger(report, limits={"RAFT_forwards": 240, "RAFT_iterations": 4800})
        class RAFT:
            def __call__(self, *args, **kwargs): return self.forward(*args, **kwargs)
            def forward(self, a, b, *, iters, test_mode): return None, 2.
        class Dynamic:
            def __init__(self): self.model = RAFT(); self.params = {"thres": 10.875, "count_num": 10}
            def get_score(self, img, flow): return flow
            def infer(self, path):
                values = []
                for _ in range(40):
                    _, flow = self.model(object(), object(), iters=20, test_mode=True); values.append(self.get_score(None, flow))
                return sum(v > 10.875 for v in values) >= 10
        module = types.SimpleNamespace(RAFT=RAFT, DynamicDegree=Dynamic); rows = []; paths = [f"clip{i}" for i in range(6)]
        original = RAFT.forward
        with q.raft_observer(module, rows, ledger, paths):
            dynamic = Dynamic()
            for path in paths: self.assertFalse(dynamic.infer(path))
            with self.assertRaisesRegex(ValueError, "Wrong/duplicate"):
                dynamic.infer(paths[0])
        self.assertIs(RAFT.forward, original); ledger.finish(); self.assertEqual(len(rows), 6)

    def test_wrong_raft_signature_blocks_before_invocation(self):
        report = {}; ledger = q.Ledger(report, limits={"RAFT_forwards": 1, "RAFT_iterations": 20}); called = []
        wrapped = q.observed_raft_forward(lambda *a, **k: called.append(1), ledger)
        with self.assertRaisesRegex(ValueError, "Original RAFT invocation"):
            wrapped(None, None, None, iters=19, test_mode=True)
        self.assertEqual(called, []); self.assertEqual(report["invoked"]["RAFT_forwards"], 0)

    def test_failed_call_keeps_predebit_and_no_completion_no_retry(self):
        report = {}; ledger = q.Ledger(report, limits={"RAFT_forwards": 1, "RAFT_iterations": 20})
        def fail(*a, **k): raise RuntimeError("injected original failure")
        with self.assertRaisesRegex(RuntimeError, "injected original failure"):
            q.observed_raft_forward(fail, ledger)(None, 1, 2, iters=20, test_mode=True)
        self.assertEqual(report["predebited"]["RAFT_forwards"], 1); self.assertEqual(report["invoked"]["RAFT_forwards"], 1)
        self.assertEqual(report["completed"]["RAFT_forwards"], 0)
        with self.assertRaisesRegex(ValueError, "budget exceeded"):
            q.observed_raft_forward(fail, ledger)(None, 1, 2, iters=20, test_mode=True)

    def test_full_8dimension_statistics_all_six_new_and_history_does_not_set_denominator(self):
        clips = [{"source": s, "mode": m, "source_id": f"s{s}", "image_sha256": f"image{s}",
                  "video": {"path": f"{s}_{m}.mp4"}} for s in q.SOURCES for m in q.MODES]
        plan = {"videos": clips, "hf_root_for_report": "/tmp"}
        reused = {s: {d: False if d == "dynamic_degree" else 10. + s for d in list(q.COMMON) + ["i2v_subject" if s == 0 else "i2v_background"]} for s in q.SOURCES}
        flows = {s: raft(f"dense{s}", 4.) for s in q.SOURCES}; completed = []
        for d in q.DIMENSIONS:
            selected = q.rows_for_dimension(clips, d)
            rows = [{"video_path": str(Path("/tmp") / v["video"]["path"]), "video_results": False if d == "dynamic_degree" else float(v["source"])} for v in selected]
            entry = {"dimension": d, "result": [0., rows]}
            if d == "dynamic_degree": entry["RAFT_observations"] = [raft(r["video_path"], 2.) for r in rows]
            completed.append(entry)
        x = q.summarize(plan, reused, flows, completed)
        self.assertEqual(x["equal_source_descriptive_means_common_six_only"]["imaging_quality"], {"dense": 6., "energy94": 6., "random94": 6.})
        self.assertEqual(x["per_source"][0]["motion"]["energy94"]["candidate_to_Dense_mean_ratio"]["value"], 1.)
        self.assertEqual(x["per_source"][0]["historical_same_byte_Dense_sanity_only"]["new_minus_old_score"]["imaging_quality"], -10.)
        self.assertEqual(x["independent_image_clusters"], 2); self.assertEqual(x["Dense_score_reuse_adds_independent_samples"], 0)
        self.assertNotIn("i2v_subject", x["equal_source_descriptive_means_common_six_only"])
        completed[0]["result"][-1][0] = completed[0]["result"][-1][1]
        with self.assertRaisesRegex(ValueError, "Duplicate/wrong source"):
            q.summarize(plan, reused, flows, completed)

    def test_per_video_result_path_and_type_gate(self):
        q.check_result([.5, [{"video_path": "one", "video_results": .5}]], ["one"], "imaging_quality")
        with self.assertRaisesRegex(ValueError, "Wrong or duplicate"):
            q.check_result([.5, [{"video_path": "wrong", "video_results": .5}]], ["one"], "imaging_quality")
        with self.assertRaisesRegex(ValueError, "Nonfinite/wrong"):
            q.check_result([False, [{"video_path": "one", "video_results": .5}]], ["one"], "dynamic_degree")


class AuthorityControls(unittest.TestCase):
    def test_cpu_receipt_missing_stale_protocol_and_RUN_sha(self):
        with self.assertRaisesRegex(ValueError, "CPU preflight NOT_BOUND"):
            q.require_cpu_receipt({}, {}, "a" * 64, Path("/tmp"))
        with tempfile.TemporaryDirectory() as t:
            hf = Path(t); plan = {"repo_sources": [], "import_environment": {}, "parent": {"RUN": {"sha256": "b" * 64}, "exit": {"sha256": "c" * 64}}}
            report = q.cpu_report(plan, "a" * 64); rp = hf / "CPU.json"; rp.write_text(json.dumps(report)); rb = q.binding(rp, hf)
            exit = {"id": q.ID, "protocol_sha256": "a" * 64, "child_returncode": 0, "process_exited": True,
                    "timeout": False, "automatic_retry": False, "run_sha256": rb["sha256"]}
            ep = hf / "EXIT.json"; ep.write_text(json.dumps(exit)); scope = {"cpu_preflight": {"RUN": rb, "launch_exit": q.binding(ep, hf)}}
            q.require_cpu_receipt(scope, plan, "a" * 64, hf)
            report["new_scored_videos"] = 4; rp.write_text(json.dumps(report)); scope["cpu_preflight"]["RUN"] = q.binding(rp, hf)
            exit["run_sha256"] = scope["cpu_preflight"]["RUN"]["sha256"]; ep.write_text(json.dumps(exit)); scope["cpu_preflight"]["launch_exit"] = q.binding(ep, hf)
            with self.assertRaisesRegex(ValueError, "CPU preflight proof"):
                q.require_cpu_receipt(scope, plan, "a" * 64, hf)
            for field, value in [("protocol_sha256", "f" * 64), ("sources_sha256", "f" * 64),
                                 ("all_CUDA_devices_hidden", False), ("NumPy_or_Torch_imported", True), ("GPU_calls", 1)]:
                report = q.cpu_report(plan, "a" * 64); report[field] = value; rp.write_text(json.dumps(report))
                scope["cpu_preflight"]["RUN"] = q.binding(rp, hf)
                exit["run_sha256"] = scope["cpu_preflight"]["RUN"]["sha256"]; ep.write_text(json.dumps(exit))
                scope["cpu_preflight"]["launch_exit"] = q.binding(ep, hf)
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "CPU preflight proof"):
                    q.require_cpu_receipt(scope, plan, "a" * 64, hf)
            report = q.cpu_report(plan, "a" * 64); rp.write_text(json.dumps(report)); scope["cpu_preflight"]["RUN"] = q.binding(rp, hf)
            exit["run_sha256"] = "f" * 64; ep.write_text(json.dumps(exit)); scope["cpu_preflight"]["launch_exit"] = q.binding(ep, hf)
            with self.assertRaisesRegex(ValueError, "Actual exit0/exact RUN SHA"):
                q.require_cpu_receipt(scope, plan, "a" * 64, hf)

    def test_job1_session2_and_dynamic_repository_authority_are_separate(self):
        plan = {"id": q.ID, "dimensions": list(q.DIMENSIONS), "costs": q.COSTS, "limits": q.LIMITS,
                "resources": q.RESOURCES, "automatic_retry": False, "assets_finalized": True, "binding_status": "BOUND_READY_FOR_REGISTRATION",
                "videos": [{"source": s, "mode": m} for s in q.SOURCES for m in q.MODES],
                "images": [{"source": s, "binding": {"sha256": str(s)}} for s in q.SOURCES],
                "parent": {"protocol": {"path": q.PARENT_PROTOCOL}},
                "legacy": {"protocol": {"path": q.LEGACY_PROTOCOL, "sha256": q.LEGACY_PROTOCOL_SHA},
                           "published_results": {"path": q.LEGACY_REPORT}, "parts": [{"part": p} for p in (0, 1)]},
                "repo_sources": [{"path": p} for p in q.REQUIRED_SOURCES], "hf_bindings": [],
                "environment": dict.fromkeys(q.ENVIRONMENT_KEYS, None), "output_HF_relative": "ffnds-derived/test-quality",
                "import_environment": {"policy": "ISOLATED_I_S_ORIGINAL_V8_SITE_NO_PROJECT_SITE_FALLBACK",
                    "limits": q.IMPORT_LIMITS, "resources": q.IMPORT_RESOURCES, "flags": ["-I", "-S"],
                    "modules": dict.fromkeys(q.IMPORT_NAMES, {}), "legacy_score_actual_module_origins": "NOT_RECORDED"}}
        scope = {"id": q.ID, "protocol_sha256": "a" * 64, "execution_status": "AUTHORIZED_NOT_STARTED",
                 "max_gpu_cards": 1, "limits": q.LIMITS, "resources": q.RESOURCES}
        active = {"user_continuation_authorization_20261027": {"status": "ACTIVE_CURRENT_USER_DIRECTED_SESSION", "new_scope_registration_allowed": True},
                  "resource_scheduling_policy_20261028": {"mode": "DYNAMIC_MAX_ACTUALLY_CALLABLE", "global_gpu_card_cap": None},
                  "current_session_resource_constraint_20261028": {"max_simultaneously_assigned_idle_GPUs": 2},
                  "experiment_authority": {"current_user_authorized_scopes": [scope]}}
        q.authorize(plan, "a" * 64, active)
        wrong = copy.deepcopy(active); wrong["resource_scheduling_policy_20261028"]["global_gpu_card_cap"] = 2
        with self.assertRaisesRegex(ValueError, "session/repository GPU authority"):
            q.authorize(plan, "a" * 64, wrong)
        wrong = copy.deepcopy(active); wrong["current_session_resource_constraint_20261028"]["max_simultaneously_assigned_idle_GPUs"] = 4
        with self.assertRaisesRegex(ValueError, "session/repository GPU authority"):
            q.authorize(plan, "a" * 64, wrong)
        wrong = copy.deepcopy(active); wrong["experiment_authority"]["current_user_authorized_scopes"][0]["max_gpu_cards"] = 2
        with self.assertRaisesRegex(ValueError, "Exact registered quality budget"):
            q.authorize(plan, "a" * 64, wrong)
        with self.assertRaisesRegex(ValueError, "Exact registered quality budget"):
            q.authorize(plan, "b" * 64, active)
        draft = copy.deepcopy(plan); draft["assets_finalized"] = False
        with self.assertRaisesRegex(ValueError, "draft is not frozen"):
            q.authorize(draft, "a" * 64, active)

    def test_forced_TF32_rejected_without_torch(self):
        for key in ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"):
            with mock.patch.dict(q.os.environ, {key: "1"}), self.assertRaisesRegex(ValueError, "Forced TF32"):
                q.precision_environment()

    def test_no_tensor_libraries_imported(self):
        self.assertNotIn("torch", sys.modules); self.assertNotIn("numpy", sys.modules)


class IsolatedEnvironmentControls(unittest.TestCase):
    def environment_fixture(self, root):
        hf = root / "hf"; hf.mkdir()
        site, source, overlay, stdlib = [hf / x for x in ("metric-site", "source", "overlay", "stdlib")]
        for path in (site, source, overlay, stdlib): path.mkdir()
        interpreter = hf / "python"; interpreter.write_text("original interpreter identity")
        old = hf / "V8_IMPORT.json"; old.write_text(json.dumps({"status": "PASS_IMPORTS_ONLY"}))
        modules = {}
        for name in ("numpy", "pyiqa", "torch"):
            path = site / name / "__init__.py"; path.parent.mkdir()
            path.write_text("raise RuntimeError('package code must not execute during spec precheck')\n")
            modules[name] = {"root": "HF", **q.binding(path, hf), "distribution": name,
                             "version": "1.26.4" if name == "numpy" else "fixture"}
        plan = {"environment": {"site": "metric-site", "source_root": "source", "overlay": "overlay"},
                "import_environment": {"python_HF_relative": "python", "python_sha256": q.sha(interpreter),
                                       "v8_actual_import_result": q.binding(old, hf), "modules": modules}, "repo_sources": []}
        return hf, plan, interpreter, stdlib

    def test_isolated_paths_reject_project_site_and_arbitrary_cwd(self):
        self.assertEqual(q.isolated_paths(["/python/lib/python3.11", str(q.ROOT / "tools")], "/python", q.ROOT / "tools"), ["/python/lib/python3.11"])
        for paths in (["/python/lib/python3.11/site-packages"], ["/tmp/shadow"], [""], [str(q.ROOT / "tools")]):
            with self.subTest(paths=paths), self.assertRaisesRegex(ValueError, "import fallback|Missing interpreter"):
                q.isolated_paths(paths, "/python", q.ROOT / "tools")

    def test_pre_Torch_selects_numpy126_and_pyiqa_spec_without_execution(self):
        with tempfile.TemporaryDirectory() as t:
            hf, plan, interpreter, stdlib = self.environment_fixture(Path(t))
            flags = types.SimpleNamespace(isolated=1, no_site=1, ignore_environment=1)
            with mock.patch.object(q.sys, "executable", str(interpreter)), mock.patch.object(q.sys, "base_prefix", str(stdlib)), \
                    mock.patch.object(q.sys, "path", [str(stdlib), str(q.ROOT / "tools")]), mock.patch.object(q.sys, "dont_write_bytecode", True):
                record = q.configure_import_environment(plan, hf, flags=flags)
                self.assertEqual(record["pre_Torch_specs"]["numpy"]["version"], "1.26.4")
                self.assertEqual(q.sys.path[:3], [str(hf / "source"), str(hf / "overlay"), str(hf / "metric-site")])
                self.assertFalse(record["project_site_fallback"])
                self.assertNotIn("numpy", sys.modules); self.assertNotIn("torch", sys.modules)

    def test_isolation_wrong_flags_preloaded_numpy_and_wrong_spec_fail_before_import(self):
        with tempfile.TemporaryDirectory() as t:
            hf, plan, interpreter, stdlib = self.environment_fixture(Path(t))
            with self.assertRaisesRegex(ValueError, "Python -I -S"):
                q.configure_import_environment(plan, hf, flags=types.SimpleNamespace(isolated=0, no_site=0, ignore_environment=0))
            flags = types.SimpleNamespace(isolated=1, no_site=1, ignore_environment=1)
            with mock.patch.dict(sys.modules, {"numpy": types.SimpleNamespace()}), self.assertRaisesRegex(ValueError, "before numeric imports"):
                q.configure_import_environment(plan, hf, flags=flags)
            with mock.patch.object(q.sys, "executable", str(interpreter)), mock.patch.object(q.sys, "base_prefix", str(stdlib)), \
                    mock.patch.object(q.sys, "path", [str(stdlib)]), mock.patch.object(q.importlib.util, "find_spec", return_value=types.SimpleNamespace(origin="/project/numpy/__init__.py")):
                with self.assertRaisesRegex(ValueError, "Pre-Torch original package spec differs: numpy"):
                    q.configure_import_environment(plan, hf, flags=flags)

    def test_module_observation_checks_real_origin_version_and_bytes(self):
        with tempfile.TemporaryDirectory() as t:
            hf, plan, interpreter, stdlib = self.environment_fixture(Path(t))
            item = plan["import_environment"]["modules"]["numpy"]; module = types.SimpleNamespace(__file__=str(hf / item["path"]))
            with mock.patch.object(q.importlib.metadata, "version", return_value="1.26.4"):
                self.assertEqual(q.module_observation(module, "numpy", plan, hf)["version"], "1.26.4")
                module.__file__ = str(hf / "wrong.py")
                with self.assertRaisesRegex(ValueError, "Actual imported module origin/version differs: numpy"):
                    q.module_observation(module, "numpy", plan, hf)
                module.__file__ = str(hf / item["path"])
            with mock.patch.object(q.importlib.metadata, "version", return_value="2.4.6"), self.assertRaisesRegex(ValueError, "Actual imported module origin/version"):
                q.module_observation(module, "numpy", plan, hf)
            (hf / item["path"]).write_text("changed package implementation")
            with self.assertRaisesRegex(ValueError, "Asset bytes/SHA differ"):
                q.module_observation(module, "numpy", plan, hf)

    def official_fixture(self, hf):
        modules = {}
        def compute(json_dir, device, submodules_list):
            raise AssertionError("imports-only must never execute compute")
        for dimension in q.DIMENSIONS:
            package = "vbench2_beta_i2v" if dimension.startswith("i2v_") else "vbench"
            path = hf / "source" / package / (dimension + ".py"); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# immutable original author fixture\n" + dimension)
            modules[package + "." + dimension] = types.SimpleNamespace(__file__=str(path), **{"compute_" + dimension: compute})
        return modules

    def test_all_eight_function_imports_are_real_callable_lookups_without_compute(self):
        with tempfile.TemporaryDirectory() as t:
            hf, plan, _, _ = self.environment_fixture(Path(t)); modules = self.official_fixture(hf)
            with mock.patch.object(q.importlib, "import_module", side_effect=modules.__getitem__) as imported:
                records = [q.original_metric_module(d, plan, hf)[1] for d in q.DIMENSIONS]
            self.assertEqual(imported.call_count, 8); self.assertEqual([r["dimension"] for r in records], list(q.DIMENSIONS))

    def test_wrong_official_origin_and_signature_rejected_without_compute(self):
        with tempfile.TemporaryDirectory() as t:
            hf, plan, _, _ = self.environment_fixture(Path(t)); modules = self.official_fixture(hf)
            module = modules["vbench.imaging_quality"]; original = module.__file__
            with mock.patch.object(q.importlib, "import_module", return_value=module):
                module.__file__ = str(hf / "wrong.py")
                with self.assertRaisesRegex(ValueError, "function origin/signature"):
                    q.original_metric_module("imaging_quality", plan, hf)
                module.__file__ = original; module.compute_imaging_quality = lambda wrong: None
                with self.assertRaisesRegex(ValueError, "function origin/signature"):
                    q.original_metric_module("imaging_quality", plan, hf)

    def test_import_CPU_receipt_exact_scope_RUN_origin_counts_resources_and_zero_NN(self):
        with tempfile.TemporaryDirectory() as t:
            hf, plan, _, _ = self.environment_fixture(Path(t)); modules = self.official_fixture(hf)
            with mock.patch.object(q.importlib, "import_module", side_effect=modules.__getitem__):
                functions = [q.original_metric_module(d, plan, hf)[1] for d in q.DIMENSIONS]
            good = q.import_report(plan, "a" * 64)
            good.update(predebited_official_module_imports=8, completed_official_module_imports=8,
                        denied_attempts=[], stateless_activation_wrapper_constructions=8, stateless_pool_wrapper_constructions=1,
                        isolation={"flags": {"isolated": 1, "no_site": 1, "ignore_environment": 1}, "project_site_fallback": False},
                        wall_seconds=1., new_HF_bytes=100, functions=functions,
                        module_origins={n: {"origin": str(hf / b["path"]), "sha256": b["sha256"], "distribution": b["distribution"], "version": b["version"]} for n,b in plan["import_environment"]["modules"].items()})
            rp, ep = hf / "IMPORT_CPU.json", hf / "EXIT.json"
            def resigned(value, *, run_sha=None):
                rp.write_text(json.dumps(value)); rb = q.binding(rp, hf)
                ep.write_text(json.dumps({"id": q.ID, "protocol_sha256": "a" * 64, "child_returncode": 0,
                    "process_exited": True, "timeout": False, "automatic_retry": False, "run_sha256": rb["sha256"] if run_sha is None else run_sha}))
                return {"cpu_import_preflight": {"RUN": rb, "launch_exit": q.binding(ep, hf)}}
            with self.assertRaisesRegex(ValueError, "imports-only CPU preflight NOT_BOUND"):
                q.require_import_receipt({}, plan, "a" * 64, hf)
            q.require_import_receipt(resigned(good), plan, "a" * 64, hf)
            for field, value, reason in [("protocol_sha256", "stale", "proof differs"), ("checkpoint_loads", 1, "proof differs"),
                    ("model_forwards", 1, "proof differs"), ("CUDA_initialized", True, "proof differs"),
                    ("completed_official_module_imports", 7, "observed budget"), ("denied_attempts", ["blocked"], "observed budget"),
                    ("wall_seconds", 301., "isolation/resource"), ("new_HF_bytes", 33 << 20, "isolation/resource")]:
                wrong = copy.deepcopy(good); wrong[field] = value
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, reason):
                    q.require_import_receipt(resigned(wrong), plan, "a" * 64, hf)
            wrong = copy.deepcopy(good); wrong["module_origins"]["numpy"]["version"] = "2.4.6"
            with self.assertRaisesRegex(ValueError, "actual origins"):
                q.require_import_receipt(resigned(wrong), plan, "a" * 64, hf)
            wrong = copy.deepcopy(good); wrong["functions"][0]["module_sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "official module bytes"):
                q.require_import_receipt(resigned(wrong), plan, "a" * 64, hf)
            with self.assertRaisesRegex(ValueError, "Actual exit0/exact RUN SHA"):
                q.require_import_receipt(resigned(good, run_sha="f" * 64), plan, "a" * 64, hf)

    def test_v2_exact_six_video_budgets(self):
        self.assertEqual(q.LIMITS["new_video_dimension_results"], 42)
        self.assertEqual(q.LIMITS["reused_Dense_video_dimension_results"], 0)
        self.assertEqual(q.LIMITS["RAFT_forwards"], 240); self.assertEqual(q.LIMITS["RAFT_iterations"], 4800)
        clips = [{"source": s, "mode": m} for s in q.SOURCES for m in q.MODES]
        self.assertEqual(sum(len(q.rows_for_dimension(clips, d)) for d in q.DIMENSIONS), 42)
        self.assertEqual(q.COSTS["imaging_quality"], {"MUSIQ": 486})
        self.assertEqual(q.COSTS["i2v_subject"], {"DINO": 246})


if __name__ == "__main__":
    unittest.main()
