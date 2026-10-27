"""CPU contracts for fixed-bank pairing, noise evidence, residency and budgets."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_ub1_consumer_disjoint20 as driver
from ffnds_stage_registered_i2v import collect


def plan_fixture():
    plan = json.loads((ROOT / "config/ub1_consumer_disjoint20_20261027_v1.json").read_text())
    plan["assets_finalized"] = True
    return plan


def good_counts():
    return {"linear_forward_calls": 1600, "activation_qdq_calls": 1600, "cache_misses": 0, "weight_transfers": 0}


class FakeConsumer:
    def begin_request(self):
        self.begun = True

    def report(self):
        return {"counts": {"main_linear": 320, "RR_linear": 320, "Q1": 160, "Q2": 160, "up": 160, "down": 160},
            "pending_RR": 0, "main_coverage": dict.fromkeys(range(320), 1), "RR_coverage": dict.fromkeys(range(160), 1)}

    def end_request(self):
        if not self.begun:
            raise RuntimeError("No consumer request")
        return self.report()


class FakeNativePipe:
    """A tiny CPU scheduler with all160 dispatch coordinates and real tensor IO."""
    def __init__(self, base, timer, initial, *, fail_after=None):
        self.base, self.timer, self.initial = base, timer, initial
        self.fail_after, self.updates = fail_after, 0
        scheduler = SimpleNamespace(step_index=0, generator=None,
            denoising_step_list=[1000, 750, 500, 250], sigmas=[1., .75, .5, .25])
        def post():
            self.updates += 1
            scheduler.latents = scheduler.latents + 1
        scheduler.step_post = post
        self.runner = SimpleNamespace(scheduler=scheduler)

    def generate(self, *, seed, image_path, prompt, negative_prompt, save_result_path):
        scheduler = self.runner.scheduler
        if seed != 42 or scheduler.generator is not None or negative_prompt != "":
            raise RuntimeError("Fresh registered source RNG was not reset")
        scheduler.generator = object()
        scheduler.latents = self.initial.clone()
        for step in range(4):
            scheduler.step_index = step
            self.base.counters["linear_forward_calls"] += 400
            self.base.counters["activation_qdq_calls"] += 400
            self.timer.events.extend({"step": step, "block": b, "elapsed_ms": .1} for b in range(40))
            scheduler.step_post()
            if self.fail_after == step:
                raise RuntimeError("Injected native step failure")
        Path(save_result_path).write_bytes(b"actual CPU mock video")


class Disjoint20Test(unittest.TestCase):
    def make_ledger(self, root):
        plan = plan_fixture()
        ledger = driver.PartLedger(root / "output/part0", 0, plan["cases"][:5], "protocol_sha", plan, root)
        ledger.start()
        return plan, ledger

    def generation(self, ledger, mode, *, dense=None, initial=None, fail_after=None, case_index=0):
        base = SimpleNamespace(counters=dict.fromkeys(good_counts(), 0))
        timer = SimpleNamespace(events=[])
        timer.report = lambda: list(timer.events)
        pipe = FakeNativePipe(base, timer, torch.zeros((1, 1, 1, 2)) if initial is None else initial,
                              fail_after=fail_after)
        runtime = base if mode == "dense" else FakeConsumer()
        contract = {**driver.VIDEO_CONTRACT, "latent_shape": [1, 1, 1, 2]}
        case = ledger.cases[case_index]
        video_info = lambda path, contract: {"path": str(path), "sha256": driver.sha256(path),
            "frames": 81, "fps": 16., "size": [832, 464], "bytes": path.stat().st_size}
        with mock.patch.multiple(torch.cuda, reset_peak_memory_stats=mock.DEFAULT, synchronize=mock.DEFAULT,
                max_memory_allocated=mock.Mock(return_value=16), max_memory_reserved=mock.Mock(return_value=32)), \
                mock.patch.object(driver, "inspect_video", side_effect=video_info):
            phase = driver.generate_video(pipe, base, runtime, mode, case, {"prompt": "CPU mock prompt"},
                Path("CPU mock image"), ledger, timer, SimpleNamespace(check=lambda: None),
                lambda **kw: None, .1, .2, contract, dense=dense)
        return phase, pipe

    def mark_core(self, phase):
        phase["core_metrics"] = {"status": "ALL_81_FRAMES_CORE_METRICS_PASS_QUALITY_PENDING"}
        return phase

    def test_exact_twenty_source_forty_job_registration_and_partitions(self):
        plan = plan_fixture()
        all_indices, all_jobs = [], []
        for part in range(4):
            cases = driver.validate_plan(plan, part)
            all_indices.extend(c["index"] for c in cases)
            all_jobs.extend(j["run_id"] for j in plan["cohort_plan"]["samples"][part * 10:part * 10 + 10])
        self.assertEqual(all_indices, list(range(20)))
        self.assertEqual(len(set(all_jobs)), 40)
        self.assertEqual(all_jobs, plan["native_registration"]["allowed_run_ids"])
        self.assertEqual(driver.validate_registered_jobs(plan)["summary"]["independent_image_clusters"], 20)
        changed = copy.deepcopy(plan)
        changed["cohort_plan"]["samples"][-1]["run_id"] = "unregistered run"
        with self.assertRaises(ValueError):
            driver.validate_plan(changed, 0)
        changed = copy.deepcopy(plan)
        changed["source_bank"]["sources"][0]["role"] = "FROZEN_CONFIRMATION"
        with self.assertRaises(ValueError):
            driver.validate_plan(changed, 0)
        with self.assertRaises(ValueError):
            driver.validate_plan({**plan, "assets_finalized": False}, 0)

    def test_active_scope_requires_protocol_and_all_forty_run_ids(self):
        plan = plan_fixture()
        active = {"user_continuation_authorization_20261027": {"id": driver.AUTHORIZATION,
            "status": "ACTIVE_CURRENT_USER_DIRECTED_SESSION"}, "experiment_authority": {
                "current_user_authorized_scopes": [{"id": driver.ID, "protocol_sha256": "sha",
                    "execution_status": "AUTHORIZED_NOT_STARTED", "allowed_run_ids": plan["native_registration"]["allowed_run_ids"]}]}}
        self.assertTrue(driver.registration_authorized(plan, "sha", active))
        self.assertFalse(driver.registration_authorized(plan, "other_sha", active))
        incomplete = copy.deepcopy(active)
        incomplete["experiment_authority"]["current_user_authorized_scopes"][0]["allowed_run_ids"].pop()
        self.assertFalse(driver.registration_authorized(plan, "sha", incomplete))
        extra = copy.deepcopy(active)
        extra["experiment_authority"]["current_user_authorized_scopes"][0]["allowed_run_ids"].append("extra")
        self.assertFalse(driver.registration_authorized(plan, "sha", extra))
        active["user_continuation_authorization_20261027"]["status"] = "PAUSED"
        self.assertFalse(driver.registration_authorized(plan, "sha", active))

    def test_candidate_records_borrow_and_restore_original_dict_even_on_exception(self):
        ffn, attention = {"weight_qdq": object()}, {"weight_qdq": object()}
        records = {"high_noise_model": {"blocks.0.ffn.0": ffn, "blocks.0.ffn.2": ffn, "blocks.0.attn.q": attention}}
        base = SimpleNamespace(records=records, _resident_weights={}, _resident_key=None)
        for _ in range(5):
            with driver.NativeRecordsView(base, "dense"):
                self.assertIs(base.records, records)
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with driver.NativeRecordsView(base, "candidate"):
                    self.assertEqual(set(base.records["high_noise_model"]), {"blocks.0.attn.q"})
                    self.assertIs(base.records["high_noise_model"]["blocks.0.attn.q"], attention)
                    raise RuntimeError("injected")
            self.assertIs(base.records, records)
            self.assertIs(base.records["high_noise_model"]["blocks.0.ffn.0"], ffn)
        base._resident_weights["still_GPU_referenced"] = object()
        with self.assertRaises(RuntimeError):
            with driver.NativeRecordsView(base, "candidate"):
                pass

    def test_actual_noise_preserves_dtype_and_checks_file_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            x = torch.tensor([1., -2., 0.], dtype=torch.bfloat16)
            a = driver.save_initial_noise(x, root / "dense.safetensors", root, lambda **kw: None, [3])
            b = driver.save_initial_noise(x.clone(), root / "candidate.safetensors", root, lambda **kw: None, [3])
            self.assertEqual(load_file(str(root / a["path"]))["latents"].dtype, torch.bfloat16)
            self.assertEqual(a["dtype"], "bfloat16")
            self.assertTrue(driver.verify_noise_pair(b, a, root))
            signed_zero = x.clone()
            signed_zero[-1] = -0.
            self.assertTrue(torch.equal(signed_zero, x))
            c = driver.save_initial_noise(signed_zero, root / "different.safetensors", root, lambda **kw: None, [3])
            with self.assertRaisesRegex(RuntimeError, "noise bytes"):
                driver.verify_noise_pair(c, a, root)
            with self.assertRaisesRegex(RuntimeError, "alias"):
                driver.verify_noise_pair(a, a, root)
            with self.assertRaises(FileExistsError):
                driver.save_initial_noise(x, root / "dense.safetensors", root, lambda **kw: None, [3])

    def test_both_arms_capture_preupdate_noise_full160_and_compatible_registered_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, ledger = self.make_ledger(root)
            dense, _ = self.generation(ledger, "dense")
            ledger.complete_phase(self.mark_core(dense))
            candidate, _ = self.generation(ledger, "candidate", dense=dense)
            ledger.complete_phase(self.mark_core(candidate))
            self.assertEqual(len(ledger.record["completed_pairs"]), 1)
            self.assertTrue(candidate["actual_initial_noise_files_byte_equal"])
            for phase in (dense, candidate):
                native_initial = load_file(str(root / phase["initial_noise"]["path"]))["latents"]
                self.assertTrue(torch.equal(native_initial, torch.zeros_like(native_initial)))
                first = load_file(str(Path(phase["generation_receipt"]["path"]).parent / "latent_step0.safetensors"))["latents"]
                self.assertTrue(torch.equal(first, torch.ones_like(first)))
                self.assertEqual(len(phase["FFN_boundary_timings"]), 160)
                self.assertEqual(phase["native_counter_delta"], good_counts())
                job_index, job, source = ledger.job(0, phase["mode"])
                receipt = driver.registered_receipt(phase, job, source, {"model": "CPU mock"})
                folder = ledger.path.parent / f"job{job_index:06d}"
                folder.mkdir()
                (folder / "RUN.json").write_text(json.dumps(receipt))
            cohort = {**plan["cohort_plan"], "sources": plan["cohort_plan"]["sources"][:1], "samples": plan["cohort_plan"]["samples"][:2]}
            verified = collect(cohort, ledger.path.parent)
            self.assertEqual(verified["phase"], "COMPLETE_REGISTERED_GENERATION_VERIFIED")
            self.assertEqual(verified["samples"][1]["variant"], "candidate")
            self.assertEqual(ledger.record["predebited_work"]["native_RR_linears"], 320)
            self.assertEqual(ledger.record["predebited_work"]["post_step_FP32_latent_files"], 8)
            broken = copy.deepcopy(candidate)
            broken["FFN_boundary_timings"][-1] = broken["FFN_boundary_timings"][0]
            with self.assertRaisesRegex(RuntimeError, "timing boundary"):
                driver.registered_receipt(broken, job, source, {})

    def test_native_partial_failure_keeps_all_predebited_work_and_blocks_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, ledger = self.make_ledger(root)
            with self.assertRaisesRegex(RuntimeError, "Injected native"):
                self.generation(ledger, "dense", fail_after=0)
            for key, amount in driver.generation_costs("dense").items():
                self.assertEqual(ledger.record["predebited_work"][key], amount)
            failed = json.loads((ledger.path / "source00_seed42/dense/PHASE.json").read_text())
            self.assertEqual(failed["completed_native_flow_steps"], 1)
            self.assertEqual(failed["observed_native_counter_delta"]["linear_forward_calls"], 400)
            self.assertEqual(ledger.record["phases"], [])
            with self.assertRaises(RuntimeError):
                ledger.reserve_generation(0, "dense")
            with self.assertRaises(RuntimeError):
                ledger.job(5, "dense")

    def test_noise_mismatch_stops_candidate_before_first_scheduler_update(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, ledger = self.make_ledger(root)
            dense, _ = self.generation(ledger, "dense")
            ledger.complete_phase(self.mark_core(dense))
            with self.assertRaisesRegex(RuntimeError, "noise bytes"):
                self.generation(ledger, "candidate", dense=dense, initial=torch.ones((1, 1, 1, 2)))
            failed = json.loads((ledger.path / "source00_seed42/candidate/PHASE.json").read_text())
            self.assertEqual(failed["completed_native_flow_steps"], 0)
            self.assertEqual(failed["observed_native_counter_delta"]["linear_forward_calls"], 400)
            self.assertEqual(ledger.record["predebited_work"]["native_full_model_flows"], 8)
            self.assertEqual(ledger.record["predebited_work"]["native_RR_linears"], 320)
            self.assertEqual(ledger.record["completed_pairs"], [])

    def test_ten_serial_generations_hit_exact_part_bounds_and_prior_pairs_survive_later_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, ledger = self.make_ledger(root)
            for index in range(5):
                dense = None
                for mode in driver.MODES:
                    phase, _ = self.generation(ledger, mode, dense=dense, case_index=index)
                    ledger.charge("core_video_reports")
                    ledger.complete_phase(self.mark_core(phase))
                    if mode == "dense":
                        dense = phase
            self.assertEqual(ledger.record["predebited_work"], driver.PART_LIMITS)
            self.assertEqual(len(ledger.record["phases"]), 10)
            self.assertEqual(len(ledger.record["completed_pairs"]), 5)
            with self.assertRaises(RuntimeError):
                ledger.reserve_generation(0, "dense")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, ledger = self.make_ledger(root)
            dense, _ = self.generation(ledger, "dense")
            ledger.complete_phase(self.mark_core(dense))
            candidate, _ = self.generation(ledger, "candidate", dense=dense)
            ledger.complete_phase(self.mark_core(candidate))
            pair_binding = copy.deepcopy(ledger.record["completed_pairs"][0])
            with self.assertRaisesRegex(RuntimeError, "Injected native"):
                self.generation(ledger, "dense", case_index=1, fail_after=0)
            self.assertEqual(ledger.record["completed_pairs"], [pair_binding])
            self.assertEqual(driver.sha256(pair_binding["path"]), pair_binding["sha256"])
            self.assertEqual(ledger.record["predebited_work"]["video_generations"], 3)

    def test_original_disjointness_and_exposure_are_bound_to_real_canonical_rows(self):
        plan = plan_fixture()
        samples = json.loads((ROOT / plan["input_manifest"]["path"]).read_text())
        recovery = json.loads((ROOT / plan["input_recovery"]["path"]).read_text())
        bank = json.loads((ROOT / plan["bank_protocol"]["path"]).read_text())
        formal = json.loads((ROOT / plan["formal20_manifest"]["path"]).read_text())
        rows = driver.validate_cohort(plan, samples, recovery, bank, formal)
        self.assertEqual(len(rows), 20)
        self.assertFalse(any(r["training24_indices"] or r["formal20_indices"] for r in rows))
        samples[0]["source_index"] += 1
        with self.assertRaises(ValueError):
            driver.validate_cohort(plan, samples, recovery, bank, formal)


if __name__ == "__main__":
    unittest.main()
