"""CPU scope and native-engine contracts for the ten-video replacement part."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_ub1_consumer_disjoint20 as original
import run_ub1_consumer_disjoint20_recovery as driver
from test_run_ub1_consumer_disjoint20 import FakeConsumer, FakeNativePipe, good_counts


def plan_fixture():
    return json.loads((ROOT / "config/ub1_consumer_disjoint20_recovery_20261027_v1.json").read_text())


def active_fixture(plan, digest="new_protocol_sha"):
    return {"user_continuation_authorization_20261027": {"id": driver.AUTHORIZATION,
        "status": "ACTIVE_CURRENT_USER_DIRECTED_SESSION"},
        "experiment_authority": {"current_user_authorized_scopes": [{"id": driver.ID,
            "protocol_sha256": digest, "allowed_run_ids": plan["native_registration"]["allowed_run_ids"],
            "max_gpu_cards": 1, "total_limits": plan["limits_total"], "execution_status": "AUTHORIZED_NOT_STARTED"}]}}


class RecoveryTest(unittest.TestCase):
    def validate(self, plan):
        template = json.loads((ROOT / driver.TEMPLATE["path"]).read_text())
        return driver.validate_plan(plan, template, driver.load_engine())

    def make_ledger(self, root):
        plan = plan_fixture()
        engine = driver.execution_engine(plan, "new_protocol_sha", root)
        ledger = engine.PartLedger(root / driver.OUTPUT / "part0", 0, plan["cases"],
            "new_protocol_sha", plan, root)
        ledger.start()
        return plan, engine, ledger

    def generation(self, engine, ledger, mode, *, dense=None, index=0, initial=None, fail_after=None):
        base = SimpleNamespace(counters=dict.fromkeys(good_counts(), 0))
        timer = SimpleNamespace(events=[])
        timer.report = lambda: list(timer.events)
        initial = torch.zeros((1, 1, 1, 2), dtype=torch.bfloat16) if initial is None else initial
        pipe = FakeNativePipe(base, timer, initial, fail_after=fail_after)
        runtime = base if mode == "dense" else FakeConsumer()
        contract = {**engine.VIDEO_CONTRACT, "latent_shape": [1, 1, 1, 2]}
        video_info = lambda path, contract: {"path": str(path), "sha256": engine.sha256(path),
            "frames": 81, "fps": 16., "size": [832, 464], "bytes": path.stat().st_size}
        with mock.patch.object(driver, "read_active", return_value=active_fixture(ledger.plan)), \
                mock.patch.multiple(torch.cuda, reset_peak_memory_stats=mock.DEFAULT, synchronize=mock.DEFAULT,
                    max_memory_allocated=mock.Mock(return_value=16), max_memory_reserved=mock.Mock(return_value=32)), \
                mock.patch.object(engine, "inspect_video", side_effect=video_info):
            phase = engine.generate_video(pipe, base, runtime, mode, ledger.cases[index],
                {"prompt": "CPU mock prompt"}, Path("CPU mock image"), ledger, timer,
                SimpleNamespace(check=lambda: None), lambda **kw: None, .1, .2, contract, dense=dense)
        return phase, pipe

    def mark_core(self, phase, ledger):
        ledger.charge("core_video_reports")
        phase["core_metrics"] = {"status": "ALL_81_FRAMES_CORE_METRICS_PASS_QUALITY_PENDING"}
        ledger.complete_phase(phase)

    def test_plan_is_exact_five_source_ten_new_jobs_one_part(self):
        plan = plan_fixture()
        cases, audit = self.validate(plan)
        self.assertEqual([c["index"] for c in cases], list(range(5)))
        self.assertEqual(audit["summary"]["independent_image_clusters"], 5)
        self.assertEqual(audit["summary"]["unique_generation_run_ids"], 10)
        template = json.loads((ROOT / driver.TEMPLATE["path"]).read_text())
        self.assertFalse(set(plan["native_registration"]["allowed_run_ids"]) &
                         set(template["native_registration"]["allowed_run_ids"]))
        for name, value in (("cases", template["cases"]), ("parts", template["parts"]),
                ("max_full_generations_total", 40), ("max_gpu_cards", 4),
                ("cohort_plan", template["cohort_plan"]), ("output_HF_relative", template["output_HF_relative"]),
                ("native_registration", template["native_registration"]),
                ("bank_manifest", {**plan["bank_manifest"], "sha256": "changed"}),
                ("private_worker_config_sha256", template["private_worker_config_sha256"])):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.validate({**plan, name: value})
        for part in (1, 2, 3, -1, False):
            with self.assertRaises(ValueError):
                driver.validate_plan(plan, template, driver.load_engine(), part)

    def test_actual_active_requires_new_sha_exact_ten_ids_and_single_gpu_budget(self):
        plan = plan_fixture()
        active = active_fixture(plan)
        self.assertTrue(driver.registration_authorized(plan, "new_protocol_sha", active))
        scope = active["experiment_authority"]["current_user_authorized_scopes"][0]
        template = json.loads((ROOT / driver.TEMPLATE["path"]).read_text())
        for name, value in (("id", driver.TEMPLATE_ID), ("protocol_sha256", driver.TEMPLATE["sha256"]),
                ("allowed_run_ids", template["native_registration"]["allowed_run_ids"]),
                ("allowed_run_ids", scope["allowed_run_ids"][:-1]),
                ("allowed_run_ids", scope["allowed_run_ids"][::-1]),
                ("allowed_run_ids", scope["allowed_run_ids"] + ["extra"]),
                ("max_gpu_cards", 2), ("total_limits", template["limits_total"]),
                ("execution_status", "CLOSED_PARTIAL_THREE_COMPLETE_ONE_NVML_MONITOR_TIMEOUT")):
            changed = copy.deepcopy(active)
            changed["experiment_authority"]["current_user_authorized_scopes"][0][name] = value
            with self.subTest(name=name):
                self.assertFalse(driver.registration_authorized(plan, "new_protocol_sha", changed))

    def test_engine_scope_adaptation_does_not_patch_original_module_or_native_functions(self):
        plan = plan_fixture()
        with tempfile.TemporaryDirectory() as directory:
            engine = driver.execution_engine(plan, "new_protocol_sha", Path(directory))
        self.assertEqual(engine.ID, driver.ID)
        self.assertEqual(original.ID, driver.TEMPLATE_ID)
        self.assertIsNot(engine.PartLedger, original.PartLedger)
        for name in ("execute", "generate_video", "save_initial_noise", "verify_noise_pair",
                "registered_receipt", "attach_core_metrics", "release_gpu_cache"):
            self.assertEqual(getattr(engine, name).__code__.co_code, getattr(original, name).__code__.co_code)
        self.assertIs(engine.UB1ConsumerRuntime, original.UB1ConsumerRuntime)
        self.assertEqual(driver.sha256(ROOT / driver.ENGINE["path"]), driver.ENGINE["sha256"])

    def test_ledger_rejects_other_parts_sources_output_and_payload_before_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, engine, ledger = self.make_ledger(root)
            for index, mode in ((5, "dense"), (19, "candidate"), (True, "dense"), (0, "neutral")):
                with self.assertRaises(RuntimeError):
                    ledger.job(index, mode)
            for part, path, cases in ((1, ledger.path, plan["cases"]),
                    (0, root / "old-output/part0", plan["cases"]), (0, ledger.path, plan["cases"][:-1])):
                with self.assertRaises(RuntimeError):
                    engine.PartLedger(path, part, cases, "new_protocol_sha", plan, root)
            with self.assertRaises(FileExistsError):
                ledger.start()

    def test_same_hook_new_identity_noise_and_full160_sites_each_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, engine, ledger = self.make_ledger(root)
            dense, _ = self.generation(engine, ledger, "dense")
            self.mark_core(dense, ledger)
            candidate, _ = self.generation(engine, ledger, "candidate", dense=dense)
            self.mark_core(candidate, ledger)
            for mode, phase in (("dense", dense), ("candidate", candidate)):
                self.assertEqual(phase["id"], driver.ID)
                self.assertEqual(phase["protocol_sha256"], "new_protocol_sha")
                self.assertTrue(phase["registered_run_id"].startswith(driver.ID + ":"))
                self.assertEqual(phase["initial_noise"]["dtype"], "bfloat16")
                self.assertEqual(len(phase["trajectory"]), 4)
                self.assertEqual(len(phase["FFN_boundary_timings"]), 160)
                self.assertEqual(phase["native_counter_delta"], good_counts())
                _, job, source = ledger.job(0, mode)
                receipt = engine.registered_receipt(phase, job, source, {})
                self.assertEqual(receipt["native_ffn_sites"], 160)
                self.assertEqual(receipt["job"]["run_id"], phase["registered_run_id"])
            self.assertTrue(engine.verify_noise_pair(candidate["initial_noise"], dense["initial_noise"], root))
            self.assertNotEqual(dense["initial_noise"]["path"], candidate["initial_noise"]["path"])
            self.assertEqual(ledger.record["predebited_work"]["video_generations"], 2)
            self.assertEqual(ledger.record["predebited_work"]["native_RR_linears"], 320)

    def test_failed_dispatch_is_fully_predebited_without_retry_or_refund(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, engine, ledger = self.make_ledger(Path(directory))
            with self.assertRaisesRegex(RuntimeError, "Injected native"):
                self.generation(engine, ledger, "dense", fail_after=0)
            for key, amount in engine.generation_costs("dense").items():
                self.assertEqual(ledger.record["predebited_work"][key], amount)
            self.assertEqual(ledger.record["phases"], [])
            with mock.patch.object(driver, "read_active", return_value=active_fixture(plan)), self.assertRaises(RuntimeError):
                ledger.reserve_generation(0, "dense")
            self.assertEqual(ledger.record["predebited_work"]["video_generations"], 1)

    def test_authority_revocation_stops_before_predebit(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, engine, ledger = self.make_ledger(Path(directory))
            with mock.patch.object(driver, "read_active", return_value={}), self.assertRaisesRegex(RuntimeError, "actual|Actual"):
                ledger.reserve_generation(0, "dense")
            self.assertTrue(all(x == 0 for x in ledger.record["predebited_work"].values()))

    def test_mismatched_native_noise_stops_candidate_without_update(self):
        for value in (torch.ones((1, 1, 1, 2), dtype=torch.bfloat16),
                torch.zeros((1, 1, 1, 2), dtype=torch.float32)):
            with self.subTest(dtype=value.dtype), tempfile.TemporaryDirectory() as directory:
                plan, engine, ledger = self.make_ledger(Path(directory))
                dense, _ = self.generation(engine, ledger, "dense")
                self.mark_core(dense, ledger)
                with self.assertRaisesRegex(RuntimeError, "noise bytes"):
                    self.generation(engine, ledger, "candidate", dense=dense, initial=value)
                phase = json.loads((ledger.path / "source00_seed42/candidate/PHASE.json").read_text())
                self.assertEqual(phase["completed_native_flow_steps"], 0)
                self.assertEqual(ledger.record["predebited_work"]["native_full_model_flows"], 8)

    def test_all_ten_jobs_stop_at_exact_upper_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, engine, ledger = self.make_ledger(Path(directory))
            for index in range(5):
                dense = None
                for mode in driver.MODES:
                    phase, _ = self.generation(engine, ledger, mode, dense=dense, index=index)
                    self.mark_core(phase, ledger)
                    if mode == "dense":
                        dense = phase
            self.assertEqual(ledger.record["predebited_work"], engine.PART_LIMITS)
            self.assertEqual(len(ledger.record["completed_pairs"]), 5)
            self.assertEqual(len({p["registered_run_id"] for p in ledger.record["phases"]}), 10)
            with mock.patch.object(driver, "read_active", return_value=active_fixture(plan)), self.assertRaises(RuntimeError):
                ledger.reserve_generation(0, "dense")

    def test_execute_never_accepts_fabricated_readiness_or_old_scope_authority(self):
        plan = plan_fixture()
        protocol = ROOT / "config/ub1_consumer_disjoint20_recovery_20261027_v1.json"
        args = SimpleNamespace(protocol=protocol, part=0, execute_native=True)
        prepared = (plan, plan["cases"], [], {}, {}, {}, {"authorized": True})
        with mock.patch.object(driver, "read_active", return_value={}), self.assertRaisesRegex(RuntimeError, "actual ACTIVE"):
            driver.execute(args, prepared)
        args.part = 1
        with self.assertRaisesRegex(RuntimeError, "part0"):
            driver.execute(args, prepared)


if __name__ == "__main__":
    unittest.main()
