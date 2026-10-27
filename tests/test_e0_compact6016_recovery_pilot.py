"""Frozen scope, source-leak prevention and actual CPU dispatch accounting."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import e0_compact6016_recovery_pilot as m
from compact6016_adapter import CompactAdapters, compact_forward, gated_loss


class Tests(unittest.TestCase):
    def plan(self):
        return json.loads((ROOT / "config/e0_compact6016_recovery_pilot_20261028_v1.json").read_text())

    def active(self):
        return {"user_continuation_authorization_20261027": {
            "id": "USER_RESUME_OFFLINE_IFC_20261027", "status": "ACTIVE_CURRENT_USER_DIRECTED_SESSION",
            "max_simultaneously_assigned_idle_GPUs": 4, "new_scope_registration_allowed": True},
            "experiment_authority": {"current_user_authorized_scopes": [{"id": m.ID,
                "protocol_sha256": "digest", "execution_status": "AUTHORIZED_NOT_STARTED",
                "allowed_run_ids": m.expected_run_ids(), "limits": m.LIMITS,
                "max_simultaneously_assigned_idle_GPUs": 1}]}}

    def test_exact_authority_run_ids_budget_and_revocation(self):
        plan, active = self.plan(), self.active()
        plan["assets_finalized"] = True
        m.authorize(plan, "digest", active, m.expected_run_ids()[0])
        with self.assertRaises(ValueError):
            m.authorize(plan, "digest", active, "foreign-run")
        active["experiment_authority"]["current_user_authorized_scopes"][0]["limits"] = {}
        with self.assertRaises(ValueError):
            m.authorize(plan, "digest", active)
        active = self.active()
        active["user_continuation_authorization_20261027"]["status"] = "REVOKED"
        with self.assertRaises(ValueError):
            m.authorize(plan, "digest", active)

    def test_operation_limits_include_every_extra_gpu_gate(self):
        self.assertEqual(m.LIMITS["student_forwards"], 384 + 1024 + 16)
        self.assertEqual(m.LIMITS["native_Q2_calls"], 1424)
        self.assertEqual(m.LIMITS["backward_calls"], 1032)
        self.assertEqual(m.LIMITS["Adam_steps"], 1024)
        self.assertEqual(m.LIMITS["BF16_forward_GEMM"], 2848)
        self.assertEqual(m.LIMITS["FP32_merge_GEMM"], 2080)
        self.assertEqual(m.LIMITS["BF16_backward_GEMM"], 3096)
        self.assertEqual(m.LIMITS["FP32_LoRA_backward_GEMM"], 4128)
        self.assertEqual(m.RESOURCE_LIMITS["new_HF_bytes"], int(1.25 * 2**30))

    def test_budget_predebit_atomic_and_failed_attempt_retains_charge(self):
        report = {}
        ledger = m.BudgetLedger(report)
        ledger.reserve({"Adam_steps": 1024}, "run")
        before = ledger.predebited.copy()
        with self.assertRaises(ValueError):
            ledger.reserve({"student_forwards": 1, "Adam_steps": 1}, "run")
        self.assertEqual(before, ledger.predebited)
        with self.assertRaises(ValueError):
            ledger.observe("native_Q2_calls")
        self.assertEqual(ledger.predebited["Adam_steps"], 1024)
        self.assertEqual(ledger.observed["Adam_steps"], 0)

    def test_heldout_energy_arrays_never_read_by_support_selection(self):
        profile = {"records": [{"index": i, "step": 0, "block": 0,
                                "group_energy": list(map(float, range(216))) if i < 16 else object()}
                               for i in range(24)]}
        self.assertEqual(m.select_energy_groups(profile, 0, 0), list(range(122, 216)))
        profile["records"][0]["index"] = 16
        with self.assertRaises(ValueError):
            m.select_energy_groups(profile, 0, 0)

    def test_heldout_training_rejected_and_random_is_data_independent(self):
        cases = {i: object() for i in range(24)}
        self.assertIs(m.training_case(cases, 0), cases[0])
        for bad in (16, 23, -1, 0., True):
            with self.assertRaises(ValueError):
                m.training_case(cases, bad)
        self.assertEqual(m.random_groups_by_position(), m.random_groups_by_position())
        self.assertTrue(all(len(groups) == len(set(groups)) == 94 for groups in m.random_groups_by_position().values()))

    def test_plan_profile_binding_support_and_rows_reject_leak(self):
        plan = self.plan()
        sources = {s["index"]: s for s in plan["sources"]}
        profile = {"schema": "ffnds.calibration.source_statistics.v2", "evidence_kind": "ACTUAL_NATIVE_OBSERVATION", "records": []}
        for case in plan["cases"]:
            source = sources[case["index"]]
            profile["records"].append({"index": case["index"], "step": case["step"], "block": case["block"],
                "source_id": source["semantic_source_id"], "image_id": source["source_id"], "seed": source["seed"],
                "source_state": "teacher_input", "payload_sha256": case["sha256"],
                "statistic_semantics": "native_gate_times_W2_group_contribution_l2_squared",
                "normalization": "mean_over_256_saved_rows", "group_energy": list(map(float, range(216)))})
        for cell in plan["cells"]:
            if cell["initialization"] == "energy94":
                cell["groups"] = m.select_energy_groups(profile, cell["step"], cell["block"])
        m.validate_plan(plan, profile)
        bad = copy.deepcopy(plan)
        bad["source_split"]["training"][-1] = 16
        with self.assertRaises(ValueError):
            m.validate_plan(bad, profile)
        bad = copy.deepcopy(plan)
        bad["cases"][0]["rows"][0] = bad["cases"][0]["rows"][1]
        with self.assertRaises(ValueError):
            m.validate_plan(bad, profile)
        bad = copy.deepcopy(profile)
        bad["records"][0]["payload_sha256"] = "different"
        with self.assertRaises(ValueError):
            m.validate_plan(plan, bad)

    def test_bound_asset_escape_and_changed_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "asset"
            path.write_bytes(b"bound")
            item = {"path": "asset", "sha256": m.sha(path), "bytes": 5}
            self.assertEqual(m.bound(root, item), path)
            path.write_bytes(b"wrong")
            with self.assertRaises(ValueError):
                m.bound(root, item)
            item["path"] = "../asset"
            with self.assertRaises(ValueError):
                m.bound(root, item)

    def test_real_cpu_dispatch_counts_forward_merge_and_backward(self):
        # Actual autograd matmuls on tiny synthetic BF16 CPU tensors. This
        # checks the ledger categorization, not native HiF4 or GPU timings.
        generator = torch.Generator().manual_seed(8)
        model = CompactAdapters((torch.randn(64, 8, generator=generator) * .1).bfloat16(),
                                (torch.randn(8, 64, generator=generator) * .1).bfloat16(), rank=3)
        x = torch.randn(4, 8, generator=generator).bfloat16()
        target = torch.randn(4, 8, generator=generator).bfloat16()
        ledger = m.BudgetLedger({})
        ledger.reserve({"FP32_merge_GEMM": 2, "BF16_forward_GEMM": 2,
                        "BF16_backward_GEMM": 3, "FP32_LoRA_backward_GEMM": 4}, "run")
        observer = m._matmul_observer(torch, ledger)
        with observer:
            with observer.during("merge"):
                w1, w2 = model.merged_weights()
            with observer.during("forward"):
                y = compact_forward(x, w1, torch.zeros(64).bfloat16(), w2, torch.zeros(8).bfloat16(),
                                    torch.ones(64), lambda value, force_fp32: value.clone(), training=True)
            value = gated_loss(y, target, torch.ones(1, 8).bfloat16())
            with observer.during("backward"):
                value.backward()
        for key in ("FP32_merge_GEMM", "BF16_forward_GEMM", "BF16_backward_GEMM", "FP32_LoRA_backward_GEMM"):
            self.assertEqual(ledger.predebited[key], ledger.observed[key], key)
        with self.assertRaises(RuntimeError):
            with observer:
                torch.ones(2, 2) @ torch.ones(2, 2)

    def test_native_quantizer_rejects_unavailable_compiled_fallback(self):
        class Backend:
            @staticmethod
            def get_hif4_api():
                return m.select_energy_groups, object(), "HiFloat4.main"
        with self.assertRaisesRegex(ValueError, "compiled native HiF4 kernels unavailable"):
            m.verify_native_quantizer(Backend(), {"repo_sources": []})


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
