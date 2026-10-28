"""CPU-only contracts for the unexecuted full80 A0 bank preparation."""
import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import e0_compact6016_full80_bank as m


def synthetic_operand(path, *, wrong_dtype=False, overlap=False):
    shapes = {"V1": [256, 5120], "V2": [256, 13824], "Y": [256, 5120],
              "gate": [1, 5120], "b2": [5120]}
    header, offset = {}, 0
    for name, shape in shapes.items():
        size = 2
        for dimension in shape:
            size *= dimension
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    if wrong_dtype:
        header["V1"]["dtype"] = "F32"
    if overlap:
        header["Y"]["data_offsets"] = header["V1"]["data_offsets"][:]
    raw = json.dumps(header).encode()
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        handle.truncate(8 + len(raw) + offset)


class Tests(unittest.TestCase):
    def plan(self):
        return json.loads((ROOT / "config/e0_compact6016_full80_bank_20261028_v1.json").read_text())

    def active(self):
        return {"user_continuation_authorization_20261027": {
            "id": "USER_RESUME_OFFLINE_IFC_20261027", "status": "ACTIVE_CURRENT_USER_DIRECTED_SESSION",
            "max_simultaneously_assigned_idle_GPUs": 4, "new_scope_registration_allowed": True},
            "experiment_authority": {"current_user_authorized_scopes": [{"id": m.ID,
                "protocol_sha256": "digest", "execution_status": "AUTHORIZED_NOT_STARTED",
                "allowed_run_ids": m.run_ids(), "limits_per_part": m.PART_LIMITS,
                "limits_total": m.TOTAL_LIMITS, "max_simultaneously_assigned_idle_GPUs": 4}]}}

    def test_partitions_exactly_eighty_shared_step_functions(self):
        all_functions = []
        for part in range(4):
            functions = m.functions_for(part)
            self.assertEqual(len(functions), 20)
            self.assertEqual(sum(e == m.EXPERTS[0] for e, _ in functions), 10)
            self.assertEqual(sum(e == m.EXPERTS[1] for e, _ in functions), 10)
            self.assertEqual(len(m.run_ids(part)), 40)
            all_functions.extend(functions)
        self.assertEqual(set(all_functions), set(m.FUNCTIONS))
        self.assertEqual(len(set(all_functions)), 80)
        self.assertEqual(m.steps_for(m.EXPERTS[0]), (0, 1))
        self.assertEqual(m.steps_for(m.EXPERTS[1]), (2, 3))
        for value in (-1, 4, True):
            with self.assertRaises(ValueError):
                m.functions_for(value)

    def test_exact_total_cost_includes_training_evaluation_and_gpu_controls(self):
        self.assertEqual(m.PART_LIMITS["student_forwards"], 40 * (256 + 96 + 2))
        self.assertEqual(m.TOTAL_LIMITS["student_forwards"], 56640)
        self.assertEqual(m.TOTAL_LIMITS["native_Q2_calls"], 56640)
        self.assertEqual(m.TOTAL_LIMITS["Adam_steps"], 40960)
        self.assertEqual(m.TOTAL_LIMITS["support_source_step_calls"], 2560)
        self.assertEqual(m.TOTAL_LIMITS["support_FP32_batched_bmm"], 69120)
        self.assertEqual(m.PART_LIMITS["backward_calls"], 40 * 257)
        self.assertEqual(m.PART_LIMITS["merged_weight_pairs"], 40 * 258)
        self.assertEqual(m.PART_LIMITS["exported_students"], 40)
        self.assertEqual(m.RESOURCE_LIMITS["compute_wall_seconds_per_part"], 2400)
        self.assertEqual(m.RESOURCE_LIMITS["external_wall_seconds_per_part"], 2700)

    def test_exact_authorization_and_cross_part_run_rejection(self):
        plan, active = self.plan(), self.active()
        plan["assets_finalized"] = True
        m.authorize(plan, "digest", active, 0, m.run_ids(0)[0])
        with self.assertRaises(ValueError):
            m.authorize(plan, "digest", active, 0, m.run_ids(1)[0])
        active["experiment_authority"]["current_user_authorized_scopes"][0]["limits_total"] = {}
        with self.assertRaises(ValueError):
            m.authorize(plan, "digest", active, 0)
        active = self.active()
        active["user_continuation_authorization_20261027"]["status"] = "REVOKED"
        with self.assertRaises(ValueError):
            m.authorize(plan, "digest", active, 0)

    def test_atomic_predebit_and_unpaid_support_bmm_rejected(self):
        ledger = m.Ledger({})
        ledger.reserve({"Adam_steps": m.PART_LIMITS["Adam_steps"]}, "run")
        before = ledger.predebited.copy()
        with self.assertRaises(ValueError):
            ledger.reserve({"student_forwards": 1, "Adam_steps": 1}, "run")
        self.assertEqual(before, ledger.predebited)
        with self.assertRaises(ValueError):
            ledger.observe("support_FP32_batched_bmm")
        self.assertEqual(ledger.observed["Adam_steps"], 0)

    def test_training_only_two_step_source_balanced_support(self):
        records = []
        for index in range(24):
            for step in (0, 1):
                records.append({"index": index, "step": step, "expert": m.EXPERTS[0], "block": 0,
                    "group_energy": [float(group + index + step) for group in range(216)] if index < 16 else object()})
        groups, mean = m.select_support(records, m.EXPERTS[0], 0)
        self.assertEqual(groups, list(range(122, 216)))
        self.assertAlmostEqual(mean[0], 8.)
        self.assertAlmostEqual(mean[-1], 223.)
        records[0]["step"] = 1
        with self.assertRaises(ValueError):
            m.select_support(records, m.EXPERTS[0], 0)

    def test_source_holdout_and_wrong_step_training_rejected(self):
        cases = {(i, s): object() for i in range(24) for s in range(4)}
        self.assertIs(m.training_case(cases, 0, 1, m.EXPERTS[0]), cases[0, 1])
        for index, step in ((16, 0), (23, 1), (-1, 0), (0, 2), (0, False), (True, 0)):
            with self.assertRaises(ValueError):
                m.training_case(cases, index, step, m.EXPERTS[0])

    def test_eighty_random_draws_reproducible_and_training_independent(self):
        first, second = m.random_supports(), m.random_supports()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 80)
        self.assertEqual(len({tuple(v) for v in first.values()}), 80)
        self.assertTrue(all(v == sorted(set(v)) and len(v) == 94 for v in first.values()))

    def test_real_safetensors_header_bytes_dtype_and_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "operand.safetensors"
            synthetic_operand(path)
            self.assertEqual(m.operand_header(path)["V2"], {"shape": [256, 13824], "dtype": "BF16"})
            synthetic_operand(path, wrong_dtype=True)
            with self.assertRaises(ValueError):
                m.operand_header(path)
            synthetic_operand(path, overlap=True)
            with self.assertRaises(ValueError):
                m.operand_header(path)
            path.write_bytes(struct.pack("<Q", 1 << 30))
            with self.assertRaises(ValueError):
                m.operand_header(path)

    def test_source_cluster_means_keep_twosteps_inside_source(self):
        records = [{"index": i, "step": s, "metric": float(i + 10 * s)} for i in range(24) for s in (2, 3)]
        result = m.source_cluster_means(records, m.EXPERTS[1], "metric")
        self.assertEqual(len(result), 24)
        self.assertEqual(result[0], {"index": 0, "value": 25.})
        records[0]["metric"] = None
        self.assertIsNone(m.source_cluster_means(records, m.EXPERTS[1], "metric")[0]["value"])
        with self.assertRaises(ValueError):
            m.source_cluster_means(records + [records[0]], m.EXPERTS[1], "metric")

    def test_actual_cpu_support_dispatch_only_counts_paid_fp32_bmm(self):
        ledger = m.Ledger({})
        ledger.reserve({"support_FP32_batched_bmm": 1, "support_logical_group_products": 8}, "run")
        observer = m._support_observer(torch, ledger)
        with observer:
            value = torch.bmm(torch.ones(8, 4, 3), torch.ones(8, 3, 2))
        self.assertEqual(tuple(value.shape), (8, 4, 2))
        self.assertEqual(ledger.observed["support_FP32_batched_bmm"], 1)
        self.assertEqual(ledger.observed["support_logical_group_products"], 8)
        with self.assertRaises(ValueError):
            with observer:
                torch.bmm(torch.ones(8, 4, 3), torch.ones(8, 3, 2))
        with self.assertRaises(RuntimeError):
            with observer:
                torch.ones(2, 2) @ torch.ones(2, 2)

    def test_expand_original_reports_960_cases_eachpart_without_stating_rows_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            hf = Path(tmp)
            plan = {"sources": [], "inputs": []}
            for index in range(24):
                source = {"index": index, "source_id": f"image{index}", "source_group": f"prompt{index}", "seed": 42 + index}
                plan["sources"].append(source)
                scopes = []
                for step in range(4):
                    for block in range(40):
                        scopes.append({"step": step, "block": block, "expert": m.EXPERTS[0 if step < 2 else 1],
                            "grid": [21, 29, 52], "rows": 256, "selected_rows": list(range(256)),
                            "sampling_seed": 11, "path": str(hf / f"input{index}_s{step}_b{block}.safetensors"),
                            "sha256": "f" * 64, "bytes": 12341624})
                record = {"index": index, "status": "PASS_NATIVE_OPERANDS_AND_SAME_PIPELINE_NOOP",
                    "input": {"input_provenance": {"image_sha256": source["source_id"]},
                              "source_group": source["source_group"], "calibration_seed": source["seed"]}, "scopes": scopes}
                path = hf / f"INPUT{index}.json"
                path.write_text(json.dumps(record))
                plan["inputs"].append({"index": index, "path": path.name, "bytes": path.stat().st_size, "sha256": m.sha(path)})
            union = []
            for part in range(4):
                cases = m.expand_cases(plan, hf, part)
                self.assertEqual(len(cases), 960)
                self.assertEqual(len({c["index"] for c in cases}), 24)
                union.extend((c["index"], c["step"], c["block"]) for c in cases)
            self.assertEqual(len(set(union)), 3840)
            plan["sources"][0]["seed"] = -1
            with self.assertRaises(ValueError):
                m.expand_cases(plan, hf, 0)

    def test_prepared_config_does_not_relabel_pilot_or_add_new_model_permissions(self):
        plan = self.plan()
        self.assertNotEqual(plan["id"], m.pilot.ID)
        self.assertEqual(plan["state_distribution"], "saved_native_teacher_post_Q1; student_own_W1_GELU_Smooth2_native_Q2")
        self.assertEqual(plan["deployment_contract"]["full_BF16_exports"], 160)
        self.assertEqual(plan["limits_total"]["full_model_calls"], 0)
        self.assertEqual(plan["limits_total"]["video_generations"], 0)
        self.assertEqual(plan["training"]["gradient_clipping"], False)
        frozen = {x["path"]: x["sha256"] for x in plan["repo_sources"]}
        for name in ("tools/e0_compact6016_recovery_pilot.py", "tools/compact6016_adapter.py"):
            self.assertEqual(m.sha(ROOT / name), frozen[name])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
