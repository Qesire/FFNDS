"""CPU-only contracts for the new sixty-function completion preparation."""
import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import e0_compact6016_full80_completion as m
import e0_compact6016_full80_bank as original


class SyntheticLineage:
    """Tiny metadata files stand for future receipts; no model tensors exist."""
    def __init__(self, path):
        self.root, self.hf = path / "repo", path / "hf"
        self.root.mkdir()
        self.hf.mkdir()
        self.plan = json.loads((ROOT / "config/e0_compact6016_full80_completion_20261028_v1.json").read_text())
        old = json.loads((ROOT / "config/e0_compact6016_full80_bank_20261028_v1.json").read_text())
        self.plan["original_protocol"] = self.put(self.root, "original.json", old)
        self.old_digest = self.plan["original_protocol"]["sha256"]
        self.random = original.random_supports()
        parts, audited = [], []
        for part in range(3):
            functions = [f for f in m.COMPLETED_FUNCTIONS if f[1] // 10 == part]
            records, cells = self.functions_and_cells(functions, old=True)
            run = {"id": m.ORIGINAL_ID, "part": part, "protocol_sha256": self.old_digest,
                   "status": "FAILED_NO_RETRY", "functions": records, "cells": cells}
            exit_record = {"id": m.ORIGINAL_ID, "part": part, "protocol_sha256": self.old_digest,
                           "launcher_returncode": 1, "process_exited": True}
            parts.append({"part": part, "run": self.put(self.hf, f"old{part}/RUN.json", run),
                          "exit": self.put(self.hf, f"old{part}/EXIT.json", exit_record)})
            audited.extend({"run_id": c["run_id"], "expert": c["expert"], "block": c["block"],
                "initialization": c["initialization"], "steps": list(m.steps_for(c["expert"])),
                "result": c["result"], "export": m._file_binding(c["export"]),
                "updates_from_completion_contract": 256} for c in cells)
        protocol = {"id": m.AUDIT_ID, "assets_finalized": True, "automatic_retry": False,
                    "original_protocol_sha256": self.old_digest, "parts": parts}
        audit_protocol = self.put(self.root, "audit_protocol.json", protocol)
        manifest = {"schema": "ffnds.compact6016.partial_completed_cells.v1",
                    "status": "AUDITED_COMPLETE_CELLS_FROM_FAILED_PARTS",
                    "original_protocol_sha256": self.old_digest, "parts": parts, "cells": audited}
        manifest_binding = self.put(self.hf, "audit/MANIFEST.json", manifest)
        receipt = {"id": m.AUDIT_ID, "status": "COMPLETE_PARTIAL_ASSET_AUDIT",
                   "protocol_sha256": audit_protocol["sha256"], "cells": 40, "functions": 20,
                   "manifest_sha256": manifest_binding["sha256"],
                   "export_bytes_verified": sum(c["export"]["bytes"] for c in audited)}
        exit_record = {"id": m.AUDIT_ID, "protocol_sha256": audit_protocol["sha256"],
                       "launcher_returncode": 0, "process_exited": True}
        audit_run = self.put(self.hf, "audit/RUN.json", receipt)
        exit_record["RUN_sha256"] = audit_run["sha256"]
        self.plan["partial_completed_assets"] = {"audit_protocol": audit_protocol,
            "manifest": manifest_binding, "RUN": audit_run,
            "launch_exit": self.put(self.hf, "audit/EXIT.json", exit_record)}

    def put(self, root, name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True))
        return {"path": name, "bytes": path.stat().st_size, "sha256": m.sha(path)}

    def change(self, root, binding, change):
        value = json.loads((root / binding["path"]).read_text())
        change(value)
        return self.put(root, binding["path"], value)

    def change_completion_run(self, receipt, change):
        """Keep the exit bound so a semantic defect reaches its own gate."""
        receipt["RUN"] = self.change(self.hf, receipt["RUN"], change)
        receipt["launch_exit"] = self.change(self.hf, receipt["launch_exit"],
            lambda record: record.update(RUN_sha256=receipt["RUN"]["sha256"]))

    def functions_and_cells(self, functions, *, old):
        records, cells = [], []
        prefix = "old" if old else "new"
        shapes = {"W1": [6016, 5120], "W2": [5120, 6016], "b1": [6016], "b2": [5120],
            "smooth1": [5120], "smooth2": [6016], "groups": [94], "channels": [6016],
            "training_lora_A1": [32, 5120], "training_lora_B1": [6016, 32],
            "training_lora_A2": [32, 6016], "training_lora_B2": [5120, 32]}
        for expert, block in functions:
            support = {"status": "COMPLETE", "heldout_used_for_support": False,
                "energy_groups": list(range(94)), "native_FP32_batched_bmm": 864,
                "records": [{"index": i, "step": s, "expert": expert, "block": block}
                            for i in range(16) for s in m.steps_for(expert)]}
            record = {"expert": expert, "block": block, "energy_groups": list(range(94)),
                "random_groups": self.random[expert, block],
                "support_file": self.put(self.hf, f"{prefix}/{expert}/{block}/SUPPORT.json", support)}
            records.append(record)
            for init in m.INITIALIZATIONS:
                run_id = (m._original_run_id(expert, block, init) if old
                          else m.function_id(expert, block) + "_" + init)
                # Small synthetic bytes intentionally do not claim a tensor
                # payload. The independent deployment loader validates bytes.
                export = self.put(self.hf, f"{prefix}/{run_id}/folded", {"fixture": run_id})
                export.update(reloaded_all_tensor_bytes_equal=True, online_LoRA_GEMM=0, online_weight_QDQ=0,
                    tensors={name: {"shape": shape, "sha256": "f" * 64,
                        "dtype": "torch.int64" if name in ("groups", "channels") else
                            "torch.float32" if name.startswith(("smooth", "training_")) else "torch.bfloat16"}
                        for name, shape in shapes.items()})
                diagnostics = [{"index": i, "step": s, "role": "TRAIN" if i < 16 else "HELDOUT_EXPOSED_DIAGNOSTIC",
                    "gated_pre_gate_relative_SSE": float(i + s + 1), "actual_BF16_gate_relative_SSE": float(i + s + 2)}
                    for i in range(24) for s in m.steps_for(expert)]
                row = {"run_id": run_id, "expert": expert, "block": block, "initialization": init,
                    "steps": list(m.steps_for(expert)), "status": m.CELL_STATUS,
                    "groups": record["energy_groups" if init == "energy94" else "random_groups"],
                    "export": export, "epoch_source_step_loss_means": [1.] * 8,
                    "zero_merge_base_weight_bitwise": [False, False],
                    "GPU_zero_LoRA_and_native_STE_byte_checks": dict.fromkeys(
                        ("post_GELU", "pre_Q2", "post_Q2", "pre_gate_output"), True),
                    "GPU_gate_gradient_norms": {"lora_A1": 0., "lora_A2": 0., "lora_B1": 1., "lora_B2": 1.},
                    "diagnostics": {phase: diagnostics for phase in ("initial", "final_folded_BF16")},
                    "source_cluster_means": {phase: {metric: m.source_cluster_means(diagnostics, expert, metric)
                        for metric in ("gated_pre_gate_relative_SSE", "actual_BF16_gate_relative_SSE")}
                        for phase in ("initial", "final_folded_BF16")}}
                cells.append({"run_id": run_id, "expert": expert, "block": block, "initialization": init,
                    "status": m.CELL_STATUS, "export": export,
                    "result": self.put(self.hf, f"{prefix}/{run_id}/RESULT.json", row)})
        return records, cells

    def completion_parts(self):
        self.plan["assets_finalized"] = True
        protocol = self.put(self.root, "completion.json", self.plan)
        parts = []
        for part in range(4):
            functions, cells = self.functions_and_cells(m.functions_for(part), old=False)
            run = {"id": m.ID, "part": part, "protocol_sha256": protocol["sha256"],
                "status": "COMPLETE_FULL80_COMPLETION_PART", "functions": functions, "cells": cells,
                "observed": m.PART_LIMITS, "predebited": m.PART_LIMITS,
                "reused_asset_chain": {"partial_audit": self.plan["partial_completed_assets"],
                    "original_protocol": self.plan["original_protocol"], "complete_cells": 40,
                    "complete_functions": 20, "partial_parameters_loaded": False}}
            exit_record = {"id": m.ID, "part": part, "protocol_sha256": protocol["sha256"],
                           "launcher_returncode": 0, "process_exited": True}
            run_binding = self.put(self.hf, f"completion{part}/RUN.json", run)
            exit_record["RUN_sha256"] = run_binding["sha256"]
            parts.append({"part": part, "RUN": run_binding,
                          "launch_exit": self.put(self.hf, f"completion{part}/EXIT.json", exit_record)})
        return protocol, parts


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
        return json.loads((ROOT / "config/e0_compact6016_full80_completion_20261028_v1.json").read_text())

    def active(self):
        return {"user_continuation_authorization_20261027": {
            "id": "USER_RESUME_OFFLINE_IFC_20261027", "status": "ACTIVE_CURRENT_USER_DIRECTED_SESSION",
            "max_simultaneously_assigned_idle_GPUs": 4, "new_scope_registration_allowed": True},
            "experiment_authority": {"current_user_authorized_scopes": [{"id": m.ID,
                "protocol_sha256": "digest", "execution_status": "AUTHORIZED_NOT_STARTED",
                "allowed_run_ids": m.run_ids(), "limits_per_part": m.PART_LIMITS,
                "limits_total": m.TOTAL_LIMITS, "max_simultaneously_assigned_idle_GPUs": 4}]}}

    def test_partitions_exactly_sixty_missing_shared_step_functions(self):
        all_functions = []
        for part in range(4):
            functions = m.functions_for(part)
            self.assertEqual(len(functions), 15)
            self.assertTrue(set(functions).isdisjoint(m.COMPLETED_FUNCTIONS))
            self.assertEqual(len(m.run_ids(part)), 30)
            all_functions.extend(functions)
        self.assertEqual(set(all_functions), set(m.REMAINING_FUNCTIONS))
        self.assertEqual(len(set(all_functions)), 60)
        self.assertEqual(m.steps_for(m.EXPERTS[0]), (0, 1))
        self.assertEqual(m.steps_for(m.EXPERTS[1]), (2, 3))
        for value in (-1, 4, True):
            with self.assertRaises(ValueError):
                m.functions_for(value)

    def test_exact_total_cost_includes_training_evaluation_and_gpu_controls(self):
        self.assertEqual(m.PART_LIMITS["student_forwards"], 30 * (256 + 96 + 2))
        self.assertEqual(m.TOTAL_LIMITS["student_forwards"], 42480)
        self.assertEqual(m.TOTAL_LIMITS["native_Q2_calls"], 42480)
        self.assertEqual(m.TOTAL_LIMITS["Adam_steps"], 30720)
        self.assertEqual(m.TOTAL_LIMITS["support_source_step_calls"], 1920)
        self.assertEqual(m.TOTAL_LIMITS["support_FP32_batched_bmm"], 51840)
        self.assertEqual(m.PART_LIMITS["backward_calls"], 30 * 257)
        self.assertEqual(m.PART_LIMITS["merged_weight_pairs"], 30 * 258)
        self.assertEqual(m.PART_LIMITS["exported_students"], 30)
        self.assertEqual(m.RESOURCE_LIMITS["compute_wall_seconds_per_part"], 7200)
        self.assertEqual(m.RESOURCE_LIMITS["external_wall_seconds_per_part"], 7500)

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

    def test_expand_original_reports_720_cases_eachpart_without_stating_rows_independent(self):
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
                self.assertEqual(len(cases), 720)
                self.assertEqual(len({c["index"] for c in cases}), 24)
                union.extend((c["index"], c["step"], c["block"]) for c in cases)
            self.assertEqual(len(set(union)), 2880)
            plan["sources"][0]["seed"] = -1
            with self.assertRaises(ValueError):
                m.expand_cases(plan, hf, 0)

    def test_prepared_config_does_not_relabel_pilot_or_add_new_model_permissions(self):
        plan = self.plan()
        self.assertNotEqual(plan["id"], m.pilot.ID)
        self.assertEqual(plan["state_distribution"], "saved_native_teacher_post_Q1; student_own_W1_GELU_Smooth2_native_Q2")
        self.assertEqual(plan["deployment_contract"]["full_BF16_exports"], 120)
        self.assertEqual(plan["limits_total"]["full_model_calls"], 0)
        self.assertEqual(plan["limits_total"]["video_generations"], 0)
        self.assertEqual(plan["training"]["gradient_clipping"], False)
        frozen = {x["path"]: x["sha256"] for x in plan["repo_sources"]}
        for name in ("tools/e0_compact6016_recovery_pilot.py", "tools/compact6016_adapter.py"):
            self.assertEqual(m.sha(ROOT / name), frozen[name])

    def test_complete_functions_are_never_new_run_ids_and_random_draws_match_original80(self):
        self.assertEqual(m.random_supports(), original.random_supports())
        self.assertEqual(len(m.run_ids()), 120)
        self.assertTrue(set(m.run_ids()).isdisjoint(original.run_ids()))
        for e, b in m.COMPLETED_FUNCTIONS:
            self.assertNotIn(m.function_id(e, b) + "_energy94", m.run_ids())
        for metric, value in original.PART_LIMITS.items():
            self.assertEqual(m.PART_LIMITS[metric] * 4, value * 3)
        plan = self.plan()
        plan["assets_finalized"] = True
        for forbidden in (original.run_ids(0)[0], m.function_id(*m.COMPLETED_FUNCTIONS[0]) + "_energy94"):
            with self.assertRaises(ValueError):
                m.authorize(plan, "digest", self.active(), 0, forbidden)
        plan["completion_contract"]["load_partial_parameters"] = True
        with self.assertRaises(ValueError):
            m.authorize(plan, "digest", self.active(), 0)

    def test_audited_failed_parents_reuse_exact40_and_signed_zero_is_not_an_output_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = SyntheticLineage(Path(tmp))
            cells = m.validate_partial_reuse(fixture.plan, fixture.hf, repo_root=fixture.root)
            self.assertEqual(len(cells), 40)
            self.assertEqual({c["producer_id"] for c in cells}, {m.ORIGINAL_ID})
            self.assertEqual({(c["expert"], c["block"]) for c in cells}, set(m.COMPLETED_FUNCTIONS))

    def test_audit_missing_actual_exit0_and_39cell_half_pair_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = SyntheticLineage(Path(tmp))
            assets = fixture.plan["partial_completed_assets"]
            assets["launch_exit"] = fixture.change(fixture.hf, assets["launch_exit"],
                lambda record: record.update(launcher_returncode=1))
            with self.assertRaisesRegex(ValueError, "Partial asset audit/actual exit0/original protocol binding differs"):
                m.validate_partial_reuse(fixture.plan, fixture.hf, repo_root=fixture.root)
        with tempfile.TemporaryDirectory() as tmp:
            fixture = SyntheticLineage(Path(tmp))
            assets = fixture.plan["partial_completed_assets"]
            assets["manifest"] = fixture.change(fixture.hf, assets["manifest"], lambda record: record["cells"].pop())
            assets["RUN"] = fixture.change(fixture.hf, assets["RUN"],
                lambda record: record.update(manifest_sha256=assets["manifest"]["sha256"]))
            assets["launch_exit"] = fixture.change(fixture.hf, assets["launch_exit"],
                lambda record: record.update(RUN_sha256=assets["RUN"]["sha256"]))
            with self.assertRaisesRegex(ValueError, "Reuse requires exactly twenty paired complete original functions"):
                m.validate_partial_reuse(fixture.plan, fixture.hf, repo_root=fixture.root)

    def test_actual_exit_receipt_must_bind_its_exact_run_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = SyntheticLineage(Path(tmp))
            assets = fixture.plan["partial_completed_assets"]
            assets["launch_exit"] = fixture.change(fixture.hf, assets["launch_exit"],
                lambda record: record.update(RUN_sha256="0" * 64))
            with self.assertRaisesRegex(ValueError, "Partial asset audit/actual exit0/original protocol binding differs"):
                m.validate_partial_reuse(fixture.plan, fixture.hf, repo_root=fixture.root)
        with tempfile.TemporaryDirectory() as tmp:
            fixture = SyntheticLineage(Path(tmp))
            protocol, parts = fixture.completion_parts()
            parts[0]["launch_exit"] = fixture.change(fixture.hf, parts[0]["launch_exit"],
                lambda record: record.update(RUN_sha256="0" * 64))
            with self.assertRaisesRegex(ValueError, "Completion part count/actual exit0/lineage differs"):
                m.assemble_manifest(protocol, parts, fixture.hf, repo_root=fixture.root)

    def test_unbound_result_changes_and_missing_old_folded_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = SyntheticLineage(Path(tmp))
            assets = fixture.plan["partial_completed_assets"]
            manifest = json.loads((fixture.hf / assets["manifest"]["path"]).read_text())
            result = fixture.hf / manifest["cells"][0]["result"]["path"]
            row = json.loads(result.read_text())
            row["status"] = "FAILED_NO_RETRY"
            result.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, "Bound asset differs"):
                m.validate_partial_reuse(fixture.plan, fixture.hf, repo_root=fixture.root)
        with tempfile.TemporaryDirectory() as tmp:
            fixture = SyntheticLineage(Path(tmp))
            assets = fixture.plan["partial_completed_assets"]
            manifest = json.loads((fixture.hf / assets["manifest"]["path"]).read_text())
            (fixture.hf / manifest["cells"][0]["export"]["path"]).unlink()
            with self.assertRaisesRegex(ValueError, "Audited folded export absent or size/path identity differs"):
                m.validate_partial_reuse(fixture.plan, fixture.hf, repo_root=fixture.root)

    def test_mixed_lineage_assembler_accepts_exact40old120new_and_keeps_failure_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = SyntheticLineage(Path(tmp))
            protocol, parts = fixture.completion_parts()
            manifest = m.assemble_manifest(protocol, parts, fixture.hf, repo_root=fixture.root)
            self.assertEqual(len(manifest["cells"]), 160)
            self.assertEqual(sum(c["producer_id"] == m.ORIGINAL_ID for c in manifest["cells"]), 40)
            self.assertEqual(sum(c["producer_id"] == m.ID for c in manifest["cells"]), 120)
            self.assertEqual(manifest["old_scope_status"], "FAILED_NO_RETRY_PRESERVED")
            self.assertEqual(len({m._cell_key(c) for c in manifest["cells"]}), 160)

    def test_mixed_lineage_rejects_exit_without_process_end_or_incomplete_new_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = SyntheticLineage(Path(tmp))
            protocol, parts = fixture.completion_parts()
            parts[0]["launch_exit"] = fixture.change(fixture.hf, parts[0]["launch_exit"],
                lambda record: record.update(process_exited=False))
            with self.assertRaisesRegex(ValueError, "Completion part count/actual exit0/lineage differs"):
                m.assemble_manifest(protocol, parts, fixture.hf, repo_root=fixture.root)
        with tempfile.TemporaryDirectory() as tmp:
            fixture = SyntheticLineage(Path(tmp))
            protocol, parts = fixture.completion_parts()
            fixture.change_completion_run(parts[0],
                lambda record: record["observed"].update(Adam_steps=0))
            exit_record = json.loads(m.bound(fixture.hf, parts[0]["launch_exit"]).read_text())
            self.assertEqual(exit_record["RUN_sha256"], parts[0]["RUN"]["sha256"])
            with self.assertRaisesRegex(ValueError, "Completion part count/actual exit0/lineage differs"):
                m.assemble_manifest(protocol, parts, fixture.hf, repo_root=fixture.root)

    def test_mixed_lineage_rejects_retraining_old_or_cross_part_function(self):
        for expert, block in (m.COMPLETED_FUNCTIONS[0], m.functions_for(1)[0]):
            with self.subTest(expert=expert, block=block), tempfile.TemporaryDirectory() as tmp:
                fixture = SyntheticLineage(Path(tmp))
                protocol, parts = fixture.completion_parts()
                def put_out_of_part_cell(record):
                    record["cells"][0].update(expert=expert, block=block,
                        run_id=m.function_id(expert, block) + "_energy94")
                fixture.change_completion_run(parts[0], put_out_of_part_cell)
                with self.assertRaisesRegex(ValueError, "New result attempts a reused or out-of-part function"):
                    m.assemble_manifest(protocol, parts, fixture.hf, repo_root=fixture.root)

    def test_mixed_lineage_rejects_rebound_duplicate_half_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = SyntheticLineage(Path(tmp))
            protocol, parts = fixture.completion_parts()
            fixture.change_completion_run(parts[0],
                lambda record: record["cells"].__setitem__(1, copy.deepcopy(record["cells"][0])))
            with self.assertRaisesRegex(ValueError, "Mixed lineage must contain exactly eighty paired functions"):
                m.assemble_manifest(protocol, parts, fixture.hf, repo_root=fixture.root)

    def test_mixed_lineage_rejects_rebound_wrong_function_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = SyntheticLineage(Path(tmp))
            protocol, parts = fixture.completion_parts()
            fixture.change_completion_run(parts[0],
                lambda record: record["functions"][0].update(expert=m.EXPERTS[1], block=39))
            with self.assertRaisesRegex(ValueError, "Completion's exact fifteen-function partition differs"):
                m.assemble_manifest(protocol, parts, fixture.hf, repo_root=fixture.root)

    def test_mixed_lineage_rejects_rebound_failed_or_foreign_scope_run(self):
        for change in ({"status": "FAILED_NO_RETRY"}, {"id": m.ORIGINAL_ID}, {"part": 1}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                fixture = SyntheticLineage(Path(tmp))
                protocol, parts = fixture.completion_parts()
                fixture.change_completion_run(parts[0], lambda record: record.update(change))
                exit_record = json.loads(m.bound(fixture.hf, parts[0]["launch_exit"]).read_text())
                self.assertEqual(exit_record["RUN_sha256"], parts[0]["RUN"]["sha256"])
                with self.assertRaisesRegex(ValueError, "Completion part count/actual exit0/lineage differs"):
                    m.assemble_manifest(protocol, parts, fixture.hf, repo_root=fixture.root)

    def test_runtime_module_uses_same_native_classes_without_global_patch(self):
        import compact6016_completion_runtime as deployment
        import compact6016_native_runtime as native
        self.assertIs(deployment.Compact6016Runtime, native.Compact6016Runtime)
        self.assertIs(deployment.Compact6016Residency, native.Compact6016Residency)
        self.assertIs(deployment.validate_deployment_pack, native.validate_deployment_pack)
        self.assertIsNot(deployment.load_bank, native.load_bank)
        self.assertEqual(native.PRODUCER_ID, m.ORIGINAL_ID)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
