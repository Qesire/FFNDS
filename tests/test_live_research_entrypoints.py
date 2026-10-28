"""Static guard for FFNDS live navigation and model/format state.

No imports of model, torch, subprocess or network; checks only active entrypoints.
Original frozen run protocols and historical reports are intentionally excluded.
"""
from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def record(path: str) -> dict:
    return json.loads(text(path))


class LiveEntrypointConsistencyTests(unittest.TestCase):
    def test_paper_and_graph_totals_follow_authoritative_indices(self):
        catalogue = record("research/literature/20261026/PRIMARY_LITERATURE_INDEX.json")
        graph = record("research/agent_protocol/GRAPH_AGENT_BINDINGS.json")
        overlay = record("research/agent_protocol/PAPER_METHOD_FOCUS_GRAPH.json")
        primary = len(catalogue["records"])
        supplements = len(overlay["supplemental_papers"])
        nodes, edges = len(graph["nodes"]), len(graph["edges"])
        self.assertEqual(len(overlay["paper_links"]), primary)
        self.assertEqual(graph["paper_method_focus_v2"]["primary_papers"], primary)
        self.assertEqual(graph["paper_method_focus_v2"]["supplemental_papers"], supplements)
        current = {
            "README.md": [f"{primary}项主文献＋{supplements}项补充",
                          f"{nodes}个稳定节点/{edges}条关系"],
            "research/RESEARCH_NAVIGATION.md":
                [f"{primary}主＋{supplements}补充", f"{nodes}节点/{edges}原始关系"],
            "research/agent_protocol/CURRENT_CONTEXT_PACKET.md":
                [f"{primary}主＋{supplements}补充", f"{nodes}节点/{edges}原始图边"],
            ".github/copilot-instructions.md":
                [f"{primary}主＋{supplements}补充", f"{nodes}节点/{edges}关系"],
            ".agents/skills/ffnds-paper-discovery/SKILL.md":
                [f"{primary}个主LID及{supplements}个补充"],
            "research/agent_protocol/README.md":
                [f"{primary}＋{supplements}篇G14机器焦点图"],
            "research/agent_protocol/NEXT_AGENT_B_SENSITIVE_OUTPUT_FIRST_20261027.md":
                [f"{primary}项主论文＋{supplements}项校准"],
            "docs/98_端到端计算图与证据注解_20261025.md":
                [f"{nodes}个节点、{edges}条有类型边、{primary}项主文献＋{supplements}项补充"],
        }
        for path, snippets in current.items():
            body = text(path)
            for snippet in snippets:
                with self.subTest(path=path, expected=snippet):
                    self.assertIn(snippet, body)

    def test_latest_completed_native_milestones_are_not_requeued(self):
        ledger = record("research/agent_protocol/CURRENT_TASK_LEDGER_20261028.json")
        tasks = {task["id"]: task for task in ledger["tasks"]}
        self.assertEqual(tasks["P0-B0-NATIVE"]["status"], "COMPLETE_NATIVE")
        for p in ("research/20261028/B_CAL24_SOURCE_PAIRED_DISCRIMINATION_RESULTS_20261028.md",
                  "research/20261028/BH_DIRECT65_H128_PREDICTABILITY_RESULTS_20261028.md",
                  "research/20261028/COMPACT6016_FULL80_RECOVERY_RESULTS_20261028.md"):
            self.assertTrue((ROOT / p).is_file())
        entries = [
            "README.md", "research/RESEARCH_NAVIGATION.md",
            "research/agent_protocol/NEXT_AGENT_B_SENSITIVE_OUTPUT_FIRST_20261027.md",
            ".agents/skills/ffnds-paper-discovery/SKILL.md",
            ".github/copilot-instructions.md",
        ]
        for p in entries:
            body = text(p)
            with self.subTest(entry=p):
                self.assertNotIn("B-H_RUNNING", body)
                self.assertNotIn("下一Agent从B0真实", body)
                self.assertNotIn("先B0原生u/head", body)
                self.assertIn("BH_DIRECT65_H128_PREDICTABILITY_RESULTS_20261028.md", body)

    def test_signed_hif4_single_contract_is_referenced_without_free_packed_codes(self):
        spec = record("research/agent_protocol/HIF4_SIGNED_FORMAT_CONTRACT_20261028.json")
        observation = record("research/agent_protocol/QMETA_OBSERVATION_CONTRACT_20261028.json")
        self.assertEqual(spec["paper"]["element"], "S1P2")
        self.assertEqual(spec["paper"]["sign_bits_per_element"], 1)
        self.assertEqual(spec["unit"]["total_bits"], 288)
        self.assertFalse(spec["native_wan_path"]["persistent_packed_code"])
        self.assertFalse(observation["hif4_signed_format_contract"]["native_qdq_exports_packed_code"])
        for p in ("README.md", "AGENTS.md",
                  "research/RESEARCH_NAVIGATION.md",
                  ".agents/skills/ffnds-paper-discovery/SKILL.md",
                  "research/agent_protocol/CURRENT_CONTEXT_PACKET.md"):
            with self.subTest(entry=p):
                self.assertIn("HIF4_SIGNED_STORAGE_AND_QDQ_SEMANTICS_AUDIT_20261028.md", text(p))

    def test_dynamic_global_gpu_policy_is_not_replaced_by_old_two_or_four(self):
        active = record("research/ACTIVE_RESEARCH.json")
        self.assertEqual(active["resource_scheduling_policy_20261028"]["mode"],
                         "DYNAMIC_MAX_ACTUALLY_CALLABLE")
        self.assertIsNone(active["resource_scheduling_policy_20261028"]["global_gpu_card_cap"])
        for p in ("README.md", "AGENTS.md",
                  "research/RESEARCH_NAVIGATION.md",
                  ".agents/skills/ffnds-paper-discovery/SKILL.md"):
            self.assertIn("DYNAMIC_GPU_RESOURCE_POLICY_20261028.md", text(p))

    def test_bf16_and_pro6000_native_nvfp4_are_independent_parallel_lanes(self):
        ledger = record("research/agent_protocol/CURRENT_TASK_LEDGER_20261028.json")
        auth = ledger["authority_and_scope"]
        devices = auth["user_local_deployment_hardware"]
        self.assertIn("NO_LOCAL_RTX_5090", devices["hardware"])
        self.assertIn("RTX_PRO_6000", devices["hardware"])
        self.assertEqual(devices["native_NVFP4_W4A4_status"],
                         "PARALLEL_R0_READINESS_ACTIVE_NATIVE_EXECUTION_GATED")
        self.assertEqual(auth["parallel_execution_policy"]["actual_pro6000_entitlement"],
                         "ACCESS_NOT_VERIFIED_CURRENT_TURN")
        self.assertIn("A100_BF16_ORIGINAL_SCIENCE", auth["parallel_execution_policy"]["lanes"])

        tasks = {task["id"]: task for task in ledger["tasks"]}
        expected = {
            "P0-SM120-R0-ENV": ("READY_STATIC", 0),
            "P1-SM120-R1-FP4-NATIVE": ("OPEN_NEEDS_REGISTERED_NATIVE", 1),
            "P1-SM120-R2-WAN-NVFP4-DENSE": ("WAITING_ON_EVIDENCE", 1),
            "P2-SM120-R3-WAN-NVFP4-CANDIDATE": ("WAITING_ON_EVIDENCE", 2),
            "P3-SM120-R4-WAN-NVFP4-VIDEO": ("WAITING_ON_EVIDENCE", 3),
        }
        for name, (status, priority) in expected.items():
            with self.subTest(task=name):
                self.assertEqual(tasks[name]["status"], status)
                self.assertEqual(tasks[name]["priority"], priority)
                self.assertIn("RTX PRO 6000", tasks[name]["track"])
        self.assertNotIn("P1-B2-STUDENT",
                         tasks["P2-SM120-R3-WAN-NVFP4-CANDIDATE"]["dependencies"])

        policy = text("research/agent_protocol/BF16_NATIVE_NVFP4_EXPERIMENT_MIGRATION_20261028.md")
        for word in ("BF16_ORIGINAL", "NVFP4_NATIVE_W4A4",
                     "HIF4_FAKE_QDQ_DIAGNOSTIC", "RTX PRO 6000 Blackwell",
                     "READY_STATIC", "ACCESS_NOT_VERIFIED"):
            self.assertIn(word, policy)
        self.assertNotIn("单张已有", policy)
        for path in ("README.md", "AGENTS.md",
                     ".agents/skills/ffnds-paper-discovery/SKILL.md",
                     "research/RESEARCH_NAVIGATION.md",
                     "research/agent_protocol/CURRENT_CONTEXT_PACKET.md",
                     "research/agent_protocol/README.md",
                     ".github/copilot-instructions.md"):
            doc = text(path)
            with self.subTest(entry=path):
                self.assertIn("RTX PRO 6000", doc)
                self.assertNotIn("用户已有5090", doc)
                self.assertIn("BF16_NATIVE_NVFP4_EXPERIMENT_MIGRATION_20261028.md", doc)
        for path in ("research/RESEARCH_NAVIGATION.md",
                     "research/agent_protocol/CURRENT_CONTEXT_PACKET.md"):
            self.assertNotIn("NVFP4执行暂缓，匹配NVFP4 checkpoint未知", text(path))

        readiness = record("research/20261028/BF16_NVFP4_R0_READINESS_20261028.json")
        self.assertIn("RTX PRO 6000", readiness["hardware"]["previous_20261026_primitive"])
        self.assertEqual(readiness["matching_NVFP4_checkpoint"]["status"], "UNKNOWN")
        native_qdq = text("upstream_audit/source_deltas/Wan2.2-I2V-A14B-W4A4/files/quant/nvfp4_rtn_backend.py")
        self.assertIn('"packed_kernel": False', native_qdq)
        self.assertIn('"compute_backend": "bf16_or_fp16_F.linear_after_QDQ"', native_qdq)


if __name__ == "__main__":
    unittest.main()
