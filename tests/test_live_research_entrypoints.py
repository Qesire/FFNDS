"""Static checks for the current reader, machine and original-blob handoff interfaces."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import check_autoresearcher_binding as binding


class CurrentHandoffTests(unittest.TestCase):
    def test_current_method_has_a_reachable_reader_and_pinned_evidence(self):
        handoff = binding.check_current_handoff()
        self.assertEqual(handoff["context_path"], "research/agent_protocol/CURRENT_CONTEXT_PACKET.md")
        self.assertEqual(handoff["historical_fast512_evidence"]["commit"], "2c4aa753761194994ee25ea8caba76a29445912d")
        self.assertEqual(handoff["historical_fast512_evidence"]["method_path"],
                         "reports/20261121/anchored_video_experiments/outputs/FFNDS_anchor_next/当前方法.md")

    def test_registered_operator_and_scope_of_end_to_end_evidence(self):
        handoff = binding.record(binding.HANDOFF)
        baseline = handoff["baseline"]
        self.assertEqual((baseline["method"], baseline["probes"], baseline["probe_layout"]),
                         ("wide8192", 64, "fixed"))
        self.assertEqual(baseline["execution_tile_rows"], 8192)
        self.assertTrue(baseline["per_position_anchors"])
        self.assertEqual(baseline["expert_schedule"], ["high", "high", "low", "low"])
        self.assertEqual(baseline["native_anchor_steps"], [0, 2])
        self.assertEqual(baseline["approximation_steps"], [1, 3])
        current = handoff["verified_video_comparison"]
        self.assertEqual((current["method"], current["sources"], current["formal_pairs"]), ("wide8192", 3, 6))
        self.assertAlmostEqual(current["relative_wall_reduction_pct"], 5.04, places=2)
        verified = handoff["historical_fast512_video_comparison"]
        self.assertEqual((verified["sources"], verified["seed"], verified["formal_runs_per_arm_per_source"]),
                         (24, 42, 1))
        self.assertEqual(verified["bitwise_equivalence_reference"], "previous anchored implementation")
        self.assertEqual(verified["bitwise_equivalent_sources"], 24)
        self.assertEqual(verified["faster_sources"], 21)
        self.assertEqual(handoff["offline_calibration_source_ids"], ["0032", "0304", "0456"])
        pending = {item["candidate"]: item["status"].lower() for item in handoff["open_work"]}
        self.assertIn("development", pending["fixed128 or rotated layout"])
        self.assertEqual(handoff["known_case"]["source"], "0951")
        self.assertIn("user", handoff["quality_decision"].lower())

    def test_archive_rejects_an_unmatched_digest_and_a_missing_registration(self):
        manifest = binding.record(binding.HANDOFF_ARCHIVE)
        changed = copy.deepcopy(manifest)
        changed["entries"][0]["source_blob_sha1"] = "0" * 40
        with self.assertRaisesRegex(SystemExit, "registered blob"):
            binding.check_handoff_archive(changed)
        with self.assertRaisesRegex(SystemExit, "file list"):
            binding.check_handoff_archive({**manifest, "entries": manifest["entries"][1:]})

    def test_stale_evidence_revision_cannot_resolve_using_current_links(self):
        original_record = binding.record

        def moved_evidence(path):
            data = original_record(path)
            if path == binding.HANDOFF:
                data["historical_fast512_evidence"]["commit"] = "0" * 40
            return data

        with patch.object(binding, "record", moved_evidence):
            with self.assertRaisesRegex(SystemExit, "commit-pinned evidence"):
                binding.check_current_handoff()


if __name__ == "__main__":
    unittest.main()