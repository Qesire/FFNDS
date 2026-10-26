"""CPU metadata fixtures for the fresh-source prerequisite controls.

The byte strings named *.mp4 are synthetic hash fixtures, never decoded media.
The strict video gate is mocked here; its own inventory tests cover that gate.
These tests establish fail-closed bookkeeping, not native or video equivalence.
"""
import builtins
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_e0_fresh_official_directional as fresh


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2))


class FreshNativeControls(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "project"
        self.hf = Path(directory.name) / "hf"
        self.core = {
            "tools/" + name: "synthetic-controlled-" + name
            for name in [
                "e0_paired_wan_controller.py",
                "e0_paired_ffn_observer.py",
                "e0_paired_weight_residency.py",
                "e0_full_depth_trace_contract.py",
                "e0_whole_flow_counterfactual.py",
            ]
        }
        self.assets = {
            "checkpoint_sha256": {"high": "synthetic-high", "low": "synthetic-low"},
            "cache_manifest_sha256": "synthetic-cache",
            "contract": "synthetic-frozen-contract",
        }
        self.bank = "synthetic-unchanged-bank"
        self.pairs = [
            {
                "id": f"dc_source{index:02d}",
                "source_index": index,
                "source_id": f"synthetic-image-{index}",
                "seed": 42 + index,
                "modes": ["dense", "candidate"],
                "save_vectors": False,
                "prompt_digest": f"synthetic-prompt-{index}",
                "role": "EXPOSED_AFFINE54_CALIBRATION",
            }
            for index in [4, 8, 16, 20]
        ]
        self.legacy = [
            {"step": step, "sha256": f"synthetic-latent-{step}",
             "shape": [16, 21, 58, 104]}
            for step in range(4)
        ]
        self.old_videos = {}
        original_bindings = {}
        for pair in self.pairs:
            index = pair["source_index"]
            directory = self.hf / "ffnds-derived" / "original" / str(index)
            directory.mkdir(parents=True)
            video = directory / "plain.mp4"
            video.write_bytes(f"SYNTHETIC_OLD_DENSE_{index}".encode())
            self.old_videos[index] = video
            phase = {"trajectory": copy.deepcopy(self.legacy), "video_sha256": digest(video)}
            receipt = directory / "RUN.json"
            write(receipt, {
                "status": "PASS_PASSIVE_CAPTURE_NOOP",
                "source_index": index,
                "source_id": pair["source_id"],
                "seed": pair["seed"],
                "phases": {"plain": phase},
            })
            original_bindings[str(index)] = {
                "HF_relative": str(directory.relative_to(self.hf)),
                "run_sha256": digest(receipt),
                **copy.deepcopy(phase),
            }
        self.prior = self.root / "config" / "phase1.json"
        write(self.prior, {
            "driver_sources_sha256": self.core,
            "pairs": self.pairs,
            "original_dense_sources": original_bindings,
        })
        self.plan = {
            "phase1_protocol": "config/phase1.json",
            "phase1_protocol_sha256": digest(self.prior),
            "driver_sources_sha256": copy.deepcopy(self.core),
            "bank_RUN_sha256": self.bank,
            "reference_model_assets_digest": fresh.object_sha(self.assets),
            "completed_phase1_controls": [],
        }
        self.receipts = {}
        self.raw = {}
        for pair in self.pairs:
            index = pair["source_index"]
            directory = self.hf / "ffnds-derived" / "phase1" / f"source{index:02d}"
            receipt = directory / "RUN.json"
            self.raw[index] = {
                "status": fresh.PASS,
                "pair": copy.deepcopy(pair),
                "seed": pair["seed"],
                "source_id": pair["source_id"],
                "protocol_sha256": digest(self.prior),
                "bank_RUN_sha256": self.bank,
                "assets": copy.deepcopy(self.assets),
                "trajectory": {"teacher": [
                    dict(row, native_dtype="torch.bfloat16", native_identity="synthetic-native")
                    for row in self.legacy
                ]},
                "videos": {"teacher": {
                    "path": "teacher.mp4", "sha256": digest(self.old_videos[index])}},
            }
            self.receipts[index] = receipt
            write(receipt, self.raw[index])
            # The mocked strict gate still receives the exact expected path.
            write(directory / "STRICT_BUNDLE.json", {"synthetic": True})
            self.plan["completed_phase1_controls"].append({
                "source_index": index,
                "RUN_HF_relative": str(receipt.relative_to(self.hf / "ffnds-derived")),
                "RUN_sha256": digest(receipt),
            })

    def rewrite_receipt(self, index=4, *, rebind=True):
        write(self.receipts[index], self.raw[index])
        if rebind:
            item = next(x for x in self.plan["completed_phase1_controls"]
                        if x["source_index"] == index)
            item["RUN_sha256"] = digest(self.receipts[index])

    def verify(self, strict_errors=None):
        original_import = builtins.__import__

        def metadata_import(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "numpy", "safetensors", "HiFloat4", "lightx2v", "cv2"}:
                raise AssertionError("Control prerequisites must remain metadata-only: " + name)
            return original_import(name, *args, **kwargs)

        with patch.object(fresh, "ROOT", self.root), \
                patch("ffnds_paired_video_strict_gate.inspect", return_value=strict_errors or []) as gate, \
                patch("builtins.__import__", side_effect=metadata_import):
            result = fresh.verify_phase1_controls(self.plan, self.hf)
        return result, gate

    def test_four_bound_controls_pass_without_model_import(self):
        _, gate = self.verify()
        self.assertEqual(gate.call_count, 4)
        self.assertEqual(
            [call.args[0] for call in gate.call_args_list],
            [self.receipts[index].parent / "STRICT_BUNDLE.json" for index in [4, 8, 16, 20]],
        )

    def test_missing_or_duplicated_source_control_is_rejected(self):
        baseline = copy.deepcopy(self.plan["completed_phase1_controls"])
        for bad in [baseline[:-1], baseline[:-1] + [copy.deepcopy(baseline[0])]]:
            with self.subTest(indices=[x["source_index"] for x in bad]):
                self.plan["completed_phase1_controls"] = bad
                with self.assertRaisesRegex(ValueError, "Four completed"):
                    self.verify()

    def test_failed_control_cannot_pass_by_rebinding_its_receipt_sha(self):
        self.raw[4]["status"] = "FAILED_PRESERVED_NO_RETRY"
        self.rewrite_receipt()
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self.verify()

    def test_receipt_bytes_drift_is_rejected(self):
        self.raw[4]["unregistered_edit"] = True
        self.rewrite_receipt(rebind=False)
        with self.assertRaisesRegex(ValueError, "receipt binding failed"):
            self.verify()

    def test_prior_protocol_file_drift_is_rejected(self):
        prior = json.loads(self.prior.read_text())
        prior["changed"] = True
        write(self.prior, prior)
        with self.assertRaisesRegex(ValueError, "phase1 protocol changed"):
            self.verify()

    def test_receipt_wrong_protocol_is_rejected_after_sha_rebinding(self):
        self.raw[4]["protocol_sha256"] = "different-phase1-protocol"
        self.rewrite_receipt()
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self.verify()

    def test_wrong_bank_or_assets_are_rejected_after_sha_rebinding(self):
        original = copy.deepcopy(self.raw[4])
        for key, value in [("bank_RUN_sha256", "different-bank"),
                           ("assets", dict(self.assets, cache_manifest_sha256="different-cache"))]:
            with self.subTest(field=key):
                self.raw[4] = copy.deepcopy(original)
                self.raw[4][key] = value
                self.rewrite_receipt()
                with self.assertRaisesRegex(ValueError, "identity differs"):
                    self.verify()

    def test_changed_controlled_math_is_rejected(self):
        self.plan["driver_sources_sha256"]["tools/e0_whole_flow_counterfactual.py"] = "changed-math"
        with self.assertRaisesRegex(ValueError, "mathematics changed"):
            self.verify()

    def test_wrong_actual_source_seed_or_pair_is_rejected(self):
        original = copy.deepcopy(self.raw[4])
        mutations = [("source_id", "wrong-source"), ("seed", 999),
                     ("pair", dict(original["pair"], prompt_digest="different-prompt"))]
        for key, value in mutations:
            with self.subTest(field=key):
                self.raw[4] = copy.deepcopy(original)
                self.raw[4][key] = value
                self.rewrite_receipt()
                with self.assertRaisesRegex(ValueError, "identity differs"):
                    self.verify()

    def test_missing_or_changed_dense_latent_is_rejected(self):
        original = copy.deepcopy(self.raw[4]["trajectory"]["teacher"])
        changed = copy.deepcopy(original)
        changed[2]["sha256"] = "different-latent"
        for trajectory in [original[:-1], changed]:
            with self.subTest(steps=len(trajectory)):
                self.raw[4]["trajectory"]["teacher"] = trajectory
                self.rewrite_receipt()
                with self.assertRaisesRegex(ValueError, "dense trajectory/video"):
                    self.verify()

    def test_changed_teacher_video_or_failed_strict_gate_is_rejected(self):
        self.raw[4]["videos"]["teacher"]["sha256"] = "different-video"
        self.rewrite_receipt()
        with self.assertRaisesRegex(ValueError, "dense trajectory/video"):
            self.verify()
        self.raw[4]["videos"]["teacher"]["sha256"] = digest(self.old_videos[4])
        self.rewrite_receipt()
        with self.assertRaisesRegex(ValueError, "strict bundle failed"):
            self.verify(strict_errors=["synthetic strict inventory failure"])

    def test_original_dense_video_bytes_are_verified_not_just_receipt_claims(self):
        self.old_videos[4].write_bytes(b"CHANGED_SYNTHETIC_ORIGINAL")
        with self.assertRaisesRegex(ValueError, "Original dense"):
            self.verify()

    def test_control_receipt_cannot_escape_the_derived_root(self):
        self.plan["completed_phase1_controls"][0]["RUN_HF_relative"] = "../../outside/RUN.json"
        with self.assertRaisesRegex(ValueError, "receipt binding failed"):
            self.verify()

    def test_invalid_prerequisite_blocks_cli_before_execution(self):
        self.plan["completed_phase1_controls"].pop()

        def preflight_fixture(*_args):
            self.verify()
            raise AssertionError("Invalid prerequisite unexpectedly passed")

        args = ["run_e0_fresh_official_directional.py", "--protocol", "synthetic.json",
                "--bank", "synthetic-bank", "--hf-root", str(self.hf),
                "--study-root", "synthetic-study", "--pair", "dc_official908_seed42",
                "--execute-native"]
        with patch.object(sys, "argv", args), \
                patch.object(fresh, "preflight", side_effect=preflight_fixture), \
                patch.object(fresh, "execute") as execute:
            with self.assertRaisesRegex(ValueError, "Four completed"):
                fresh.main()
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
