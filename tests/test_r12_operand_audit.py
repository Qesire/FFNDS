"""Evidence boundaries and original-byte checks for the CPU-only R12 audit."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from audit_r12_saved_operands import historical_headroom, verified_safetensors


class SavedOperandAuditTests(unittest.TestCase):
    def test_restore_rank_reversals_do_not_equal_large_conditional_gain(self):
        rows = json.loads((ROOT / "research/20261021/R7D1_results.json").read_text())["rows"]
        result = historical_headroom(rows)
        self.assertEqual(result["best_fixed_restore_arm"], "B")
        self.assertAlmostEqual(result["restore_oracle_gain_over_best_fixed_fraction"], .0017102224117051534)
        self.assertEqual(result["matched_R12_actions_observed"], 0)
        self.assertGreater(result["global_main_MAC_fractions"]["A"], .5)
        self.assertGreater(result["leave_one_out_total_loss"], result["total_loss_by_restore_arm"]["B"])

    def test_original_tensor_bytes_and_geometry_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "tiny.safetensors"
            raw = struct.pack("<2f", 1., 2.)
            header = json.dumps({"x": {"shape": [2], "dtype": "F32", "data_offsets": [0, 8]}}).encode()
            p.write_bytes(struct.pack("<Q", len(header)) + header + raw)
            file_sha = hashlib.sha256(p.read_bytes()).hexdigest()
            expected = {"x": {"shape": [2], "dtype": "torch.float32", "sha256": hashlib.sha256(raw).hexdigest()}}
            with patch("audit_r12_saved_operands.locator", return_value="test/tiny.safetensors"):
                item, _, _ = verified_safetensors(p, file_sha, expected)
                self.assertEqual(item["tensors"]["x"]["dtype"], "F32")
                expected["x"]["shape"] = [1, 2]
                with self.assertRaisesRegex(ValueError, "original record"):
                    verified_safetensors(p, file_sha, expected)
                p.write_bytes(p.read_bytes()[:-1] + b"\x01")
                with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                    verified_safetensors(p, file_sha)


if __name__ == "__main__":
    unittest.main()
