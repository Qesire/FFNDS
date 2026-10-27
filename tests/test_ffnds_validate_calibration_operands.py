"""Native operand manifest must match every true calibration row coordinate."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from ffnds_validate_calibration_operands import check


class OperandValidationTests(unittest.TestCase):
    def test_source_row_draw_matches_native_operand_index(self):
        with tempfile.TemporaryDirectory() as dir:
            p = Path(dir) / "x.safetensors"
            p.write_bytes(b"header checked only in actual native tensor environment")
            actual, index = [], []
            for step in range(4):
                for block in range(40):
                    r = {"source_id": "s0", "step": step, "block": block,
                         "source_state": "teacher_input", "expert": "high_noise_model" if step < 2 else "low_noise_model",
                         "selected_row_ids": [2, 4, 7], "n_rows": 3, "population_rows": 100}
                    actual.append(r)
                    index.append({**r, "operand_path": str(p),
                                  "tensor_boundary": "original_native_Q1_V1_Q2_V2_W2_Y_b2_gate"})
            plan = {"schema": "ffnds.cohort.v1",
                    "phase": "REGISTERED_CALIBRATION_ROW_PLAN",
                    "mode": "CUSTOM_PAIRED", "samples": [],
                    "sources": [{"source_id": "s0", "image_id": "i0", "prompt_id": "p0",
                                 "role": "CALIBRATION"}], "calibration": actual,
                    "training_rows_per_position": 3}
            self.assertEqual(check(plan, index)["native_scope_count"], 160)
            index[3]["selected_row_ids"] = [1, 4, 7]
            with self.assertRaisesRegex(ValueError, "sampled rows"):
                check(plan, index)


if __name__ == "__main__":
    unittest.main()
