import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import e0_b_final_block_capture as original
import e0_b_final_block_capture_head_asset as revised


class HeadAssetLocal(unittest.TestCase):
    def test_actual_head_modulation_is_bound_at_native_capture(self):
        def record(value):
            return {"raw_tensor_sha256": value}
        source = {"head_weight_runtime_KN": record("w"), "head_bias": record("b"),
                  "head_modulation": record("m")}
        captured = {"actual_head_operator": {"weight": record("w"), "bias": record("b")},
                    "capture_file": {"tensors": {"head_modulation": record("m")}}}
        revised.verify_actual_head_asset(captured, source)
        captured["capture_file"]["tensors"]["head_modulation"] = record("changed")
        with self.assertRaises(ValueError):
            revised.verify_actual_head_asset(captured, source)

    def test_original_model_calls_and_resources_unchanged(self):
        self.assertEqual(revised.RESOURCES, original.RESOURCES)
        self.assertEqual({k: revised.LIMITS[k] for k in original.LIMITS}, original.LIMITS)
        self.assertEqual(revised.SOURCE_COST, original.SOURCE_COST)
        self.assertEqual(revised.FLOW_COST, original.FLOW_COST)
        self.assertEqual(revised.run_ids(), original.run_ids())
        self.assertEqual(revised.LIMITS["CPU_head_weight_transpose_elements"], 2 * 64 * 5120)


if __name__ == "__main__":
    unittest.main()
