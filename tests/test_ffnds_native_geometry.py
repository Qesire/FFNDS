"""Historical native case receipts to 160-site geometry."""
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from ffnds_native_geometry import extract
from ffnds_experiment_plans import save_new


class NativeGeometryTests(unittest.TestCase):
    def test_complete_native_capture(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            bank = {"schema": "ffnds.source.bank.v1", "sources": [
                {"source_id": "dataset:animal:0", "source_group": "animal:0",
                 "image_id": "image0", "prompt_id": "prompt0", "role": "CALIBRATION"}]}
            save_new(p / "bank.json", bank)
            case = {"status": "PASS_NATIVE_OPERANDS_AND_SAME_PIPELINE_NOOP",
                    "input": {"source_group": "animal:0"},
                    "scopes": [{"step": st, "block": block, "grid": [21, 29, 52],
                                "rows": 2, "selected_rows": [0, 1]}
                               for st in range(4) for block in range(40)]}
            save_new(p / "case.json", case)
            save_new(p / "collection.json", {
                "status": "PASS_CALIBRATION24_SHARD_FULL160_OPERANDS_AND_NOOP",
                "completed": [{"input_report": str(p / "case.json")}]})
            rows = extract([p / "collection.json"], p / "bank.json")
            self.assertEqual(len(rows), 160)
            self.assertEqual(rows[0]["population_rows"], 21 * 29 * 52)
            case["scopes"].pop()
            save_new(p / "bad.json", case)
            save_new(p / "broken.json", {
                "status": "PASS_CALIBRATION24_SHARD_FULL160_OPERANDS_AND_NOOP",
                "completed": [{"input_report": str(p / "bad.json")}]})
            with self.assertRaisesRegex(ValueError, "all160"):
                extract([p / "broken.json"], p / "bank.json")


if __name__ == "__main__":
    unittest.main()
