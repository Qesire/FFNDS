import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import e0_b_final_block_capture_local_package as local
from local_python_package_mirror import inventory


class LocalPackageCapture(unittest.TestCase):
    def test_checked_import_rejects_source_change_and_external_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "transformers"
            package.mkdir()
            (package / "__init__.py").write_text("value = 1\n")
            rows = inventory(package)
            finder = local.CheckedPackageFinder(root, rows)
            saved = sys.path[:]
            try:
                sys.path.insert(0, str(root))
                spec = finder.find_spec("transformers")
                self.assertEqual(spec.origin, str(package / "__init__.py"))
                (package / "__init__.py").write_text("value = 2\n")
                with self.assertRaises(ValueError):
                    finder.find_spec("transformers")
                other = root / "other" / "transformers"
                other.mkdir(parents=True)
                (other / "__init__.py").write_text("value = 1\n")
                sys.path.insert(0, str(other.parent))
                with self.assertRaises(ValueError):
                    finder.find_spec("transformers")
                self.assertIsNone(finder.find_spec("unrelated"))
            finally:
                sys.path[:] = saved

    def test_native_call_and_resource_budgets_are_unchanged(self):
        self.assertEqual(local.native.LIMITS["complete_Dense_flows"], 4)
        self.assertEqual(local.native.LIMITS["native_FFN_calls"], 160)
        self.assertEqual(local.native.RESOURCES["compute_wall_seconds"], 1200)
        self.assertEqual(local.native.RESOURCES["GPU"], 1)


if __name__ == "__main__":
    unittest.main()
