from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from preflight_ar2 import audit_sampling


class SamplingProvenance(unittest.TestCase):
    def test_lossless_index_conversion_preserves_recovered_source_marker(self):
        scope = {"selected_rows": [0, 2, 7], "design_weights": [1, 2, 3], "coordinates": None,
                 "grid": [2, 2, 2], "coordinate_provenance": "RECONSTRUCTED_DETERMINISTIC_NOT_RUNTIME_OBSERVED"}
        result = audit_sampling(scope)
        self.assertEqual(result["coordinate_provenance"], scope["coordinate_provenance"])
        self.assertIn("NOT_A_NEW_RUNTIME_OBSERVATION", result["coordinate_index_view"])

    def test_inconsistent_saved_coordinate_is_never_silently_reconstructed(self):
        scope = {"selected_rows": [0, 7], "design_weights": [1, 1], "coordinates": [[0, 0, 0], [1, 1, 0]],
                 "grid": [2, 2, 2], "coordinate_provenance": "OBSERVED_DQ_CAPTURE"}
        with self.assertRaises(ValueError):
            audit_sampling(scope)


if __name__ == "__main__":
    unittest.main()
