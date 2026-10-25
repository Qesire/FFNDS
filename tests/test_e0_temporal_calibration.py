import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from analyze_e0_temporal_calibration import evaluate, feature, fit_affine, score


def example(index, scale=2.0):
    old = np.arange(1, 13, dtype=np.float64).reshape(4, 3)
    return {"source_index": index, "block": 39, "target_step": 1, "expert": "high",
            "previous_payload_sha256": "previous", "current_payload_sha256": "current",
            "previous": old, "current": scale * old, "input_previous": old / 2,
            "input_current": old, "gate": np.array([[.2, 2.0, 0.0]]), "bias": np.ones((1, 3))}


class TemporalMechanismAccounting(unittest.TestCase):
    def test_held_out_output_cannot_change_its_predictor(self):
        cases = [example(i) for i in range(6)]
        cases[-1]["current"] *= 100
        first = evaluate(cases, ["reuse", "fixed_scale"])[-1]
        cases[-1]["current"] *= 10
        second = evaluate(cases, ["reuse", "fixed_scale"])[-1]
        self.assertAlmostEqual(first["predictors"]["fixed_scale"]["calibration"]["scale"], 2.0)
        self.assertEqual(first["predictors"]["fixed_scale"]["calibration"],
                         second["predictors"]["fixed_scale"]["calibration"])
        self.assertEqual(first["calibration_source_indices"], list(range(5)))
        self.assertNotEqual(first["oracle_scalar_coefficient"], second["oracle_scalar_coefficient"])

    def test_weighted_affine_recovers_known_scale_and_shift(self):
        cases = [example(i) for i in range(3)]
        shift = np.array([[4.0, -3.0, 2.0]])
        for i, case in enumerate(cases):
            case["previous"] = case["previous"] + i
            case["gate"] = np.array([[i + 1.0, .2, .5]])
            case["current"] = .7 * case["previous"] + shift
        scale, offset = fit_affine(cases)
        self.assertAlmostEqual(scale, .7)
        np.testing.assert_allclose(offset, shift)
        for case in cases:
            self.assertLess(score(case, scale * case["previous"] + offset)["relative_SSE"], 1e-28)

    def test_input_features_do_not_read_target_and_error_splits_orthogonally(self):
        case = example(0)
        first = feature(case, "input_row_RMS_bias_preserving_calibrated")
        case["current"] = 7 * case["previous"] + np.array([[3.0, -4.0, 1.0]])
        second = feature(case, "input_row_RMS_bias_preserving_calibrated")
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        result = score(case, case["previous"])
        self.assertAlmostEqual(result["relative_SSE"], result["mean_error_over_target_energy"] +
                               result["centered_error_over_target_energy"])


if __name__ == "__main__":
    unittest.main()
