"""Static HiF4 signed-storage and current fake-QDQ semantics regression.

This file uses only the stdlib. It never imports CUDA/PyTorch, starts GPU/model,
assumes a packed HiF4 artifact exists, or changes pinned source files.
"""
from __future__ import annotations

import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "research/agent_protocol/HIF4_SIGNED_FORMAT_CONTRACT_20261028.json"


def read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


class SignedHiF4FormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads(CONTRACT.read_text(encoding="utf-8"))

    def test_sign_is_in_each_four_bit_element_not_magnitude_metadata(self):
        fmt = self.spec["paper"]
        unit = self.spec["unit"]
        self.assertEqual(fmt["element"], "S1P2")
        self.assertEqual((fmt["element_bits"], fmt["sign_bits_per_element"],
                          fmt["integer_bits"], fmt["fractional_bits"]), (4, 1, 1, 2))
        self.assertEqual(fmt["representable_levels"], [i / 4 for i in range(-7, 8)])
        self.assertEqual(unit["elements"], 64)
        self.assertEqual(unit["element_code_bits_total"], 256)
        self.assertEqual(unit["scale_metadata_bits_total"], 8 + 8 + 16)
        self.assertEqual(unit["total_bits"], 288)
        self.assertEqual(unit["effective_bits_per_value"], 4.5)
        for raw, value in (("0001", 0.25), ("1001", -0.25),
                           ("0111", 1.75), ("1111", -1.75)):
            code = int(raw, 2)
            sign = -1 if code & 0b1000 else 1
            magnitude = code & 0b0111
            self.assertEqual(sign * magnitude / 4, value)
        self.assertTrue(fmt["positive_and_negative_zero_codewords_in_paper"])

    def test_pinned_cuda_preserves_sign_and_generates_unsigned_magnitude_scales(self):
        cu = read(self.spec["source_proofs"][0]["path"])
        for needle in ("x_int & 0x80000000", "x_int & 0x7FFFFFFF",
                       "float x_sign", "float out = x_sign * e * mant3 * lv0;",
                       "lv2[i] = abs(lv3[i*4])",
                       "lv3[i*4] = round_mant<N>(lv3[i*4]",
                       "res_shared[thread_idx * 64 + i] = lv3[i]"):
            with self.subTest(marker=needle):
                self.assertIn(needle, cu)
        self.assertIn("void hifx_quant_cuda_bf16(torch::Tensor x, torch::Tensor result, int N)", cu)
        self.assertNotIn("torch::Tensor codes, torch::Tensor scales", cu)

    def test_software_int8_signed_codes_are_not_native_packed_storage(self):
        py = read(self.spec["source_proofs"][1]["path"])
        self.assertIn("codes: torch.Tensor                      # signed int8", py)
        self.assertIn("codes = torch.round(normalized * 4.0).clamp(-7, 7).to(torch.int8)", py)
        self.assertIn("dequantized = (codes.float() / 4.0 * full_scale)", py)
        native = self.spec["native_wan_path"]
        self.assertTrue(native["codec_sign_preserved_by_source"])
        self.assertFalse(native["persistent_packed_code"])
        self.assertFalse(native["scale_and_code_out_args"])
        self.assertTrue(native["actual_BF16_result"])
        self.assertFalse(native["native_bitwise_negative_zero_confirmed"])
        self.assertEqual(native["software_int8_vs_native_rounding_bitwise_equal"], "NOT_MEASURED")

    def test_observation_contract_uses_format_native_sign_without_free_export(self):
        observation = json.loads(read("research/agent_protocol/QMETA_OBSERVATION_CONTRACT_20261028.json"))
        sign = observation["hif4_signed_format_contract"]
        self.assertEqual(sign["path"], "research/agent_protocol/HIF4_SIGNED_FORMAT_CONTRACT_20261028.json")
        self.assertTrue(sign["signed_native_element_code"])
        self.assertFalse(sign["native_qdq_exports_packed_code"])
        self.assertFalse(sign["scale_metadata_contains_element_sign"])
        self.assertIn("S1P2_HAS_NATIVE_SIGN_BIT", self.spec["research_rules"])
        self.assertIn("CHARGE_FUSED_OBSERVER_OR_BF16_SCAN_REAL_COST", self.spec["research_rules"])


if __name__ == "__main__":
    unittest.main()
