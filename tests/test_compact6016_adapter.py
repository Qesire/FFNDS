"""CPU-only synthetic numerical contracts, not native CUDA evidence."""
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from compact6016_adapter import CompactAdapters, compact_forward, error_metrics, gated_loss, native_q2_ste, same_bits


class Tests(unittest.TestCase):
    def test_exact_ste_forward_preserves_quantizer_bytes_and_identity_backward(self):
        x = torch.tensor([-.0, .00390625, 1., 256.], dtype=torch.bfloat16, requires_grad=True)
        calls = []
        def quantizer(value, *, force_fp32):
            self.assertTrue(force_fp32)
            calls.append(1)
            result = value.detach().clone()
            result[1] = 0
            return result
        y = native_q2_ste(x, quantizer)
        expected = quantizer(x, force_fp32=True)
        self.assertTrue(same_bits(y, expected))
        y.float().sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))
        self.assertEqual(len(calls), 2)

    def test_student_recomputes_own_hidden_and_quantizer_input(self):
        generator = torch.Generator().manual_seed(3)
        x = torch.randn(3, 8, generator=generator).bfloat16()
        w1 = torch.randn(64, 8, generator=generator).bfloat16()
        w2 = torch.randn(8, 64, generator=generator).bfloat16()
        b1, b2 = torch.zeros(64).bfloat16(), torch.zeros(8).bfloat16()
        smooth = torch.linspace(.5, 2, 64)
        seen = []
        def quantizer(value, *, force_fp32):
            seen.append(value.detach().clone())
            return (value.float().round()).bfloat16()
        result = compact_forward(x, w1, b1, w2, b2, smooth, quantizer)
        expected_input = (F.gelu(F.linear(x, w1, b1), approximate="tanh").float() / smooth).bfloat16()
        self.assertTrue(same_bits(seen[0], expected_input))
        expected = F.linear(expected_input.float().round().bfloat16(), w2, b2)
        self.assertTrue(same_bits(result, expected))

    def test_only_fp32_lora_is_trainable_and_merge_does_not_quantize(self):
        generator = torch.Generator().manual_seed(4)
        w1 = torch.randn(64, 8, generator=generator).bfloat16()
        w2 = torch.randn(8, 64, generator=generator).bfloat16()
        model = CompactAdapters(w1, w2, rank=3)
        self.assertEqual(len(list(model.parameters())), 4)
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
        self.assertFalse(model.base_W1.requires_grad)
        self.assertFalse(model.base_W2.requires_grad)
        with torch.no_grad():
            model.lora_B1.fill_(.125)
        merged = model.merged_weights()
        expected = (w1.float() + model.lora_B1 @ model.lora_A1).bfloat16()
        self.assertTrue(same_bits(merged[0], expected))
        self.assertEqual(merged[0].dtype, torch.bfloat16)

    def test_zero_adapter_forward_and_both_weight_gradient_paths(self):
        generator = torch.Generator().manual_seed(5)
        w1 = (torch.randn(64, 8, generator=generator) * .1).bfloat16()
        w2 = (torch.randn(8, 64, generator=generator) * .1).bfloat16()
        x = torch.randn(4, 8, generator=generator).bfloat16()
        target = torch.randn(4, 8, generator=generator).bfloat16()
        model = CompactAdapters(w1, w2, rank=3)
        b1, b2, smooth = torch.zeros(64).bfloat16(), torch.zeros(8).bfloat16(), torch.ones(64)
        q = lambda value, force_fp32: (value.float() * 2).round().bfloat16() / 2
        actual = compact_forward(x, *[model.merged_weights()[0], b1, model.merged_weights()[1], b2, smooth, q], training=True)
        reference = compact_forward(x, w1, b1, w2, b2, smooth, q)
        self.assertTrue(same_bits(actual, reference))
        gated_loss(actual, target, torch.ones(1, 8).bfloat16()).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            if "lora_B" in name:
                self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_source_row_mean_centered_identity_and_zero_energy_status(self):
        target = torch.tensor([[1., 2.], [3., 4.], [5., 2.]], dtype=torch.bfloat16)
        prediction = target + torch.tensor([[1., -.5], [2., .5], [3., 0.]], dtype=torch.bfloat16)
        gate = torch.tensor([[.3, 1.1]], dtype=torch.bfloat16)
        result = error_metrics(prediction, target, gate)
        self.assertAlmostEqual(result["source_row_mean_error_SSE"] + result["source_row_centered_error_SSE"], result["gated_pre_gate_SSE"], places=10)
        self.assertGreater(result["source_row_centered_teacher_energy"], 0)
        self.assertEqual(result["source_row_centered_status"], "MEASURED")
        constant = torch.ones(3, 2).bfloat16()
        result = error_metrics(constant * 2, constant, gate)
        self.assertIsNone(result["source_row_centered_relative_SSE"])
        self.assertEqual(result["source_row_centered_status"], "NOT_MEASURED_ZERO_CENTERED_TEACHER_ENERGY")
        self.assertIsNone(error_metrics(constant, constant, gate)["source_row_mean_error_fraction"])

    def test_native_bf16_gate_metric_is_separate_from_training_loss(self):
        target = torch.tensor([[1.1015625, .6015625], [.80078125, 1.203125]], dtype=torch.bfloat16)
        prediction = target + torch.tensor([[.0078125, -.00390625], [.00390625, .0078125]], dtype=torch.bfloat16)
        gate = torch.tensor([[.30078125, .69921875]], dtype=torch.bfloat16)
        result = error_metrics(prediction, target, gate)
        self.assertNotEqual(result["gated_pre_gate_relative_SSE"], result["actual_BF16_gate_relative_SSE"])
        self.assertAlmostEqual(float(gated_loss(prediction, target, gate)), result["gated_pre_gate_relative_SSE"], places=8)
        with self.assertRaises(ValueError):
            gated_loss(prediction, torch.zeros_like(target), gate)

    def test_folded_bf16_pack_reload_preserves_all_bits(self):
        from safetensors.torch import load_file, save_file
        tensors = {"W1": torch.tensor([-.0, 1.125]).bfloat16(),
                   "smooth2": torch.tensor([1., 2.]), "groups": torch.tensor([0, 3])}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pack.safetensors"
            save_file(tensors, str(path))
            loaded = load_file(str(path))
            self.assertTrue(all(same_bits(value, loaded[key]) for key, value in tensors.items()))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
