"""CPU software controls. Tiny arithmetic is not a native model experiment."""
import ast
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import unittest
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import b_final_block_capture as capture
import e0_b_final_block_capture as driver


def test_exact_bf16_widening_preserves_signed_zero_and_finite_bits():
    values = torch.tensor([0., -0., 1., -2., .00001, 3.125], dtype=torch.bfloat16)
    raw = values.float().view(torch.uint8).numpy().tobytes()
    assert capture.exact_bf16_from_fp32_bytes(raw) == values.view(torch.uint8).numpy().tobytes()


def test_state_import_rejects_rounding_and_nonfinite():
    for bits in [0x3F800001, 0x7F800000, 0x7FC00000, 0xFF800000]:
        with unittest.TestCase().assertRaises(ValueError):
            capture.exact_bf16_from_fp32_bytes(struct.pack("<I", bits))


def test_row_draw_uses_native_ordinal_not_local_subset_index():
    from ffnds_experiment_plans import row_draw
    rows, seed = row_draw(31668, 256, base_seed=42, source_index=895, step=0, block=39, state_pair=True)
    assert len(rows) == len(set(rows)) == 256
    assert (rows, seed) != row_draw(31668, 256, base_seed=42, source_index=1, step=0, block=39, state_pair=True)


def test_budget_rejects_extra_call_and_preserves_failed_debit():
    record = {}
    ledger = driver.WorkLedger(record)
    ledger.reserve({"complete_Dense_flows": 4}, driver.run_ids()[0])
    with unittest.TestCase().assertRaises(ValueError):
        ledger.reserve({"complete_Dense_flows": 1}, driver.run_ids()[0])
    assert record["predebited"]["complete_Dense_flows"] == 4
    assert record["observed"] == {}
    with unittest.TestCase().assertRaises(ValueError):
        ledger.observe({"head_replays": 1})
    with unittest.TestCase().assertRaises(ValueError):
        ledger.reserve({"complete_Dense_flows": 1}, "other_source")


def test_state_hash_binds_all_numeric_fields_and_rejects_unknown_objects():
    scheduler = SimpleNamespace(config={"path": "/private/worker", "flag": False}, latents=torch.ones(2),
                                generator=torch.Generator().manual_seed(42), transformer_infer=SimpleNamespace())
    old = capture.scheduler_state(scheduler)
    assert "/private" not in json.dumps(old)
    scheduler.latents[0] += 1
    assert capture.scheduler_state(scheduler)["sha256"] != old["sha256"]
    scheduler.unknown = object()
    with unittest.TestCase().assertRaises(TypeError):
        capture.scheduler_state(scheduler)


def test_patch_restores_inherited_method_and_own_override_on_failure():
    class Owner:
        def call(self):
            return 3
    owner = Owner()
    with unittest.TestCase().assertRaises(RuntimeError), capture.patch_method(owner, "call", lambda: 4):
        assert owner.call() == 4
        raise RuntimeError()
    assert owner.call() == 3 and "call" not in vars(owner)
    owner.call = lambda: 7
    with capture.patch_method(owner, "call", lambda: 5):
        assert owner.call() == 5
    assert owner.call() == 7


def test_instrumented_scheduler_preparation_methods_are_explicit_source_references():
    scheduler = SimpleNamespace(latents=torch.ones(2), prepare=lambda: None, prepare_latents=lambda: None)
    recorded = capture.scheduler_state(scheduler)
    for name in ("prepare", "prepare_latents"):
        assert "non_numeric_instrumentation_callable" in recorded["fields"][name]
        assert recorded["fields"][name]["source_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    scheduler.some_unregistered_method = lambda: None
    with unittest.TestCase().assertRaises(TypeError):
        capture.scheduler_state(scheduler)


def original_post_class():
    """Execute the actual frozen class body without importing native envs."""
    source = driver.ROOT / ".runtime/code/LightX2V/lightx2v/models/networks/wan/infer/post_infer.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WanPostInfer")
    env = {"torch": torch, "math": math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), env)
    return env["WanPostInfer"]


def test_actual_post_source_returns_FP32_flow_and_native_alias_assignment():
    post = original_post_class()({"out_dim": 2, "clean_cuda_cache": False})
    head = torch.arange(48).reshape(6, 8).to(torch.bfloat16)
    pre = SimpleNamespace(grid_sizes=SimpleNamespace(tuple=(1, 2, 3)))
    result = post.infer(head, pre)[0]
    assert result.shape == (2, 1, 4, 6) and result.dtype == torch.float32
    # Real no-CFG WanModel assignment preserves this FP32 object; there is no
    # intervening BF16 cast before scheduler.step_post.
    scheduler = SimpleNamespace()
    scheduler.noise_pred_cond = scheduler.noise_pred_guided = scheduler.noise_pred = result
    assert scheduler.noise_pred is result and scheduler.noise_pred.dtype == torch.float32


class IdentityOp:
    def apply(self, value):
        return value + torch.zeros((), dtype=value.dtype)


class TinyNorm:
    eps = 1e-6
    def _get_actual_weight(self):
        return None
    def _get_actual_bias(self):
        return None
    def apply(self, value):
        return torch.nn.functional.layer_norm(value, (4,), eps=self.eps)


class TinyHead:
    has_lora_branch = False
    def __init__(self):
        self.weight = (torch.arange(32).reshape(4, 8).float() / 80).to(torch.bfloat16).t().contiguous().t()
        self.bias = torch.arange(8).to(torch.bfloat16) / 16
    def _get_actual_weight(self):
        return self.weight
    def _get_actual_bias(self):
        return self.bias
    def apply(self, value):
        self.last = torch.addmm(self.bias, value, self.weight)
        return self.last


class TinyRuntime:
    def __init__(self, adapter):
        self.adapter = adapter
        self.counters = dict.fromkeys(("linear_forward_calls", "activation_qdq_calls", "weight_transfers", "cache_misses"), 0)
    def apply(self, mm, value, *, expert, step_index, block_index, module_name):
        q = self.adapter.native_hif4_activation_qdq(value, force_fp32=True)
        self.counters["linear_forward_calls"] += 1
        self.counters["activation_qdq_calls"] += 1
        return q + torch.tensor(.001, dtype=q.dtype)


class TinyInfer:
    infer_dtype = sensitive_layer_dtype = torch.bfloat16
    def __init__(self, runtime, index):
        self.runtime, self.index, self.block_idx, self.step = runtime, index, 0, 0
    def linear(self, value, name):
        expert = "high_noise_model" if self.index == 0 else "low_noise_model"
        return self.runtime.apply(None, value, expert=expert, step_index=self.step,
                                  block_index=self.block_idx, module_name=f"blocks.{self.block_idx}." + name)
    def infer_block(self, block, x, pre):
        for i in range(8):
            x = self.linear(x, "attention." + str(i))
        y = self.infer_ffn(block.compute_phases[2], x, x, None, None,
                           torch.tensor([.125, -.25, .5, 1.], dtype=torch.bfloat16).reshape(1, 1, 4))
        return self.post_process(x, y, torch.tensor([.125, -.25, .5, 1.], dtype=torch.bfloat16).reshape(1, 1, 4), pre)
    def infer_ffn(self, phase, x, attn, shift, scale, gate=None):
        y = self.linear(x, "ffn.0")
        y = torch.nn.functional.gelu(y, approximate="tanh")
        return self.linear(y, "ffn.2")
    def post_process(self, x, y, gate, pre=None):
        x.add_(y * gate.squeeze())
        return x
    def infer_non_blocks(self, weights, x, e):
        modulation = weights.head_modulation.tensor
        e = (modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = weights.norm.apply(x)
        x.mul_(1 + e[1].squeeze()).add_(e[0].squeeze())
        return weights.head.apply(x)


class TinyPre:
    def infer(self, weights, inputs):
        x = weights.patch_embedding.apply(torch.arange(24).reshape(6, 4).to(torch.bfloat16) / 100)
        e = weights.time_embedding_2.apply(weights.time_embedding_0.apply(torch.ones(1, 4, dtype=torch.bfloat16) / 32))
        e0 = weights.time_projection_1.apply(e).unsqueeze(1).expand(1, 6, 4)
        context = weights.text_embedding_2.apply(weights.text_embedding_0.apply(e))
        return SimpleNamespace(x=x, embed=e, embed0=e0, context=context,
            cos_sin=torch.ones(6, 2, dtype=torch.complex64), grid_sizes=SimpleNamespace(tuple=(1, 2, 3)))


def tiny_models(runtime):
    result = []
    for index in range(2):
        weights = SimpleNamespace(norm=TinyNorm(), head=TinyHead(), head_modulation=SimpleNamespace(tensor=torch.ones(1, 2, 4, dtype=torch.bfloat16) / 64))
        pw = SimpleNamespace(**{name: IdentityOp() for name in
            ("time_embedding_0", "time_embedding_2", "time_projection_1", "text_embedding_0", "text_embedding_2", "patch_embedding")})
        result.append(SimpleNamespace(transformer_infer=TinyInfer(runtime, index), transformer_weights=weights,
            pre_weight=pw, pre_infer=TinyPre(), post_infer=original_post_class()({"out_dim": 2, "clean_cuda_cache": False})))
    return result


def invoke(model, step):
    infer = model.transformer_infer
    infer.step = step
    pre = model.pre_infer.infer(model.pre_weight, {})
    x = pre.x
    block = SimpleNamespace(compute_phases=[SimpleNamespace(), SimpleNamespace(), SimpleNamespace()])
    for b in range(40):
        infer.block_idx = b
        x = infer.infer_block(block, x, pre)
    head = infer.infer_non_blocks(model.transformer_weights, x, pre.embed)
    return head, model.post_infer.infer(head, pre)[0]


@contextmanager
def tiny_fixture():
    with ExitStack() as stack:
        for name, value in {"M": 6, "D": 4, "OUT": 8, "GRID": (1, 2, 3)}.items():
            stack.enter_context(patch.object(capture, name, value))
        stack.enter_context(patch.object(torch.cuda, "get_rng_state", lambda: torch.get_rng_state().clone()))
        adapter = SimpleNamespace(native_hif4_activation_qdq=lambda value, force_fp32=True: value)
        runtime = TinyRuntime(adapter)
        models = tiny_models(runtime)
        packs = {}
        def save(run_id, name, tensors):
            packs[(run_id, name)] = {k: v.clone() for k, v in tensors.items()}
            return {"path": run_id + "/" + name, "tensors": {k: capture.tensor_record(v) for k, v in tensors.items()}}
        yield adapter, runtime, models, packs, save


def test_full_tap_is_byte_identical_and_captures_actual_combined_modulation(tiny):
    adapter, runtime, models, packs, save = tiny
    head_before, flow_before = invoke(models[0], 0)
    original_qdq, original_apply = adapter.native_hif4_activation_qdq, runtime.apply
    with capture.FinalBlockTap(runtime, adapter, models, {0: [0, 2, 4], 3: [0, 2, 4]}, save) as tap:
        tap.begin("run", 0)
        head, flow = invoke(models[0], 0)
        assert head is models[0].transformer_weights.head.last
        record = tap.finish(flow, torch.ones(2, 1, 4, 6, dtype=torch.bfloat16), torch.ones(2, 1, 4, 6, dtype=torch.bfloat16))
    assert torch.equal(head.view(torch.uint8), head_before.view(torch.uint8))
    assert torch.equal(flow.view(torch.uint8), flow_before.view(torch.uint8))
    payload = packs[("run", "capture")]
    assert payload["flow"].dtype == torch.float32
    assert payload["head_shift"].shape == (1, 1, 4)
    assert torch.equal(payload["head_shift"], torch.full((1, 1, 4), 3 / 64, dtype=torch.bfloat16))
    assert record["native_noop"]["CPU_BF16_writeback_mul_add_byte_equal"]
    assert record["actual_boundary_counts"]["linears"] == 400
    assert record["actual_head_operator"]["weight"]["stride"] == [1, 4]
    assert adapter.native_hif4_activation_qdq is original_qdq and runtime.apply == original_apply


def test_tap_rejects_false_writeback_and_restores_every_patch(tiny):
    adapter, runtime, models, packs, save = tiny
    original = models[0].transformer_infer.infer_block
    with unittest.TestCase().assertRaisesRegex(ValueError, "writeback"), capture.FinalBlockTap(runtime, adapter, models, {0: [0, 2, 4]}, save) as tap:
        tap.begin("run", 0)
        _, flow = invoke(models[0], 0)
        tap.pending["r_rows"] = torch.ones_like(tap.pending["r_rows"])
        tap.finish(flow, torch.zeros(2, 1, 4, 6, dtype=torch.bfloat16), torch.zeros(2, 1, 4, 6, dtype=torch.bfloat16))
    assert models[0].transformer_infer.infer_block == original
    assert "infer_block" not in vars(models[0].transformer_infer)


def test_tap_rejects_reordered_blocks_before_execution(tiny):
    adapter, runtime, models, packs, save = tiny
    with capture.FinalBlockTap(runtime, adapter, models, {0: [0]}, save) as tap:
        tap.begin("run", 0)
        models[0].transformer_infer.block_idx = 1
        with unittest.TestCase().assertRaisesRegex(ValueError, "forty"):
            models[0].transformer_infer.infer_block(SimpleNamespace(compute_phases=[]), torch.ones(6, 4), None)
    assert runtime.counters["linear_forward_calls"] == 0


def test_no_registration_cannot_authorize_frozen_plan():
    # Registration itself must bind exact protocol/run IDs; the CPU draft is
    # not executable merely because the user allows up to four idle cards.
    path = driver.ROOT / "config/e0_b_final_block_capture_20261028_v1.json"
    if not path.exists():
        raise unittest.SkipTest("Draft config is produced after code tests")
    plan = json.loads(path.read_text())
    with unittest.TestCase().assertRaises(ValueError):
        driver.authorize(plan, "0" * 64, {})


class CPUCaptureControls(unittest.TestCase):
    test_exact_widening = staticmethod(test_exact_bf16_widening_preserves_signed_zero_and_finite_bits)
    test_reject_invalid_state = staticmethod(test_state_import_rejects_rounding_and_nonfinite)
    test_original_row_identity = staticmethod(test_row_draw_uses_native_ordinal_not_local_subset_index)
    test_fixed_budget = staticmethod(test_budget_rejects_extra_call_and_preserves_failed_debit)
    test_complete_state_hash = staticmethod(test_state_hash_binds_all_numeric_fields_and_rejects_unknown_objects)
    test_patch_cleanup = staticmethod(test_patch_restores_inherited_method_and_own_override_on_failure)
    test_preparation_method_identity = staticmethod(test_instrumented_scheduler_preparation_methods_are_explicit_source_references)
    test_original_post_FP32 = staticmethod(test_actual_post_source_returns_FP32_flow_and_native_alias_assignment)
    test_registration_required = staticmethod(test_no_registration_cannot_authorize_frozen_plan)

    def test_native_tap_noop(self):
        with tiny_fixture() as tiny:
            test_full_tap_is_byte_identical_and_captures_actual_combined_modulation(tiny)

    def test_reject_false_writeback(self):
        with tiny_fixture() as tiny:
            test_tap_rejects_false_writeback_and_restores_every_patch(tiny)

    def test_reject_reordered_blocks(self):
        with tiny_fixture() as tiny:
            test_tap_rejects_reordered_blocks_before_execution(tiny)


if __name__ == "__main__":
    unittest.main()
