"""Independent CPU converter/layout proofs and bounded C5 runner contracts."""
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F
from torch.sparse._semi_structured_conversions import (
    _calculate_meta_reordering_scatter_offsets,
    sparse_semi_structured_from_dense_cutlass)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import activation24_fused_cutlass_pack as packing
import e0_activation24_fused_cutlass_probe as driver
from e0_activation24_spmm_probe import select_top2


class FusedCutlassTest(unittest.TestCase):
    def plan(self):
        return json.loads((ROOT / "config/e0_activation24_fused_cutlass_probe_20261027_v1.json").read_text())

    def compare_original_cpu(self, value):
        masked = F.pad(select_top2(value), (0, 0, 0, packing.padded_rows(value.shape[0]) - value.shape[0]))
        reference = sparse_semi_structured_from_dense_cutlass(masked)
        proposed = packing.cpu_pack_proof(value)
        for left, right in zip(proposed, reference):
            self.assertEqual(left.dtype, right.dtype)
            self.assertEqual(left.shape, right.shape)
            self.assertTrue(torch.equal(left.view(torch.uint8), right.view(torch.uint8)))

    def test_exact_installed_converter_source_is_the_frozen_reference(self):
        module = sys.modules["torch.sparse._semi_structured_conversions"]
        actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        self.assertEqual(actual, packing.CONVERTER_SHA256)

    def test_all_sixteen_nonzero_patterns_follow_original_boolean_encoding(self):
        for flags in itertools.product((False, True), repeat=4):
            raw = [0x3f80 + j * 128 if flag else 0 for j, flag in enumerate(flags)]
            row = torch.tensor(raw, dtype=torch.uint16).view(torch.bfloat16)
            packed, meta = sparse_semi_structured_from_dense_cutlass(row.repeat(32, 16))
            first, second = packing.canonical_indices(flags)
            self.assertEqual(meta.flatten()[packing.metadata_offset(0, 0, 32, 4)].item() & 15,
                             first | (second << 2))
            self.assertEqual(packed[0, :2].view(torch.uint16).tolist(), [raw[first], raw[second]])

    def test_zero_padding_and_signed_zero_do_not_encode_top2_support(self):
        for bits, code in (([0, 0, 0, 0], 14), ([0x8000, 0, 0x8000, 0], 14),
                ([0x3f80, 0x8000, 0, 0], 8), ([0x8000, 0x3f80, 0, 0], 9),
                ([0x8000, 0, 0x3f80, 0], 14), ([0x8000, 0, 0, 0x3f80], 14)):
            values, actual = packing.select_pack_group_bits(bits)
            self.assertEqual(actual, code)
            self.assertNotIn(0x8000, values)
        # Stable top2 selects indices0,1 in the first tuple, but the converter
        # must encode0,2. This is a concrete counterexample to direct support packing.
        self.assertEqual(packing.select_pack_group_bits([0x3f80, 0, 0, 0]), ((0x3f80, 0), 8))

    def test_swizzle_is_bijective_and_matches_original_scatter_offsets(self):
        for rows, columns in ((32, 4), (64, 8), (96, 16), (32, 320), (64, 864)):
            proposed = [packing.metadata_offset(r, c, rows, columns)
                        for r in range(rows) for c in range(columns)]
            self.assertEqual(sorted(proposed), list(range(rows * columns)))
            original = _calculate_meta_reordering_scatter_offsets(rows, columns, torch.int16, "cpu")
            self.assertEqual(proposed, original.tolist())

    def test_two_fixed_edge_inputs_match_original_packed_and_meta_bytes(self):
        for shape in driver.EDGE_SHAPES:
            with self.subTest(shape=shape):
                self.compare_original_cpu(packing.edge_input(*shape))

    def test_signed_zero_subnormals_ties_and_extremes_exhaustive_finite_pool(self):
        pool = (0, 0x8000, 0x0001, 0x8001, 0x3f80, 0xbf80, 0x7f7f, 0xff7f)
        raw = list(itertools.product(pool, repeat=4))
        value = torch.tensor(raw, dtype=torch.uint16).view(torch.bfloat16).reshape(64, 256)
        self.compare_original_cpu(value)
        for bad in (0x7f80, 0xff80, 0x7fc0):
            with self.assertRaises(ValueError):
                packing.select_pack_group_bits([bad, 0, 0, 0])

    def test_actual_triton_dsl_cpu_interpreter_matches_original_without_cuda(self):
        script = """
import sys,torch
sys.path.insert(0,TOOLS)
import activation24_fused_cutlass_pack as p
from e0_activation24_spmm_probe import select_top2
from torch.sparse._semi_structured_conversions import sparse_semi_structured_from_dense_cutlass as original
assert not torch.cuda.is_initialized()
for shape in [(35,64),(33,128)]:
 x=p.edge_input(*shape);m,k=x.shape;mp=p.padded_rows(m)
 packed=torch.empty((mp,k//2),dtype=torch.bfloat16);meta=torch.empty((mp,k//16),dtype=torch.int16)
 p._kernel()[((mp*(k//4)+255)//256,)](x,packed,meta,m,k,mp,False,256,num_warps=4)
 stable=torch.nn.functional.pad(select_top2(x),(0,0,0,mp-m));ref,rm=original(stable)
 assert torch.equal(packed.view(torch.uint8),ref.view(torch.uint8))
 assert torch.equal(meta.view(torch.uint8),rm.view(torch.uint8))
 masked=torch.empty_like(stable)
 p._kernel()[((mp*(k//4)+255)//256,)](x,masked,x,m,k,mp,True,256,num_warps=4)
 assert torch.equal(masked.view(torch.uint8),stable.view(torch.uint8))
assert not torch.cuda.is_initialized()
print('CPU_INTERPRETER_BYTES_PASS_CUDA_FALSE')
""".replace("TOOLS", repr(str(ROOT / "tools")))
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30,
            env={**os.environ, "TRITON_INTERPRET": "1", "CUDA_VISIBLE_DEVICES": ""})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CPU_INTERPRETER_BYTES_PASS_CUDA_FALSE", result.stdout)

    def test_exact_operation_budget_includes_both_online_paths_and_correctness(self):
        counts = dict.fromkeys(driver.LIMITS, 0)
        # Two small gates plus two full gates, all performed in the future run.
        driver.charge(counts, {"edge_tensor_upload": 2, "random_tensor_init": 4,
            "finite_input_check": 6, "stable_reference_select": 4, "fused_select_pack": 4,
            "torch_converter": 4, "cutlass_wrapper": 6, "packed_meta_byte_check": 4,
            "materialize_selector": 2, "masked_byte_check": 2,
            "dense_gemm": 2, "cutlass_sparse_gemm": 2})
        for _ in range(2):
            for cost in driver.TIMING_COSTS.values():
                for _ in range(7):
                    driver.charge(counts, cost)
        self.assertEqual(counts, driver.LIMITS)
        before = counts.copy()
        with self.assertRaises(RuntimeError):
            driver.charge(counts, {"cutlass_sparse_gemm": 1, "fits": 1})
        self.assertEqual(counts, before)

    def test_byte_gate_rejects_signed_zero_difference_and_cpu_proof_is_not_gpu_gate(self):
        packed = torch.zeros((32, 32), dtype=torch.bfloat16)
        meta = torch.zeros((32, 4), dtype=torch.int16)
        reference = SimpleNamespace(packed=packed.clone(), meta=meta.clone())
        proof = driver.byte_conformance(packed, meta, reference)
        with self.assertRaisesRegex(RuntimeError, "Actual GPU"):
            driver.require_gpu_gate({"gpu_converter_byte_gate": proof})
        packed.view(torch.uint16)[0, 0] = 0x8000
        self.assertTrue(torch.equal(packed, reference.packed))
        with self.assertRaisesRegex(RuntimeError, "byte conformance"):
            driver.byte_conformance(packed, meta, reference)
        with self.assertRaises(RuntimeError):
            driver.require_gpu_gate({})

    def test_registered_gpu_scope_does_not_inherit_old_c4_authority(self):
        plan = self.plan()
        driver.validate_plan(plan)
        active = {"user_continuation_authorization_20261027": {"id": "USER_RESUME_OFFLINE_IFC_20261027",
            "status": "ACTIVE_CURRENT_USER_DIRECTED_SESSION", "new_scope_registration_allowed": True,
            "max_simultaneously_assigned_idle_GPUs": 4}, "experiment_authority": {"current_user_authorized_scopes": [
                {"id": driver.ID, "protocol_sha256": "new_sha", "execution_status": "AUTHORIZED_NOT_STARTED"}]}}
        driver.authorize(plan, "new_sha", active)
        for key, value in (("id", "E0_ACTIVATION24_CUTLASS_BACKEND_PROBE_20261027"),
                           ("protocol_sha256", "old_sha"), ("execution_status", "CLOSED")):
            changed = copy.deepcopy(active)
            changed["experiment_authority"]["current_user_authorized_scopes"][0][key] = value
            with self.assertRaises(ValueError):
                driver.authorize(plan, "new_sha", changed)
        for key, value in (("backend", "CUSPARSELT"), ("block_groups", 512),
                ("byte_conformance_required_in_this_GPU_run", False), ("automatic_retry", True)):
            with self.assertRaises(ValueError):
                driver.validate_plan({**plan, key: value})
        with self.assertRaisesRegex(RuntimeError, "Explicit"):
            driver.run(SimpleNamespace(execute_gpu=False))


if __name__ == "__main__":
    unittest.main()
