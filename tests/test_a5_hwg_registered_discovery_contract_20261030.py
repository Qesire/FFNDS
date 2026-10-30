"""Tiny software/branch fixtures, no real saved observations or CUDA."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import a5_hwg_matched_capture_20261030 as guard
original_resources = guard.RES
import a5_hwg_registered_delta_discovery_20261030 as m
from a5_hwg_registered_static_fit_20261030 import validate_metadata_manifest
guard.RES = original_resources


class DiscoveryContractFixtures(unittest.TestCase):
    def test_all_80_pairs_charge_every_58arm_branch(self):
        from a5_hwg_e0_native_replay_20261030 import arm_specs
        specs = arm_specs()
        self.assertEqual(m.CALL_LIMITS['support_select'], 80 * len(specs))
        self.assertEqual(m.CALL_LIMITS['candidate_W2_FP32'], 80 * (len(specs) - 1))
        self.assertEqual(m.CALL_LIMITS['native_Q2'], 160)
        self.assertEqual(m.CALL_LIMITS['native_sample_verify'], 160)
        self.assertEqual(m.CALL_LIMITS['proxy_old_W1'], 80)

    def test_zero_update_changes_gate_using_current_native_branch(self):
        old = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=torch.bfloat16)
        rows = torch.tensor([0, 2])
        residual = torch.tensor([[10., 20.], [30., 40.]], dtype=torch.bfloat16)
        gate = torch.tensor([-2., .5], dtype=torch.bfloat16)
        delta = old[rows] * gate
        sampled = {'R': residual, 'gate_broadcast': gate,
                   'Delta_writeback_operand': delta, 'X_next': residual + delta}
        result = m.sample_postwrite(old, torch.zeros_like(old), rows, sampled)
        self.assertEqual(result['gated_writeback_relative_L2'], 0.)
        self.assertEqual(result['post_residual_relative_L2'], 0.)
        self.assertEqual(result['rows'], 2)

    def test_paid_rank_identity_and_opposite_ranking(self):
        ascending = torch.arange(216, dtype=torch.float32).repeat_interleave(64).reshape(1, -1)
        scores = {'paid_delta_diagonal': ascending,
                  'a5_delta_hwg': ascending, 'a5_current_hwg': ascending.flip(1)}
        result = m.ranking_diagnostic(scores)
        self.assertEqual(result['a5_delta_hwg']['stable_tie_rank_correlation'], 1.)
        self.assertEqual(result['a5_delta_hwg']['K64_top80_support_Jaccard'], 1.)
        self.assertEqual(result['a5_current_hwg']['stable_tie_rank_correlation'], -1.)
        self.assertEqual(result['a5_current_hwg']['K64_top80_support_Jaccard'], 0.)

    def test_post_verification_same_path_metadata_substitution_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            packets = [{'step': s, 'block': b, 'metadata':
                        {'path': f'step{s}_block{b:02d}.json', 'bytes': 20, 'sha256': 'original'}}
                       for s in range(4) for b in range(40)]
            manifest = {'status': 'PASS_CAPTURE_STRUCTURE_ONLY', 'full_V1_files': 160,
                        'completed_scopes': 160, 'scopes': packets}
            raw = json.dumps(manifest).encode()
            (root / 'manifest.json').write_bytes(raw)
            ref = {'path': 'manifest.json', 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
            run = {'phases': [{'arm': 'plain'}, {'arm': 'observed', 'capture_manifest': ref}]}
            source = {'metadata': {f'step{p["step"]}_block{p["block"]:02d}': dict(p['metadata']) for p in packets}}
            validate_metadata_manifest(source, run, root, guard.ReadBudget(1 << 20))
            source['metadata']['step2_block37']['sha256'] = 'replacement_at_same_original_path_and_schema'
            with self.assertRaisesRegex(RuntimeError, 'independently verified producer manifest'):
                validate_metadata_manifest(source, run, root, guard.ReadBudget(1 << 20))

    def test_shadow_native_root_rejected_before_adapter_import(self):
        expected = ROOT / '.runtime/code/HiF4_Sp/HiFloat4'
        wrong = SimpleNamespace(submodule_search_locations=['/tmp/shadow/HiFloat4'], origin='/tmp/shadow/HiFloat4/__init__.py')
        with patch.object(m.importlib.util, 'find_spec', return_value=wrong):
            with self.assertRaisesRegex(RuntimeError, 'native root search path'):
                m.preflight_native_root('HiFloat4', expected)


if __name__ == '__main__': unittest.main()
