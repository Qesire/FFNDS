"""Source/noise/Dense reuse and native full160 boundary software controls."""
import copy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import e0_raw_bf16_temporal_video20 as m


class Meter:
    def __init__(self): self.counts = {}
    def call(self, rid, costs, fn, *a, **kw):
        result = fn(*a, **kw)
        for k, v in costs.items(): self.counts[k] = self.counts.get(k, 0)+v
        return result


class VideoControls(unittest.TestCase):
    def test_full_depth_budget_has_only_forty_new_videos(self):
        for part in (0, 1):
            v = m.limits(part)
            self.assertEqual(v['video_generations'], 20)
            self.assertEqual(v['native_full_model_flows'], 80)
            self.assertEqual(v['native_FFN_calls'], 3200)
            self.assertEqual(v['BF16_factor_GEMM'], 6400)
            self.assertEqual(v['original_W2_H2D_skips'], 3240)
            self.assertEqual(v['factor_GPU_H2D_tensors'], 20*80*3)
            self.assertEqual(v['state_tensor_D2H'], 200)
            self.assertEqual(v['reused_Dense_registration_writes'], 10)
            self.assertEqual(v['extra_teacher_flows'], 0)

    def test_two_arm_order_balances_each_part_and_names_are_unique(self):
        for part in (0, 1):
            cases = [{'index': i, 'native_source_ordinal': 100+i} for i in range(part, 20, 2)]
            self.assertEqual(sum(m.phase_order(c)[0] == 'dcac10' for c in cases), 5)
            self.assertEqual(len({m.run_id(c, f) for c in cases for f in m.MODES}), 20)

    def test_wrong_raw_noise_rejected_even_when_initial_noise_matches(self):
        state = {k: {'bytes': 1, 'sha256': k, 'raw_tensor_sha256': k, 'shape': [1], 'dtype': 'bfloat16'}
                 for k in ('raw_noise_FP32', 'initial_noise')}
        phase = {'run_id': 'candidate', 'states': copy.deepcopy(state)}
        dense = {'states': copy.deepcopy(state)}
        meter = Meter(); m.match_noise(phase, dense, meter)
        self.assertEqual(meter.counts, {'matched_reference_noise_byte_checks': 2})
        dense['states']['raw_noise_FP32']['raw_tensor_sha256'] = 'wrong'
        with self.assertRaises(ValueError): m.match_noise(phase, dense, Meter())

    def test_shared_physical_factor_identity_spans_both_steps(self):
        self.assertEqual(m.coordinates({'expert': 1, 'block': 39, 'family': 'dcac10'}), [(1, 2, 39), (1, 3, 39)])
        with self.assertRaises(ValueError): m.coordinates({'expert': 1, 'block': 40, 'family': 'dcac10'})

    def test_core_uses_real_reused_Dense_receipt_and_path(self):
        import ffnds_video_metrics as metrics
        case = {'image': {'sha256': 'image'}, 'prompt_sha256': 'prompt'}
        phase = {'run_id': 'candidate', 'protocol_sha256': 'new', 'generation_receipt': {'sha256': 'newGEN'},
            'video': {'path': '/hf/candidate.mp4'}, 'generation_wall_seconds_including_encoders_decode_observers': 1.,
            'FFN_boundary_CUDA_seconds': .2, 'peak_cuda_allocated_bytes': 100}
        dense = {'protocol_sha256': 'original', 'generation': {'sha256': 'originalGEN'}, 'video': {'path': 'old/dense.mp4'}}
        contexts = []
        def evaluate(path, reference, context):
            self.assertEqual(reference, Path('/hf/old/dense.mp4'))
            self.assertEqual(context['reference_identity']['run_sha256'], 'originalGEN')
            self.assertEqual(context['identity']['run_sha256'], 'newGEN')
            contexts.append(context); return {}
        counter = SimpleNamespace(write_json=lambda *_: None, binding=lambda *_: {'sha256': 'CORE'})
        with patch.dict(sys.modules, {'cv2': SimpleNamespace(setNumThreads=lambda _: None)}), \
             patch.object(metrics, 'evaluate', side_effect=evaluate), patch.object(metrics, 'validate', return_value=[]):
            m.core_report({'models': []}, phase, case, dense, Path('/hf'), Path('/out'), counter, Meter(), 2.)
        self.assertEqual(len(contexts), 1)

    def test_staging_reuses_Dense_instead_of_claiming_a_new_generation(self):
        # The unchanged generic verifier must see an actual Dense file/noise.
        from ffnds_stage_registered_i2v import collect
        with TemporaryDirectory() as temp:
            root = Path(temp); source = {'source_id': 's', 'image_id': 'image', 'prompt_id': 'p',
                'role': 'DEVELOPMENT', 'conditioning_id': 'condition', 'prompt': 'prompt',
                'image_path': str(root/'image.png'), 'image_name': 'image.png'}
            (root/'image.png').write_bytes(b'image')
            jobs = []; phases = []
            for mode in ('dense', *m.MODES):
                video = root/(mode+'.mp4'); video.write_bytes(mode.encode())
                noise = root/(mode+'.sf'); noise.write_bytes(b'actual identical noise')
                jobs.append({'source_id': 's', 'variant': mode, 'sample_index': 0, 'seed': 42, 'run_id': mode,
                    'asset_id': mode, 'initial_noise_id': 'same', 'conditioning_id': 'condition',
                    'scheduler_id': 'same', 'frames': 81, 'fps': 16, 'width': 832, 'height': 464})
                phases.append({'native_counter_delta': {'linear_forward_calls': 1600},
                    'states': {'initial_noise': {'path': noise.name}}, 'video': {'path': str(video)},
                    'generation_receipt': {'sha256': mode+'GEN'}})
            plan = {'schema': 'ffnds.cohort.v1', 'mode': 'CUSTOM_PAIRED', 'phase': 'REGISTERED_REAL_GENERATION_PLAN',
                'purpose': 'DEVELOPMENT', 'sources': [source], 'samples': jobs, 'required_variants': ['dense', *m.MODES],
                'expected_samples_per_source_per_variant': 1}
            for i, phase in enumerate(phases):
                folder = root/f'job{i:06d}'; folder.mkdir()
                record = m.registration_record(plan, i, phase, root, reused=i == 0)
                (folder/'RUN.json').write_text(json.dumps(record))
            self.assertEqual(len(collect(plan, root)['samples']), 3)
            self.assertEqual(json.loads((root/'job000000/RUN.json').read_text())['generation_origin'],
                             'REUSED_ORIGINAL_DENSE_NOT_A_NEW_GENERATION')
            (root/'value10.sf').write_bytes(b'different noise')
            with self.assertRaises(ValueError): collect(plan, root)


if __name__ == '__main__': unittest.main()
