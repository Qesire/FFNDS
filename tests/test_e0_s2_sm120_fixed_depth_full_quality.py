import copy
import importlib.util
from pathlib import Path
import unittest
import tempfile
import types
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('full_quality_test', ROOT / 'tools/e0_s2_sm120_fixed_depth_full_quality.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class FullQualityTests(unittest.TestCase):
    def test_budget_is_eight_real_functions_for_twelve_clips(self):
        self.assertEqual(len(m.DIMENSIONS), 8)
        self.assertEqual(m.LIMITS['video_dimension_results'], 12 * 8)
        self.assertEqual(m.COSTS['dynamic_degree'], {'RAFT': 480, 'RAFT_iterations': 9600})
        self.assertEqual(m.COSTS['i2v_subject']['DINO'], 12 * 82)
        self.assertEqual(m.COSTS['i2v_background']['DreamSim_embed'], 12 * 82)
        self.assertNotIn('temporal_flickering', m.DIMENSIONS)

    def test_all_axes_paired_source_equal_and_reject_missing_clip(self):
        hf = Path('/tmp/fake-quality-hf')
        plan = {'shard': 0, 'videos': [{'ordinal': ordinal, 'mode': mode,
                 'video': {'path': f'{ordinal}_{mode}.mp4'}}
                for ordinal in m.SHARDS[0] for mode in m.MODES]}
        dimensions = []
        for dimension in m.DIMENSIONS:
            rows = [{'video_path': str(hf / v['video']['path']), 'video_results': (v['mode'] != 'dense') if dimension == 'dynamic_degree' else float(m.MODES.index(v['mode'])), 'image_path':'vbench2_beta_i2v/data/crop/16-9/' + str(v['ordinal']) + '.png'}
                    for v in plan['videos']]
            dimensions.append({'dimension': dimension, 'result': [0.0, rows]})
        summary = m.summarize(plan, hf, dimensions)
        self.assertEqual(summary['independent_image_sources'], 3)
        for axis in summary['per_axis'].values():
            self.assertEqual(axis['source_equal_means']['dense'], 0.)
            self.assertEqual(axis['source_equal_paired_differences'][m.MODES[1]], 1.)
        broken = copy.deepcopy(dimensions)
        broken[0]['result'][1].pop()
        with self.assertRaises(ValueError): m.summarize(plan, hf, broken)

    def test_conflicting_binding_rejected(self):
        def b(path, digest='a'): return {'path': path, 'bytes': 1, 'sha256': digest * 64}
        plan = {'repo_sources': [], 'hf_bindings': [b('x')], 'parents': [], 'videos': [],
                'images': [{'binding': b('x', 'b')}],
                'reused_import_proof': {'protocol': b('p'), 'RUN': b('r'), 'exit': b('e')},
                'import_environment': {'modules': {}}}
        with self.assertRaises(ValueError): m.all_bindings(plan)

    def test_loader_bound_path_and_class_origins(self):
        with tempfile.TemporaryDirectory() as tmp:
            hf = Path(tmp); weight = hf / 'weight.pth'; weight.write_bytes(b'weights')
            binding = m.base.binding(weight, hf)
            class Module:
                def __init__(self): pass
            class Model(Module): pass
            torch = types.SimpleNamespace(load=lambda path: 'loaded',
                jit=types.SimpleNamespace(load=lambda path: 'jit'), nn=types.SimpleNamespace(Module=Module))
            safetensors = types.ModuleType('safetensors'); safetensors.__path__ = []
            safe_torch = types.ModuleType('safetensors.torch'); safe_torch.load_file = lambda path: 'safe'
            report = {'active_dimension':'imaging_quality', 'actual_checkpoint_loads':[], 'actual_model_classes':[]}
            with mock.patch.dict('sys.modules', {'safetensors':safetensors, 'safetensors.torch':safe_torch}):
                restore = m.install_model_audit(torch, {'hf_bindings':[binding]}, hf, report, lambda: None)
                try:
                    self.assertEqual(torch.load(weight), 'loaded')
                    self.assertEqual(torch.jit.load(weight), 'jit')
                    self.assertEqual(safe_torch.load_file(weight), 'safe')
                    Model()
                    self.assertEqual([v['loader'] for v in report['actual_checkpoint_loads']],
                                     ['torch.load', 'torch.jit.load', 'safetensors.torch.load_file'])
                    self.assertEqual(report['actual_model_classes'][0]['class'], 'FullQualityTests.test_loader_bound_path_and_class_origins.<locals>.Model')
                    with self.assertRaises(ValueError): torch.load(hf / 'unregistered')
                finally: restore()
            self.assertEqual(torch.load(weight), 'loaded')

    def test_result_validation_detects_tampered_ledger_and_axis(self):
        hf = Path('/tmp/fake-quality-hf')
        videos = [{'ordinal': ordinal, 'mode': mode, 'video': {'path': f'{ordinal}_{mode}.mp4'}}
                  for ordinal in m.SHARDS[0] for mode in m.MODES]
        weight = {'path':'asset'+m.motion.WEIGHT_SUFFIX, 'bytes':1, 'sha256':m.motion.WEIGHT_SHA}
        raft_source = {'path':'asset/vbench/third_party/RAFT/core/raft.py', 'bytes':1, 'sha256':'a'*64}
        plan = {'id':'test', 'shard':0, 'videos':videos, 'hf_bindings':[weight,raft_source]}
        dimensions = []
        for dimension in m.DIMENSIONS:
            rows = [{'video_path':str(hf / v['video']['path']),
                     'video_results':False if dimension == 'dynamic_degree' else 1.0, 'image_path':'vbench2_beta_i2v/data/crop/16-9/' + str(v['ordinal']) + '.png'} for v in videos]
            dimensions.append({'dimension':dimension, 'result':[0.0,rows]})
        observations = [{'video_path':str(hf/v['video']['path']), 'scores':[0.0]*40,
                         'params':{'thres':10.875,'count_num':10}, 'official_binary':False} for v in videos]
        report = {'id':'test', 'status':m.STATUS, 'predebited':copy.deepcopy(m.LIMITS),
                  'invoked':copy.deepcopy(m.LIMITS), 'completed':copy.deepcopy(m.LIMITS),
                  'predebited_outer_model_work':copy.deepcopy(m.COSTS), 'completed_dimensions':dimensions,
                  'source_summary':m.summarize(plan,hf,dimensions), 'official_module_functions':[{}]*8,
                  'RAFT_observations':observations, 'actual_checkpoint_loads':[{'dimension':'dynamic_degree','binding':weight}],
                  'actual_RAFT_class':raft_source,
                  'actual_model_classes':[{'dimension':'dynamic_degree','module':'x','class':'X','origin':None}],
                  'native_dir_fd_audit':{'denied_attempts':[]}, 'native_metric_import_audit':{'denied_attempts':[]},
                  'assigned_GPU_resources':{'violations':[],'sampled_isolation_pass':True}}
        m.validate_result(plan,report,hf)
        for key in ('ledger','axis','weight','observation'):
            bad = copy.deepcopy(report)
            if key == 'ledger': bad['completed']['RAFT_forwards'] -= 1
            if key == 'axis': bad['completed_dimensions'][0]['result'][1].pop()
            if key == 'weight': bad['actual_checkpoint_loads'][0]['binding']['sha256'] = 'f'*64
            if key == 'observation': bad['RAFT_observations'][0]['official_binary'] = True
            with self.assertRaises(ValueError): m.validate_result(plan,bad,hf)

    def test_actual_raft_decoder_and_iteration_budgets(self):
        self.assertEqual(m.LIMITS['decoded_frames'], 12 * 81)
        self.assertEqual(m.LIMITS['RAFT_forwards'], 12 * 40)
        self.assertEqual(m.LIMITS['RAFT_iterations'], 12 * 40 * 20)
        self.assertEqual(m.RESOURCES['GPU'], 1)
        self.assertNotIn('session_max_GPUs', m.RESOURCES)

if __name__ == '__main__': unittest.main()
