import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('new12_motion_test', ROOT / 'tools/e0_s2_new12_dense_motion_metrics.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class MotionTests(unittest.TestCase):
    def test_three_shards_cover_unique_new_sources(self):
        self.assertEqual(len(set(sum((list(s) for s in m.SHARDS), []))), 12)
        self.assertEqual(m.LIMITS['decoded_frames'], 4 * 81)
        self.assertEqual(m.LIMITS['RAFT_forwards'], 4 * 40)
        self.assertEqual(m.LIMITS['RAFT_iterations'], 4 * 40 * 20)

    def test_original_summary_and_invalid_values(self):
        scores = [float(i) for i in range(40)]
        row = {'scores': scores, 'params': {'thres': 10.875, 'count_num': 10}, 'official_binary': True}
        summary = m.base.validate_raft(row)
        self.assertEqual(summary['mean_top5percent_flow_px_per_2frame_pair'], 19.5)
        self.assertEqual(len(summary['five_fixed_8pair_windows']), 5)
        row['scores'][0] = float('nan')
        with self.assertRaises(ValueError): m.base.validate_raft(row)


    def test_original_observer_records_exact_scores_and_work(self):
        class RAFT:
            def forward(self, x, y, **kw): return None
        class DynamicDegree:
            params = {'thres': 10.875, 'count_num': 10}
            def get_score(self, x, y): return 3.0
            def infer(self, path):
                model = RAFT()
                for _ in range(40):
                    model.forward(1, 2, iters=20, test_mode=True)
                    self.get_score(1, 2)
                return False
        class Module: pass
        Module.RAFT = RAFT; Module.DynamicDegree = DynamicDegree
        original = RAFT.forward; rows = []; report = {}
        ledger = m.base.Ledger(report, limits={'RAFT_forwards':40, 'RAFT_iterations':800})
        with m.base.raft_observer(Module, rows, ledger, ['clip']):
            self.assertFalse(DynamicDegree().infer('clip'))
        self.assertIs(RAFT.forward, original)
        self.assertEqual(rows[0]['scores'], [3.0]*40)
        self.assertEqual(report['completed'], {'RAFT_forwards':40, 'RAFT_iterations':800})
        ledger.finish()

    def test_actual_decode_counter_and_no_retry_on_short_read(self):
        class Device:
            def __init__(self, n): self.n = n
            def read(self):
                self.n -= 1
                return (self.n >= 0, object() if self.n >= 0 else None)
            def release(self): pass
        class Cv: VideoCapture = lambda self, path: Device(int(path))
        class Module: cv2 = Cv()
        report = {}; ledger = m.base.Ledger(report, limits={'decoded_frames': 81})
        old = m.install_decoder_counter(Module, ledger)
        try:
            cap = Module.cv2.VideoCapture('81')
            for _ in range(82): cap.read()
            cap.release(); self.assertEqual(report['completed']['decoded_frames'], 81)
        finally: Module.cv2.VideoCapture = old
        report = {}; ledger = m.base.Ledger(report, limits={'decoded_frames': 81})
        old = m.install_decoder_counter(Module, ledger)
        try:
            cap = Module.cv2.VideoCapture('2')
            for _ in range(3): cap.read()
            with self.assertRaises(ValueError): cap.release()
            self.assertEqual(report['completed']['decoded_frames'], 2)
        finally: Module.cv2.VideoCapture = old


    def test_execution_monitor_independent_from_old_template(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); old = directory / 'old-nvidia-smi'; current = directory / 'nvidia-smi'
            old.write_bytes(b'historical driver'); current.write_bytes(b'new driver')
            target = {'path':str(current), 'bytes':current.stat().st_size, 'sha256':m.base.sha(current)}
            old_binding = {'path':str(old), 'bytes':old.stat().st_size, 'sha256':m.base.sha(old)}
            plan = {'execution_monitor_binary':target, 'environment':{'nvidia_smi':old_binding}}
            with patch.object(m.shutil, 'which', return_value=str(current)):
                self.assertEqual(m.verify_execution_monitor(plan), current.resolve())
            with patch.object(m.shutil, 'which', return_value=str(old)):
                with self.assertRaises(ValueError): m.verify_execution_monitor(plan)
            current.write_bytes(b'wrong driver')
            with patch.object(m.shutil, 'which', return_value=str(current)):
                with self.assertRaises(ValueError): m.verify_execution_monitor(plan)

    def test_worker_private_config_binding(self):
        import tempfile
        import worker_identity
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'worker.json'; path.write_bytes(b'private configuration')
            plan = {'execution_private_worker_config_sha256':m.base.sha(path)}
            with patch.object(worker_identity, 'require_worker'), patch.object(worker_identity, 'private_config_path', return_value=path):
                m.verify_execution_worker(plan)
                path.write_bytes(b'changed configuration')
                with self.assertRaises(ValueError): m.verify_execution_worker(plan)

    def test_authorization_exact_digest_resources(self):
        scope = {'id':'x', 'protocol_sha256':'abc', 'execution_status':'AUTHORIZED_NOT_STARTED',
                 'limits':copy.deepcopy(m.LIMITS), 'resources':copy.deepcopy(m.RESOURCES), 'max_gpu_cards':1}
        active = {'user_continuation_authorization_20261027':{'new_scope_registration_allowed':True},
                  'experiment_authority':{'current_user_authorized_scopes':[scope]}}
        from unittest.mock import patch
        with patch.object(m, 'validate'):
            self.assertIs(m.authorize({'id':'x'}, 'abc', active), scope)
            for broken in ('protocol_sha256','limits','resources','max_gpu_cards'):
                bad = copy.deepcopy(active); bad['experiment_authority']['current_user_authorized_scopes'][0][broken] = None
                with self.assertRaises(ValueError): m.authorize({'id':'x'}, 'abc', bad)

    def test_actual_parent_lineage_and_bound_draft(self):
        hf = Path('/home/xiang/.cache/huggingface')
        if not (hf / m.PARENT_HF.format(0) / 'results/RUN.json').exists(): self.skipTest('Parent assets unavailable')
        for shard in range(3):
            plan = m.draft(hf, shard)
            m.validate(plan); m.verify_lineage(plan, hf)
            if shard == 0:
                frozen = copy.deepcopy(plan)
                frozen.update(status='FROZEN_REGISTERED', assets_finalized=True, binding_status='BOUND_READY_FOR_REGISTRATION')
                with self.assertRaises(ValueError): m.validate(frozen, True)
                frozen['execution_monitor_binary'] = {'path':'/usr/bin/nvidia-smi', 'bytes':1243808,
                    'sha256':'384896f294f80e8e5b59c11dbb04d3fdc028cf9bb01d7839154b96f94e966f4e'}
                frozen['execution_private_worker_config_sha256'] = 'a'*64
                m.validate(frozen, True)
            self.assertEqual([v['ordinal'] for v in plan['videos']], list(m.SHARDS[shard]))
            bad = copy.deepcopy(plan); bad['videos'][0]['video']['sha256'] = '0'*64
            with self.assertRaises(ValueError): m.verify_lineage(bad, hf)

    def test_closed_result_rejects_tampering(self):
        hf = Path('/home/xiang/.cache/huggingface')
        if not (hf / m.PARENT_HF.format(0) / 'results/RUN.json').exists(): self.skipTest('Parent assets unavailable')
        plan = m.draft(hf, 0)
        rows = []
        for v in plan['videos']:
            r = {'ordinal':v['ordinal'], 'video_sha256':v['video']['sha256'],
                 'video_path':str(hf / v['video']['path']), 'status':'COMPLETE',
                 'scores':[1.0]*40, 'params':{'thres':10.875,'count_num':10}, 'official_binary':False}
            r['summary'] = m.base.validate_raft(r); rows.append(r)
        report = {'id':plan['id'], 'status':m.STATUS, 'RAFT_observations':rows,
                  'source_equal_mean_top5percent_flow_px':1.0,
                  'official_dynamic_degree_only':[0.0, [{'video_path':r['video_path'], 'video_results':False} for r in rows]],
                  'actual_RAFT_checkpoint_loads':[next(b for b in plan['raft_assets'] if b['path'].endswith(m.WEIGHT_SUFFIX))],
                  'actual_RAFT_source':next(b for b in plan['raft_assets'] if b['path'].endswith('/vbench/third_party/RAFT/core/raft.py')),
                  'assigned_GPU_resources':{'violations':[], 'sampled_isolation_pass':True}}
        for key in ('predebited','invoked','completed'): report[key] = copy.deepcopy(m.LIMITS)
        m.validate_result(plan, report, hf)
        for mutation in ('score', 'weight', 'budget', 'binary'):
            bad = copy.deepcopy(report)
            if mutation == 'score': bad['RAFT_observations'][0]['scores'][0] = 9.0
            if mutation == 'weight': bad['actual_RAFT_checkpoint_loads'] = []
            if mutation == 'budget': bad['completed']['RAFT_forwards'] -= 1
            if mutation == 'binary': bad['official_dynamic_degree_only'][1][0]['video_results'] = True
            with self.assertRaises(ValueError): m.validate_result(plan, bad, hf)

if __name__ == '__main__': unittest.main()
