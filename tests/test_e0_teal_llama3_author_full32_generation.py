import importlib.util
import os
import subprocess
import tempfile
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('finite_generation', ROOT / 'tools/e0_teal_llama3_author_full32_generation.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class Tests(unittest.TestCase):
    def test_import_has_no_scientific_libraries(self):
        self.assertFalse({'torch', 'triton', 'transformers'} & set(sys.modules))

    def test_histogram_paths_are_complete_and_unique(self):
        paths = m.hist_keyset()
        self.assertEqual(len(paths), 64)
        self.assertIn('author/histograms/layer-31/self_attn/histograms.pt', paths)
        self.assertNotIn('author/histograms/layer-32/mlp/histograms.pt', paths)

    def test_truezero_sets_all_224_thresholds(self):
        model = SimpleNamespace(layers=[SimpleNamespace(attention=SimpleNamespace(), feed_forward=SimpleNamespace()) for _ in range(32)])
        m.set_truezero(model)
        result = m.thresholds(model)
        self.assertEqual(len(result), 32)
        self.assertTrue(all(v == 0 for row in result for v in row['values'].values()))
        with self.assertRaises(Exception):
            m.set_truezero(SimpleNamespace(layers=model.layers[:-1]))

    def test_predebit_limit_enforced_before_persist(self):
        state = {'predebited': {'model_loads': 0}}
        calls = []
        m.budget_before(state, lambda: calls.append(state['predebited']['model_loads']), 'model_loads', 3)
        self.assertEqual(calls, [3])
        with self.assertRaises(Exception):
            m.budget_before(state, lambda: calls.append(4), 'model_loads')
        self.assertEqual(calls, [3])

    def test_missing_actual_parent_cannot_freeze(self):
        p = {'id':m.ID,'design':m.DESIGN,'resources':m.RESOURCES,'cpu_resources':m.CPU_RESOURCES,'limits':m.LIMITS,'output':m.OUT,'cpu_output':m.CPU,'source_bindings':[{'path':x} for x in m.SOURCES], 'author_sources':[{'path':x} for x in m.AUTHOR_FILES], 'parent_model':{'sha256':m.MODEL_SHA,'bytes':m.MODEL_SIZE},'parent_receipts':[1]*4, 'status':'FROZEN_PENDING_REGISTRATION','review':None,'worker_config_sha256':'a'*64,'histograms':None}
        m.validate(p)
        with self.assertRaises(Exception):
            m.validate(p, True)

    def test_actual_parent_exit_is_mandatory(self):
        for receipt in ({}, {'id':'E0_TEAL_LLAMA3_AUTHOR_GRAB_ACTS_20261030','process_exited':True,'child_returncode':1}):
            with self.assertRaises(Exception):
                m.check_hist_parent({}, receipt, 'a'*64)

    def test_cache_capacity_checks_every_layer(self):
        tensor = lambda shape: SimpleNamespace(shape=shape)
        cache = lambda n: SimpleNamespace(k_cache=tensor((1,8,n,128)), v_cache=tensor((1,8,n,128)))
        model = SimpleNamespace(max_seq_length=80,max_batch_size=1,causal_mask=tensor((80,80)),layers=[SimpleNamespace(attention=SimpleNamespace(kv_cache=cache(80))) for _ in range(32)])
        self.assertEqual(m.cache_identity(model, 9)['capacity'], 80)
        model.layers[31].attention.kv_cache = cache(72)
        with self.assertRaises(Exception):m.cache_identity(model, 9)

    def test_observer_rejects_duplicate_projection(self):
        rows=[{'layer':i,'projection':p} for i in range(32) for p in ('q','k','v','o','up','gate','down')]
        m.check_observer(rows)
        rows[-1]=rows[0]
        with self.assertRaises(Exception):m.check_observer(rows)

    def test_fixed_generation_plan_and_decode_budget(self):
        sequence=m.DESIGN['sequence']
        self.assertEqual(len(sequence),9)
        self.assertEqual({arm:sequence.count(arm) for arm in m.DESIGN['arms']},dict.fromkeys(m.DESIGN['arms'],3))
        self.assertEqual(m.LIMITS['generation_prefills'],len(sequence))
        self.assertEqual(m.LIMITS['generation_decode_calls'],len(sequence)*(m.DESIGN['max_new_tokens']-1))

    def test_exact_job_spec_isolated_and_parent_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'protocol.json';path.write_text('{}')
            cpu=m.job_spec(path,Path(tmp)/'cpu.json',Path(tmp)/'profile.json',True)
            gpu=m.job_spec(path,Path(tmp)/'gpu.json',Path(tmp)/'profile.json',False)
            self.assertIn('CUDA_VISIBLE_DEVICES=',cpu['argv'])
            self.assertIn('-I',cpu['argv']);self.assertIn('-I',gpu['argv'])
            self.assertIn('preflight',cpu['argv']);self.assertIn('run',gpu['argv'])
            self.assertEqual(cpu['external_timeout_seconds'],m.CPU_RESOURCES['outer_seconds'])
            self.assertEqual(gpu['external_timeout_seconds'],m.RESOURCES['outer_seconds'])
            args=SimpleNamespace(out=Path(tmp)/'frozen.json',worker_profile=Path(tmp)/'profile.json',hist_run=None,hist_exit=None,cpu_job_spec=Path(tmp)/'cpu.json',gpu_job_spec=Path(tmp)/'gpu.json')
            with self.assertRaises(Exception):m.prepare(args)
            self.assertFalse(args.out.exists())

    def test_author_cuda_conditional_import_without_gpu(self):
        code='''import sys, importlib.util, torch
torch.cuda.is_available=lambda:True
import transformers.utils as u
assert importlib.util.find_spec("flash_attn") is None
assert not u.is_flash_attn_2_available()
sys.path.insert(0,sys.argv[1])
import e0_teal_llama3_author_full32_generation as producer
producer.author_import()
'''
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
        subprocess.run([m.ENV,'-I','-B','-c',code,str(ROOT/'tools')],env=env,check=True,timeout=90)

if __name__ == '__main__':
    unittest.main()
