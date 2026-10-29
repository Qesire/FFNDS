import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('sm120_fixed_test',ROOT/'tools/e0_s2_new12_sm120_fixed_depth_videos.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class SM120PlanTests(unittest.TestCase):
    def test_shard_partition_and_independent_runids(self):
        ordinals=[n for shard in m.SHARDS for n in shard]
        self.assertEqual(len(ordinals),12);self.assertEqual(set(ordinals),set(m.acquisition.ORDINALS))
        for part in range(4):
            case={'native_source_ordinal':898,'index':0}
            a=m.run_id({**case,'_control_runtime':'original'},'dense')
            b=m.run_id({**case,'_control_runtime':'fixed'},'dense')
            self.assertNotEqual(a,b)
            self.assertNotEqual(a,m.run_id(case,'dense'))
            self.assertEqual(m.order(case)[0],'dense')
            self.assertEqual(len(set(m.order(case))),4)

    def test_work_counts_and_prefetch_are_actual_distinct_quantities(self):
        records=[m.expected('dense')]*2+[m.expected(mode) for _ in range(3) for mode in m.MODES]
        limits=m.limits(0)
        self.assertEqual(sum(r['factor_GEMMs'] for r in records),2400)
        self.assertEqual(sum(r['original_W2_skips'] for r in records),1212)
        self.assertEqual(sum(r['original_W2_skips']+r['original_W2_prefetches'] for r in records),2268)
        self.assertEqual(sum(r['dense_compute_sites'] for r in records),1040)
        self.assertEqual(limits['raw_Default_MM_loads'],812)
        self.assertEqual(limits['buffer_Default_MM_loads'],40)
        self.assertEqual(limits['BF16_factor_GEMM'],2400)
        self.assertEqual(limits['original_W2_H2D_skips'],1212)
        self.assertEqual(limits['native_FFN_calls'],2240)
        self.assertEqual((m.expected('value_early10_dense')['original_W2_prefetches'],
                          m.expected('value_late10_dense')['original_W2_prefetches']),(42,40))
        self.assertEqual(limits['video_generations'],14)
        self.assertEqual(limits['core_CPU_reports'],9)
        self.assertEqual(limits['matched_reference_noise_byte_checks'],18)

    def test_two_controls_must_match_all_ten_states_and_video(self):
        state={'sha256':'a'*64,'raw_tensor_sha256':'b'*64,'bytes':8,'shape':[2],'dtype':'bfloat16'}
        old={'run_id':'a','native_source_ordinal':898,'mode':'dense','seed':42,
             'states':{str(i):copy.deepcopy(state) for i in range(10)},
             'video':{'sha256':'c'*64,'bytes':55,'frames':81,'fps':16,'size':[832,464]}}
        new=copy.deepcopy(old);new['run_id']='b'
        class Ledger:
            def __init__(self):self.calls=[]
            def call(self,rid,cost,fn):self.calls.append((rid,cost));return fn()
        ledger=Ledger();m.validate_neutral(old,new,ledger);self.assertEqual(len(ledger.calls),11)
        for field in ('state','video','id'):
            bad=copy.deepcopy(new)
            if field=='state':bad['states']['3']['raw_tensor_sha256']='d'*64
            if field=='video':bad['video']['sha256']='e'*64
            if field=='id':bad['run_id']='a'
            with self.assertRaises(ValueError):m.validate_neutral(old,bad,Ledger())

    def test_same_source_generated_dense_first_and_registration(self):
        sources=[m.native.read(ROOT/m.SOURCE_CONFIG.format(i)) for i in (0,1)]
        for part in range(4):
            plan={'id':m.scope_id(part),'cases':m.case_selection(sources,part),
                  'source_cohort':sources[0]['source_bank']}
            manifest=m.registration(plan,Path('/tmp/hf'))
            self.assertEqual(len(manifest['samples']),12)
            self.assertEqual(manifest['required_variants'],list(m.MODES))
            for case in plan['cases']:
                jobs=[x for x in manifest['samples'] if x['source_id']==case['source_id']]
                self.assertEqual(jobs[0]['variant'],'dense')
                self.assertEqual({j['variant'] for j in jobs},set(m.MODES))
                self.assertEqual(len({j['run_id'] for j in jobs}),4)


    def test_draft_and_frozen_validation_with_mock_closed_master_locator(self):
        from unittest.mock import patch
        first=m.native.read(ROOT/m.SOURCE_CONFIG.format(0))
        worker={'models':first['models'], 'private_worker_config_sha256':'a'*64,
                'repo_sources':[], 'output_HF_relative':'ffnds-derived/mock/worker/results'}
        real_read=m.native.read;real_bound=m.native.bound;real_binding=m.native.binding
        hf=Path('/tmp/sm120-synthetic-hf')
        def fake_read(path):
            if Path(path)==ROOT/m.MASTER:return copy.deepcopy(worker)
            return real_read(path)
        def fake_bound(base,item):
            if item['path']==m.MASTER:return ROOT/m.MASTER
            return real_bound(base,item)
        def fake_binding(path,base):
            path=Path(path)
            if not path.exists():
                return {'path':str(path.relative_to(base)), 'bytes':1, 'sha256':'b'*64}
            return real_binding(path,base)
        with patch.object(m.native,'read',side_effect=fake_read), patch.object(m.native,'bound',side_effect=fake_bound), patch.object(m.native,'binding',side_effect=fake_binding):
            for part in range(4):
                plan=m.draft(part,hf)
                m.validate(plan)
                self.assertEqual(len(plan['run_ids']),14)
                frozen=copy.deepcopy(plan);frozen.update(status='FROZEN_REGISTERED',assets_finalized=True)
                m.validate(frozen,True)
                bad=copy.deepcopy(frozen);bad['repo_sources'].pop()
                with self.assertRaises(ValueError):m.validate(bad,True)
                bad=copy.deepcopy(frozen);bad['cases'][0]['image']['sha256']='0'*64
                with self.assertRaises(ValueError):m.validate(bad,True)

    def test_original_and_fixed_transfer_policy_selection(self):
        saved=m.fixed._original_geometry
        try:
            m.fixed._original_geometry=lambda b,d,i,r:r.mode
            old=SimpleNamespace(mode='dense',objects={})
            self.assertEqual(m.transfer_geometry(None,None,9,old),'dense')
            self.assertEqual(m.transfer_geometry(None,None,10,old),'dense')
            fixed=m.FixedLayerW2Runtime.__new__(m.FixedLayerW2Runtime)
            fixed.objects={};fixed.is_factor=lambda i:i>=10
            self.assertEqual(m.transfer_geometry(None,None,9,fixed),'dense')
            self.assertEqual(m.transfer_geometry(None,None,10,fixed),'factor')
        finally:m.fixed._original_geometry=saved

if __name__=='__main__':unittest.main()
