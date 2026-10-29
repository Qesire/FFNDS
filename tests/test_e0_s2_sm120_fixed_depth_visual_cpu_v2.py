import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('sm120_visual_test',ROOT/'tools/e0_s2_sm120_fixed_depth_visual_cpu_v2.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class Reader:
    def __init__(self,n):self.n=n
    def read(self):
        if self.n:
            self.n-=1;return True,object()
        return False,None


class Renderer:
    def __init__(self):self.seen=[]
    def add(self,frame,source,index):self.seen.append((source['ordinal'],source['mode'],index))
    def finish(self,page):return 'page'+str(page)


class SM120VisualTests(unittest.TestCase):
    def test_all_81_frames_and_both_pages_per_arm(self):
        source={'ordinal':911,'mode':'value_early10_dense'};report={};tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);ledger=m.JournalLedger(report,Path(tmp.name)/'j');self.addCleanup(ledger.close);renderer=Renderer();pages=[]
        m.extract(Reader(81),renderer,source,ledger,lambda image,p,positions:pages.append((image,p,positions)))
        self.assertEqual(renderer.seen,[(911,'value_early10_dense',i) for i in range(81)])
        self.assertEqual([p[1] for p in pages],[0,1]);self.assertEqual(pages[0][2],list(range(42)))
        self.assertEqual(pages[1][2],list(range(42,81))+[None]*3)
        self.assertEqual(report['completed']['successful_decoded_frames'],81)
        self.assertEqual(report['completed']['eof_checks'],1);self.assertEqual(report['completed']['JPEG_pages'],2)

    def test_short_and_extra_fail_without_completed_coverage(self):
        source={'ordinal':911,'mode':'dense'}
        for count in (17,82):
            report={};tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);ledger=m.JournalLedger(report,Path(tmp.name)/'j');self.addCleanup(ledger.close)
            with self.assertRaises(ValueError):m.extract(Reader(count),Renderer(),source,ledger,lambda *_:None)
            self.assertLess(report['completed']['JPEG_pages'],2)

    def test_source_arm_identity_and_fixed_shards(self):
        self.assertEqual(len({o for p in m.motion.SHARDS for o in p}),12)
        self.assertEqual(len({m.output(i) for i in range(4)}),4)
        with self.assertRaises(ValueError):m.identity(4)
        self.assertEqual(m.LIMITS['videos'],12);self.assertEqual(m.LIMITS['successful_decoded_frames'],12*81)
        self.assertEqual(m.LIMITS['JPEG_pages'],24);self.assertEqual(m.LIMITS['video_encodes'],0)

    def test_input_hash_and_later_stat_change(self):
        with tempfile.TemporaryDirectory() as d:
            hf=Path(d);(hf/'video').write_bytes(b'video');b=m.base.binding(hf/'video',hf)
            with mock.patch.object(m,'inputs',return_value=[('HF',b)]):
                rows=m.capture_inputs({},hf);m.check_stats(rows,hf)
                (hf/'video').write_bytes(b'changed')
                with self.assertRaises(ValueError):m.check_stats(rows,hf)
                with self.assertRaises(ValueError):m.capture_inputs({},hf)

    def test_parent_must_close_and_match_frozen_rows(self):
        plan={'parent':{'id':'p'},'part':0,'videos':[{'ordinal':911,'mode':'dense'}]}
        with mock.patch.object(m,'parent_rows',return_value=plan['videos']) as parent:
            m.verify_parent(plan,Path('/tmp'));parent.assert_called_once()
        with mock.patch.object(m,'parent_rows',return_value=[{'ordinal':911,'mode':'other'}]):
            with self.assertRaises(ValueError):m.verify_parent(plan,Path('/tmp'))
        with mock.patch.object(m,'parent_rows',side_effect=ValueError('EXIT not closed')):
            with self.assertRaises(ValueError):m.verify_parent(plan,Path('/tmp'))

    def test_distinct_runs_may_have_equal_video_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            hf=Path(d);cases=[];phases=[{'run_id':'control1'},{'run_id':'control2'}];gens={}
            for o in m.motion.SHARDS[0]:
                c={'native_source_ordinal':o,'index':0,'seed':42,'source_id':str(o),
                   'image':{'sha256':str(o)},'prompt_sha256':'p'};cases.append(c)
                for mode in m.motion.video_parent.order(c):
                    name=f'{o}_{mode}';(hf/name).write_bytes(b'same video')
                    p={'id':'parent','protocol_sha256':'digest','numeric_path':'BF16_ORIGINAL','seed':42,
                       'native_source_ordinal':o,'source_id':str(o),'image_sha256':str(o),'prompt_sha256':'p',
                       'run_id':name,'mode':mode,'generation_receipt':{'path':'gen_'+name},'states':{},
                       'video':{'path':str(hf/name),'frames':81,'size':[832,464],'fps':16}}
                    phases.append(p);gens['gen_'+name]={k:p[k] for k in ('states','video','run_id','mode')}
            parent={'id':'parent','part':0,'cases':cases,'run_ids':[p['run_id'] for p in phases]}
            run={'id':'parent','part':0,'protocol_sha256':'digest','phases':phases}
            proof={'protocol':{'path':m.motion.PARENTS.format(0),'sha256':'digest'},'RUN':{'path':'run'},'exit':{'path':'exit'}}
            def read_bound(root,b):
                return parent if b['path']==proof['protocol']['path'] else run if b['path']=='run' else {} if b['path']=='exit' else gens[b['path']]
            with mock.patch.object(m.base,'read_bound',side_effect=read_bound),mock.patch.object(m.base,'exited'), \
                 mock.patch.object(m.motion.video_parent,'verify_closed'), \
                 mock.patch.object(m.base,'clip_binding',side_effect=lambda v,h:{'path':Path(v['path']).name}), \
                 mock.patch.object(m.base,'bound',side_effect=lambda root,b:hf/b['path']):
                rows=m.parent_rows(proof,hf,0);self.assertEqual(len(rows),12)

    def test_input_collision_includes_source_generation_and_states(self):
        def b(path,sha='a'):return {'path':path,'bytes':1,'sha256':sha*64}
        row={'video':b('video'),'generation':b('gen'),'case':{'image':b('image')},'states':{'x':b('state')}}
        plan={'repo_sources':[],'parent':{'protocol':b('protocol'),'RUN':b('run'),'exit':b('exit')},'videos':[row],'v1_failure':{'protocol':b('v1protocol'),'RUN':b('v1run'),'exit':b('v1exit')}}
        self.assertEqual(len(m.inputs(plan)),10)
        bad=copy.deepcopy(plan);bad['videos'][0]['states']['x']=b('video','b')
        with self.assertRaises(ValueError):m.inputs(bad)

    def test_failed_operation_journal_retains_debit_and_replays(self):
        with tempfile.TemporaryDirectory() as d:
            report={};path=Path(d)/'journal';ledger=m.JournalLedger(report,path)
            ledger.call('successful_decoded_frames',lambda:True)
            with self.assertRaises(RuntimeError):ledger.call('JPEG_pages',lambda:(_ for _ in ()).throw(RuntimeError('fail')))
            ledger.close();self.assertEqual(m.replay_journal(path),{k:report[k] for k in m.SIDES})
            self.assertEqual(report['predebited']['JPEG_pages'],1);self.assertEqual(report['completed']['JPEG_pages'],0)
            with path.open('a') as f:f.write('{"side":"completed","key":"JPEG_pages"}\n{"side":"completed","key":"JPEG_pages"}\n')
            with self.assertRaises(ValueError):m.replay_journal(path)

    def test_v2_resource_and_pinned_failure(self):
        self.assertEqual(m.RESOURCES['wall_seconds'],900);self.assertEqual(m.RESOURCES['new_HF_bytes'],96<<20)
        self.assertIn('v2',m.output(0));self.assertNotEqual(m.identity(0),'E0_S2_SM120_FIXED_DEPTH_ALL81_VISUAL_CPU_20261029_PART0')
        self.assertEqual(m.failure_expected()['RUN']['sha256'],m.V1_SHAS['RUN'])

    def test_torch_finder_rejects_import(self):
        with self.assertRaises(ImportError):m.old.NoTorch().find_spec('torch.nn')
        self.assertIsNone(m.old.NoTorch().find_spec('numpy'))

if __name__=='__main__':unittest.main()
