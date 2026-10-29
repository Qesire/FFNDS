import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('new12_visual_test',ROOT/'tools/e0_s2_new12_dense_visual_cpu.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class VisualTests(unittest.TestCase):
    def test_all_positions_unique_continuous_and_explicit_blanks(self):
        pages=[m.page_positions(0),m.page_positions(1)]
        self.assertEqual([n for page in pages for n in page if n is not None],list(range(81)))
        self.assertEqual(pages[1][-3:],[None,None,None])
        self.assertEqual([len(x) for x in pages],[42,42])
        with self.assertRaises(ValueError):m.page_positions(2)

    def test_short_clip_preserves_actual_budget(self):
        class Reader:
            def __init__(self,n):self.remaining=n
            def read(self):
                if self.remaining<=0:return False,None
                self.remaining-=1;return True,object()
        class Renderer:
            def add(self,frame,ordinal,index):pass
            def finish(self,page):return page
        report={};ledger=m.Ledger(report);pages=[]
        with self.assertRaises(ValueError):
            m.extract_one(Reader(43),Renderer(),{'native_source_ordinal':911},ledger,
                          lambda image,page,positions:pages.append((page,positions)))
        self.assertEqual(report['predebited']['successful_decoded_frames'],44)
        self.assertEqual(report['invoked']['successful_decoded_frames'],44)
        self.assertEqual(report['completed']['successful_decoded_frames'],43)
        self.assertEqual(report['completed']['JPEG_pages'],1)
        self.assertEqual(pages,[(0,list(range(42)))])

    def test_exact_frames_and_identity_each_page(self):
        class Reader:
            def __init__(self):self.i=0
            def read(self):
                i=self.i;self.i+=1
                return (True,i) if i<81 else (False,None)
        class Renderer:
            def __init__(self):self.seen=[]
            def add(self,frame,ordinal,index):self.seen.append((frame,ordinal,index))
            def finish(self,page):return page
        report={};ledger=m.Ledger(report);renderer=Renderer();pages=[]
        m.extract_one(Reader(),renderer,{'native_source_ordinal':979},ledger,
                      lambda image,page,positions:pages.append((page,positions)))
        self.assertEqual(renderer.seen,[(i,979,i) for i in range(81)])
        self.assertEqual(pages,[(0,m.page_positions(0)),(1,m.page_positions(1))])
        self.assertEqual(report['completed']['successful_decoded_frames'],81)
        self.assertEqual(report['completed']['JPEG_pages'],2)
        self.assertEqual(report['completed']['eof_checks'],1)

    def test_extra_frame_aborts_before_page1(self):
        class Reader:
            def read(self):return True,object()
        class Renderer:
            def add(self,frame,ordinal,index):pass
            def finish(self,page):return page
        report={};ledger=m.Ledger(report);pages=[]
        with self.assertRaises(ValueError):
            m.extract_one(Reader(),Renderer(),{'native_source_ordinal':911},ledger,
                          lambda image,page,positions:pages.append(page))
        self.assertEqual(pages,[0]);self.assertEqual(report['completed']['successful_decoded_frames'],81)
        self.assertEqual(report['completed']['eof_checks'],1)

    def test_exact_active_registration(self):
        scope={'id':m.ID,'protocol_sha256':'abc','execution_status':'AUTHORIZED_NOT_STARTED',
               'resources':copy.deepcopy(m.RESOURCES),'limits':copy.deepcopy(m.LIMITS),'max_gpu_cards':0}
        active={'user_continuation_authorization_20261027':{'new_scope_registration_allowed':True},
                'experiment_authority':{'current_user_authorized_scopes':[scope]}}
        with patch.object(m,'validate'):
            self.assertEqual(m.authorize({},'abc',active),scope)
            wrong=copy.deepcopy(active);wrong['experiment_authority']['current_user_authorized_scopes'][0]['max_gpu_cards']=1
            with self.assertRaises(ValueError):m.authorize({},'abc',wrong)


    def test_rendered_frame_label_and_empty_cells(self):
        try:
            import numpy as np
        except ImportError:self.skipTest('NumPy unavailable')
        class CV:
            INTER_AREA=3;FONT_HERSHEY_SIMPLEX=0;LINE_AA=16
            def __init__(self):self.labels=[]
            def resize(self,frame,shape,interpolation):return np.full((178,320,3),17,dtype=np.uint8)
            def putText(self,image,text,*args):self.labels.append(text)
        cv=CV();renderer=m.Sheets(cv,np);frame=np.zeros((464,832,3),dtype=np.uint8)
        renderer.add(frame,911,42);renderer.add(frame,911,80);image=renderer.finish(1)
        self.assertEqual(image.shape,(1414,1920,3))
        self.assertIn('source 911  frame 42  2.625s',cv.labels)
        self.assertIn('source 911  frame 80  5.000s',cv.labels)
        self.assertEqual(cv.labels.count('EMPTY'),3)

    def test_actual_closed_parent_assets(self):
        hf=Path('/home/xiang/.cache/huggingface')
        if not (hf/'ffnds-derived/e0_s2_new12_dense_motion_metrics_20261029/shard0/attempt1/worker/results/RUN.json').exists():
            self.skipTest('Actual parent not available')
        report=m.base.read(ROOT/m.REPORT);m.verify_parent(report,hf)
        plan=m.draft(hf);m.validate(plan)
        self.assertEqual(len(plan['sources']),12)
        bad=copy.deepcopy(plan);bad['sources'][0]['video']['sha256']='0'*64
        with self.assertRaises(ValueError):m.validate(bad)
        bad=copy.deepcopy(report);bad['sources'][0]['prompt']='changed'
        with self.assertRaises(ValueError):m.verify_parent(bad,hf)

if __name__=='__main__':unittest.main()
