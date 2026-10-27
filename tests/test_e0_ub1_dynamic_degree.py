import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
p=Path(__file__).resolve().parents[1]/'tools/e0_ub1_dynamic_degree.py';spec=importlib.util.spec_from_file_location('ub1',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class InstrumentationTests(unittest.TestCase):
    def test_forward_delegation_and_precharge_limit(self):
        seen=[]; sentinel=object();counts={'forwards':159};im=SimpleNamespace(shape=(1,3,464,832))
        def original(*a,**kw):seen.append((a,kw));return sentinel
        wrapped=m.counted_forward(original,counts)
        self.assertIs(wrapped('model',im,im,iters=20,test_mode=True),sentinel)
        self.assertEqual(seen[0][1],{'iters':20,'test_mode':True})
        with self.assertRaises(RuntimeError):wrapped('model',im,im,iters=20,test_mode=True)
        self.assertEqual(len(seen),1)
    def test_score_preserves_return(self):
        scores=[];value=1.25;wrapped=m.observed_score(lambda *a,**k:value,scores)
        self.assertIs(wrapped('self','frame','flow'),value);self.assertEqual(scores,[1.25])

class PreflightTests(unittest.TestCase):
    def test_pending_and_wrong_authorization_rejected(self):
        plan={'id':m.ID,'limits':dict(m.LIMITS),'helper_sha256':m.PENDING,'v8_protocol_sha256':'a'*64,'v8_run_sha256':'b'*64}
        active={'experiment_authority':{'current_user_authorized_scopes':[]}}
        with self.assertRaises(ValueError):m.authorize(plan,'test',active)
        plan['helper_sha256']='c'*64
        with self.assertRaises(ValueError):m.authorize(plan,'test',active)
    def test_invalid_asset_rejected_before_model(self):
        import tempfile,json,hashlib
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad';p.write_text('bad')
            plan={'tool':'missing','test':'missing','helper':'missing','v8_protocol':'missing','tool_sha256':'0'*64}
            with self.assertRaises((ValueError,FileNotFoundError)):m.verify_preflight(plan,Path(d))

class MonitorAuditTests(unittest.TestCase):
    def test_trusted_exact_query_and_untrusted_denied(self):
        from unittest.mock import patch
        audit=None
        def trusted():return audit('subprocess.Popen',('nvidia-smi',['nvidia-smi','-i','2','--query-gpu=memory.used','--format=csv,noheader,nounits'],None,None))
        audit=m.install_audit(Path('/tmp/scope'),Path('/tmp/model.pth'),trusted.__code__,Path('/usr/bin/nvidia-smi'),'2')
        with patch.object(m.shutil,'which',return_value='/usr/bin/nvidia-smi'):
            trusted()
            with self.assertRaises(RuntimeError):audit('subprocess.Popen',('nvidia-smi',['nvidia-smi','-i','2','--query-gpu=memory.used','--format=csv,noheader,nounits'],None,None))
            with self.assertRaises(RuntimeError):audit('subprocess.Popen',('nvidia-smi',['nvidia-smi','-i','2','--query-gpu=uuid','--format=csv,noheader,nounits'],None,None))
    def test_failed_forward_is_charged(self):
        counts={'forwards':0,'iterations':0};im=SimpleNamespace(shape=(1,3,464,832))
        def fail(*a,**kw):raise RuntimeError('forward failed')
        with self.assertRaises(RuntimeError):m.counted_forward(fail,counts)(None,im,im,iters=20,test_mode=True)
        self.assertEqual(counts,{'forwards':1,'iterations':20})

class VideoAccountingTests(unittest.TestCase):
    def test_video_charged_before_failure_and_order_enforced(self):
        counts={'videos':0}
        def clip():
            m.charge_video(counts,['a','b'],'a')
            raise RuntimeError('decode failure')
        with self.assertRaises(RuntimeError):clip()
        self.assertEqual(counts['videos'],1)
        with self.assertRaises(RuntimeError):m.charge_video(counts,['a','b'],'a')
        self.assertEqual(counts['videos'],1)
