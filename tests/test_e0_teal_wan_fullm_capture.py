import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from e0_teal_wan_fullm_capture import CaptureTransaction, RAW_BYTES

class V:
    shape = (6, 4)

def test_transaction_preserves_gate_and_actual_grid():
    gate = object(); events = []
    c = CaptureTransaction(lambda x: ('copy', x), lambda p, g: events.append((p.copy(), g)), {(0, 0)}, (6, 4))
    assert c.begin(0, 0, 0, gate)
    v = V(); c.operand(0, 0, 0, v); c.finish(0, 0, 0, gate, (1, 2, 3)); c.verify_complete()
    assert events[0][0]['gate'] is gate and events[0][0]['h'] == ('copy', v)
    assert events[0][1] == (1, 2, 3) and RAW_BYTES == 875556864

def test_missing_operand_wrong_gate_or_grid_rejected():
    gate = object(); c = CaptureTransaction(lambda x: x, lambda *_: None, {(2, 20)}, (6, 4))
    c.begin(2, 20, 1, gate)
    with unittest.TestCase().assertRaises(ValueError): c.finish(2, 20, 1, gate, (1, 2, 3))
    c.operand(2, 20, 1, V())
    with unittest.TestCase().assertRaises(ValueError): c.finish(2, 20, 1, object(), (1, 2, 3))
    with unittest.TestCase().assertRaises(ValueError): c.finish(2, 20, 1, gate, (1, 2, 4))
    c.finish(2, 20, 1, gate, (1, 2, 3))
    with unittest.TestCase().assertRaises(ValueError): c.begin(2, 20, 1, gate)

if __name__ == "__main__":
    test_transaction_preserves_gate_and_actual_grid()
    test_missing_operand_wrong_gate_or_grid_rejected()

def test_draft_authorization_fails_before_access():
    from e0_teal_wan_fullm_capture import authorize, ID, RESOURCES
    class Forbidden:
        def read(self, *_): raise AssertionError('draft read ACTIVE')
    with unittest.TestCase().assertRaises(ValueError):
        authorize({'id': ID, 'resources': RESOURCES, 'assets_finalized':False,
                   'status':'DRAFT_NOT_REGISTERED'}, '0'*64, Forbidden(), Path('/unused'))

def test_commit_failure_does_not_mark_site_complete():
    def fail(*_): raise OSError('write failed')
    gate=object(); c=CaptureTransaction(lambda x:x, fail, {(0,0)}, (6,4))
    c.begin(0,0,0,gate); c.operand(0,0,0,V())
    with unittest.TestCase().assertRaises(OSError): c.finish(0,0,0,gate,(1,2,3))
    assert not c.seen and c.pending is not None
    with unittest.TestCase().assertRaises(ValueError): c.verify_complete()

if __name__ == '__main__':
    test_draft_authorization_fails_before_access()
    test_commit_failure_does_not_mark_site_complete()

def test_tap_delegates_original_and_records_exact_input():
    from types import SimpleNamespace
    from contextlib import ExitStack, contextmanager
    from e0_teal_wan_fullm_capture import make_tap_class
    class Native:
        @staticmethod
        def need(ok, msg):
            if not ok: raise ValueError(msg)
        @staticmethod
        @contextmanager
        def patch_method(owner, name, fn):
            old=getattr(owner,name); setattr(owner,name,fn)
            try: yield
            finally: setattr(owner,name,old)
    mm=object(); gate=object(); result=object(); posted=object(); captured=[]
    class Value:
        shape=(6,4); dtype='BF16'
        def stride(self): return (4,1)
    h=Value()
    class Infer:
        block_idx=0
        def infer_ffn(self, phase,x,attn,shift,scale,g):
            assert tap.mm_call(mm,h,{},lambda *_:result) is result
            return result
        def post_process(self,x,y,g,pre=None):
            assert y is result and g is gate
            return posted
    infer=Infer(); model=SimpleNamespace(transformer_infer=infer)
    class Parent:
        def __init__(self):
            self.runner=SimpleNamespace(model=SimpleNamespace(model=[model])); self.guard=SimpleNamespace(torch=SimpleNamespace(bfloat16='BF16'))
            self.mapping={id(mm):('block',0,'ffn.2')}; self.block=(0,0); self.stack=ExitStack()
        def step(self,expert): return 0
        def install(self): pass
        def mm_call(self,mm,value,meta,original): return original(mm,value)
        def finish(self): pass
    tapclass=make_tap_class(SimpleNamespace(LightTap=Parent),Native)
    c=CaptureTransaction(lambda value:('snapshot',value),lambda p,g:captured.append((p.copy(),g)),{(0,0)},(6,4))
    tap=tapclass(capture=c); tap.install()
    assert infer.infer_ffn(None,None,None,None,None,gate) is result
    pre=SimpleNamespace(grid_sizes=SimpleNamespace(tuple=(1,2,3)))
    assert infer.post_process(None,result,gate,pre) is posted
    tap.finish(); tap.stack.close()
    assert captured[0][0]['h']==('snapshot',h) and captured[0][0]['h_stride']==(4,1)

if __name__ == '__main__': test_tap_delegates_original_and_records_exact_input()

def test_single_file_limit_restored_and_small_limit_enforced():
    import resource, tempfile
    from e0_teal_wan_fullm_capture import single_file_bound, SINGLE_FILE_LIMIT, RAW_BYTES
    assert SINGLE_FILE_LIMIT > RAW_BYTES + (1 << 20) and SINGLE_FILE_LIMIT > 128 << 20
    before=resource.getrlimit(resource.RLIMIT_FSIZE)
    with tempfile.TemporaryDirectory() as directory:
        with single_file_bound(1024):
            with open(Path(directory)/'small','wb',buffering=0) as stream:
                assert stream.write(b'ok') == 2
            assert (Path(directory)/'small').read_bytes() == b'ok'
            with open(Path(directory)/'partial','wb',buffering=0) as stream:
                assert stream.write(b'x'*2048)==1024
            assert (Path(directory)/'partial').stat().st_size <= 1024
    assert resource.getrlimit(resource.RLIMIT_FSIZE)==before

if __name__ == '__main__': test_single_file_limit_restored_and_small_limit_enforced()

def test_job_contract_rejects_wrong_local_path():
    import hashlib, json, tempfile
    from types import SimpleNamespace
    from unittest.mock import patch
    from e0_teal_wan_fullm_capture import verify_job, OUTPUT, ID, CPU_RESOURCES
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory); hf=root/'hf'; proto=root/'p.json'; jobfile=root/'job.json'; local=root/'local'; profile=root/'profile'
        out=hf/OUTPUT; result=out.parent.parent/'cpu_preflight/worker/results'; digest='a'*64
        env=['env','FFNDS_WORKER_CONFIG='+str(profile),'OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','PATH=/usr/bin:/bin']
        child=['taskset','-c','0,1,2,3',str(root/'.venv/bin/python'),'-I','-B',str(root/'tools/e0_teal_wan_fullm_capture.py')]
        tail=['--protocol',str(proto),'--hf-root',str(hf),'--local-root',str(local),'--job-spec',str(jobfile)]
        job={'id':ID,'protocol_sha256':digest,'cwd':str(root),'hf_root':str(hf),'protocol_path':str(proto),
             'output_dir':str(result.parent),'run_json':str(result/'RUN.json'),'external_timeout_seconds':CPU_RESOURCES['outer_seconds'],
             'argv':env+['CUDA_VISIBLE_DEVICES=']+child+['preflight','--out',str(result)]+tail}
        jobfile.write_text(json.dumps(job)); scope={'cpu_preflight_spec_sha256':hashlib.sha256(jobfile.read_bytes()).hexdigest(),'cpu_affinity':[0,1,2,3]}
        args=SimpleNamespace(job_spec=jobfile,hf_root=hf,protocol=proto,local_root=local,out=result)
        with patch('os.sched_getaffinity',return_value={0,1,2,3}):
            verify_job(args,{},digest,scope,root,profile,'cpu')
            args.local_root=root/'different'
            with unittest.TestCase().assertRaises(ValueError): verify_job(args,{},digest,scope,root,profile,'cpu')

if __name__ == '__main__': test_job_contract_rejects_wrong_local_path()

def test_failed_serializer_keeps_and_accounts_partial():
    import tempfile
    from e0_teal_wan_fullm_capture import save_reserved_packet
    class Counter:
        def __init__(self): self.events=[]
        def reserve_payload(self,n): self.events.append(('reserve',n))
        def observe_external(self,path): self.events.append(('observe',path.read_bytes()))
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'partial'; counter=Counter()
        def fail():
            path.write_bytes(b'partial')
            raise OSError('serializer failed')
        with unittest.TestCase().assertRaises(OSError): save_reserved_packet(counter,path,7,fail)
        assert counter.events==[('reserve',7),('observe',b'partial')] and path.exists()
        class BrokenCounter(Counter):
            def observe_external(self,path): raise OSError('hash failed')
        with unittest.TestCase().assertRaises(OSError): save_reserved_packet(BrokenCounter(),path,7,lambda:None)
        assert path.read_bytes()==b'partial'

if __name__ == '__main__': test_failed_serializer_keeps_and_accounts_partial()

def test_asset_read_failure_preserves_predebits_and_unknown():
    import copy
    from e0_teal_wan_fullm_capture import verify_static_assets
    class Native:
        def bound(self,*_): raise OSError('source hash interrupted')
    plan={'repo_sources':[{'bytes':17}], 'local_package_inventory_payload_bytes':40,
          'local_transformers':{'inventory':{'bytes':3}}, 'model_view_metadata':[], 'cases':[], 'original_dense_references':[]}
    report={}; checkpoints=[]
    with unittest.TestCase().assertRaises(OSError):
        verify_static_assets(plan,Native(),Path('/x'),Path('/x'),Path('/x'),report,lambda:checkpoints.append(copy.deepcopy(report)))
    assert checkpoints and report['static_asset_reads']['source_files']=={'predebited_bytes':17,'completed_bytes':'UNKNOWN'}
    assert report['static_asset_reads']['package_files']=={'predebited_bytes':40,'completed_bytes':'UNKNOWN'}
    assert report['static_asset_reads']['package_inventory']['predebited_bytes']==12

if __name__ == '__main__': test_asset_read_failure_preserves_predebits_and_unknown()

def test_isolated_python_imports_without_numeric_modules():
    import subprocess
    root=Path(__file__).resolve().parents[1]
    code="import sys,importlib.util; sys.path.insert(0,sys.argv[1]); import e0_teal_wan_fullm_capture; spec=importlib.util.spec_from_file_location('v3',sys.argv[2]); p=importlib.util.module_from_spec(spec); spec.loader.exec_module(p); assert 'torch' not in sys.modules and 'numpy' not in sys.modules"
    result=subprocess.run([str(root/'.venv/bin/python'),'-I','-B','-c',code,str(root/'tools'),str(root/'tools/e0_teal_wan_all160_video_v3.py')],capture_output=True,text=True,check=False)
    assert result.returncode==0, result.stderr

if __name__ == '__main__': test_isolated_python_imports_without_numeric_modules()
