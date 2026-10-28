"""Real stdlib audit subprocesses; no Torch/metric imports/model/GPU."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import e0_compact6016_consumer_quality_v3 as q


CHILD=r'''
import ast,json,os,pathlib,sys,tempfile
root,scope,outside,mode=map(pathlib.Path,sys.argv[1:5])
mode=str(mode);sys.path.insert(0,str(root/'tools'))
import e0_compact6016_consumer_quality_v3 as q
spec=q.importlib.util.spec_from_file_location('v8_review',root/'tools/e0_official_metric_environment_v8.py')
helper=q.importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
tmp=scope/'tmp';tmp.mkdir();tempfile.tempdir=str(tmp)
result={'denied_attempts':[]};denied,original,_=helper.import_audit_guard(scope,result)
audit=original if mode=='legacy_probe' else q.normalize_dir_fd_audit(original,scope,result,denied)
fd=None
if mode in ('outside_fd','file_fd','escape','closed_fd','symlink'):
    if mode=='outside_fd':fd=os.open(outside,os.O_RDONLY|os.O_DIRECTORY)
    elif mode=='file_fd':fd=os.open(scope/'ordinary',os.O_RDONLY)
    else:fd=os.open(scope,os.O_RDONLY|os.O_DIRECTORY)
    if mode=='closed_fd':os.close(fd)
source=root/'tools/e0_official_metric_environment_v8.py'
if mode in ('legacy_probe','probe'):
    probe_file=pathlib.Path(sys.argv[5]) if len(sys.argv)>5 else None
    if probe_file:
        tree=ast.parse(probe_file.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_probe_link_follow_symlinks')
        ns={'tempfile':tempfile,'Path':pathlib.Path,'os':os,'_HAS_LINK':hasattr(os,'link')}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(probe_file),'exec'),ns)
        probe=ns['_probe_link_follow_symlinks']
    else:
        def probe():
            with tempfile.TemporaryDirectory() as d:
                item=pathlib.Path(d,'probe-source');item.touch();os.link(item,pathlib.Path(d,'probe-link'),follow_symlinks=False)
            return True
sys.addaudithook(audit)
try:
    if mode in ('legacy_probe','probe'):
        assert probe() is True
        with tempfile.TemporaryDirectory() as d:
            child=pathlib.Path(d,'nested');child.mkdir();(child/'file').write_text('tiny')
    elif mode=='outside_fd':os.unlink('victim',dir_fd=fd)
    elif mode=='escape':os.unlink('../outside/victim',dir_fd=fd)
    elif mode=='closed_fd':os.unlink('ordinary',dir_fd=fd)
    elif mode=='file_fd':os.unlink('ordinary',dir_fd=fd)
    elif mode=='symlink':os.unlink('escape_link',dir_fd=fd)
    elif mode=='checkpoint':(scope/'weights.safetensors').read_bytes()
    elif mode=='network':
        import socket;socket.socket()
    else:raise AssertionError(mode)
    result['outcome']='PASS'
except RuntimeError as error:result.update(outcome='DENIED',error=str(error))
finally:
    if fd is not None and mode!='closed_fd':os.close(fd)
result.update(torch_imported='torch' in sys.modules,numpy_imported='numpy' in sys.modules,
              outside_file_untouched=(outside/'victim').is_file())
print(json.dumps(result))
'''


class RealAuditControls(unittest.TestCase):
    def child(self,mode,*,probe=None):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp);scope=parent/'scope';outside=parent/'outside';cwd=parent/'cwd'
            for d in (scope,outside,cwd):d.mkdir()
            (scope/'ordinary').write_text('inside');(outside/'victim').write_text('outside')
            (scope/'weights.safetensors').write_bytes(b'not-a-model')
            (scope/'escape_link').symlink_to(outside/'victim')
            args=[sys.executable,'-I','-S','-c',CHILD,str(ROOT),str(scope),str(outside),mode]
            if probe:args.append(str(probe))
            run=subprocess.run(args,cwd=cwd,capture_output=True,text=True,timeout=20)
            self.assertEqual(run.returncode,0,run.stderr)
            result=json.loads(run.stdout)
            self.assertFalse(result['torch_imported']);self.assertFalse(result['numpy_imported'])
            self.assertTrue((outside/'victim').is_file())
            return result

    def test_real_TemporaryDirectory_dirfd_cleanup_regresses_old_v8_then_passes_v3(self):
        old=self.child('legacy_probe');self.assertEqual(old['outcome'],'DENIED')
        self.assertIn('FILESYSTEM_MUTATION_OUTSIDE_SCOPE',old['error'])
        actual=self.child('probe');self.assertEqual(actual['outcome'],'PASS')
        self.assertEqual(actual['denied_attempts'],[])
        counter=actual['dir_fd_normalization'];self.assertGreaterEqual(counter['normalized_events'],4)
        self.assertEqual({e['event'] for e in counter['events']},{'os.remove','os.rmdir'})

    def test_actual_pinned_filelock_probe_function_runs_without_importing_filelock(self):
        path=Path('/home/xiang/.cache/huggingface/derived/ffnds/E0_official_metric_environment_v2/venv/lib/python3.11/site-packages/filelock/_strict.py')
        if not path.is_file():self.skipTest('Registered filelock source unavailable; portable stdlib regression is separate')
        result=self.child('probe',probe=path)
        self.assertEqual(result['outcome'],'PASS');self.assertEqual(result['denied_attempts'],[])
        self.assertGreater(result['dir_fd_normalization']['normalized_events'],0)

    def test_outside_directory_fd_is_rejected_before_real_unlink(self):
        result=self.child('outside_fd');self.assertEqual(result['outcome'],'DENIED')
        self.assertIn('FD_OUTSIDE_SCOPE',result['error']);self.assertTrue(result['outside_file_untouched'])

    def test_parent_escape_is_rejected_before_real_unlink(self):
        result=self.child('escape');self.assertEqual(result['outcome'],'DENIED')
        self.assertIn('TARGET_OUTSIDE_SCOPE',result['error'])

    def test_closed_and_nondirectory_fds_are_rejected(self):
        for mode in ('closed_fd','file_fd'):
            with self.subTest(mode=mode):
                result=self.child(mode);self.assertEqual(result['outcome'],'DENIED')
                self.assertIn('DIR_FD_AUDIT:',result['error'])

    def test_original_final_symlink_target_policy_is_preserved(self):
        result=self.child('symlink');self.assertEqual(result['outcome'],'DENIED')
        self.assertIn('TARGET_OUTSIDE_SCOPE',result['error'])

    def test_checkpoint_and_network_original_guards_remain_live(self):
        for mode,reason in (('checkpoint','CHECKPOINT_FILE_READ'),('network','NETWORK_OR_SUBPROCESS')):
            with self.subTest(mode=mode):
                result=self.child(mode);self.assertEqual(result['outcome'],'DENIED');self.assertIn(reason,result['error'])


class ContractControls(unittest.TestCase):
    def test_only_two_real_FD_events_are_rewritten_and_original_tuple_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            scope=Path(tmp);(scope/'value').write_text('x');fd=os.open(scope,os.O_RDONLY|os.O_DIRECTORY)
            seen=[];result={};denied=lambda why:lambda:(_ for _ in ()).throw(RuntimeError(why))
            audit=q.normalize_dir_fd_audit(lambda e,v:seen.append((e,v)),scope,result,denied)
            try:
                values=('value',fd);audit('os.remove',values)
                self.assertEqual(values,('value',fd));self.assertEqual(seen[-1],('os.remove',(str(scope/'value'),fd)))
                for event,values in (('open',('x','r',0)),('os.rename',('a','b',fd,fd)),('os.remove',('x',-1)),('os.rmdir',('x',None))):
                    audit(event,values);self.assertIs(seen[-1][1],values)
                with self.assertRaisesRegex(RuntimeError,'INVALID_ARGUMENTS'):audit('os.remove',('value',True))
            finally:os.close(fd)

    def test_native_common_guard_is_delegated_not_replaced(self):
        import e0_ub1_consumer_official_quality as common
        with tempfile.TemporaryDirectory() as tmp:
            scope=Path(tmp);fd=os.open(scope,os.O_RDONLY|os.O_DIRECTORY);result={}
            old=common.audit_scope(scope,None,Path('/not/a/binary'),'unused')
            denied=lambda why:lambda:(_ for _ in ()).throw(RuntimeError(why))
            audit=q.normalize_dir_fd_audit(old,scope,result,denied)
            try:
                audit('os.remove',('inside',fd))
                for event,args in (('socket.__new__',()),('subprocess.Popen',('other',['other'])),('open',(str(scope.parent/'bad'),'w',os.O_WRONLY))):
                    with self.subTest(event=event),self.assertRaises(RuntimeError):audit(event,args)
            finally:os.close(fd)

    def test_v2_v8_sources_and_six_video_scope_are_exact_and_unchanged(self):
        self.assertEqual(q.base.sha(q.BASE_PATH),q.BASE_SHA)
        self.assertEqual(q.base.sha(ROOT/'tools/e0_official_metric_environment_v8.py'),q.V8_SHA)
        self.assertEqual(q.base.LIMITS['new_video_dimension_results'],42)
        self.assertEqual(q.base.LIMITS['RAFT_forwards'],240);self.assertEqual(q.base.LIMITS['RAFT_iterations'],4800)
        self.assertEqual(q.base.IMPORT_LIMITS['official_module_imports'],8)
        self.assertNotEqual(q.ID,q.V2_ID)

    def test_import_proof_contains_new_adapter_and_failed_v2_identity(self):
        plan={'repo_sources':[],'import_environment':{},'directory_fd_audit_policy':q.POLICY,
              'failed_import_predecessor_v2':{'RUN':{'sha256':q.V2_RUN_SHA}}}
        proof=q.import_report(plan,'protocol')
        self.assertEqual(proof['id'],q.ID);self.assertEqual(proof['failed_import_v2_RUN_sha256'],q.V2_RUN_SHA)
        self.assertEqual(proof['directory_fd_audit_policy_sha256'],q.base.state_sha(q.POLICY))
        self.assertEqual(proof['numeric_path'],'HIF4_FAKE_QDQ_DIAGNOSTIC')

    def test_import_receipt_requires_actual_successful_normalization_events(self):
        plan={};scope={};proof={'RUN':{}}
        good={'dir_fd_normalization':{'policy':q.POLICY,'normalized_events':1,'events':[{}]}}
        with mock.patch.object(q,'_require_import_receipt',return_value=proof),mock.patch.object(q.base,'read_bound',return_value=good):
            self.assertEqual(q.require_import_receipt(scope,plan,'digest',Path('/tmp')),proof)
        for bad in ({},{'dir_fd_normalization':{'policy':q.POLICY,'normalized_events':0,'events':[]}}):
            with mock.patch.object(q,'_require_import_receipt',return_value=proof),mock.patch.object(q.base,'read_bound',return_value=bad),self.assertRaisesRegex(ValueError,'observations absent'):
                q.require_import_receipt(scope,plan,'digest',Path('/tmp'))

    def test_native_wrapper_persists_new_identity_and_restores_original_factory(self):
        import e0_ub1_consumer_official_quality as common
        with tempfile.TemporaryDirectory() as tmp:
            hf=Path(tmp);protocol=hf/'protocol.json'
            plan={'output_HF_relative':'results','directory_fd_audit_policy':q.POLICY,
                'failed_import_predecessor_v2':{'RUN':{'sha256':q.V2_RUN_SHA}}}
            protocol.write_text(json.dumps(plan));args=types.SimpleNamespace(protocol=protocol,hf_root=hf)
            original=common.audit_scope
            def fake_run(args):
                out=hf/'results';out.mkdir();fd=os.open(out,os.O_RDONLY|os.O_DIRECTORY)
                try:common.audit_scope(out,None,Path('/unused'),'unused')('os.remove',('inside',fd))
                finally:os.close(fd)
                record={'id':q.ID,'protocol_sha256':q.base.sha(protocol),'status':'SYNTHETIC_COMPLETE_NO_MODEL'}
                q.base.write(out/'RUN.json',record);return record
            with mock.patch.object(q,'_run',side_effect=fake_run):q.run(args)
            self.assertIs(common.audit_scope,original)
            result=q.base.read(hf/'results/RUN.json')
            self.assertEqual(result['native_dir_fd_audit']['dir_fd_normalization']['normalized_events'],1)
            self.assertEqual(result['numeric_path'],'HIF4_FAKE_QDQ_DIAGNOSTIC')
            with self.assertRaisesRegex(ValueError,'no failed-output overwrite'):q.run(args)

    def fixture_failure(self):
        run={'id':q.V2_ID,'status':'FAILED_IMPORTS_ONLY_PRESERVED_NO_RETRY','protocol_sha256':q.V2_PROTOCOL_SHA,
            'predebited_official_module_imports':1,'completed_official_module_imports':0,'functions':[],
            'CUDA_initialized':False,'denied_attempts':['FILESYSTEM_MUTATION_OUTSIDE_SCOPE'],
            'denied_attempt_details':[{'stack':[{'function':'_probe_link_follow_symlinks','file':'/site/filelock/_strict.py'},
                                              {'function':'_rmtree_safe_fd','file':'/stdlib/shutil.py'}]}]}
        run.update(dict.fromkeys(('GPU_calls','checkpoint_loads','model_constructions','parameter_creations','buffer_creations','model_forwards','official_scores'),0))
        exited={'id':q.V2_ID,'protocol_sha256':q.V2_PROTOCOL_SHA,'child_returncode':1,'process_exited':True,
                'timeout':False,'automatic_retry':False,'run_sha256':q.V2_RUN_SHA}
        plan={'failed_import_predecessor_v2':dict.fromkeys(('protocol','RUN','exit','stderr'),{})}
        return plan,run,exited

    def test_failed_v2_exact_exit_work_and_reason_are_retained(self):
        plan,run,exited=self.fixture_failure()
        with mock.patch.object(q.base,'read_bound',side_effect=[{'id':q.V2_ID},run,exited]),mock.patch.object(q.base,'bound'):
            q.verify_failed_import_v2(plan,Path('/tmp'))
        for field,value in (('official_scores',1),('completed_official_module_imports',1),('CUDA_initialized',True),('denied_attempts',[])):
            changed=copy.deepcopy(run);changed[field]=value
            with self.subTest(field=field),mock.patch.object(q.base,'read_bound',side_effect=[{'id':q.V2_ID},changed,exited]),mock.patch.object(q.base,'bound'),self.assertRaises(ValueError):
                q.verify_failed_import_v2(plan,Path('/tmp'))
        with mock.patch.object(q.base,'read_bound',side_effect=[{'id':q.V2_ID},run,dict(exited,child_returncode=0)]),mock.patch.object(q.base,'bound'),self.assertRaises(ValueError):
            q.verify_failed_import_v2(plan,Path('/tmp'))

    def test_wrapper_import_has_no_numeric_or_GPU_imports(self):
        command="import sys;sys.path.insert(0,'tools');import e0_compact6016_consumer_quality_v3;assert 'torch' not in sys.modules;assert 'numpy' not in sys.modules"
        completed=subprocess.run([sys.executable,'-c',command],cwd=ROOT,capture_output=True,text=True,timeout=20)
        self.assertEqual(completed.returncode,0,completed.stderr)


if __name__=='__main__':unittest.main()
