import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('teal_probe_test',ROOT/'tools/e0_teal_author_kernel_probe.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ProbeTests(unittest.TestCase):
    def test_source_bound_and_safe_author_shape(self):
        p=m.draft();m.validate(p)
        self.assertEqual(p['design']['input_shape'],[1,1,4096]);self.assertEqual(p['design']['weight_shape'],[14336,4096])
        self.assertEqual(14336%512,0);self.assertEqual(4096%128,0)
        self.assertEqual(m.LIMITS['sparse_forward'],2+2*(10+20))
        self.assertEqual(m.LIMITS['dense_benchmark'],2*(10+20))
        self.assertEqual(m.LIMITS['author_triton_launch_entry'],62)
        self.assertIsNone(p['review']);self.assertFalse(p['assets_finalized'])
        source=(ROOT/m.SOURCE/'kernels/sparse_gemv.py').read_text()
        self.assertIn('hidden_states.shape[1] == 1 else torch.matmul(hidden_states, weights.T)',source)
        self.assertIn('mask=idx[:, None]',source)

    def test_fingerprint_freezes_scope_and_excludes_review_pointer(self):
        p=m.draft();old=m.fingerprint(p);p['review']={'path':'review','sha256':'a'*64,'bytes':1}
        self.assertEqual(old,m.fingerprint(p));p['design']=copy.deepcopy(p['design']);p['design']['seed']=7
        self.assertNotEqual(old,m.fingerprint(p))
        with self.assertRaises(ValueError):m.validate(p)

    def test_frozen_requires_review_and_worker_binding(self):
        p=m.draft();p.update(status='FROZEN_PENDING_REGISTRATION',assets_finalized=True)
        with self.assertRaises(ValueError):m.validate(p,True)
        p['environment']['worker_config_sha256']='a'*64;p['environment']['prepared_environment']={};p['environment']['driver_version']='550.54.14';p['environment']['monitor_binary']={}
        m.validate(p,True)

    def test_balanced_order_and_failure_diagnostics_persisted(self):
        self.assertEqual(m.paired_order(0),('sparse','dense'))
        self.assertEqual(m.paired_order(1),('dense','sparse'))
        report={};saved=[]
        with self.assertRaises(ValueError):
            m.record_then_require(report,'correctness',{'relative_l2':1.0},lambda:saved.append(copy.deepcopy(report)),False,'mismatch')
        self.assertEqual(saved[0]['correctness']['relative_l2'],1.0)

    def test_actual_byte_binding_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'file';path.write_bytes(b'correct');binding=m.bind(path,Path(tmp))
            self.assertEqual(m.bound(Path(tmp),binding),path)
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):m.bound(Path(tmp),binding)

    def test_supervisor_exact_argv_and_receipt_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);hf=root/'hf';hf.mkdir();profile=root/'profile';profile.write_bytes(b'private')
            protocol=root/'protocol.json';protocol.write_bytes(b'protocol');spec=root/'spec.json'
            args=SimpleNamespace(protocol=protocol,job_spec=spec,hf_root=hf)
            plan={'environment':{'worker_config_sha256':m.sha(profile),'prepared_environment':{'venv_path':str(root/'venv')}}}
            argv=['env','FFNDS_WORKER_CONFIG='+str(profile),'bash',str(m.ROOT/'scripts/worker_teal_gpu0_uuid_exec.sh'),
                  'taskset','-c','0,1',str(root/'venv/bin/python'),'-B',str(m.ROOT/m.SELF),'run',
                  '--protocol',str(protocol),'--hf-root',str(hf),'--job-spec',str(spec)]
            out=hf/m.OUTPUT
            job={'argv':argv,'id':m.ID,'protocol_sha256':'digest','protocol_path':str(protocol),'cwd':str(m.ROOT),
                 'hf_root':str(hf),'output_dir':str(out.parent),'run_json':str(out/'RUN.json'),
                 'external_timeout_seconds':m.RESOURCES['outer_seconds']}
            m.verify_job_spec(job,args,plan,'digest')
            for changed in ('cpu','launcher','output','timeout'):
                bad=copy.deepcopy(job)
                if changed=='cpu':bad['argv'][6]='2,3'
                if changed=='launcher':bad['argv'][3]='wrong'
                if changed=='output':bad['output_dir']=str(hf)
                if changed=='timeout':bad['external_timeout_seconds']=9999
                with self.assertRaises(ValueError):m.verify_job_spec(bad,args,plan,'digest')

    def test_fallback_and_review_contract_bind_real_artifact_paths(self):
        self.assertEqual(m.LIMITS['fallback_forward'],1)
        self.assertEqual(m.LIMITS['fallback_matmul'],1)
        self.assertEqual(m.LIMITS['fallback_reference'],1)
        self.assertEqual(m.LIMITS['fallback_masked_reference'],1)
        self.assertIn('tools/durable_worker_job.py',m.HELPERS)
        self.assertIn('tools/assigned_gpu_resources.py',m.HELPERS)
        self.assertTrue(m.INDEPENDENT_REVIEW.endswith('/INDEPENDENT_REVIEW_V1.md'))
        self.assertTrue(m.AUTHOR_RESPONSE.endswith('/AUTHOR_RESPONSE_V1.md'))

    def test_review_chain_requires_real_recheck_artifact(self):
        p=m.draft();p.update(status='FROZEN_PENDING_REGISTRATION',assets_finalized=True)
        p['environment'].update(worker_config_sha256='a'*64,prepared_environment={},driver_version='550',monitor_binary={})
        scope={'id':m.ID,'protocol_sha256':'d','execution_status':'AUTHORIZED_NOT_STARTED',
               'resources':m.RESOURCES,'limits':m.LIMITS,'max_gpu_cards':1,'launch_spec_sha256':'c'*64,
               'review':{'path':'review.json','bytes':1,'sha256':'b'*64}}
        active={'user_continuation_authorization_20261027':{'new_scope_registration_allowed':True},
                'experiment_authority':{'current_user_authorized_scopes':[scope]}}
        review={'review_schema':'FFNDS_PREEXEC_REVIEW_V1','scope_id':m.ID,'state':'REVIEWER_RECHECK_APPROVED',
                'reviewer_disposition':'REVISE','recheck_disposition':'APPROVE','author_agent_id':'author',
                'reviewer_agent_id':'reviewer','responses_to_review':['reply'],'adversarial_counterexample':'counterexample',
                'approval_scope_fingerprint':m.fingerprint(p),'author_review':{'path':m.AUTHOR_RESPONSE},
                'independent_review':{'path':m.INDEPENDENT_REVIEW},'reviewer_recheck':{'path':m.REVIEWER_RECHECK},
                'critical_source_hashes':{b['path']:b['sha256'] for b in p['repo_sources']+p['source_files']}}
        with patch.object(m,'read',side_effect=[active,review]),patch.object(m,'bound',return_value=Path('file')) as binding:
            self.assertEqual(m.authorize(p,'d'),scope);self.assertEqual(binding.call_count,4)
        bad=copy.deepcopy(review);del bad['reviewer_recheck']
        with patch.object(m,'read',side_effect=[active,bad]),patch.object(m,'bound',return_value=Path('file')):
            with self.assertRaises(ValueError):m.authorize(p,'d')

    def test_review_rejects_same_agent_and_changed_fingerprint(self):
        p=m.draft();p.update(status='FROZEN_PENDING_REGISTRATION',assets_finalized=True)
        p['environment']['worker_config_sha256']='a'*64;p['environment']['prepared_environment']={};p['environment']['driver_version']='550.54.14';p['environment']['monitor_binary']={}
        scope={'id':m.ID,'protocol_sha256':'d','execution_status':'AUTHORIZED_NOT_STARTED',
               'resources':m.RESOURCES,'limits':m.LIMITS,'max_gpu_cards':1,'review':{'path':'r','sha256':'b'*64,'bytes':1}}
        active={'user_continuation_authorization_20261027':{'new_scope_registration_allowed':True},
                'experiment_authority':{'current_user_authorized_scopes':[scope]}}
        review={'review_schema':'FFNDS_PREEXEC_REVIEW_V1','scope_id':m.ID,'state':'REVIEWER_RECHECK_APPROVED',
                'reviewer_disposition':'REVISE','recheck_disposition':'APPROVE','author_agent_id':'a',
                'reviewer_agent_id':'a','responses_to_review':['reply'],'adversarial_counterexample':'counterexample',
                'approval_scope_fingerprint':m.fingerprint(p)}
        with patch.object(m,'read',side_effect=[active,review]),patch.object(m,'bound',return_value=Path('r')):
            with self.assertRaises(ValueError):m.authorize(p,'d')
        review['reviewer_agent_id']='b';review['approval_scope_fingerprint']='wrong'
        with patch.object(m,'read',side_effect=[active,review]),patch.object(m,'bound',return_value=Path('r')):
            with self.assertRaises(ValueError):m.authorize(p,'d')

if __name__=='__main__':unittest.main()
