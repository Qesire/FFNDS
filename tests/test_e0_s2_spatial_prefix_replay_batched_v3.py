import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from tools import e0_s2_spatial_prefix_replay_batched_v3 as m


class V3Controls(unittest.TestCase):
    def test_block_prefix_and_exact_costs(self):
        limits=m.parent_limits(8)
        self.assertEqual(limits['raw_W2_tensor_reads'],2)
        self.assertEqual(limits['packet_reads'],24)
        self.assertEqual(limits['BF16_original_W2_GEMM'],24)
        self.assertEqual(limits['BF16_factor_GEMM'],64)
        self.assertEqual(limits['rFFT_calls'],256)
        self.assertEqual(limits['rowLN_calls'],80)
        for n in (0,3,5,160):
            with self.assertRaises(ValueError):m.parent_limits(n)

    def test_execution_order_exact_families_and_coverage(self):
        positions=m.canonical_positions(1,['a','b'])[:4]
        rows=[{'block':block,'source_id':source,'step':step,'family':family}
              for block,source,step in positions for family in ('value10','dcac10')]
        self.assertEqual(m.audit_records(rows,1,['a','b'],4),positions)
        bad=[dict(x) for x in rows];bad[2]['step']=2
        with self.assertRaises(ValueError):m.audit_records(bad,1,['a','b'],4)
        with self.assertRaises(ValueError):m.audit_records(rows[:-1],1,['a','b'],4)

    def test_parent_exit_and_ledger_require_exact_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ident='E0_S2_SPATIAL52_SAME_H_LOCAL_PREFIX_REPLAY_LAUNCH_V2_20261029_PART0'
            sources=['a','b']; positions=m.canonical_positions(0,sources)[:4]
            rows=[{'block':b,'source_id':s,'step':t,'family':f} for b,s,t in positions for f in ('value10','dcac10')]
            (root/'positions').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            counts=m.parent_limits(4);(root/'journal').write_text(json.dumps({'kind':'predebit','costs':counts})+'\n')
            plan={'id':ident,'part':0,'input_index':{'cases':[{'source_id':s} for s in sources]}}
            run={'id':ident,'protocol_sha256':'p','status':'FAILED_LOCAL_PREFIX_REPLAY_PRESERVED_NO_RETRY',
                 'error_type':'ValueError','error':'Exact replay not registered','assigned_GPU_resources':{'violations':[]},
                 'completed_positions':4,'operation_ledger':{'predebited':counts,'observed':counts},
                 'positions':{'path':str(root/'positions')},'operations':{'path':str(root/'journal')}}
            receipt={'id':ident,'protocol_sha256':'p','run_sha256':'r','process_exited':True,
                     'child_returncode':1,'automatic_retry':False,'timeout':False}
            for name,value in (('plan',plan),('run',run),('exit',receipt)):(root/name).write_text(json.dumps(value))
            proof={key:{'path':str(root/name),'sha256':sha} for key,name,sha in
                   (('protocol','plan','p'),('RUN','run','r'),('exit','exit','e'))}
            with patch.object(m.native,'bound',side_effect=lambda _,item:Path(item['path'])):
                self.assertEqual(m.verify_parent(root,proof)['completed_positions'],4)
                receipt['process_exited']=False;(root/'exit').write_text(json.dumps(receipt))
                with self.assertRaises(ValueError):m.verify_parent(root,proof)
                receipt['process_exited']=True;(root/'exit').write_text(json.dumps(receipt))
                receipt['timeout']=True;(root/'exit').write_text(json.dumps(receipt))
                with self.assertRaises(ValueError):m.verify_parent(root,proof)
                receipt['timeout']=False;(root/'exit').write_text(json.dumps(receipt))
                run['error']='different failure';(root/'run').write_text(json.dumps(run))
                with self.assertRaises(ValueError):m.verify_parent(root,proof)
                run['error']='Exact replay not registered'
                run['operation_ledger']['observed']={};(root/'run').write_text(json.dumps(run))
                with self.assertRaises(ValueError):m.verify_parent(root,proof)

    def test_comparison_catches_null_bool_and_near_threshold(self):
        old={'reference_energy':1e-24,'result':None,'flag':True,'value':1.0,'candidate_phase_radians':3.141592653589793}
        same={'reference_energy':1e-24,'result':None,'flag':True,'value':1.0+1e-9,'candidate_phase_radians':-3.141592653589793}
        result=m.compare_records(old,same);self.assertTrue(result['passed']);self.assertEqual(result['near_reference_paths'],['$'])
        bad=dict(same,result=0.0,flag=False,value=1.01)
        result=m.compare_records(old,bad);self.assertFalse(result['passed']);self.assertEqual(len(result['failures']),3)

    def test_controls_cannot_overlap_remaining_or_each_other(self):
        with self.assertRaises(ValueError):m.work_limits([(1,'a',0)],[(1,'a',0),(0,'b',1)])
        budget=m.work_limits([(1,'a',0),(1,'b',0)],[(0,'a',0),(0,'b',1)])
        self.assertEqual(budget['raw_W2_tensor_reads'],2)
        self.assertEqual(budget['source_position_records'],8)

    def test_three_controls_exact_counters_and_exhaustion(self):
        work=m.construct_workload(0,['a','b'],8,(0,1,7));lim=work['limits']
        count=len(work['remaining'])+3
        self.assertEqual(lim['rFFT_calls'],32*count)
        self.assertEqual(lim['gY_FP64_products'],10*count)
        self.assertEqual(lim['rowLN_calls'],10*count)
        report={};journal=[];meter=m.Meter({'rFFT_calls':16},report,journal.append)
        self.assertEqual(meter.work({'rFFT_calls':16},lambda:'ok'),'ok')
        self.assertEqual(report['predebited'],report['observed'])
        with self.assertRaises(ValueError):meter.work({'rFFT_calls':1},lambda:None)
        failed={};meter=m.Meter({'rFFT_calls':16},failed,lambda _:None)
        def error():raise RuntimeError('intentional')
        with self.assertRaises(RuntimeError):meter.work({'rFFT_calls':16},error)
        self.assertEqual(failed['predebited'],{'rFFT_calls':16});self.assertEqual(failed['observed'],{})

    def test_unbound_draft_is_explicitly_not_executable(self):
        draft=m.make_unbound(0)
        self.assertEqual(draft['native_runner'],'IMPLEMENTED_NOT_RUN')
        self.assertIn('NOT_BOUND',draft['status'])
        self.assertEqual(draft['resources']['wall_seconds'],3600)

    def test_bound_draft_reuses_parent_index_and_rejects_malformed_controls(self):
        parent={'part':0,'completed_positions':8,'source_order':['a','b'],
                'parent_plan':{'repo_sources':[],'input_index_sha256':'a'*64}}
        with patch.object(m,'verify_parent',return_value=parent):
            plan=m.draft(m.ROOT,0,{'protocol':'synthetic'})
        self.assertNotIn('input_index',plan)
        self.assertEqual(plan['control_indices'],[0,1,7]);m.validate(plan)
        plan['control_indices']=[0,1,6]
        with self.assertRaises(ValueError):m.validate(plan)

    def test_workload_and_complete_unique_merge(self):
        sources=['a','b']; work=m.construct_workload(0,sources,8,(0,7))
        self.assertEqual(len(work['remaining']),152); self.assertEqual(len(work['controls']),2)
        rows=[{'block':b,'source_id':s,'step':t,'family':f}
              for b,s,t in m.canonical_positions(0,sources) for f in ('value10','dcac10')]
        self.assertEqual(m.merge_records(0,sources,rows[:16],8,rows[16:]),rows)
        bad=list(rows[16:]);bad[0]=rows[0]
        with self.assertRaises(ValueError):m.merge_records(0,sources,rows[:16],8,bad)

    def test_cpu_adapter_agrees_with_numpy_without_resizing(self):
        import numpy as np
        from tools import raw_bf16_spatial_prefix_diagnostics as old
        rng=np.random.default_rng(4);t=rng.normal(size=(21,52,2))+2
        s=t+.03*rng.normal(size=t.shape);cs=s+.02*rng.normal(size=t.shape);ct=t+.01*rng.normal(size=t.shape)
        masks={'r':[0,1,2]}
        got=m.cpu_batched_pair((t,s,cs,ct),(t,cs),masks,spectrum=True)
        expected={'view':old.temporal_views(t,s,cs,ct,masks,np),
                  'bridge':old.native_rounding_bridge(t,cs,t,cs,masks,np)}
        self.assertTrue(m.compare_records(expected,got)['passed'])

    def test_actual_batched_fft_count_matches_registered_pair_budget(self):
        import numpy as np
        import torch
        rng=np.random.default_rng(17);a=rng.normal(size=(21,52,2))
        values=[a+i*.02*rng.normal(size=a.shape) for i in range(4)]
        with patch.object(torch.fft,'rfft',wraps=torch.fft.rfft) as fft:
            m.cpu_batched_pair(values,values[:2],{'r':[0,1]},spectrum=True)
            self.assertEqual(fft.call_count,16)
        with patch.object(torch.fft,'rfft',wraps=torch.fft.rfft) as fft:
            m.cpu_batched_pair(values,values[:2],{'r':[0,1]},spectrum=False)
            self.assertEqual(fft.call_count,0)


if __name__=='__main__':unittest.main()
