"""Portable synthetic controls. No metric imports, weights or CUDA work."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import e0_original_bf16_consumer_quality as q


def fixture():
    return {'id':q.ID,'numeric_path':'BF16_ORIGINAL','dimensions':list(q.DIMENSIONS),
      'limits':copy.deepcopy(q.LIMITS),'costs':copy.deepcopy(q.COSTS),'resources':copy.deepcopy(q.RESOURCES),
      'automatic_retry':False,'videos':[{'source':s,'mode':'dense'} for s in (0,12)],
      'images':[{'source':s,'binding':{'sha256':str(s)}} for s in (0,12)],
      'parent':{'protocol':{'path':q.PARENT,'sha256':q.PARENT_SHA},'RUN':{'sha256':q.PARENT_RUN_SHA},'exit':{'sha256':q.PARENT_EXIT_SHA}},
      'reused_import_proof':{'protocol':{'path':q.METRIC_PROTOCOL,'sha256':q.METRIC_SHA},'RUN':{'sha256':q.IMPORT_RUN_SHA},'exit':{'sha256':q.IMPORT_EXIT_SHA}},
      'directory_fd_audit_policy':copy.deepcopy(q.v4.v3.POLICY),'activation_bootstrap_policy':copy.deepcopy(q.v4.POLICY),
      'repo_sources':[{'path':s} for s in sorted(q.REQUIRED)],'import_environment':{},'output_HF_relative':'ffnds-derived/new-quality/results'}


class RawQualityTest(unittest.TestCase):
    def test_exact_two_source_contract_and_no_failed_predecessors(self):
        p=fixture();q.validate(p)
        for change in ('alias','order','scope','failure_chain','wrong_parent','wrong_proof','escaped','retry'):
            bad=copy.deepcopy(p)
            if change=='alias':bad['images'][1]['binding']=bad['images'][0]['binding']
            elif change=='order':bad['videos'].reverse()
            elif change=='scope':bad['limits']['new_video_dimension_results']=42
            elif change=='failure_chain':bad['failed_import_predecessor_v3']={}
            elif change=='wrong_parent':bad['parent']['RUN']['sha256']='0'*64
            elif change=='wrong_proof':bad['reused_import_proof']['RUN']['sha256']='0'*64
            elif change=='escaped':bad['output_HF_relative']='ffnds-derived/../outside'
            else:bad['automatic_retry']=True
            with self.subTest(change=change),self.assertRaises(ValueError):q.validate(bad)
        with self.assertRaises(ValueError):q.validate(p,True)
        p.update(assets_finalized=True,binding_status='BOUND_READY_FOR_REGISTRATION');q.validate(p,True)

    def test_dispatch_two_common_clips_and_one_each_i2v(self):
        p=fixture();sizes=[len(q.base.rows_for_dimension(p['videos'],d)) for d in q.DIMENSIONS]
        self.assertEqual(sizes,[2,2,2,2,2,2,1,1]);self.assertEqual(sum(sizes),14)
        self.assertEqual(q.base.rows_for_dimension(p['videos'],'i2v_subject')[0]['source'],0)
        self.assertEqual(q.base.rows_for_dimension(p['videos'],'i2v_background')[0]['source'],12)
        self.assertEqual(q.COSTS['dynamic_degree'],{'RAFT':80,'RAFT_iterations':1600})

    def test_failure_keeps_both_predebits_before_invocation(self):
        report={};events=[];ledger=q.base.Ledger(report,limits={'official_dimension_calls':1,'new_video_dimension_results':2},persist=lambda:events.append(copy.deepcopy(report)))
        with self.assertRaisesRegex(RuntimeError,'original compute failed'):
            with ledger('official_dimension_calls'),ledger('new_video_dimension_results',2):
                self.assertEqual(report['predebited'],{'official_dimension_calls':1,'new_video_dimension_results':2})
                self.assertEqual(report['completed'],{'official_dimension_calls':0,'new_video_dimension_results':0})
                raise RuntimeError('original compute failed')
        self.assertEqual(report['invoked'],report['predebited']);self.assertEqual(report['completed']['new_video_dimension_results'],0)
        with self.assertRaises(ValueError):
            with ledger('official_dimension_calls'):pass
        with self.assertRaises(ValueError):ledger.finish()
        self.assertGreaterEqual(len(events),4)

    def test_all_exact_budgets_close(self):
        report={};ledger=q.base.Ledger(report,limits=q.LIMITS)
        with ledger('reused_core_reports',2):pass
        for d in q.DIMENSIONS:
            count=2 if d in q.COMMON else 1
            with ledger('official_dimension_calls'),ledger('new_video_dimension_results',count):
                if d=='dynamic_degree':
                    for _ in range(80):
                        with ledger('RAFT_forwards'),ledger('RAFT_iterations',20):pass
        ledger.finish();self.assertEqual(report['completed'],q.LIMITS)

    def test_source_summary_continuous_and_binary_are_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            hf=Path(temp);videos=[]
            for s in (0,12):
                p=hf/f'{s}.mp4';p.write_bytes(f'clip{s}'.encode());videos.append({'source':s,'mode':'dense','video':q.base.binding(p,hf)})
            entries=[]
            for d in q.DIMENSIONS:
                chosen=q.base.rows_for_dimension(videos,d)
                rows=[{'video_path':str(hf/x['video']['path']),'video_results':False if d=='dynamic_degree' else (0.25 if x['source']==0 else 0.75)} for x in chosen]
                entries.append({'dimension':d,'result':[0.0,rows]})
            raft=[{'video_path':str(hf/x['video']['path']),'scores':[1.0 if x['source']==0 else 2.0]*40,'params':{'thres':10.875,'count_num':10},'official_binary':False} for x in videos]
            result=q.summarize({'videos':videos},hf,entries,raft)
            self.assertEqual(result['source_equal_common6_means']['imaging_quality'],0.5)
            self.assertEqual(result['source_equal_common6_means']['dynamic_degree'],0.0)
            self.assertEqual(result['continuous_RAFT_per_source']['12']['mean_top5percent_flow_px_per_2frame_pair'],2.0)
            self.assertEqual(len(result['continuous_RAFT_per_source']['0']['five_fixed_8pair_windows']),5)
            self.assertEqual(result['independent_image_clusters'],2);self.assertFalse(result['quality_accepted'])
            self.assertEqual(result['i2v_subject_source0'],0.25);self.assertEqual(result['i2v_background_source12'],0.75)
            raft[0]['scores'].pop()
            with self.assertRaises(ValueError):q.summarize({'videos':videos},hf,entries,raft)

    def test_bad_result_duplicate_wrong_clip_and_nonfinite_rejected(self):
        paths=['/clip0.mp4','/clip12.mp4']
        for rows in ([{'video_path':paths[0],'video_results':0.1}]*2,
                     [{'video_path':p,'video_results':float('nan')} for p in paths],
                     [{'video_path':'/outside.mp4','video_results':True}]):
            with self.subTest(rows=rows),self.assertRaises(ValueError):q.base.check_result([0,rows],paths,'imaging_quality')
        with self.assertRaises(ValueError):q.base.check_result([0,[{'video_path':p,'video_results':0.0} for p in paths]],paths,'dynamic_degree')

    def test_cpu_proof_must_be_independent_actual_exit_zero(self):
        p=fixture();digest='a'*64
        with tempfile.TemporaryDirectory() as temp:
            hf=Path(temp);run=hf/'CPU.json';q.base.write(run,q.cpu_report(p,digest));binding=q.base.binding(run,hf)
            exit=hf/'EXIT.json'
            receipt={'id':q.ID,'protocol_sha256':digest,'child_returncode':0,'process_exited':True,'timeout':False,'automatic_retry':False,'run_sha256':binding['sha256']}
            q.base.write(exit,receipt);scope={'cpu_preflight':{'RUN':binding,'launch_exit':q.base.binding(exit,hf)}}
            q.require_cpu(scope,p,digest,hf)
            receipt['child_returncode']=1;q.base.write(exit,receipt);scope['cpu_preflight']['launch_exit']=q.base.binding(exit,hf)
            with self.assertRaises(ValueError):q.require_cpu(scope,p,digest,hf)
            with self.assertRaises(ValueError):q.require_cpu({},p,digest,hf)

    def test_inspect_helpers_import_no_numerical_package(self):
        code="import sys; from tools import e0_original_bf16_consumer_quality; assert not {'torch','numpy','pyiqa','imgaug'} & set(sys.modules)"
        completed=subprocess.run([sys.executable,'-c',code],cwd=q.ROOT,capture_output=True,text=True)
        self.assertEqual(completed.returncode,0,completed.stderr)

    def test_run_is_independent_and_reuses_successful_v4_helpers(self):
        import ast
        module=ast.parse(Path(q.__file__).read_text());run=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        calls=[ast.unparse(n.func) for n in ast.walk(run) if isinstance(n,ast.Call)]
        self.assertNotIn('base.run',calls);self.assertNotIn('v4.base.run',calls)
        self.assertIn('v4.metric_module',calls);self.assertIn('v4.v3.normalize_dir_fd_audit',calls)
        self.assertIn('base.configure_import_environment',calls);self.assertIn('base.raft_observer',calls)
        self.assertIn('summarize',calls)


if __name__=='__main__':unittest.main()
