"""Synthetic stdlib-only controls for the future 60-video quality scope."""
import ast
import copy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import e0_raw_bf16_video20_quality as m


def fixture():
    hf=Path('/tmp/ffnds-quality-synthetic');videos=[]
    for s in range(20):
        stratum='subject' if s<10 else 'background' if s<17 else 'camera'
        for mode in m.MODES:
            videos.append({'source':s,'source_id':str(s),'mode':mode,'image_sha256':str(s),'stratum':stratum,
                'conditioning_identity':'NEW_CAMERA_PROMPT_NOT_HISTORICAL_CONDITIONING' if s>=17 else 'ORIGINAL_PROMPT',
                'i2v_dimension':'i2v_subject' if s<10 else 'i2v_background','video':{'path':f'{s}/{mode}.mp4'}})
    dimensions=[]
    for dim in m.DIMENSIONS:
        rows=[{'video_path':str((hf/v['video']['path']).resolve()),'video_results':False if dim=='dynamic_degree' else (1. if v['mode']=='dense' else 2.)}
            for v in m.rows_for_dimension(videos,dim)]
        dimensions.append({'dimension':dim,'result':[0.,rows]})
    observations=[{'video_path':str((hf/v['video']['path']).resolve()),'scores':[1. if v['mode']=='dense' else 2.]*40,
        'params':{'thres':10.875,'count_num':10},'official_binary':False} for v in videos]
    return {'videos':videos},hf,dimensions,observations


class QualityControls(unittest.TestCase):
    def test_composite_failed_prefix_repair_and_complete_v2(self):
        cases=[{'index':i,'part':i%2,'native_source_ordinal':100+i} for i in range(20)]
        cohort={'sources':cases};parents=[];objects={'cohort':cohort};repair_parents=[];repair_phases=[]
        def binding(name):return {'path':name,'sha256':name}
        common={'source_cohort':binding('cohort'),'prior_development_criteria':{},'numeric_path':'BF16_ORIGINAL',
                'factor_parent':{},'models':[],'runtime_overlay':{},'runtime_versions':{}}
        for part in (0,1):
            oldid='E0_RAW_BF16_FULL160_VIDEO20_20261029_PART'+str(part)
            old={'part':part,'kind':'FAILED_V1_PREFIX','protocol':binding('oldp'+str(part)),'RUN':binding('oldr'+str(part)),'exit':binding('olde'+str(part))}
            new={'part':part,'kind':'COMPLETE_V2_REMAINING9','protocol':binding('newp'+str(part)),'RUN':binding('newr'+str(part)),'exit':binding('newe'+str(part))}
            parents.extend([old,new]);repair_parents.append({k:old[k] for k in ('part','protocol','RUN','exit')})
            oldphases=[];newphases=[]
            for c in [x for x in cases if x['part']==part]:
                for mode in m.MODES:
                    phase={'native_source_ordinal':c['native_source_ordinal'],'mode':mode,'generation_receipt':binding(f"g{c['index']}{mode}"),
                        'states':{},'native_operand_transfer_evidence':{},'core_metrics':{}}
                    if c['index']<2:
                        oldphases.append(phase);repair_phases.append({**phase,'parent_remains_failed':True,
                            'failed_native_parent_RUN':old['RUN'],'failed_native_parent_exit':old['exit']})
                    else:newphases.append(phase)
            oldproto={**common,'id':oldid,'cases':[x for x in cases if x['part']==part]}
            newproto={**common,'id':'E0_RAW_BF16_FULL160_VIDEO20_REMAINING18_20261029_PART'+str(part)+'_V2',
                'cases':[x for x in cases if x['part']==part and x['index']>=2], 'limits':{'work':1},
                'failed_predecessor':{'protocol':old['protocol'],'RUN':old['RUN'],'launch_exit':old['exit']}}
            oldrun={'id':oldid,'protocol_sha256':old['protocol']['sha256'],'numeric_path':'BF16_ORIGINAL',
                'operation_ledger':{'forbidden_attempts':[]},'phases':oldphases}
            newrun={'id':newproto['id'],'protocol_sha256':new['protocol']['sha256'],'numeric_path':'BF16_ORIGINAL',
                'status':'COMPLETE_RAW_BF16_FULL160_VIDEO20_V2_REMAINING9_PART_NINE_TRIADS','phases':newphases,'triads':[{}]*9,
                'actual_full160_videos':27,'actual_flow_steps':108,'actual_FFN_positions':4320,
                'operation_ledger':{'forbidden_attempts':[],'predebited':{'work':1},'observed':{'work':1}}}
            objects.update({old['protocol']['path']:oldproto,old['RUN']['path']:oldrun,old['exit']['path']:{'child_returncode':1},
                new['protocol']['path']:newproto,new['RUN']['path']:newrun,new['exit']['path']:{}})
        rp={'protocol':binding('repairp'),'RUN':binding('repairr'),'exit':binding('repaire')}
        objects['repairp']={'id':'repair','parents':repair_parents,'resources':{'GPU':0},'limits':{'paired_CORE_evaluations':4}}
        objects['repairr']={'id':'repair','protocol_sha256':'repairp','status':'COMPLETE_CPU_PREFIX_CORE_REPAIR_SIX_VALID_GENERATIONS_PARENTS_REMAIN_FAILED',
            'operation_ledger':{'predebited':4,'observed':4},'GPU_calls':0,'model_calls':0,'new_generations':0,'tensor_loads':0,
            'parents':repair_parents,'phases':repair_phases};objects['repaire']={}
        plan={'source_cohort':binding('cohort'),'prefix_repair':rp,'parents':parents,'prior_development_criteria':{}}
        def validate_repair(p,frozen):
            m.base.require(frozen is True and p['resources']=={'GPU':0} and p['limits']=={'paired_CORE_evaluations':4},'repair scope differs')
        prefix=SimpleNamespace(ID='repair',validate_parent=lambda *_a,**_k:None,validate_plan=validate_repair)
        spec=SimpleNamespace(loader=SimpleNamespace(exec_module=lambda _module:None))
        with patch.object(m.importlib.util,'spec_from_file_location',return_value=spec),patch.object(m.importlib.util,'module_from_spec',return_value=prefix),patch.object(m.base,'read_bound',side_effect=lambda _root,b:objects[b['path']]),patch.object(m.base,'exited',return_value=None):
            self.assertEqual(len(m.generation_metadata(plan,Path('/tmp'))[0]),60)
            for kind in ('old_success','wrong_repair','wrong_resources','duplicate','missing_v2'):
                changed=copy.deepcopy(objects);p=copy.deepcopy(plan)
                if kind=='old_success':changed['olde0']['child_returncode']=0
                elif kind=='wrong_repair':changed['repairr']['operation_ledger']['observed']=3
                elif kind=='wrong_resources':changed['repairp']['resources']['GPU']=1
                elif kind=='duplicate':changed['newr0']['phases'][0]=copy.deepcopy(changed['oldr0']['phases'][0])
                else:p['parents']=[x for x in p['parents'] if x['kind']!='COMPLETE_V2_REMAINING9' or x['part']!=0]
                with patch.object(m.base,'read_bound',side_effect=lambda _root,b:changed[b['path']]):
                    if kind=='missing_v2':
                        # Metadata closure alone is checked against all60 by verify_generation.
                        self.assertNotEqual(len(m.generation_metadata(p,Path('/tmp'))[0]),60)
                    else:
                        with self.assertRaises((ValueError,KeyError,StopIteration)):m.generation_metadata(p,Path('/tmp'))

    def test_fixed_scene_and_budget(self):
        p,_,_,_=fixture()
        self.assertEqual(len(m.DIMENSIONS),8);self.assertNotIn('temporal_flickering',m.DIMENSIONS)
        self.assertEqual(sum(len(m.rows_for_dimension(p['videos'],d)) for d in m.DIMENSIONS),420)
        self.assertEqual((len(m.rows_for_dimension(p['videos'],'i2v_subject')),len(m.rows_for_dimension(p['videos'],'i2v_background'))),(30,30))
        self.assertEqual(sum(v['stratum']=='camera' for v in p['videos']),9)
        self.assertEqual((m.LIMITS['RAFT_forwards'],m.LIMITS['RAFT_iterations']),(2400,48000))

    def test_source_equal_paired_axes_and_continuous_motion(self):
        plan,hf,dims,obs=fixture();result=m.summarize(plan,hf,dims,obs)
        self.assertEqual(len(result['per_source']),20)
        self.assertEqual(result['per_axis']['i2v_subject']['source_count'],10)
        self.assertEqual(result['per_axis']['i2v_background']['source_count'],10)
        self.assertEqual(result['per_axis']['imaging_quality']['paired_differences']['shared']['mean'],1.)
        self.assertEqual(result['continuous_motion_comparisons']['shared']['ratio_of_source_equal_means'],2.)
        self.assertEqual(result['continuous_motion_comparisons']['shared']['mean_of_defined_source_ratios'],2.)
        self.assertEqual(result['per_axis']['dynamic_degree']['source_equal_means']['dense'],0.)

    def test_missing_duplicate_wrong_and_nan_rejected(self):
        plan,hf,dims,obs=fixture()
        with self.assertRaises(ValueError):m.summarize(plan,hf,dims[:-1],obs)
        for mutation in ('missing','wrong','nan'):
            altered=copy.deepcopy(dims);rows=altered[0]['result'][-1]
            if mutation=='missing':rows.pop()
            elif mutation=='wrong':rows[0]['video_path']='/unregistered'
            else:rows[0]['video_results']=float('nan')
            with self.assertRaises((ValueError,RuntimeError)):m.summarize(plan,hf,altered,obs)
        with self.assertRaises(ValueError):m.summarize(plan,hf,dims,obs[:-1]+[obs[0]])
        wrong=copy.deepcopy(obs);wrong[0]['scores'][0]=float('nan')
        with self.assertRaises(ValueError):m.summarize(plan,hf,dims,wrong)

    def test_failed_predebit_retains_attempt_and_no_repeat(self):
        report={};ledger=m.base.Ledger(report,limits={'official_dimension_calls':1})
        with self.assertRaises(RuntimeError):
            with ledger('official_dimension_calls'):raise RuntimeError('evaluator failed')
        self.assertEqual(report['predebited']['official_dimension_calls'],1)
        self.assertEqual(report['invoked']['official_dimension_calls'],1)
        self.assertEqual(report['completed']['official_dimension_calls'],0)
        with self.assertRaises(ValueError):
            with ledger('official_dimension_calls'):pass

    def test_cpu_receipt_must_be_real_exit_zero(self):
        run={'sha256':'run'};proof={'RUN':run,'launch_exit':{'sha256':'exit'}}
        report={'status':'ok','wall_seconds':1.,'byte_checks':[],'byte_checks_sha256':m.base.state_sha([])}
        receipt={'id':m.ID,'protocol_sha256':'digest','run_sha256':'run','process_exited':True,'child_returncode':0,'timeout':False,'automatic_retry':False}
        def get(_root,binding):return report if binding is run else receipt
        with patch.object(m,'cpu_report',return_value={'status':'ok'}),patch.object(m,'all_byte_bindings',return_value=[]),patch.object(m.base,'read_bound',side_effect=get):
            self.assertEqual(m.require_cpu({'cpu_preflight':proof},{},'digest',Path('/tmp')),proof)
            receipt['child_returncode']=False
            with self.assertRaises(ValueError):m.require_cpu({'cpu_preflight':proof},{},'digest',Path('/tmp'))
            with self.assertRaises(ValueError):m.require_cpu({}, {},'digest',Path('/tmp'))

    def test_own_finite_loop_and_stdlib_import(self):
        tree=ast.parse((ROOT/'tools/e0_raw_bf16_video20_quality.py').read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        self.assertTrue(any(isinstance(n,ast.For) and isinstance(n.iter,ast.Name) and n.iter.id=='DIMENSIONS' for n in ast.walk(run)))
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id in ('base','raw') and n.func.attr=='run' for n in ast.walk(run)))
        code="import sys;sys.path.insert(0,'tools');import e0_raw_bf16_video20_quality;assert not {'torch','numpy'} & set(sys.modules)"
        subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)


if __name__=='__main__':unittest.main()
