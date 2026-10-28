"""V2 remaining-source and paired CORE regression; stdlib fixtures only."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import e0_raw_bf16_full160_video20_v2 as m
import ffnds_video_metrics as metrics


class V2Controls(unittest.TestCase):
    def test_fixed_remaining_scope_counts(self):
        for part in (0,1):
            lim=m.limits(part)
            expected={'video_generations':27,'native_full_model_flows':108,'native_FFN_calls':4320,
                'BF16_factor_GEMM':5760,'original_W2_H2D_skips':2916,'prefetch_transfer_observations':4374,
                'stage_CUDA_event_pairs':189,'state_tensor_D2H':270,'matched_triad_state_byte_checks':36,
                'historical_state_byte_checks':10 if part==0 else 0}
            for key,value in expected.items():self.assertEqual(lim[key],value)
            cases=[{'index':i,'native_source_ordinal':100+i} for i in range(2+part,20,2)]
            self.assertEqual(len(cases),9)
            self.assertEqual(len({m.run_id(c,mode) for c in cases for mode in m.MODES}),27)

    def test_missing_reference_identity_is_real_validator_counterexample(self):
        report={'reference':{'sha256':'dense'},'reference_identity':None,'identity':{},'cost':{}}
        with self.assertRaises(AttributeError):metrics.validate(report)

    def test_failed_prefix_exit_count_scope_and_generation_gates(self):
        case={'index':0,'part':0,'native_source_ordinal':894,'image':{'sha256':'image'},'prompt_sha256':'prompt'}
        remaining=[{'index':2+i*2,'part':0} for i in range(9)]
        cohort={'sources':[case]+remaining};states={k:{'sha256':k} for k in
            {'raw_noise_FP32','initial_noise'}|{f'{x}_step{s}' for s in range(4) for x in ('flow','latent')}}
        saved=[];phases=[];objects={}
        for mode in m.MODES:
            video={'path':'/hf/'+mode,'bytes':1,'sha256':mode};gen={'path':mode+'GEN','sha256':mode+'sha'}
            evidence={'path':mode+'evidence'};core={'path':'denseCORE'} if mode=='dense' else None
            phase={'native_source_ordinal':894,'source_index':0,'mode':mode,'generation_receipt':gen,
                'states':copy.deepcopy(states),'native_operand_transfer_evidence':evidence,'video':video,
                'status':'PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE'}
            if core:phase['core_metrics']=core
            phases.append(phase);saved.append({'mode':mode,'generation':gen,'states':copy.deepcopy(states),
                'native_evidence':evidence,'video':video,'core':core})
            objects[gen['path']]={'status':phase['status'],'protocol_sha256':'proto','seed':42,'all160_FFN_covered':True,
                'completed_native_flow_steps':4,'states':copy.deepcopy(states),'video':video,'mode':mode,
                'image_sha256':'image','prompt_sha256':'prompt','native_counter_delta':{'linear_forward_calls':1600,
                'activation_qdq_calls':0,'weight_quantizations':0}}
            objects[evidence['path']]={'all160_FFN_sites':[[s,b] for s in range(4) for b in range(40)]}
        objects['denseCORE']={'candidate':{'sha256':'dense'},'identity':{'source_id':'image','input_prompt_digest':'prompt','run_sha256':'densesha'}}
        oldid='E0_RAW_BF16_FULL160_VIDEO20_20261029_PART0';proto={'id':oldid,'source_cohort':{'path':'cohort'},
            'factor_parent':{},'models':[],'runtime_overlay':{},'runtime_versions':{},'output_HF_relative':'old','cases':[case]+remaining}
        counts={'video_generations':3,'native_full_model_flows':12,'native_FFN_calls':480,'native_FFN_linears':960,
            'native_FP32_flow_files':12,'native_BF16_post_files':12,'core_CPU_reports':2,'source_registration_writes':0}
        report={'id':oldid,'protocol_sha256':'proto','status':'FAILED_FULL160_VIDEO20_PART_PRESERVED_NO_RETRY',
            'error_type':'AttributeError','error':"'NoneType' object has no attribute 'get'",'operation_ledger':{'observed':counts,'forbidden_attempts':[]},'phases':phases,'triads':[]}
        receipt={'id':oldid,'protocol_sha256':'proto','run_sha256':'run','process_exited':True,'child_returncode':1,'timeout':False,'automatic_retry':False}
        objects.update({'config/e0_raw_bf16_full160_video20_part0_20261029_v1.json':proto,'RUN':report,'exit':receipt,'cohort':cohort})
        plan={'part':0,'source_cohort':{'path':'cohort'},'factor_parent':{},'models':[],'runtime_overlay':{},'runtime_versions':{},
            'output_HF_relative':'new','cases':remaining,'failed_predecessor':{'protocol':{'path':'config/e0_raw_bf16_full160_video20_part0_20261029_v1.json','sha256':'proto'},
            'RUN':{'path':'RUN','sha256':'run'},'launch_exit':{'path':'exit'},'completed_prefix':{'source':case,'registered_jobs':'NOT_CREATED_IN_FAILED_V1','phases':saved}}}
        with patch.object(m.native,'bound',side_effect=lambda _root,b:b['path']),patch.object(m.native,'read',side_effect=lambda path:objects[str(path)]),patch.object(metrics,'validate',return_value=[]):
            m.verify_failed_prefix(plan,Path('/hf'))
            for change in ('exit0','wrongsha','repeat','missinggen','counts'):
                p=copy.deepcopy(plan);r=copy.deepcopy(receipt);rep=copy.deepcopy(report);gen=copy.deepcopy(objects['denseGEN'])
                if change=='exit0':r['child_returncode']=0
                elif change=='wrongsha':r['run_sha256']='different'
                elif change=='repeat':p['cases']=[case]+remaining[:-1]
                elif change=='missinggen':gen['completed_native_flow_steps']=3
                else:rep['operation_ledger']['observed']['native_FFN_calls']=479
                altered={**objects,'exit':r,'RUN':rep,'denseGEN':gen}
                with patch.object(m.native,'read',side_effect=lambda path:altered[str(path)]):
                    with self.assertRaises(ValueError):m.verify_failed_prefix(p,Path('/hf'))

    def test_core_candidate_uses_same_dense_generation_identity(self):
        case={'image':{'sha256':'imageSHA'},'prompt_sha256':'promptSHA'}
        plan={'models':[]};phases=[]
        for mode in m.MODES:
            phases.append({'mode':mode,'run_id':mode,'protocol_sha256':'protocol',
                'generation_receipt':{'sha256':'denseGEN' if mode=='dense' else mode+'GEN'},
                'video':{'path':'/tmp/'+mode+'.mp4'},'generation_wall_seconds_including_encoders_decode_observers':1.,
                'FFN_boundary_CUDA_seconds':.2,'peak_cuda_allocated_bytes':0})
        calls=[]
        def evaluate(candidate,reference,context):
            calls.append((candidate,reference,copy.deepcopy(context)))
            if reference is not None:
                identity=context['reference_identity']
                self.assertEqual({k:identity[k] for k in ('source_id','seed','input_prompt_digest','run_sha256')},
                    {'source_id':'imageSHA','seed':42,'input_prompt_digest':'promptSHA','run_sha256':'denseGEN'})
                self.assertEqual(reference,Path('/tmp/dense.mp4'))
            return {'ok':True}
        class Counter:
            def write_json(self,*_):pass
            def binding(self,*_):return {'sha256':'test'}
        class Ledger:
            def call(self,_rid,_cost,fn,*a,**kw):return fn(*a,**kw)
        cv2=SimpleNamespace(setNumThreads=lambda n:None)
        with patch.dict(sys.modules,{'cv2':cv2}),patch.object(metrics,'evaluate',side_effect=evaluate),patch.object(metrics,'validate',return_value=[]):
            out=m.core_reports(plan,phases,case,Path('/tmp'),Path('/tmp'),Counter(),Ledger(),1.)
        self.assertEqual(len(out),3);self.assertEqual(len(calls),3)


if __name__=='__main__':unittest.main()
