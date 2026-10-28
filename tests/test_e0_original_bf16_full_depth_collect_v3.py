"""Portable controls for exact asymmetric V3 completion; no real tensors."""
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('ffnds_v3_test',ROOT/'tools/e0_original_bf16_full_depth_collect_v3.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class V3Controls(unittest.TestCase):
    def test_stdlib_and_public_globals_unchanged(self):
        code='''import sys
sys.path.insert(0,'tools')
import e0_original_bf16_full_depth_collect as v1
import e0_original_bf16_full_depth_collect_v2 as v2
before=(v1.scope_id(0),v1.limits(),v2.scope_id(0),v2.limits())
import e0_original_bf16_full_depth_collect_v3
assert before==(v1.scope_id(0),v1.limits(),v2.scope_id(0),v2.limits())
assert not {'torch','numpy'} & set(sys.modules)
'''
        subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)

    def test_asymmetric_budget_and_original_selection(self):
        self.assertEqual(m.CASE_INDICES,{0:[0,2,3],1:[0,3]})
        self.assertEqual(m.COMPLETION['remaining_calibration_native_ordinals_by_part'],{'0':[896,240],'1':[304]})
        for part,n in ((0,3),(1,2)):
            lim=m.limits(part)
            expected={'original_expert_loads':2,'raw_Default_MM_loads':812,'buffer_Default_MM_loads':40,
              'T5_raw_Default_MM_loads':168,'T5_buffer_Default_MM_loads':7,'T5_Default_MM_calls':168*n,
              'T5_checkpoint_block_buffer_loads':24*n,'input_encoder_requests':n,'T5_positive_prompt_encodes':n,
              'VAE_encodes':n,'VAE_decodes':n-1,'video_generations':n-1,'core_CPU_reports':n-1,
              'scheduler_prepare':n,'FP32_initial_noise_draws':n,'scheduler_step_pre':4*n,'scheduler_step_post':4*n,
              'native_full_model_flows':4*n,'native_blocks':160*n,'native_FFN_calls':160*n,'FFN_CUDA_event_pairs':160*n,
              'native_FFN_linears':320*n,'native_non_FFN_linears':1280*n,'pre_Default_MM_calls':20*n,
              'head_Default_MM_calls':4*n,'patch_Conv3D_calls':4*n,'pre_infer_calls':4*n,'final_LayerNorm_calls':4*n,
              'post_unpatchify_calls':4*n,'original_checkpoint_block_buffer_loads':162*n,'offload_buffer_resets':2*n,
              'native_prefix_replays_no_decode':1,'row_index_uploads':n,'row_index_upload_bytes':n*4*40*256*8,
              'raw_postwrite_packets':160*n,'packet_tensor_D2H':160*n*9,
              'packet_tensor_D2H_bytes':160*n*17596416,'state_tensor_D2H':10*n,'state_tensor_D2H_bytes':60802560*n,
              'neutral_state_byte_checks':10,'video_geometry_checks':n-1,'source_registration_writes':n-1}
            for key,want in expected.items():self.assertEqual(lim[key],want,(part,key))
        self.assertEqual(m.RESOURCES['wall_seconds'],1800)
        self.assertEqual(m.RESOURCES['outer_seconds'],2100)
        with self.assertRaises(ValueError):m.limits(2)

    def test_predecessor_must_be_failed_and_prefix_really_complete(self):
        old={'id':'parent','limits':{'packets':4},'cases':[{'native_source_ordinal':11,'role':'development','source_id':'dev'},
          {'native_source_ordinal':22,'role':'calibration','source_id':'cal'}]}
        refs=[{'native_source_ordinal':x['native_source_ordinal'],'role':x['role'],'GENERATION':{'path':'gen'+str(i)}} for i,x in enumerate(old['cases'])]
        previous={'protocol':{'sha256':'proto'},'RUN':{'path':'run','sha256':'runsha'},'launch_exit':{'path':'exit'},'completed_prefix_generations':refs}
        run={'id':'parent','protocol_sha256':'proto','status':'FAILED_FULL_DEPTH_PART_PRESERVED_NO_RETRY',
          'operation_ledger':{'forbidden_attempts':[],'observed':{'packets':3},'predebited':{'packets':4}},
          'phases':[{'native_source_ordinal':c['native_source_ordinal'],'generation_receipt':refs[i]['GENERATION']} for i,c in enumerate(old['cases'])]}
        exit_={'id':'parent','protocol_sha256':'proto','run_sha256':'runsha','process_exited':True,'child_returncode':-2,'automatic_retry':False}
        gens=[]
        for c in old['cases']:
            gens.append({'id':'parent','protocol_sha256':'proto','status':'COMPLETE_RAW_BF16_TEACHER_STATE_FULL160',
              'numeric_path':'BF16_ORIGINAL','source_state':'teacher_input','native_source_ordinal':c['native_source_ordinal'],
              'source_id':c['source_id'],'role':c['role'],'completed_native_flow_steps':4,'native_ffn_sites':160,
              'packets':[{'step':s,'block':b} for s in range(4) for b in range(40)],'states':{str(i):{} for i in range(10)}})
        docs={'run':run,'exit':exit_,'gen0':gens[0],'gen1':gens[1]}
        plan={'failed_predecessor':previous,'cases':[old['cases'][0],{'native_source_ordinal':33}]}
        with mock.patch.object(m,'parent_protocol',return_value=old),mock.patch.object(m.native,'bound',side_effect=lambda root,item:item['path']),mock.patch.object(m.native,'read',side_effect=lambda p:docs[p]):
            self.assertEqual(m.verify_predecessor(plan,ROOT),previous)
            docs['gen1']=copy.deepcopy(gens[1]);docs['gen1']['packets'].pop()
            with self.assertRaises(ValueError):m.verify_predecessor(plan,ROOT)
            docs['gen1']=gens[1];docs['exit']=dict(exit_,child_returncode=0)
            with self.assertRaises(ValueError):m.verify_predecessor(plan,ROOT)
            docs['exit']=exit_;docs['run']=copy.deepcopy(run);docs['run']['phases'].append({'native_source_ordinal':33,'generation_receipt':{'path':'unexpected'}})
            with self.assertRaises(ValueError):m.verify_predecessor(plan,ROOT)


if __name__=='__main__':unittest.main()
