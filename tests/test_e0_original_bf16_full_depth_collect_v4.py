"""Synthetic V4 scope and monitor controls; no NVML, model or GPU call."""
import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('v4_test',ROOT/'tools/e0_original_bf16_full_depth_collect_v4.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
import raw_bf16_assigned_gpu_monitor_v4 as mon


class V4Controls(unittest.TestCase):
    def test_public_modules_and_stdlib_only(self):
        code='''import sys
sys.path.insert(0,'tools')
import startup_gpu_reservation as st
import e0_original_bf16_full_depth_collect as a
import e0_original_bf16_full_depth_collect_v2 as b
import e0_original_bf16_full_depth_collect_v3 as c
before=(st.UUIDAssignedGPUResources,st.gpu_uuid,a.limits(),b.limits(),c.limits(0),c.limits(1))
import e0_original_bf16_full_depth_collect_v4
assert before==(st.UUIDAssignedGPUResources,st.gpu_uuid,a.limits(),b.limits(),c.limits(0),c.limits(1))
assert not {'torch','numpy'}&set(sys.modules)
'''
        subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)

    def test_exact_one_calibration_and_scope(self):
        lim=m.limits();expected={'original_expert_loads':2,'raw_Default_MM_loads':812,'buffer_Default_MM_loads':40,
          'T5_raw_Default_MM_loads':168,'T5_buffer_Default_MM_loads':7,'T5_Default_MM_calls':168,
          'T5_checkpoint_block_buffer_loads':24,'input_encoder_requests':1,'T5_positive_prompt_encodes':1,
          'VAE_encodes':1,'VAE_decodes':1,'video_generations':1,'core_CPU_reports':1,
          'scheduler_prepare':1,'FP32_initial_noise_draws':1,'scheduler_step_pre':4,'scheduler_step_post':4,
          'native_full_model_flows':4,'native_blocks':160,'native_FFN_calls':160,'FFN_CUDA_event_pairs':160,
          'native_FFN_linears':320,'native_non_FFN_linears':1280,'pre_Default_MM_calls':20,'head_Default_MM_calls':4,
          'original_checkpoint_block_buffer_loads':162,'offload_buffer_resets':2,'native_prefix_replays_no_decode':0,
          'neutral_state_byte_checks':0,'row_index_uploads':1,'packet_safetensor_writes':160,'state_tensor_D2H':10,
          'video_geometry_checks':1,'source_registration_writes':1}
        for key,n in expected.items():self.assertEqual(lim[key],n,key)
        self.assertEqual(m.COMPLETION['native_source_ordinal'],240);self.assertEqual(m.COMPLETION['role'],'calibration')
        self.assertEqual(m.RESOURCES['wall_seconds'],900);self.assertEqual(m.RESOURCES['outer_seconds'],1200)
        self.assertEqual(m.RESOURCES['new_HF_bytes'],4<<30)

    def test_first_case_calibration_is_registered(self):
        import ffnds_cohort_preflight
        case={'role':'calibration','native_source_ordinal':240,'source_id':'s','image_sha256':'img',
          'prompt_sha256':'prompt-hash','prompt':'p','image':{'path':'image.png'},'conditioning_id':'c'}
        with mock.patch.object(ffnds_cohort_preflight,'audit',return_value={'errors':[], 'summary':{'independent_image_clusters':1}}):
            result=m.registration({'id':m.ID,'cases':[case]},ROOT)
        self.assertEqual(len(result['samples']),1)
        self.assertEqual(result['samples'][0]['run_id'],m.run_id(case))
        self.assertEqual(result['sources'][0]['role'],'CALIBRATION')

    def test_monitor_timeout_interval_close_and_stop(self):
        monitor=object.__new__(mon.AssignedMonitorV4);monitor.gpu='0';monitor.samples=[];monitor.violations=[];monitor.stop=threading.Event();monitor.thread=None
        with mock.patch.object(mon.subprocess,'run',return_value=mock.Mock(stdout='a,b\n')) as run:
            self.assertEqual(monitor.query(['fake']),[['a','b']]);self.assertEqual(run.call_args.kwargs['timeout'],30)
        monitor.query=mock.Mock(side_effect=lambda args:(monitor.stop.set() or [[str(os.getpid()+100000),'4']]))
        monitor.sample();self.assertEqual(monitor.query.call_count,1)
        self.assertEqual(monitor.violations,['Foreign compute appeared on the assigned physical GPU'])
        monitor.check=mock.Mock();monitor.stop=mock.Mock();monitor.stop.wait=mock.Mock(return_value=True);monitor.sample=mock.Mock()
        fake_thread=mock.Mock();fake_thread.is_alive.return_value=False
        def make_thread(*args,**kw):kw['target']();return fake_thread
        with mock.patch.object(mon.threading,'Thread',side_effect=make_thread):monitor.start()
        monitor.stop.wait.assert_called_once_with(5);monitor.close();fake_thread.join.assert_called_once_with(timeout=35)

    def test_reused_dev_parent_must_be_failed_and_all_ten_match(self):
        states={str(i):{'raw_tensor_sha256':str(i),'shape':[1],'dtype':'bf16','path':'state'+str(i)} for i in range(10)}
        phase={'native_source_ordinal':894,'generation_receipt':{'path':'gen'},'states':states}
        old={'id':'old','cases':[{'controls':copy.deepcopy(states)}]}
        prev={'protocol':{'sha256':'p'},'RUN':{'path':'run','sha256':'r'},'launch_exit':{'path':'exit'}}
        reused={'native_source_ordinal':894,'GENERATION':phase['generation_receipt'],'states':states}
        run={'id':'old','protocol_sha256':'p','status':'FAILED_FULL_DEPTH_PART_PRESERVED_NO_RETRY','local_development_neutral_passed':True,'phases':[phase]}
        exit_={'id':'old','protocol_sha256':'p','run_sha256':'r','process_exited':True,'child_returncode':1,'automatic_retry':False}
        gen={'id':'old','protocol_sha256':'p','status':'COMPLETE_RAW_BF16_TEACHER_STATE_FULL160','native_source_ordinal':894,
          'role':'development','all_four_original_flow_and_post_bytes_passed':True,'completed_native_flow_steps':4,'native_ffn_sites':160,'states':states}
        plan={'failed_predecessor':prev,'reused_development_neutral':reused};docs={'run':run,'exit':exit_,'gen':gen}
        with mock.patch.object(m,'parent_protocol',return_value=old),mock.patch.object(m.native,'bound',side_effect=lambda root,item:item['path']),mock.patch.object(m.native,'read',side_effect=lambda p:docs[p]):
            m.verify_predecessor(plan,ROOT)
            docs['exit']=dict(exit_,child_returncode=0)
            with self.assertRaises(ValueError):m.verify_predecessor(plan,ROOT)
            docs['exit']=exit_;docs['gen']={**gen,'states':{}}
            with self.assertRaises(ValueError):m.verify_predecessor(plan,ROOT)
            docs['gen']=gen;docs['run']=copy.deepcopy(run);docs['run']['phases'].append({'native_source_ordinal':240,'generation_receipt':{'path':'already'}})
            with self.assertRaises(ValueError):m.verify_predecessor(plan,ROOT)


if __name__=='__main__':unittest.main()
