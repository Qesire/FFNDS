"""Verify one already generated video from a closed failed capture; no generation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import e0_original_bf16_videos as native
from worker_identity import private_config_path, require_worker

ROOT=Path(__file__).resolve().parents[1]
ID='E0_RAW_FULL_DEPTH_VIDEO_FINALIZE_ORD288'
RESOURCES={'GPU':0,'CPU_threads':2,'wall_seconds':120,'outer_seconds':150,'new_HF_bytes':1<<20}
LIMITS={'existing_video_full_decodes':1,'new_video_generations':0,'model_calls':0,'GPU_calls':0}


def run(args):
    start=time.monotonic();plan=native.read(args.protocol);digest=native.sha(args.protocol)
    native.need(plan['id']==ID and plan['status']=='FROZEN_REGISTERED' and plan['resources']==RESOURCES and plan['limits']==LIMITS,'Exact video verification scope differs')
    active=native.read(ROOT/'research/ACTIVE_RESEARCH.json');scopes=[s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id')==ID]
    native.need(active['user_continuation_authorization_20261027']['new_scope_registration_allowed'] is True and len(scopes)==1 and scopes[0]['protocol_sha256']==digest and scopes[0]['execution_status']=='RUNNING_FIXED_SCOPE' and scopes[0]['resources']==RESOURCES and scopes[0]['limits']==LIMITS,'Registered video verification required')
    native.need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and len(os.sched_getaffinity(0))==2 and not {'torch'}&set(sys.modules),'Fresh CPU verification with hidden CUDA required')
    require_worker();native.need(native.sha(private_config_path())==plan['private_worker_config_sha256'],'Execution worker differs')
    for b in plan['repo_sources']:native.bound(ROOT,b)
    old=native.read(native.bound(ROOT,plan['parent']['protocol']));prior=native.read(native.bound(args.hf_root,plan['parent']['RUN']));ex=native.read(native.bound(args.hf_root,plan['parent']['exit']))
    native.need(prior['id']==ex['id']==old['id'] and prior['protocol_sha256']==ex['protocol_sha256']==plan['parent']['protocol']['sha256'] and ex['run_sha256']==plan['parent']['RUN']['sha256'] and ex['process_exited'] is True and ex['child_returncode']==1 and ex['timeout'] is False and not ex['automatic_retry'],'Closed original failed scope identity differs')
    native.need(prior['status']=='FAILED_FULL_DEPTH_PART_PRESERVED_NO_RETRY' and prior['error_type']=='TimeoutError' and not prior['assigned_GPU_resources']['violations'],'Original timeout/isolation identity differs')
    phase=next(p for p in prior['phases'] if p['native_source_ordinal']==288)
    native.need(phase['role']=='calibration' and len(phase['packets'])==160 and len(phase['states'])==10 and phase['status']=='FAILED_SOURCE_FULL_DEPTH_PRESERVED_NO_RETRY' and phase['error_type']=='TimeoutError','Existing source capture boundary differs')
    video=native.bound(args.hf_root,plan['video']);expected=(args.hf_root/old['output_HF_relative']/phase['run_id']/'output.mp4').resolve();native.need(video==expected,'Existing video locator differs')
    out=(args.hf_root/plan['output_HF_relative']).resolve();native.need(out.is_relative_to(args.hf_root.resolve()) and not out.exists(),'Fresh confined verification result required');out.mkdir(parents=True)
    report={'id':ID,'protocol_sha256':digest,'status':'STARTED_EXISTING_VIDEO_VERIFICATION','native_source_ordinal':288,'parent':plan['parent'],'video':plan['video'],'original_native_scope_remains_failed':True,'automatic_retry':False,'model_calls':0,'GPU_calls':0,'new_video_generations':0,'predebited_existing_video_full_decodes':1,'observed_existing_video_full_decodes':0}
    def timeout(*_):raise TimeoutError('CPU existing video verification wall120 exhausted')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120);native.write(out/'RUN.json',report)
    try:
        from run_ub1_consumer_videos import inspect_video
        result=inspect_video(video,native.VIDEO)
        native.need(result['sha256']==plan['video']['sha256'] and result['bytes']==plan['video']['bytes'],'Decoded existing video bytes differ')
        native.need('torch' not in sys.modules,'Neural framework imported during CPU video verification')
        report.update(status='COMPLETE_VIDEO_FULL_DECODE',decoded_frames=81,decoded_fps=16,decoded_size=[832,464],inspection=result,observed_existing_video_full_decodes=1)
    except BaseException as error:
        report.update(status='FAILED_VIDEO_VERIFICATION_PRESERVED_NO_RETRY',error_type=type(error).__name__,error=str(error));raise
    finally:
        signal.alarm(0);report['wall_seconds']=time.monotonic()-start;native.write(out/'RUN.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--hf-root',type=Path,required=True);args=p.parse_args();args.hf_root=args.hf_root.resolve();print(json.dumps({'status':run(args)['status']}))


if __name__=='__main__':main()
