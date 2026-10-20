"""Verify all160 native local dense teacher FFNs and whole trajectories on two inputs."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import torch

from ar2_checkpoint import atomic_json,digest
from distill_assets import ROOT,verified_assets
from fit_cstatic_centered import load_training_reports
from native_teacher_capture import NativeTeacherCapture,TeacherRecords
from run_ar2_restart import tensor_sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--collections',type=Path,nargs=2,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();torch.set_num_threads(8)
    records,bindings,capture=load_training_reports(args.collections,expected_inputs=24,expected_rows=256)
    kwargs,cache,assets=verified_assets()
    if capture['assets']!=assets:raise RuntimeError('Dense reference model/cache assets differ')
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'JOB.lock').open('w');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (args.out/'RUN.json').exists():raise RuntimeError('Select a fresh immutable native teacher control')
    identity={'collections':bindings,'assets':assets,'protocol_sha256':digest(ROOT/'config/onpolicy_residual_op1.json'),
        'sources':{str(p):digest(p) for p in [Path(__file__),ROOT/'tools/native_teacher_capture.py',
            ROOT/'.runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/lightx2v_wan_distill.py',
            ROOT/'.runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/wan_i2v.py',
            ROOT/'.runtime/code/LightX2V/lightx2v/models/networks/wan/infer/transformer_infer.py']},
        'torch':torch.__version__,'GPU':torch.cuda.get_device_name()}
    meta={'status':'RUNNING','identity':identity,'completed':[],'quality_or_speed_acceptance':False}
    atomic_json(args.out/'RUN.json',meta)
    from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
    teacher=TeacherRecords(cache);started=time.perf_counter()
    try:
        pipe=adapter.build_lightx2v_distill_pipeline(**kwargs);runtime=adapter.HIF4CacheRuntime(cache['path'])
        adapter.activate_lightx2v_hooks(runtime=runtime)
        try:
            for index in [0,1]:
                case=records[index];sample=case['input'];folder=args.out/f'input{index:02d}';folder.mkdir()
                trajectory=[];before=dict(runtime.counters);scheduler=pipe.runner.scheduler
                original_post=scheduler.step_post
                def post():
                    result=original_post();latent=scheduler.latents.detach().float().cpu().contiguous()
                    if not bool(torch.isfinite(latent).all()):raise RuntimeError('Nonfinite teacher-control latent')
                    trajectory.append({'step':int(scheduler.step_index),'sha256':tensor_sha(latent),'shape':list(latent.shape)})
                    return result
                scheduler.step_post=post;scheduler.generator=None;video=folder/'output.mp4'
                observer=NativeTeacherCapture(runtime,teacher,folder,index,dense_control=True)
                try:
                    with torch.inference_mode(),observer:
                        pipe.generate(seed=42+index,image_path=sample['img_paths'][0],prompt=sample['prompt'],
                            negative_prompt='',save_result_path=str(video))
                    torch.cuda.synchronize()
                finally:scheduler.step_post=original_post
                delta={k:runtime.counters[k]-before[k] for k in before}
                if delta['linear_forward_calls']!=1600 or delta['activation_qdq_calls']!=1600 or delta['cache_misses']:
                    raise RuntimeError('Incomplete original dense bridge coverage')
                if trajectory!=case['trajectories']['plain'] or digest(video)!=case['videos']['plain']['sha256']:
                    raise RuntimeError('Native teacher observation changed actual dense latents/video bytes')
                if (len(observer.scopes)!=160 or {(s['step'],s['block']) for s in observer.scopes}!=
                        {(s,b) for s in range(4) for b in range(40)} or any(not s['Q1_teacher_equals_actual'] or
                            not s['Q2_teacher_equals_actual'] or not s['Y_teacher_equals_actual_dense'] for s in observer.scopes)):
                    raise RuntimeError('Not every original full native FFN matched bitwise')
                result={'status':'PASS_NATIVE_DENSE_TEACHER_FULL160_AND_TRAJECTORY_NOOP','index':index,
                    'reference_input_report_sha256':next(e['input_report_sha256'] for binding in bindings for e in binding['inputs'] if e['index']==index),
                    'trajectory':trajectory,'video_sha256':digest(video),'original_hif4_counter_delta':delta,
                    'offline_extra_full_teacher_GEMMs':sum(s['extra_full_teacher_GEMMs'] for s in observer.scopes),
                    'scopes':observer.scopes,'quality_or_speed_acceptance':False}
                atomic_json(folder/'INPUT.json',result)
                meta['completed'].append({'index':index,'report':str(folder/'INPUT.json'),'sha256':digest(folder/'INPUT.json')})
                atomic_json(args.out/'RUN.json',meta);print(f'CONTROL input{index}: all160 Q1/Q2/Y exact, all4 latents/video exact',flush=True)
        finally:adapter.deactivate_lightx2v_hooks()
        meta.update(status='PASS_NATIVE_DENSE_TEACHER_BOTH_INPUTS_FULL160_AND_NOOP',elapsed_seconds=time.perf_counter()-started,
            teacher_shards_verified=len(teacher.verified))
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error));raise
    finally:atomic_json(args.out/'RUN.json',meta)


if __name__=='__main__':main()
