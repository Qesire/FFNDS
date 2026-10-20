"""Collect native local dense targets on one frozen C24 student trajectory per input."""
import argparse
from contextlib import nullcontext
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time

import imageio.v2 as imageio
import torch

from ar2_checkpoint import atomic_json,canonical_json,digest
from distill_assets import ROOT,verified_assets
from fit_cstatic_centered import load_training_reports
from native_teacher_capture import NativeTeacherCapture,TeacherRecords
from run_ar2_restart import tensor_sha


class CaptureWithGeometry(NativeTeacherCapture):
    def observe(self,state,result):
        # Host metadata only; the validated full-native teacher call and all
        # numerical checks remain in the unchanged parent implementation.
        geometry={'native_V1_shape':list(state['actual_v1'].shape),
            'native_V1_stride':list(state['actual_v1'].stride()),
            'native_BaseY_shape':list(state['base_y'].shape)}
        super().observe(state,result)
        self.scopes[-1].update(geometry)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--shard',type=int,choices=[0,1],required=True)
    ap.add_argument('--collections',type=Path,nargs=2,required=True)
    ap.add_argument('--student-policy',type=Path,required=True)
    ap.add_argument('--teacher-control',type=Path,required=True)
    ap.add_argument('--raw-root',type=Path,default=Path('/home/xiang/.cache/huggingface/ffnds-derived/onpolicy_c24_ffn_operands_20261021'))
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();torch.set_num_threads(8)
    if not args.raw_root.resolve().is_relative_to(Path.home()/'.cache/huggingface/ffnds-derived'):
        raise RuntimeError('On-policy raw tensors must remain in shared home HF cache')
    if shutil.disk_usage('/var/tmp').free<80*1024**3:raise RuntimeError('Preserve80GiB node reserve')
    sources,bindings,capture=load_training_reports(args.collections,expected_inputs=24,expected_rows=256)
    for label in ['C24','VM1']:
        quality=json.loads((ROOT/f'runs/worker_restart_20261020/formal20_{label}_comparison/COMPARISON.json').read_text())
        if quality['status']!='COMPLETED_PAIRED_FORMAL20_THREE_METRIC_COMPARISON' or quality['formal20_quality_pass']:
            raise RuntimeError('OP1 bulk collection is conditional on both prior quality rejections')
    control=json.loads(args.teacher_control.read_text())
    observer_path=ROOT/'tools/native_teacher_capture.py';protocol_path=ROOT/'config/onpolicy_residual_op1.json'
    if (control['status']!='PASS_NATIVE_DENSE_TEACHER_BOTH_INPUTS_FULL160_AND_NOOP'
            or control['identity']['sources'][str(observer_path)]!=digest(observer_path)
            or control['identity']['protocol_sha256']!=digest(protocol_path)):
        raise RuntimeError('Same-source full native teacher/trajectory control must pass first')
    policy=json.loads(args.student_policy.read_text())
    if policy['experiment']!='RR1_CALIBRATION24_FIXED_RANK256_LAMBDA1_20261021':
        raise RuntimeError('OP1 must collect exactly the frozen C24 student')
    expected_positions={}
    for binding in policy['fit_reports'].values():
        if digest(binding['path'])!=binding['sha256']:raise RuntimeError('Frozen C24 base report changed')
        expected_positions.update({(p['step'],p['block']):p for p in json.loads(Path(binding['path']).read_text())['positions']})
    for index,case in sources.items():
        sample=case['input']
        if sample['calibration_seed']!=42+index or digest(sample['img_paths'][0])!=sample['input_provenance']['image_sha256']:
            raise RuntimeError('Original24 images/seeds changed')
    kwargs,cache,assets=verified_assets()
    if capture['assets']!=assets or control['identity']['assets']!=assets:raise RuntimeError('Teacher/student assets differ')
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'JOB.lock').open('w');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity={'protocol_sha256':digest(protocol_path),'teacher_control_sha256':digest(args.teacher_control),
        'student_policy_sha256':digest(args.student_policy),'calibration_collections':bindings,'assets':assets,
        'shard':args.shard,'input_count':24,'rows_per_scope':256,'torch':torch.__version__,
        'sources':{str(p):digest(p) for p in [Path(__file__),observer_path,ROOT/'tools/trained_packed_runtime.py',
            ROOT/'tools/reduced_rank_residual.py',ROOT/'.runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/lightx2v_wan_distill.py']}}
    fingerprint=hashlib.sha256(canonical_json(identity).encode()).hexdigest();report=args.out/'RUN.json'
    completed=[]
    if report.exists():
        previous=json.loads(report.read_text())
        if previous['identity']!=identity:raise RuntimeError('On-policy collection identity changed; select new output')
        completed=previous['completed']
    for entry in completed:
        if digest(entry['input_report'])!=entry['input_report_sha256']:raise RuntimeError('Completed input report changed')
    meta={'status':'RUNNING','fingerprint':fingerprint,'identity':identity,'completed':completed,'quality_or_speed_acceptance':False}
    atomic_json(report,meta)
    teacher=TeacherRecords(cache)
    from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
    try:
        pending=[i for i in range(24) if i%2==args.shard and i not in {r['index'] for r in completed}]
        if pending:
            pipe=adapter.build_lightx2v_distill_pipeline(**kwargs)
            from packed_reference_runtime import create_runtime
            runtime=create_runtime(cache['path'],args.student_policy);adapter.activate_lightx2v_hooks(runtime=runtime)
            try:
                for index in pending:
                    sample=sources[index]['input'];parent=args.raw_root/fingerprint/f'input{index:02d}';parent.mkdir(parents=True,exist_ok=True)
                    if digest(sample['img_paths'][0])!=sample['input_provenance']['image_sha256']:
                        raise RuntimeError('Frozen source image changed before use')
                    attempt=1
                    while (parent/f'attempt{attempt}').exists():attempt+=1
                    out=parent/f'attempt{attempt}';out.mkdir()
                    observer=CaptureWithGeometry(runtime,teacher,out,index)
                    trajectories={};videos={};phase_counters={}
                    for phase in ['plain','observed']:
                        scheduler=pipe.runner.scheduler;original_post=scheduler.step_post;trajectory=[]
                        def post():
                            result=original_post();latent=scheduler.latents.detach().float().cpu().contiguous()
                            if not bool(torch.isfinite(latent).all()):raise RuntimeError('Nonfinite student trajectory')
                            trajectory.append({'step':int(scheduler.step_index),'sha256':tensor_sha(latent),'shape':list(latent.shape)})
                            return result
                        scheduler.step_post=post;scheduler.generator=None;runtime.begin_request();before=dict(runtime.counters)
                        video=out/f'{phase}.mp4';started=time.perf_counter()
                        try:
                            with torch.inference_mode(),observer if phase=='observed' else nullcontext():
                                pipe.generate(seed=42+index,image_path=sample['img_paths'][0],prompt=sample['prompt'],
                                    negative_prompt='',save_result_path=str(video))
                            torch.cuda.synchronize()
                        finally:scheduler.step_post=original_post
                        delta={k:runtime.counters[k]-before[k] for k in before}
                        packed=runtime.report()['packed_reference']
                        if (delta['linear_forward_calls']!=1600 or delta['activation_qdq_calls']!=1600 or delta['cache_misses']
                                or packed['actual_FFN_linear_calls']!=320 or packed['residual_correction']['actual_positions']!=160
                                or packed['residual_correction']['pending_predictions']):raise RuntimeError('Incomplete frozen student coverage')
                        if [r['step'] for r in trajectory]!=list(range(4)):raise RuntimeError('Incomplete four-step student trajectory')
                        trajectories[phase]=trajectory;phase_counters[phase]=delta
                        videos[phase]={'path':str(video),'sha256':digest(video),'elapsed_including_observation':time.perf_counter()-started}
                        atomic_json(out/f'PHASE_{phase}.json',{'phase':phase,'trajectory':trajectory,'video':videos[phase],
                            'seed':42+index,'scheduler_generator_reset':True,'native_counter_delta':delta})
                        print(f'DONE OP1 input{index:02d} {phase}',flush=True)
                    if trajectories['plain']!=trajectories['observed'] or videos['plain']['sha256']!=videos['observed']['sha256']:
                        raise RuntimeError('Local dense teacher observation changed student latent/video bytes; operands rejected')
                    if len(observer.scopes)!=160 or {(s['step'],s['block']) for s in observer.scopes}!={(s,b) for s in range(4) for b in range(40)}:
                        raise RuntimeError('On-policy input lacks all160 positions')
                    for scope in observer.scopes:
                        info=expected_positions[scope['step'],scope['block']]
                        if (scope['expert']!=('high_noise_model' if scope['step']<2 else 'low_noise_model')
                                or scope['gate_sha256']!=info['gate_sha256']
                                or scope['original_b2_sha256']!=info['original_b2_sha256']
                                or scope['effective_base_bias_sha256']!=info['policies']['C']['modes']['centered_gain_bias']['b2_sha256']):
                            raise RuntimeError('Native observed expert/gate/original or folded bias differs from frozen C24')
                    with imageio.get_reader(videos['observed']['path']) as reader:
                        frames,info=reader.count_frames(),reader.get_meta_data()
                    if frames!=81 or abs(float(info['fps'])-16)>.01:raise RuntimeError('Student video frame/fps differs')
                    atomic_json(out/'SCOPES.json',observer.scopes)
                    path=out/'INPUT.json'
                    atomic_json(path,{'status':'PASS_NATIVE_ONPOLICY_FULL160_AND_NOOP','fingerprint':fingerprint,'index':index,
                        'input':sample,'trajectories':trajectories,'videos':videos,'scopes':observer.scopes,
                        'native_counter_deltas':phase_counters,'observed_video':{'frames':frames,'fps':info['fps'],'size':info['size']},
                        'offline_extra_full_teacher_GEMMs':sum(s['extra_full_teacher_GEMMs'] for s in observer.scopes),
                        'no_op_limit':'same initialized frozen student pipeline, independently reset private seed generator',
                        'quality_or_speed_acceptance':False})
                    completed.append({'index':index,'input_report':str(path),'input_report_sha256':digest(path)})
                    atomic_json(report,meta);print(f'COMMITTED OP1 input{index:02d}: all160 targets and no-op PASS',flush=True)
            finally:adapter.deactivate_lightx2v_hooks()
        if {r['index'] for r in completed}!={i for i in range(24) if i%2==args.shard}:
            raise RuntimeError('Incomplete on-policy shard')
        meta['status']='PASS_ONPOLICY_CALIBRATION24_SHARD_AND_NOOP'
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error));raise
    finally:atomic_json(report,meta)


if __name__=='__main__':main()
