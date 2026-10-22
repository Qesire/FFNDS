"""Generate exactly one dense/candidate full free-running video pair.

Refuses model, torch and CUDA imports until the separate registered scope and
direct user authority are verified. Candidate replaces block20 in four steps.
"""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import shutil
import time

from rollout_contract import (Budget,METHOD,ROOT,atomic,read,require_registration,sha)


def resource_check(root,p):
    used=sum(f.stat().st_size for f in root.rglob('*') if f.is_file())
    if used>p['resources']['new_output_bytes_max'] or shutil.disk_usage(root).free<80<<30:
        raise RuntimeError('Paired video output/free-space cap exceeded')
    return used


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    p=require_registration()
    # This is the first torch import in the executable path.
    import torch
    from safetensors.torch import load_file,save_file
    from historical_source_snapshot import HistoricalSources
    from resident_ffn_runtime import make_resident,unused_raw_placeholders
    from restored_assets import verified_restored_assets
    from rollout_mean_contrast import MeanContrastHook
    from run_ar2_restart import tensor_sha
    from worker_identity import require_worker
    require_worker()
    if torch.cuda.device_count()!=1:raise RuntimeError('One locked physical GPU required')
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    root=Path(os.environ['HF_HOME'])/'derived/ffnds/R15V0_paired_rollout'
    if root.exists():raise RuntimeError('Preserve the first paired rollout; no automatic retry')
    root.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(root.parent).free<(80<<30)+(1<<28):raise RuntimeError('Insufficient HF output reserve')
    root.mkdir();budget=Budget(root)
    report={'id':METHOD,'status':'PREPARING_PAIRED_FULL_VIDEO','protocol_sha256':sha(ROOT/p['self_path']),
            'requests':[],'new_training_steps':0,'intermediate_latent_reset_to_DQ':False,
            'video_quality_or_Pareto_acceptance':False}
    report_path=root/'RUN.json';atomic(report_path,report)
    try:
        print('R15 paired video: verifying original frozen assets',flush=True)
        kwargs,cache,assets=verified_restored_assets(HistoricalSources())
        samples=read(ROOT/'.runtime/inputs/calibration24/canonical.json')
        sample=dict(samples[0]);sample['seed']=sample['calibration_seed'];sample['image_path']=sample['img_paths'][0]
        if sample['seed']!=42 or sha(sample['image_path'])!=p['image_sha256']:
            raise RuntimeError('Frozen paired source0/seed/image changed')
        from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
        print('R15 paired video: loading original native runtime',flush=True)
        runtime=adapter.HIF4CacheRuntime(cache['path'])
        parameters={};factor_tensors={};factor_records=[]
        affine_records=read(ROOT/'research/20261021/R9A1_native_affine_results.json')['affine_matrices']
        with budget.call('factors'):
            cpu_rng,cuda_rng=torch.get_rng_state(),torch.cuda.get_rng_state()
            try:
                for expert_id,expert in enumerate(('high_noise_model','low_noise_model')):
                    row=next(a for a in affine_records if (0 if expert_id==0 else 2) in a['shared_steps'])
                    affine_path=Path(cache['path']).parents[1]/row['HF_derived_relative']
                    if sha(affine_path)!=row['sha256']:raise RuntimeError('Original frozen affine changed')
                    A=load_file(str(affine_path),device='cpu')['matrix_BF16'].to('cuda')
                    if tensor_sha(A)!=row['tensor_sha256']['matrix']:raise RuntimeError('Frozen A tensor changed')
                    w=runtime.records[expert]['blocks.20.ffn.0']['weight_qdq'].to('cuda',dtype=torch.float32)
                    torch.manual_seed(p['factorization']['seed']+expert_id)
                    began=time.monotonic()
                    left,values,right=torch.svd_lowrank(w,q=256,niter=2)
                    U=(left*values.sqrt()).bfloat16().contiguous()
                    V=(values.sqrt()[:,None]*right.T).bfloat16().contiguous()
                    parameters[expert]={'A':A,'U':U,'V':V}
                    for key,value in [('U',U),('V',V)]:factor_tensors[f'{expert}.{key}']=value.cpu().contiguous()
                    factor_records.append({'expert':expert,'rank':256,'weight_shape':list(w.shape),
                        'weight_tensor_sha256':tensor_sha(w.bfloat16()),
                        'factor_tensor_sha256':{'U':tensor_sha(U),'V':tensor_sha(V)},
                        'approx_retained_weight_energy_fraction':float(values.square().sum()/w.square().sum()),
                        'elapsed_seconds':time.monotonic()-began})
                    del w,left,values,right
                    print('R15 paired video: fixed weight factorization '+str(expert_id)+' complete',flush=True)
            finally:
                torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
        save_file(factor_tensors,str(root/'factors.safetensors'));del factor_tensors
        report['factorization']={'records':factor_records,'file_sha256':sha(root/'factors.safetensors')}
        with unused_raw_placeholders(adapter) as audit:
            pipe=adapter.build_lightx2v_distill_pipeline(**kwargs)
        if len(audit['names'])!=400 or audit['calls']<800:
            raise RuntimeError('Original dense bridge placeholder coverage differs')
        report['residency']=make_resident(runtime)
        report['assets']=assets;report['image_sha256']=p['image_sha256'];report['seed']=42
        report['input_prompt_sha256']=__import__('hashlib').sha256(sample['prompt'].encode()).hexdigest()
        atomic(report_path,report)
        adapter.activate_lightx2v_hooks(runtime=runtime)
        try:
            for arm in p['arms']:
                resource_check(root,p)
                before=dict(runtime.counters)
                scheduler=pipe.runner.scheduler;old_post=scheduler.step_post
                scheduler.generator=None;trajectory=[];initial=[]
                def post():
                    step=int(scheduler.step_index)
                    if step==0:initial.append(tensor_sha(scheduler.latents.detach().float().cpu().contiguous()))
                    result=old_post()
                    latent=scheduler.latents.detach().float().cpu().contiguous()
                    if not bool(torch.isfinite(latent).all()):raise RuntimeError('Nonfinite free student state')
                    trajectory.append({'step':step,'raw_tensor_sha256':tensor_sha(latent),'shape':list(latent.shape)})
                    print(f'R15 paired video: {arm} completed free student step{step}',flush=True)
                    return result
                scheduler.step_post=post
                hook=MeanContrastHook(runtime,parameters,p['expected_grid']) if arm!='dense_DQ' else None
                video=root/(arm+'.mp4');torch.cuda.reset_peak_memory_stats();began=time.monotonic()
                try:
                    with budget.call(arm),torch.inference_mode(),(hook if hook else nullcontext()):
                        pipe.generate(seed=42,image_path=sample['image_path'],prompt=sample['prompt'],
                                      negative_prompt='',save_result_path=str(video))
                        torch.cuda.synchronize()
                finally:scheduler.step_post=old_post
                counters={k:runtime.counters[k]-before[k] for k in before}
                if (counters['linear_forward_calls']!=1600 or counters['activation_qdq_calls']!=1600
                        or counters['cache_misses'] or counters['weight_transfers'] or audit['fallback_calls']
                        or [r['step'] for r in trajectory]!=list(range(4)) or len(initial)!=1):
                    raise RuntimeError('Incomplete paired native model coverage or fallback')
                peak=torch.cuda.max_memory_allocated()
                if peak>p['resources']['peak_allocated_bytes_max']:raise RuntimeError('Paired model allocation cap exceeded')
                request={'arm':arm,'trajectory':trajectory,'initial_noise_sha256':initial[0],
                         'video_sha256':sha(video),'video_HF_relative':str(video.relative_to(Path(os.environ['HF_HOME']))),
                         'counter_delta':counters,'candidate_events':hook.events if hook else [],
                         'elapsed_seconds_including_observation':time.monotonic()-began,'peak_allocated_bytes':peak,
                         'performance_benchmark':False,'free_student_states_no_reset':True}
                report['requests'].append(request);report['new_output_bytes']=resource_check(root,p)
                atomic(report_path,report)
                print('R15 paired video: '+arm+' video written',flush=True)
        finally:adapter.deactivate_lightx2v_hooks()
        if (len(report['requests'])!=2 or report['requests'][0]['initial_noise_sha256']!=report['requests'][1]['initial_noise_sha256']
                or len(report['requests'][1]['candidate_events'])!=8):
            raise RuntimeError('Paired initial noise or actual candidate coverage differs')
        report['status']='COMPLETE_PAIRED_FULL_VIDEO_AWAITING_DECODE_AND_REVIEW'
        report['matched_initial_noise']=True
    except BaseException as error:
        report.update(status='FAILED_FIXED_SMOKE_DEBIT_PRESERVED',error_type=type(error).__name__)
        raise
    finally:
        report['budget']=budget.state;atomic(report_path,report);budget.close()
    print(json.dumps({'status':report['status'],'videos':len(report['requests']),
                      'matched_initial_noise':report['matched_initial_noise']}),flush=True)


if __name__=='__main__':main()
