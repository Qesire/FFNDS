"""Checkpointed formal20 generation with one persistent, explicitly reseeded pipeline."""
import argparse
import fcntl
import importlib.metadata
import json
from pathlib import Path
import time

import torch
from safetensors.torch import save_file

from ar2_checkpoint import atomic_json, canonical_json, digest
from distill_assets import ROOT, verified_assets
from run_ar2_restart import tensor_sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs',type=Path,default=ROOT/'.runtime/inputs/formal20/canonical.json')
    ap.add_argument('--indices',type=int,nargs='+',required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--packed-policy',type=Path)
    args=ap.parse_args()
    args.out=args.out.resolve()
    from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
    torch.set_num_threads(8)
    samples=json.loads(args.inputs.read_text())
    recovery_path=ROOT/'.runtime/inputs/formal20/recovery_report.json'
    recovery=json.loads(recovery_path.read_text())
    if recovery['status']!='PASS_NEW_FORMAL20_INPUT_RECOVERY' or recovery['canonical_sha256']!=digest(args.inputs):
        raise RuntimeError('Frozen formal input identity changed')
    if (len(samples)!=20 or len({r['source_group'] for r in samples})!=20 or
            len(set(args.indices))!=len(args.indices) or any(i not in range(20) for i in args.indices)):
        raise RuntimeError('Invalid independent formal input slice')
    for row in samples:
        if row['seed']!=42 or row['dimension'] not in ['i2v_subject','i2v_background'] or digest(row['image_path'])!=row['input_provenance']['image_sha256']:
            raise RuntimeError('Formal seed/dimension/image changed')
    args.out.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'JOB.lock').open('w')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    kwargs,cache,assets=verified_assets()
    identity={'assets':assets,'inputs_sha256':digest(args.inputs),'input_recovery_sha256':digest(recovery_path),
              'indices':args.indices,'arm':'D-Q-PACKED-TRAINED-REFERENCE' if args.packed_policy else 'D-Q',
              'packed_policy_sha256':digest(args.packed_policy) if args.packed_policy else None,
              'sources':{str(p):digest(p) for p in [Path(__file__),ROOT/'tools/distill_assets.py']},
              'versions':{n:importlib.metadata.version(n) for n in ['torch','torchvision','transformers','diffusers','safetensors','numpy','triton']},
              'native_sources':{str(p):digest(p) for p in [Path(adapter.__file__),
                  ROOT/'.runtime/code/LightX2V/lightx2v/models/networks/wan/infer/transformer_infer.py',
                  ROOT/'.runtime/code/LightX2V/lightx2v/common/ops/mm/mm_weight.py',
                  ROOT/'.runtime/code/LightX2V/lightx2v/models/schedulers/wan/step_distill/scheduler.py',
                  ROOT/'.runtime/code/LightX2V/lightx2v/models/schedulers/wan/scheduler.py']},
              'seed_semantics':'scheduler.generator=None before every native request'}
    report=args.out/'RUN.json'
    completed=[]
    if report.exists():
        previous=json.loads(report.read_text())
        if previous['identity']!=identity:
            raise RuntimeError('Formal generation identity changed; select new output')
        completed=previous['completed']
    for row in completed:
        if digest(row['case_report'])!=row['case_report_sha256']:
            raise RuntimeError('Formal case report changed')
        case=json.loads(Path(row['case_report']).read_text())
        if case['status']!='PASS_FORMAL_INPUT_GENERATION_ONLY' or digest(case['video']['path'])!=case['video']['sha256']:
            raise RuntimeError('Formal video changed')
    meta={'status':'RUNNING','identity':identity,'completed':completed,'quality_or_speed_acceptance':False}
    atomic_json(report,meta)
    try:
        pending=[i for i in args.indices if i not in {r['index'] for r in completed}]
        if pending:
            pipe=adapter.build_lightx2v_distill_pipeline(**kwargs)
            adapter.validate_effective_pipeline_config_against_contract(pipe.runner.config)
            if args.packed_policy:
                from packed_reference_runtime import create_runtime
                runtime=create_runtime(cache['path'],args.packed_policy)
            else:
                runtime=adapter.HIF4CacheRuntime(cache['path'])
            adapter.activate_lightx2v_hooks(runtime=runtime)
            try:
                for index in pending:
                    case=generate_one(pipe,runtime,samples[index],index,args.out,identity)
                    completed.append({'index':index,'case_report':str(case),'case_report_sha256':digest(case)})
                    atomic_json(report,meta)
                    print(f'COMMITTED formal input{index:02d}, arm={identity["arm"]}',flush=True)
            finally:
                adapter.deactivate_lightx2v_hooks()
        if len(completed)!=len(args.indices) or {r['index'] for r in completed}!=set(args.indices):
            raise RuntimeError('Incomplete/duplicate formal generation slice')
        meta['status']='PASS_FORMAL_GENERATION_SLICE_ONLY'
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error));raise
    finally:
        atomic_json(report,meta)


def generate_one(pipe,runtime,sample,index,out,identity):
    parent=out/f'input{index:02d}'
    parent.mkdir(parents=True,exist_ok=True)
    attempt=1
    while (parent/f'attempt{attempt}').exists():
        attempt+=1
    case=parent/f'attempt{attempt}'
    case.mkdir()
    scheduler=pipe.runner.scheduler
    original_post=scheduler.step_post
    trajectory=[]
    before=dict(runtime.counters)
    if hasattr(runtime,'begin_request'):
        runtime.begin_request()
    scheduler.generator=None
    torch.cuda.reset_peak_memory_stats()

    def post():
        result=original_post()
        latent=scheduler.latents.detach().float().cpu().contiguous()
        if not bool(torch.isfinite(latent).all()):
            raise RuntimeError('Nonfinite formal trajectory')
        step=int(scheduler.step_index)
        path=case/f'latent_step{step}.safetensors'
        save_file({'latents':latent},str(path))
        trajectory.append({'step':step,'denoising_step':int(scheduler.denoising_step_list[step]),
                           'sigma':float(scheduler.sigmas[step]),'shape':list(latent.shape),
                           'raw_tensor_sha256':tensor_sha(latent),'path':str(path),'file_sha256':digest(path)})
        atomic_json(case/'TRAJECTORY.json',trajectory)
        return result

    scheduler.step_post=post
    video=case/'output.mp4'
    started=time.perf_counter()
    try:
        with torch.inference_mode():
            pipe.generate(seed=42,image_path=sample['image_path'],prompt=sample['prompt'],negative_prompt='',save_result_path=str(video))
        torch.cuda.synchronize()
    finally:
        scheduler.step_post=original_post
    elapsed=time.perf_counter()-started
    if [r['step'] for r in trajectory]!=list(range(4)) or [r['denoising_step'] for r in trajectory]!=[1000,750,500,250]:
        raise RuntimeError('Incomplete formal four-step trajectory')
    counters={k:runtime.counters[k]-before[k] for k in before}
    if counters['linear_forward_calls']!=1600 or counters['activation_qdq_calls']!=1600 or counters['cache_misses']:
        raise RuntimeError('Formal run lacks complete native HiF4 coverage')
    runtime_report=runtime.report()
    if identity['packed_policy_sha256'] and runtime_report['packed_reference']['actual_FFN_linear_calls']!=320:
        raise RuntimeError('Formal packed run lacks all160 FFN positions')
    import imageio.v2 as imageio
    with imageio.get_reader(video) as reader:
        frames,info=reader.count_frames(),reader.get_meta_data()
    if frames!=81 or abs(float(info['fps'])-16)>.01:
        raise RuntimeError('Formal frame count/fps changed')
    result={'status':'PASS_FORMAL_INPUT_GENERATION_ONLY','index':index,'input':sample,
            'arm':identity['arm'],'model_assets':identity['assets'],'scheduler_generator_reset':True,
            'trajectory':trajectory,'hif4_counter_delta':counters,'hif4_runtime':runtime_report,
            'video':{'path':str(video),'sha256':digest(video),'frames':frames,'fps':info['fps'],'size':info['size']},
            'generation_wall_seconds_including_observation':elapsed,
            'peak_cuda_memory_bytes':torch.cuda.max_memory_allocated(),'quality_or_speed_acceptance':False}
    atomic_json(case/'INPUT.json',result)
    return case/'INPUT.json'


if __name__=='__main__':
    main()
