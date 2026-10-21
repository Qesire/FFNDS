"""Versioned three-arm UB2 benchmark at fixed formal input0."""
import argparse
import fcntl
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ar2_checkpoint import atomic_json,digest
from benchmark_resources import ResourceMonitor
from distill_assets import ROOT,verified_assets
from resident_ffn_runtime import FullFFNProfiler,make_resident,unused_raw_placeholders
from run_ar2_restart import tensor_sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm',choices=['DQ','UB1','UB2'],required=True)
    ap.add_argument('--reference-case',type=Path,required=True)
    ap.add_argument('--policy',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--payload-root',type=Path,required=True)
    ap.add_argument('--measured-repeats',type=int,default=3)
    args=ap.parse_args()
    if args.measured_repeats != 3 or bool(args.policy)!=(args.arm!='DQ'):
        ap.error('Exactly one warmup+3 measured requests; policy required only for UB1/UB2')
    if (not args.payload_root.resolve().is_relative_to(Path('/var/tmp/ffnds-xiang/huggingface/ffnds-derived'))
            or shutil.disk_usage('/var/tmp').free<82*1024**3):
        raise RuntimeError('Benchmark raw outputs require node HF cache and80GiB reserve plus2GiB allowance')
    args.payload_root.mkdir(parents=True,exist_ok=True)
    r=ROOT/'runs/worker_restart_20261020'
    condition=json.loads((r/'terminal_dense_ub2_controller/RUN.json').read_text())
    equivalence=json.loads((r/'formal20_UB2_nonresident_equivalence/RUN.json').read_text())
    active_policy=r/'terminal_dense_ub2_fit/UB2_policy.json'
    if (condition['status']!='COMPLETED_UB2_BOTH_OBSERVED_COHORT_QUALITY_PASSED'
            or not condition['both_observed_cohort_quality_pass']
            or condition['policy_sha256']!=digest(active_policy)
            or equivalence['status']!='PASS_FORMAL_GENERATION_SLICE_ONLY'
            or equivalence['identity']['packed_policy_sha256']!=digest(active_policy)
            or len(equivalence['completed'])!=20
            or not all(v['paired_latents_and_video_equal'] for v in equivalence['completed'])):
        raise RuntimeError('Finish both quality gates and all20 active resident/nonresident equivalence first')
    supplemental=json.loads((r/'disjoint20_UB2_nonresident_equivalence/RUN.json').read_text())
    if (supplemental['status']!='PASS_FORMAL_GENERATION_SLICE_ONLY'
            or supplemental['identity']['packed_policy_sha256']!=digest(active_policy)
            or len(supplemental['completed'])!=20
            or not all(v['paired_latents_and_video_equal'] for v in supplemental['completed'])):
        raise RuntimeError('Both complete20-source active equivalence cohorts are required')
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'JOB.lock').open('w');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    report=args.out/'RUN.json'
    if report.exists():raise RuntimeError('Timing runs are immutable; select a fresh output')
    reference=json.loads(args.reference_case.read_text())
    sample=json.loads((ROOT/'.runtime/inputs/formal20/canonical.json').read_text())[0]
    expected_policy=(None if args.arm=='DQ' else (r/'ub1_step_budget_fit/UB1_policy.json' if args.arm=='UB1' else active_policy))
    if (reference['status']!='PASS_FORMAL_INPUT_GENERATION_ONLY' or reference['index']!=0 or reference['input']!=sample
            or digest(sample['image_path'])!=sample['input_provenance']['image_sha256']
            or (args.policy and digest(args.policy)!=digest(expected_policy))
            or digest(reference['video']['path'])!=reference['video']['sha256']):
        raise RuntimeError('Exact fixed same-input reference and frozen arm policy required')
    packed=reference['hif4_runtime'].get('packed_reference')
    if (bool(packed)!=(args.arm!='DQ') or (packed and packed['policy_sha256']!=digest(args.policy))):
        raise RuntimeError('Reference latent trajectory belongs to another arm/policy')
    kwargs,cache,assets=verified_assets()
    if reference['model_assets']!=assets:raise RuntimeError('Reference model/cache differs')
    reference_latents=[]
    for row in reference['trajectory']:
        if digest(row['path'])!=row['file_sha256']:raise RuntimeError('Reference latent file changed')
        with safe_open(row['path'],framework='pt',device='cpu') as handle:
            raw=tensor_sha(handle.get_tensor('latents'))
        if raw!=row['raw_tensor_sha256']:raise RuntimeError('Reference latent bits changed')
        reference_latents.append(raw)
    if len(reference_latents)!=4:raise RuntimeError('Four reference native steps required')
    names=['run_terminal_dense_benchmark.py','resident_ffn_runtime.py','trained_packed_runtime.py',
           'reduced_rank_residual.py','benchmark_resources.py','step_budget_runtime.py',
           'terminal_dense_runtime.py','terminal_dense_validation.py','compare_ffn_benchmark.py',
           'compare_terminal_dense_benchmark.py']
    identity={'assets':assets,'input':sample,'arm':args.arm,
        'reference_meta_sha256':digest(args.reference_case),'reference_raw_latents':reference_latents,
        'policy_sha256':digest(args.policy) if args.policy else None,
        'protocol_sha256':digest(ROOT/'config/terminal_dense_benchmark_ub2.json'),
        'sources':{str(ROOT/'tools'/n):digest(ROOT/'tools'/n) for n in names},
        'torch':torch.__version__,'device':torch.cuda.get_device_name(),'TF32':False,
        'versions':{n:importlib.metadata.version(n) for n in ['torch','numpy','safetensors','transformers','diffusers','triton']},
        'cuda_runtime':torch.version.cuda,'measured_repeats':3}
    meta={'status':'RUNNING','identity':identity,'requests':[],'full_project_acceptance':False}
    atomic_json(report,meta)
    profiler=None;monitor=None
    try:
        if args.measured_repeats:
            monitor=ResourceMonitor();monitor.start()
        from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
        started=time.perf_counter()
        with unused_raw_placeholders(adapter) as audit:
            pipe=adapter.build_lightx2v_distill_pipeline(**kwargs)
        meta['raw_placeholder_audit']={k:sorted(v) if isinstance(v,set) else v for k,v in audit.items()}
        if len(audit['names'])!=400 or audit['calls']<800:
            raise RuntimeError('Expected all400 target names in both native experts')
        if args.policy:
            if args.arm == 'UB2':
                from terminal_dense_runtime import create_runtime
            else:
                from step_budget_runtime import create_runtime
            runtime=create_runtime(cache['path'],args.policy)
        else:
            runtime=adapter.HIF4CacheRuntime(cache['path'])
        meta['residency']=make_resident(runtime)
        meta['setup_seconds']=time.perf_counter()-started
        adapter.activate_lightx2v_hooks(runtime=runtime)
        profiler=FullFFNProfiler(adapter);profiler.install();runtime.full_ffn_profiler_active=True
        scheduler=pipe.runner.scheduler;original_post=scheduler.step_post
        for repetition in range(1+args.measured_repeats):
            if monitor is not None:monitor.check()
            folder=args.payload_root/f'request{repetition:02d}';folder.mkdir()
            trajectory=[];before=dict(runtime.counters)
            if hasattr(runtime,'begin_request'):runtime.begin_request()
            profiler.begin_request();scheduler.generator=None
            torch.cuda.reset_peak_memory_stats()

            def post():
                result=original_post()
                latent=scheduler.latents.detach().float().cpu().contiguous()
                step=int(scheduler.step_index);path=folder/f'latent_step{step}.safetensors'
                save_file({'latents':latent},str(path))
                trajectory.append({'step':step,'raw_tensor_sha256':tensor_sha(latent),'path':str(path),
                    'sha256':digest(path),'shape':list(latent.shape)})
                return result

            scheduler.step_post=post;video=folder/'output.mp4';started=time.perf_counter()
            try:
                with torch.inference_mode():
                    pipe.generate(seed=sample['seed'],image_path=sample['image_path'],prompt=sample['prompt'],
                        negative_prompt='',save_result_path=str(video))
                torch.cuda.synchronize()
            finally:
                scheduler.step_post=original_post
            counters={k:runtime.counters[k]-before[k] for k in before}
            if monitor is not None:monitor.check()
            if (counters['linear_forward_calls']!=1600 or counters['activation_qdq_calls']!=1600
                    or counters['cache_misses'] or counters['weight_transfers'] or audit['fallback_calls']):
                raise RuntimeError('Incomplete resident bridge coverage or placeholder fallback')
            if [r['step'] for r in trajectory]!=list(range(4)) or [r['raw_tensor_sha256'] for r in trajectory]!=reference_latents:
                raise RuntimeError('Resident/profiling interventions changed the paired trajectory')
            request={'repetition':repetition,'warmup':repetition==0,'trajectory':trajectory,
                'FFN_timing':profiler.report(),'hif4_counter_delta':counters,'hif4_runtime':runtime.report(),
                'wall_seconds_including_attention_VAE_observation':time.perf_counter()-started,
                'peak_cuda_memory_bytes':torch.cuda.max_memory_allocated(),
                'video':{'path':str(video),'sha256':digest(video)},'four_latents_bitwise_equal':True,'video_bytes_equal':digest(video)==reference['video']['sha256']}
            if not request['video_bytes_equal']:raise RuntimeError('Profiled video bytes changed')
            if args.arm=='UB2':
                from terminal_dense_validation import validate_runtime
                validate_runtime(request['hif4_runtime']['packed_reference'],digest(args.policy))
            elif args.arm=='UB1':
                from compare_ffn_benchmark import validate_candidate_runtime
                validate_candidate_runtime(request['hif4_runtime']['packed_reference'],digest(args.policy))
            shared_request=args.out/f'request{repetition:02d}_REQUEST.json'
            atomic_json(shared_request,request)
            meta['requests'].append({'report':str(shared_request),'sha256':digest(shared_request),
                'warmup':repetition==0,'FFN_wall_ms':request['FFN_timing']['wall_ms'],
                'FFN_cuda_ms':request['FFN_timing']['cuda_ms'],'four_latents_bitwise_equal':True,'video_bytes_equal':True})
            atomic_json(report,meta)
            print(f'RESIDENT request{repetition}: all4 latents bitwise, full160 FFN wall_ms={request["FFN_timing"]["wall_ms"]:.3f}',flush=True)
        meta['status']='PASS_RESIDENT_FULL160_AND_PAIRED_LATENT_EQUIVALENCE'
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error));raise
    finally:
        if profiler is not None:profiler.restore()
        if monitor is not None:
            monitor.close();meta['resources']=monitor.report()
            if monitor.violations:
                meta.update(status='FAILED_RESUMABLE',error='Resource isolation failed; timings unusable')
        atomic_json(report,meta)
    if meta['status']!='PASS_RESIDENT_FULL160_AND_PAIRED_LATENT_EQUIVALENCE':
        raise RuntimeError(meta.get('error','Resident benchmark failed'))
    print(json.dumps({k:v for k,v in meta.items() if k not in ['identity','raw_placeholder_audit']},indent=2))


if __name__=='__main__':
    main()
