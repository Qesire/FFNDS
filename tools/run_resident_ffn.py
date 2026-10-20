"""Verify resident native inference and collect complete160-position FFN times."""
import argparse
import fcntl
import importlib.metadata
import json
from pathlib import Path
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
    ap.add_argument('--input-manifest',type=Path,required=True)
    ap.add_argument('--reference-meta',type=Path,required=True)
    ap.add_argument('--policy',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--measured-repeats',type=int,default=0)
    args=ap.parse_args()
    if args.measured_repeats not in [0,3]:
        ap.error('Choose smoke only (0), or the predeclared3 repeats after one warmup')
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'JOB.lock').open('w');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    report=args.out/'RUN.json'
    if report.exists():raise RuntimeError('Select a fresh resident run; completed runs are immutable')
    sample=json.loads(args.input_manifest.read_text())
    if len(sample)!=1:raise RuntimeError('One exact paired input required')
    sample=sample[0];reference=json.loads(args.reference_meta.read_text())
    if (reference['status']!='PASS' or reference['input_record']!=sample
            or bool(args.policy)!=(reference['arm']=='D-Q-PACKED-REFERENCE')
            or (args.policy and digest(args.policy)!=reference['packed_policy_sha256'])):
        raise RuntimeError('Reference input/arm/policy differs')
    kwargs,cache,assets=verified_assets()
    if cache['sha256']!=reference['quantized_cache_identity']['sha256']:
        raise RuntimeError('Reference quantized cache differs')
    reference_latents=[]
    for row in reference['trajectory']:
        if digest(row['path'])!=row['sha256']:raise RuntimeError('Reference latent file changed')
        with safe_open(row['path'],framework='pt',device='cpu') as handle:
            reference_latents.append(tensor_sha(handle.get_tensor('latents')))
    identity={'assets':assets,'input_manifest_sha256':digest(args.input_manifest),'input':sample,
        'reference_meta_sha256':digest(args.reference_meta),'reference_raw_latents':reference_latents,
        'policy_sha256':digest(args.policy) if args.policy else None,
        'protocol_sha256':digest(ROOT/'config/resident_ffn_benchmark.json'),
        'sources':{str(p):digest(p) for p in [Path(__file__),ROOT/'tools/resident_ffn_runtime.py',
            ROOT/'tools/trained_packed_runtime.py',ROOT/'tools/reduced_rank_residual.py',ROOT/'tools/benchmark_resources.py',
            ROOT/'tools/packed_reference_runtime.py',ROOT/'tools/output_variance_runtime.py',ROOT/'tools/output_variance_kernel.py',
            ROOT/'tools/step_budget_runtime.py']},
        'torch':torch.__version__,'device':torch.cuda.get_device_name(),'TF32':False,
        'versions':{n:importlib.metadata.version(n) for n in ['torch','numpy','safetensors','transformers','diffusers','triton']},
        'cuda_runtime':torch.version.cuda,
        'measured_repeats':args.measured_repeats}
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
            from packed_reference_runtime import create_runtime
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
            folder=args.out/f'request{repetition:02d}';folder.mkdir()
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
                'video':{'path':str(video),'sha256':digest(video)},'four_latents_bitwise_equal':True}
            atomic_json(folder/'REQUEST.json',request)
            meta['requests'].append({'report':str(folder/'REQUEST.json'),'sha256':digest(folder/'REQUEST.json'),
                'warmup':repetition==0,'FFN_wall_ms':request['FFN_timing']['wall_ms'],
                'FFN_cuda_ms':request['FFN_timing']['cuda_ms'],'four_latents_bitwise_equal':True})
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
