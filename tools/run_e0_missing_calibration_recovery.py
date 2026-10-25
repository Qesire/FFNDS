"""Passive missing-block calibration operands, one unchanged native dense video.

Collects original fields without applying a sparse action.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

from analyze_ffnds_observations import load_arrays
from e0_missing_calibration_capture import MissingCalibrationCapture as ContiguousCapture
from ffnds_observation_common import sha256, write_json
from prepare_ffnds_observations import preflight

ROOT=Path(__file__).resolve().parents[1]


def tensor_sha(value):
    import hashlib
    import torch
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()



def execute(args):
    p=json.loads(args.protocol.read_text())
    info=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    if int(info['MemAvailable'].split()[0]) < p['minimum_available_host_gib']*(1<<20):
        raise RuntimeError('Host memory reserve unavailable before model loading')
    if not args.execute_native or args.index not in p['source_indices']:
        raise ValueError('Explicit native flag and registered source required')
    for name,expected in p['driver_sources_sha256'].items():
        if sha256(ROOT/name)!=expected:
            raise ValueError('Frozen observer/driver source changed')
    mp=ROOT/p['accepted_observations']
    if sha256(mp)!=p['accepted_observations_sha256']:
        raise ValueError('Accepted manifest changed')
    if preflight(ROOT,p)['status']!='PREPARED_STATIC_PREFLIGHT':
        raise RuntimeError('Native static preflight failed')
    for name in ['HiF4_Sp','LightX2V','Wan2.2-I2V-A14B-W4A4']:
        sys.path.insert(0,str(ROOT/'.runtime/code'/name))
    for name in ['HiFloat4','HiFloat4/hif4_gpu']:
        sys.path.insert(0,str(ROOT/'.runtime/code/HiF4_Sp'/name))
    import torch
    from historical_source_snapshot import HistoricalSources
    from restored_assets import verified_restored_assets
    from startup_gpu_reservation import reserve_before_assets
    from worker_identity import require_worker
    from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
    require_worker()
    if 'A100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Frozen native control requires A100')
    accepted=json.loads(mp.read_text())
    source=next(s for s in accepted['sources'] if s['index']==args.index)
    run_path=args.raw_root/f"source{args.index:02d}_attempt1/RUN.json"
    if sha256(run_path)!=source['run_sha256']:
        raise ValueError('Original run changed')
    original=json.loads(run_path.read_text())
    inputs=ROOT/p['input_default'];sample=json.loads(inputs.read_text())[args.index]
    image=Path(sample.get('image_path') or sample['img_paths'][0])
    seed=sample.get('seed',sample.get('calibration_seed'))
    if sha256(image)!=source['source_id'] or seed!=source['seed']:
        raise ValueError('Original image/seed changed')
    args.out.mkdir(parents=True,exist_ok=False)
    snapshot=args.out/'EXECUTED_SOURCES';snapshot.mkdir()
    for name in p['driver_sources_sha256']:
        target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    shutil.copyfile(args.protocol,snapshot/args.protocol.name)
    record={'status':'STARTED_NOT_ACCEPTED','protocol_sha256':sha256(args.protocol),
            'source_index':args.index,'source_id':source['source_id'],'seed':seed,
            'source_group':source['source_group'],'generations_started':0,
            'extra_native_FFN_calls':0,'standalone_norm_calls':0,'training_updates':0,
            'quality_or_speed_acceptance':False,'automatic_retry':False}
    write_json(args.out/'RUN.json',record)
    monitor=None;observer=None;runtime=None;torch.set_num_threads(8)
    try:
        reservation,monitor,receipt=reserve_before_assets(torch)
        record['early_CUDA_reservation']=receipt
        write_json(args.out/'RUN.json',record)
        kwargs,cache,assets=verified_restored_assets(HistoricalSources())
        record['assets']=assets;monitor.check()
        pipe=adapter.build_lightx2v_distill_pipeline(**kwargs)
        adapter.validate_effective_pipeline_config_against_contract(pipe.runner.config)
        runtime=adapter.HIF4CacheRuntime(cache['path']);adapter.activate_lightx2v_hooks(runtime=runtime)
        scheduler=pipe.runner.scheduler;old_post=scheduler.step_post;trajectory=[]
        def post():
            value=old_post();latent=scheduler.latents.detach().float().cpu().contiguous()
            trajectory.append({'step':int(scheduler.step_index),'sha256':tensor_sha(latent),'shape':list(latent.shape)})
            return value
        scheduler.step_post,scheduler.generator=post,None
        observer=ContiguousCapture(runtime,args.out/'observed',source['source_id'],p,cache['identity'])
        record['generations_started']=1
        write_json(args.out/'RUN.json',record)
        started=time.perf_counter()
        try:
            if shutil.disk_usage(args.out).free<p['disk_reserve_gib']*(1<<30):
                raise RuntimeError('Disk reserve reached')
            with torch.inference_mode(),observer:
                pipe.generate(seed=seed,image_path=str(image),prompt=sample['prompt'],negative_prompt='',
                              save_result_path=str(args.out/'dense_observed.mp4'))
            torch.cuda.synchronize();monitor.check()
        finally:
            scheduler.step_post=old_post
        if len(observer.scopes)!=len(p['blocks'])*len(p['steps']):
            raise RuntimeError('Incomplete adjacent capture')
        for scope in observer.scopes:
            prior=next((s for s in original['scopes'] if (s['step'],s['block'])==(scope['step'],scope['block'])),None)
            if prior is None:
                continue
            actual=load_arrays(args.out/'observed'/scope['path'])
            old_path=run_path.parent/'observed'/prior['path']
            if sha256(old_path)!=prior['sha256']:
                raise RuntimeError('Original payload changed')
            old=load_arrays(old_path)
            ai=[scope['row_plan']['rows'].index(r) for r in scope['row_plan']['uniform_rows']]
            oi=[prior['row_plan']['rows'].index(r) for r in scope['row_plan']['uniform_rows']]
            if any(not __import__('numpy').array_equal(actual[name][ai],old[name][oi]) for name in ContiguousCapture.fields):
                raise RuntimeError('Passive native sampled field identity failed')
            scope['historical_uniform_selected_fields_exact']=True
        counts=dict(runtime.counters)
        video_sha=sha256(args.out/'dense_observed.mp4')
        if counts!=source['native_counts_each_phase'] or trajectory!=source['trajectory'] or video_sha!=source['video_sha256']:
            raise RuntimeError('Historical dense trajectory/video/call-count identity failed')
        record.update(status='PASS_NATIVE_MISSING_CALIBRATION_OPERANDS_NOOP',trajectory=trajectory,
                      video_sha256=video_sha,native_counts=counts,scopes=observer.scopes,
                      elapsed_including_observation=time.perf_counter()-started)
        if sum(f.stat().st_size for f in args.out.rglob('*') if f.is_file())>p['max_output_gib_per_source']*(1<<30):
            raise RuntimeError('Output budget exceeded')
    except BaseException as error:
        record.update(status='FAILED_PRESERVED',error_type=type(error).__name__,error=str(error));raise
    finally:
        adapter.deactivate_lightx2v_hooks()
        if observer is not None:
            record['scopes']=observer.scopes
        if runtime is not None:
            record['native_counts_observed']=dict(runtime.counters)
        if monitor is not None:
            monitor.close();record['assigned_GPU_resources']=monitor.report()
        write_json(args.out/'RUN.json',record)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--index',type=int,required=True)
    p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--execute-native',action='store_true');execute(p.parse_args())


if __name__=='__main__':main()
