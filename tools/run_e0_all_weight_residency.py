"""Fixed dense numerical-identity/runtime study of weight allocation policies.

One mode per fresh process, three complete videos, no changed FFN arithmetic.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

from ffnds_observation_common import sha256,write_json
from ffnds_all_weight_residency import AllWeightResidency as DenseWeightResidency
from prepare_ffnds_observations import preflight
from run_e0_intrinsic_native_support import tensor_sha

ROOT=Path(__file__).resolve().parents[1]


class FFNStageTimer:
    def __init__(self,adapter):self.adapter=adapter;self.events=[]
    def __enter__(self):
        import torch
        from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
        self.cls=WanTransformerInfer;self.old=self.cls.infer_ffn;owner=self
        def ffn(infer,phase,x,attn,shift,scale,gate=None):
            if getattr(infer,'use_compile',False):raise RuntimeError('Frozen native timing requires noncompiled FFN')
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();value=owner.old(infer,phase,x,attn,shift,scale,gate);end.record()
            owner.events.append((int(owner.adapter._step_ctx.get()),int(infer.block_idx),start,end))
            return value
        self.cls.infer_ffn=ffn;return self
    def report(self):
        return [{'step':s,'block':b,'elapsed_ms':start.elapsed_time(end)} for s,b,start,end in self.events]
    def __exit__(self,*error):self.cls.infer_ffn=self.old


def execute(a):
    p=json.loads(a.protocol.read_text())
    if not a.execute_native or a.mode not in p['modes']:raise ValueError('Registered explicit native mode required')
    for reference in p['memory_references']:
        source=a.out.parent.parent/reference['HF_relative']
        if sha256(source)!=reference['sha256']:
            raise ValueError('Original memory-control run changed')
    for name,h in p['driver_sources_sha256'].items():
        if sha256(ROOT/name)!=h:raise ValueError('Frozen driver/source changed')
    if preflight(ROOT,p)['status']!='PREPARED_STATIC_PREFLIGHT':raise RuntimeError('Static native preflight failed')
    manifest=ROOT/p['accepted_observations']
    if sha256(manifest)!=p['accepted_observations_sha256']:raise ValueError('Original reference changed')
    if sha256(ROOT/p['input_default'])!=p['input_default_sha256'] or sha256(ROOT/'.runtime/hif4_rebuilt_cache.json')!=p['cache_manifest_sha256']:
        raise ValueError('Original canonical input or cache manifest changed')
    for name in ['HiF4_Sp','LightX2V','Wan2.2-I2V-A14B-W4A4']:sys.path.insert(0,str(ROOT/'.runtime/code'/name))
    for name in ['HiFloat4','HiFloat4/hif4_gpu']:sys.path.insert(0,str(ROOT/'.runtime/code/HiF4_Sp'/name))
    import torch
    from historical_source_snapshot import HistoricalSources
    from restored_assets import verified_restored_assets
    from startup_gpu_reservation import reserve_before_assets
    from worker_identity import require_worker
    from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
    require_worker()
    if 'A100' not in torch.cuda.get_device_name(0):raise RuntimeError('Frozen native A100 required')
    references=json.loads(manifest.read_text());inputs=json.loads((ROOT/p['input_default']).read_text())
    a.out.mkdir(parents=True,exist_ok=False);snap=a.out/'EXECUTED_SOURCES';snap.mkdir()
    for name in p['driver_sources_sha256']:
        target=snap/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    shutil.copyfile(a.protocol,snap/a.protocol.name)
    stop=a.out.parent/'STUDY_FAILURE.json'
    record={'status':'STARTED_NOT_ACCEPTED','protocol_sha256':sha256(a.protocol),'mode':a.mode,
            'generations_started':0,'generations_completed':0,'phases':[],
            'new_weight_decomposition':0,'training_updates':0,'official_quality_calls':0,
            'automatic_retry':False,'quality_or_speed_acceptance':False}
    write_json(a.out/'RUN.json',record);monitor=None;runtime=None;policy=None
    try:
        if stop.exists():raise RuntimeError('Another registered mode failed; no new generation started')
        reservation,monitor,receipt=reserve_before_assets(torch);record['CUDA_reservation']=receipt
        torch.set_num_threads(8)
        kwargs,cache,assets=verified_restored_assets(HistoricalSources());record['assets']=assets
        pipe=adapter.build_lightx2v_distill_pipeline(**kwargs)
        adapter.validate_effective_pipeline_config_against_contract(pipe.runner.config)
        runtime=adapter.HIF4CacheRuntime(cache['path']);adapter.activate_lightx2v_hooks(runtime=runtime)
        expected_bytes=sum(v['weight_qdq'].numel()*v['weight_qdq'].element_size()
                           for records in runtime.records.values() for v in records.values()
                           if hasattr(v,'get') and torch.is_tensor(v.get('weight_qdq')))
        device_total=torch.cuda.get_device_properties(0).total_memory
        estimated=expected_bytes+p['reference_peak_physical_MiB']*(1<<20)+p['memory_safety_bytes']
        record['memory_precheck']={'decoded_weight_source_bytes':expected_bytes,'device_total_bytes':device_total,
                                  'reference_peak_plus_all_weights_and_safety':estimated}
        if expected_bytes>p['max_retained_bytes'] or estimated>device_total:
            record['status']='SKIPPED_MEMORY_HEADROOM_NO_GENERATION'
            return
        scheduler=pipe.runner.scheduler;old_post=scheduler.step_post;trajectory=[]
        def post():
            result=old_post();latent=scheduler.latents.detach().float().cpu().contiguous()
            trajectory.append({'step':int(scheduler.step_index),'sha256':tensor_sha(latent),'shape':list(latent.shape)})
            return result
        scheduler.step_post=post
        try:
            with torch.inference_mode(),DenseWeightResidency(runtime,a.mode,p['max_retained_bytes']) as policy,FFNStageTimer(adapter) as timer:
                for order,index in enumerate(p['source_sequence']):
                    if stop.exists():raise RuntimeError('Registered study failure stops further generations')
                    if shutil.disk_usage(a.out).free<p['disk_reserve_gib']*(1<<30):raise RuntimeError('Disk reserve reached')
                    reference=next(s for s in references['sources'] if s['index']==index)
                    sample=inputs[index];image=Path(sample.get('image_path') or sample['img_paths'][0])
                    seed=sample.get('seed',sample.get('calibration_seed'))
                    if sha256(image)!=reference['source_id'] or seed!=reference['seed']:raise ValueError('Image/seed changed')
                    video=a.out/f'phase{order:02d}_source{index:02d}.mp4'
                    trajectory=[];timer.events.clear();scheduler.generator=None
                    before=dict(runtime.counters);before_policy=policy.snapshot()
                    record['generations_started']+=1;write_json(a.out/'RUN.json',record)
                    started=time.perf_counter()
                    pipe.generate(seed=seed,image_path=str(image),prompt=sample['prompt'],negative_prompt='',save_result_path=str(video))
                    torch.cuda.synchronize();wall=time.perf_counter()-started;monitor.check()
                    counts={k:runtime.counters[k]-before[k] for k in before}
                    stats=policy.snapshot();delta={k:stats[k]-before_policy[k] for k in stats if k not in ['retained_bytes','peak_retained_bytes']}
                    expected=dict(reference['native_counts_each_phase'])
                    expected['weight_transfers']-=delta['all_retained_hits']
                    timings=timer.report();video_sha=sha256(video)
                    if counts!=expected or trajectory!=reference['trajectory'] or video_sha!=reference['video_sha256']:
                        raise RuntimeError('Dense numerical identity or native count control failed')
                    if len(timings)!=160 or {(v['step'],v['block']) for v in timings}!={(s,b) for s in range(4) for b in range(40)}:
                        raise RuntimeError('Incomplete native FFN boundary timing')
                    if delta['FFN_weight_requests']!=320:raise RuntimeError('Unexpected FFN weight request count')
                    expected_hits=(160 if order==0 else 320) if a.mode=='all_resident_no_flush' else 0
                    if delta['FFN_retained_hits']!=expected_hits or delta['all_retained_hits']!=(800 if order==0 else 1600):raise RuntimeError('Full residency ledger mismatch')
                    phase={'order':order,'source_index':index,'source_id':reference['source_id'],'seed':seed,
                           'FFN_cache_state':'empty at process entry' if order==0 else 'after previous registered generation',
                           'wall_seconds_including_video_save':wall,'native_counts':counts,
                           'policy_delta':delta,'retained_bytes_end':stats['retained_bytes'],
                           'FFN_boundary_total_ms':sum(v['elapsed_ms'] for v in timings),
                           'FFN_boundary_timings':timings,'trajectory':trajectory,'video_sha256':video_sha,
                           'identity':'EXACT_HISTORICAL_LATENTS_AND_VIDEO'}
                    record['phases'].append(phase);record['generations_completed']+=1
                    write_json(a.out/'RUN.json',record)
                    print(f"dense mode {a.mode} phase{order} source{index:02d}: exact original; wall={wall:.3f}s FFN={phase['FFN_boundary_total_ms']/1000:.3f}s retained_hits={expected_hits}",flush=True)
        finally:scheduler.step_post=old_post
        if sum(f.stat().st_size for f in a.out.rglob('*') if f.is_file())>p['max_output_gib_per_mode']*(1<<30):raise RuntimeError('Output budget exceeded')
        record['status']='PASS_DENSE_RESIDENCY_IDENTITY_AND_TIMING'
    except BaseException as e:
        record.update(status='FAILED_PRESERVED',error_type=type(e).__name__,error=str(e))
        if not stop.exists():write_json(stop,{'mode':a.mode,'error_type':type(e).__name__,'generation_attempts':record['generations_started']})
        raise
    finally:
        adapter.deactivate_lightx2v_hooks()
        if policy is not None:record['final_policy']=policy.snapshot()
        if runtime is not None:record['native_counts_observed']=dict(runtime.counters)
        if monitor is not None:monitor.close();record['assigned_GPU_resources']=monitor.report()
        write_json(a.out/'RUN.json',record)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--mode',choices=DenseWeightResidency.modes,required=True)
    ap.add_argument('--execute-native',action='store_true');execute(ap.parse_args())
