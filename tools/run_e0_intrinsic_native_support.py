"""Passive full-M GELU support observation, one unchanged native dense video.

This measures occupied analytical tiles, not a sparse kernel or its speed.
"""
import argparse
import contextvars
import json
from pathlib import Path
import shutil
import sys
import time

from analyze_ffnds_observations import load_arrays
from e0_intrinsic_math import support_tile_statistics
from e0_intrinsic_temporal_support import innovation_support
from ffnds_observation_common import sha256, write_json
from prepare_ffnds_observations import preflight

ROOT=Path(__file__).resolve().parents[1]


def tensor_sha(value):
    import hashlib
    import torch
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class SupportObserver:
    def __init__(self,runtime,adapter,protocol,accepted_path,accepted_run):
        self.runtime,self.adapter,self.protocol=runtime,adapter,protocol
        self.accepted_path,self.accepted_run=accepted_path,accepted_run
        self.current=contextvars.ContextVar('intrinsic_selected_q2',default=None)
        self.scopes=[];self.previous={}

    def __enter__(self):
        import torch
        owner=self
        self.old_apply,self.old_qdq=self.runtime.apply,self.adapter.native_hif4_activation_qdq
        def qdq(value,*,force_fp32=True):
            result=owner.old_qdq(value,force_fp32=force_fp32)
            record=owner.current.get()
            if record is not None:
                if tuple(result.shape)!=(31668,13824) or not force_fp32:
                    raise ValueError('Changed full-M Q2 boundary')
                record['post_Q2_zero_fraction']=1-float(torch.count_nonzero(result))/result.numel()
                record['Q2_calls']+=1
                q=result.to(dtype=torch.bfloat16)
                record['post_Q2_support']=support_tile_statistics(q,owner.protocol['retained_fractions'],
                                                                 tuple(owner.protocol['row_tiles']))
                key=(record['expert'],record['block'])
                if record['step'] in [0,2]:
                    if key in owner.previous:
                        raise ValueError('Duplicate previous-expert cache')
                    owner.previous[key]=q.detach().cpu().contiguous()
                else:
                    previous=owner.previous.pop(key)
                    record['same_expert_innovation_support']=innovation_support(
                        q,previous,owner.protocol['retained_fractions'],tuple(owner.protocol['row_tiles']))
                    del previous
            return result
        def apply(mm,value,*,expert,step_index,block_index,module_name):
            kw=dict(expert=expert,step_index=step_index,block_index=block_index,module_name=module_name)
            if block_index not in owner.protocol['blocks'] or not module_name.endswith('.ffn.2'):
                return owner.old_apply(mm,value,**kw)
            if tuple(value.shape)!=(31668,13824):
                raise ValueError('Wrong original GELU shape')
            scope=next(s for s in owner.accepted_run['scopes'] if
                       (s['step'],s['block'])==(step_index,block_index))
            path=owner.accepted_path.parent/'observed'/scope['path']
            if sha256(path)!=scope['sha256']:
                raise ValueError('Original operand identity changed')
            record={'step':step_index,'block':block_index,'expert':expert,'Q2_calls':0,
                    'original_payload_sha256':scope['sha256'],
                    'support':support_tile_statistics(value,owner.protocol['retained_fractions'],
                                                     tuple(owner.protocol['row_tiles']))}
            token=owner.current.set(record)
            try:
                result=owner.old_apply(mm,value,**kw)
            finally:
                owner.current.reset(token)
            old=load_arrays(path)
            ids=[scope['row_plan']['rows'].index(row) for row in scope['row_plan']['uniform_rows']]
            rows=torch.tensor(scope['row_plan']['uniform_rows'],device=result.device)
            actual=result.index_select(0,rows).detach().cpu().contiguous()
            reference=torch.from_numpy(old['ffn_output'][ids]).to(dtype=actual.dtype)
            record['sampled_native_output_sha256']=tensor_sha(actual)
            record['saved_sampled_native_output_sha256']=tensor_sha(reference)
            if record['Q2_calls']!=1 or record['sampled_native_output_sha256']!=record['saved_sampled_native_output_sha256']:
                raise RuntimeError('Passive observer changed original sampled FFN output')
            owner.scopes.append(record)
            print(f"native full-M support step{step_index} block{block_index:02d}: exact saved FFN samples",flush=True)
            return result
        self.runtime.apply,self.adapter.native_hif4_activation_qdq=apply,qdq
        return self

    def __exit__(self,*error):
        self.runtime.apply,self.adapter.native_hif4_activation_qdq=self.old_apply,self.old_qdq
        if not any(error):
            positions={(s['step'],s['block']) for s in self.scopes}
            if len(self.scopes)!=20 or len(positions)!=20 or self.previous:
                raise RuntimeError('Incomplete passive support coverage')


def execute(args):
    p=json.loads(args.protocol.read_text())
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
        observer=SupportObserver(runtime,adapter,p,run_path,original)
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
        counts=dict(runtime.counters)
        video_sha=sha256(args.out/'dense_observed.mp4')
        if counts!=source['native_counts_each_phase'] or trajectory!=source['trajectory'] or video_sha!=source['video_sha256']:
            raise RuntimeError('Historical dense trajectory/video/call-count identity failed')
        record.update(status='PASS_NATIVE_FULL_M_SUPPORT_OBSERVATION_NOOP',trajectory=trajectory,
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
