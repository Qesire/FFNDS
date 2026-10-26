"""SHA-bound CPU-only posthoc metrics for the six existing fixed videos."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from ffnds_observation_common import sha256,write_json
from ffnds_video_metrics import evaluate,validate
from ffnds_video_metric_gate import inspect

ROOT=Path(__file__).resolve().parents[1]


def object_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def run(args):
    plan=json.loads(args.protocol.read_text())
    for name,digest in plan['bound_sources'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Frozen metric source changed: '+name)
    previous=json.loads((ROOT/plan['previous_result']).read_text())
    references=json.loads((ROOT/'research/20261025/E0_NATIVE_OBSERVATIONS.json').read_text())
    inputs=json.loads((ROOT/plan['input_default']).read_text())
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(args.out.parent).free<plan['disk_reserve_gib']*(1<<30):raise RuntimeError('Disk reserve')
    args.out.mkdir(exist_ok=False)
    snapshot=args.out/'EXECUTED_SOURCES';snapshot.mkdir()
    for name in plan['bound_sources']:
        target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    shutil.copyfile(args.protocol,snapshot/args.protocol.name)
    bundle={'schema':'ffnds.video.bundle.v1','expected_video_count':6,'videos':[]}
    completed={'id':plan['id'],'status':'STARTED_NOT_ACCEPTED','protocol_sha256':sha256(args.protocol),
               'analysis_sha256':sha256(Path(__file__)),'videos':[],'new_generations':0,'GPU_calls':0,
               'official_quality_calls':0,'quality_accepted':False,'legacy_sources_unchanged':True}
    write_json(args.out/'RUN.json',completed)
    try:
        for run in previous['runs']:
            path=args.hf_root/run['raw_binding']['HF_relative']
            if sha256(path)!=run['raw_binding']['sha256']:raise ValueError('Original generation receipt changed')
            raw=json.loads(path.read_text())
            for phase in raw['phases']:
                source=phase['source_index'];sample=inputs[source]
                ref=next(s for s in references['sources'] if s['index']==source)
                video=path.parent/phase['video']
                if sha256(video)!=phase['video_sha256']:raise ValueError('Video changed')
                image=Path(sample.get('image_path') or sample['img_paths'][0])
                if sha256(image)!=ref['source_id'] or phase['seed']!=ref['seed']:raise ValueError('Input identity changed')
                prompt_digest=object_sha({'image_sha256':ref['source_id'],'prompt':sample['prompt'],
                                          'negative_prompt':'','contract':raw['assets']['contract']})
                model_binding={'checkpoint_sha256':raw['assets']['checkpoint_sha256'],
                               'cache_manifest_sha256':raw['assets']['cache_manifest_sha256'],
                               'prototype_family':run['mode'],
                               'bank_RUN_sha256':raw['bank_RUN_sha256'] if run['mode']!='dense' else None}
                identity={'source_id':ref['source_id'],'seed':ref['seed'],'expected_frames':plan['expected_frames'],
                          'input_prompt_digest':prompt_digest,'model_digest':object_sha(model_binding),
                          'protocol_sha256':raw['protocol_sha256'],'run_sha256':sha256(path)}
                metadata={'identity':identity,'cost':{
                    'generation_wall_seconds':phase['wall_seconds_including_video_save_and_common_latent_hash'],
                    'ffn_boundary_seconds':phase['FFN_boundary_total_ms']/1000,
                    'setup_seconds':raw['setup_seconds_including_assets_pipeline_packs_neutral_and_weight_preload'],
                    'peak_gpu_memory_mib':raw['assigned_GPU_resources']['peak_physical_used_memory_MiB']}}
                dense=next(r for r in previous['runs'] if r['mode']=='dense')
                dense_path=args.hf_root/dense['raw_binding']['HF_relative']
                if sha256(dense_path)!=dense['raw_binding']['sha256']:raise ValueError('Dense receipt changed')
                dense_raw=json.loads(dense_path.read_text())
                dense_phase=next(p for p in dense_raw['phases'] if p['source_index']==source)
                reference=dense_path.parent/dense_phase['video'] if run['mode']!='dense' else None
                if reference:
                    if sha256(reference)!=dense_phase['video_sha256']:raise ValueError('Dense video changed')
                    if phase['seed']!=dense_phase['seed']:raise ValueError('Unpaired seed')
                    metadata['reference_identity']={**identity,'run_sha256':sha256(dense_path)}
                report=evaluate(video,reference,metadata)
                errors=validate(report)
                if errors:raise ValueError('; '.join(errors))
                name=f"{run['mode']}_source{source:02d}"
                target=args.out/(name+'_metrics.json');write_json(target,report)
                write_json(args.out/(name+'_identity.json'),{'identity':identity,'model_binding':model_binding,
                                                            'original_execution_receipt':str(path),**metadata})
                entry={'source_id':ref['source_id'],'seed':ref['seed'],'variant':run['mode'],
                       'video_path':str(video),'video_sha256':sha256(video),
                       'run_receipt_path':str(path),'run_sha256':sha256(path),
                       'metric_report_path':target.name,'metric_report_sha256':sha256(target)}
                if reference:
                    entry.update(reference_video_path=str(reference),reference_video_sha256=sha256(reference),
                                 reference_run_receipt_path=str(dense_path),reference_run_sha256=sha256(dense_path))
                bundle['videos'].append(entry)
                completed['videos'].append({'mode':run['mode'],'source_index':source,'seed':ref['seed'],
                                            'metric_path':target.name,'metric_sha256':sha256(target),
                                            'identity':identity,'metrics':report['metrics']})
                write_json(args.out/'RUN.json',completed)
                print('Core metrics complete:',name,flush=True)
        write_json(args.out/'BUNDLE.json',bundle)
        failures=inspect(args.out/'BUNDLE.json')
        if failures:raise ValueError('; '.join(failures))
        completed.update(status='PASS_SIX_EXISTING_VIDEO_CORE_DIAGNOSTICS_NO_QUALITY_ACCEPTANCE',
                         bundle_sha256=sha256(args.out/'BUNDLE.json'),core_metric_gate_pass=True)
        if sum(p.stat().st_size for p in args.out.rglob('*') if p.is_file())>plan['max_output_gib']*(1<<30):
            raise RuntimeError('Metric output cap')
    except BaseException as exc:
        completed.update(status='FAILED_PRESERVED',error_type=type(exc).__name__,error=str(exc));raise
    finally:write_json(args.out/'RUN.json',completed)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['protocol','hf-root','out']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
