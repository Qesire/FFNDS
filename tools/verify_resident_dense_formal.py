"""Validate the resident dense D-Q baseline on all20 frozen formal sources."""
import argparse
import fcntl
import json
from pathlib import Path

import torch

from ar2_checkpoint import atomic_json,digest
from distill_assets import ROOT,verified_assets
from generate_formal_suite import generate_one
from resident_ffn_runtime import make_resident,unused_raw_placeholders


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference-suite',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    source=json.loads(args.reference_suite.read_text());source_id=source['identity']
    if (source['status']!='PASS_FORMAL_GENERATION_SLICE_ONLY' or len(source['completed'])!=20
            or source_id['arm']!='D-Q' or source_id['packed_policy_sha256'] is not None
            or source_id['indices']!=list(range(20))):
        raise RuntimeError('Complete immutable dense formal20 required')
    samples=json.loads((ROOT/'.runtime/inputs/formal20/canonical.json').read_text())
    kwargs,cache,assets=verified_assets()
    if assets!=source_id['assets'] or source_id['inputs_sha256']!=digest(ROOT/'.runtime/inputs/formal20/canonical.json'):
        raise RuntimeError('Dense source/cache/input identity differs')
    references={row['index']:row for row in source['completed']}
    if set(references)!=set(range(20)):raise RuntimeError('Distinct20 source indices required')
    identity={**source_id,'reference_suite_sha256':digest(args.reference_suite),
        'resident_execution':{'protocol_sha256':digest(ROOT/'config/resident_ffn_benchmark.json'),
            'sources':{str(p):digest(p) for p in [Path(__file__),ROOT/'tools/resident_ffn_runtime.py',ROOT/'tools/generate_formal_suite.py']}}}
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'JOB.lock').open('w');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=args.out/'RUN.json';previous={}
    if path.exists():
        previous=json.loads(path.read_text())
        if previous['identity']!=identity:raise RuntimeError('Resident dense identity changed')
    completed=previous.get('completed',[])
    for row in completed:
        if digest(row['case_report'])!=row['case_report_sha256'] or not row['four_latents_bitwise_equal'] or not row['video_bytes_equal']:
            raise RuntimeError('Completed equivalence record changed')
    meta={**previous,'status':'RUNNING','identity':identity,'completed':completed,'quality_or_speed_acceptance':False}
    atomic_json(path,meta)
    try:
        pending=[i for i in range(20) if i not in {r['index'] for r in completed}]
        if pending:
            from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
            with unused_raw_placeholders(adapter) as audit:
                pipe=adapter.build_lightx2v_distill_pipeline(**kwargs)
            if len(audit['names'])!=400 or audit['calls']<800:raise RuntimeError('All covered raw matrices must be placeholders')
            runtime=adapter.HIF4CacheRuntime(cache['path']);meta['residency']=make_resident(runtime)
            meta['raw_placeholder_audit']={k:sorted(v) if isinstance(v,set) else v for k,v in audit.items()}
            adapter.activate_lightx2v_hooks(runtime=runtime)
            try:
                for index in pending:
                    ref=references[index]
                    if digest(ref['case_report'])!=ref['case_report_sha256']:raise RuntimeError('Dense reference case changed')
                    reference=json.loads(Path(ref['case_report']).read_text())
                    if reference['input']!=samples[index]:raise RuntimeError('Dense conditioning differs')
                    case_path=generate_one(pipe,runtime,samples[index],index,args.out,identity)
                    case=json.loads(case_path.read_text())
                    if ([r['raw_tensor_sha256'] for r in case['trajectory']]!=[r['raw_tensor_sha256'] for r in reference['trajectory']]
                            or case['video']['sha256']!=reference['video']['sha256'] or audit['fallback_calls']):
                        raise RuntimeError(f'Resident dense input{index:02d} differs')
                    completed.append({'index':index,'case_report':str(case_path),'case_report_sha256':digest(case_path),
                        'reference_case':ref['case_report'],'reference_case_sha256':ref['case_report_sha256'],
                        'four_latents_bitwise_equal':True,'video_bytes_equal':True})
                    atomic_json(path,meta);print(f'VERIFIED resident D-Q formal{index:02d}: all4 latents and video bytes identical',flush=True)
            finally:
                adapter.deactivate_lightx2v_hooks()
        if len(completed)!=20 or {r['index'] for r in completed}!=set(range(20)):
            raise RuntimeError('Dense all20 equivalence required')
        meta['status']='PASS_RESIDENT_DQ_FORMAL20_ALL_LATENTS_AND_VIDEO_BYTES_EQUIVALENT'
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error));raise
    finally:
        atomic_json(path,meta)


if __name__=='__main__':main()
