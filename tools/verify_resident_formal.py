"""Generate resident formal20 and require every latent/video to match its source."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time

import torch

from ar2_checkpoint import atomic_json,canonical_json,digest
from distill_assets import ROOT,verified_assets
from generate_formal_suite import generate_one
from resident_ffn_runtime import make_resident,unused_raw_placeholders


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference-suite',type=Path,required=True)
    ap.add_argument('--policy',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'JOB.lock').open('w');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
    from packed_reference_runtime import create_runtime
    samples=json.loads((ROOT/'.runtime/inputs/formal20/canonical.json').read_text())
    source=json.loads(args.reference_suite.read_text());source_id=source['identity']
    kwargs,cache,assets=verified_assets()
    if (source_id['assets']!=assets or source_id['packed_policy_sha256']!=digest(args.policy)
            or source_id['inputs_sha256']!=digest(ROOT/'.runtime/inputs/formal20/canonical.json')
            or source_id['indices']!=list(range(20))):
        raise RuntimeError('Reference source/model/policy differs from the full20 protocol')
    identity={**source_id,'reference_identity_sha256':hashlib.sha256(canonical_json(source_id).encode()).hexdigest(),
        'resident_execution':{'protocol_sha256':digest(ROOT/'config/resident_ffn_benchmark.json'),
            'sources':{str(p):digest(p) for p in [Path(__file__),ROOT/'tools/resident_ffn_runtime.py',ROOT/'tools/generate_formal_suite.py']},
            'four_latents_and_video_must_equal_reference':True}}
    report=args.out/'RUN.json';completed=[];previous={}
    if report.exists():
        previous=json.loads(report.read_text())
        if previous['identity']!=identity:raise RuntimeError('Resident formal run identity changed')
        completed=previous['completed']
    for row in completed:
        if digest(row['case_report'])!=row['case_report_sha256'] or digest(row['reference_case'])!=row['reference_case_sha256']:
            raise RuntimeError('Completed equivalence record changed')
    meta={'status':'RUNNING','identity':identity,'completed':completed,'quality_or_speed_acceptance':False}
    for key in ['residency','raw_placeholder_audit']:
        if key in previous:meta[key]=previous[key]
    atomic_json(report,meta)
    try:
        pending=[i for i in range(20) if i not in {r['index'] for r in completed}]
        if pending:
            with unused_raw_placeholders(adapter) as audit:
                pipe=adapter.build_lightx2v_distill_pipeline(**kwargs)
            if len(audit['names'])!=400 or audit['calls']<800:raise RuntimeError('All bridge target matrices must be placeholders')
            runtime=create_runtime(cache['path'],args.policy)
            meta['residency']=make_resident(runtime)
            meta['raw_placeholder_audit']={k:sorted(v) if isinstance(v,set) else v for k,v in audit.items()}
            adapter.activate_lightx2v_hooks(runtime=runtime)
            try:
                for index in pending:
                    while True:
                        source=json.loads(args.reference_suite.read_text())
                        if source['identity']!=source_id:raise RuntimeError('Source identity changed during verification')
                        if source['status']=='FAILED_RESUMABLE':raise RuntimeError('Source generation failed; preserve resident partial results')
                        record=next((r for r in source['completed'] if r['index']==index),None)
                        if record is not None:break
                        time.sleep(5)
                    if digest(record['case_report'])!=record['case_report_sha256']:
                        raise RuntimeError('Reference case changed')
                    reference=json.loads(Path(record['case_report']).read_text())
                    if reference['input']!=samples[index]:raise RuntimeError('Reference conditioning differs')
                    case_path=generate_one(pipe,runtime,samples[index],index,args.out,identity)
                    case=json.loads(case_path.read_text())
                    if ([r['raw_tensor_sha256'] for r in case['trajectory']]!=[r['raw_tensor_sha256'] for r in reference['trajectory']]
                            or case['video']['sha256']!=reference['video']['sha256'] or audit['fallback_calls']):
                        raise RuntimeError(f'Resident intervention changed formal input{index:02d}')
                    completed.append({'index':index,'case_report':str(case_path),'case_report_sha256':digest(case_path),
                        'reference_case':record['case_report'],'reference_case_sha256':record['case_report_sha256'],
                        'four_latents_bitwise_equal':True,'video_bytes_equal':True})
                    atomic_json(report,meta)
                    print(f'VERIFIED resident formal{index:02d}: all4 latents and video bytes identical',flush=True)
            finally:
                adapter.deactivate_lightx2v_hooks()
        deadline=time.monotonic()+60
        while True:
            source=json.loads(args.reference_suite.read_text())
            if source['status']!='RUNNING' or time.monotonic()>=deadline:break
            time.sleep(2)
        if (source['status']!='PASS_FORMAL_GENERATION_SLICE_ONLY' or len(completed)!=20
                or {r['index'] for r in completed}!=set(range(20))):
            raise RuntimeError('Source and resident suites must both finish all20')
        meta.update(status='PASS_RESIDENT_FORMAL20_ALL_LATENTS_AND_VIDEO_BYTES_EQUIVALENT',
            reference_suite_sha256=digest(args.reference_suite))
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error));raise
    finally:
        atomic_json(report,meta)


if __name__=='__main__':main()
