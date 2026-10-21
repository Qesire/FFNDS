"""Prove LR2's identity wrapper on four fixed cases across both observed cohorts."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil

import torch

from ar2_checkpoint import atomic_json,digest
from distill_assets import ROOT,verified_assets
from generate_formal_suite import generate_one
from late_block_rank_runtime import create_runtime
from late_block_validation import validate_runtime
from resident_ffn_runtime import make_resident,unused_raw_placeholders


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--policy',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--payload-root',type=Path,required=True)
    args = ap.parse_args()
    policy = json.loads(args.policy.read_text())
    if not policy['identity_control']:
        raise RuntimeError('Identity proof requires the explicit LR2 identity policy')
    if (not args.payload_root.resolve().is_relative_to(Path('/var/tmp/ffnds-xiang/huggingface/ffnds-derived'))
            or shutil.disk_usage('/var/tmp').free < 82*1024**3):
        raise RuntimeError('Node HF cache and80GiB reserve plus2GiB allowance required')
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    r = ROOT/'runs/worker_restart_20261020'
    cohorts = {'formal20':r/'formal20_UB1/FULL_RUN.json','disjoint20':r/'disjoint20_UB1/RUN.json'}
    samples = {name:json.loads((ROOT/f'.runtime/inputs/{name}/canonical.json').read_text()) for name in cohorts}
    suites = {name:json.loads(p.read_text()) for name,p in cohorts.items()}
    kwargs,cache,assets = verified_assets()
    for name,value in suites.items():
        if (value['status'] != 'PASS_FORMAL_GENERATION_SLICE_ONLY'
                or value['identity']['packed_policy_sha256'] != policy['parent_policy']['sha256']
                or value['identity']['assets'] != assets
                or value['identity']['inputs_sha256'] != digest(ROOT/f'.runtime/inputs/{name}/canonical.json')
                or {v['index'] for v in value['completed']} != set(range(20))):
            raise RuntimeError('Exact complete original UB1 reference suite required')
    sources = [Path(__file__),ROOT/'tools/generate_formal_suite.py',ROOT/'tools/late_block_rank_runtime.py',
               ROOT/'tools/late_block_validation.py',ROOT/'tools/resident_ffn_runtime.py']
    identity = {'policy_sha256':digest(args.policy),'packed_policy_sha256':digest(args.policy),
                'arm':'D-Q-PACKED-TRAINED-REFERENCE','assets':assets,
                'reference_suites':{str(p):digest(p) for p in cohorts.values()},
                'sources':{str(p):digest(p) for p in sources},'payload_root':str(args.payload_root.resolve()),
                'indices':{'formal20':[0,10],'disjoint20':[0,10]}}
    args.out.mkdir(parents=True,exist_ok=True)
    lock = (args.out/'JOB.lock').open('w')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    report = args.out/'RUN.json'
    completed = []
    if report.exists():
        prior = json.loads(report.read_text())
        if prior['identity'] != identity:
            raise RuntimeError('Identity-proof source/protocol changed')
        completed = prior['completed']
    for row in completed:
        if digest(row['case_report']) != row['case_report_sha256']:
            raise RuntimeError('Completed identity case changed')
    meta = {'status':'RUNNING','identity':identity,'completed':completed,'quality_or_speed_acceptance':False}
    atomic_json(report,meta)
    from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
    try:
        if len(completed) < 4:
            with unused_raw_placeholders(adapter) as audit:
                pipe = adapter.build_lightx2v_distill_pipeline(**kwargs)
            if len(audit['names']) != 400 or audit['calls'] < 800:
                raise RuntimeError('All original covered raw matrices must use strict placeholders')
            runtime = create_runtime(cache['path'],args.policy)
            meta['residency'] = make_resident(runtime)
            adapter.activate_lightx2v_hooks(runtime=runtime)
            try:
                for name in cohorts:
                    for index in [0,10]:
                        if any(v['cohort'] == name and v['index'] == index for v in completed):
                            continue
                        sample = samples[name][index]
                        if digest(sample['image_path']) != sample['input_provenance']['image_sha256'] or sample['seed'] != 42:
                            raise RuntimeError('Fixed source/seed changed')
                        refrow = next(v for v in suites[name]['completed'] if v['index'] == index)
                        if digest(refrow['case_report']) != refrow['case_report_sha256']:
                            raise RuntimeError('Original UB1 case changed')
                        reference = json.loads(Path(refrow['case_report']).read_text())
                        if reference['input'] != sample:
                            raise RuntimeError('Reference conditioning differs')
                        casepath = generate_one(pipe,runtime,sample,index,args.payload_root.resolve()/name,identity)
                        case = json.loads(casepath.read_text())
                        validate_runtime(case['hif4_runtime']['packed_reference'],digest(args.policy),require_timing=False,allow_identity=True)
                        if ([v['raw_tensor_sha256'] for v in case['trajectory']] != [v['raw_tensor_sha256'] for v in reference['trajectory']]
                                or case['video']['sha256'] != reference['video']['sha256'] or audit['fallback_calls']):
                            raise RuntimeError('LR2 identity mode changed original UB1 latents/video')
                        shared = args.out/f'{name}_input{index:02d}_{casepath.parent.name}.json'
                        shutil.copyfile(casepath,shared)
                        completed.append({'cohort':name,'index':index,'case_report':str(shared),'case_report_sha256':digest(shared),
                                          'four_latents_bitwise_equal':True,'video_bytes_equal':True})
                        atomic_json(report,meta)
                        print(f'LR2 identity {name} input{index:02d}',flush=True)
            finally:
                adapter.deactivate_lightx2v_hooks()
        if len(completed) != 4 or {(v['cohort'],v['index']) for v in completed} != {(c,i) for c in cohorts for i in [0,10]}:
            raise RuntimeError('Exactly four fixed identity cases required')
        meta['status'] = 'PASS_LR2_FOUR_CASE_IDENTITY_LATENTS_AND_VIDEO_BYTES'
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error))
        raise
    finally:
        atomic_json(report,meta)


if __name__ == '__main__':
    main()
