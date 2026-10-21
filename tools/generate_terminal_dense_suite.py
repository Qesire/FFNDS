"""Generate one observed20-source cohort with the source-bound active UB2 runtime."""
import argparse
import fcntl
import importlib.metadata
import json
from pathlib import Path
import shutil

import torch

from ar2_checkpoint import atomic_json,digest
from distill_assets import ROOT,verified_assets
from generate_formal_suite import generate_one
from terminal_dense_runtime import create_runtime,effective_groups
from terminal_dense_validation import validate_runtime
from resident_ffn_runtime import make_resident,unused_raw_placeholders


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cohort',choices=['formal20','disjoint20'],required=True)
    ap.add_argument('--policy',type=Path,required=True)
    ap.add_argument('--identity-proof',type=Path,required=True)
    ap.add_argument('--terminal-flow-proof',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--payload-root',type=Path,required=True)
    ap.add_argument('--nonresident',action='store_true')
    ap.add_argument('--reference-suite',type=Path)
    args = ap.parse_args()
    if args.nonresident and not args.reference_suite:
        ap.error('Nonresident equivalence requires the completed resident reference suite')
    policy = json.loads(args.policy.read_text())
    effective_groups(policy)
    if policy['identity_control']:
        raise RuntimeError('Active quality generation cannot use the identity policy')
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = ROOT/f'.runtime/inputs/{args.cohort}/canonical.json'
    recovery_path = inputs.with_name('recovery_report.json')
    samples,recovery = json.loads(inputs.read_text()),json.loads(recovery_path.read_text())
    expected_status = ('PASS_NEW_FORMAL20_INPUT_RECOVERY' if args.cohort == 'formal20'
                       else 'PASS_FROZEN_SOURCE_DISJOINT20_INPUTS')
    if (recovery['status'] != expected_status or recovery['canonical_sha256'] != digest(inputs)
            or len(samples) != 20 or len({v['source_group'] for v in samples}) != 20):
        raise RuntimeError('Exact original frozen20-source conditioning required')
    for sample in samples:
        if (sample['seed'] != 42 or sample['dimension'] not in ['i2v_subject','i2v_background']
                or digest(sample['image_path']) != sample['input_provenance']['image_sha256']):
            raise RuntimeError('Frozen source/seed changed')
    if (not args.payload_root.resolve().is_relative_to(Path('/var/tmp/ffnds-xiang/huggingface/ffnds-derived'))
            or shutil.disk_usage('/var/tmp').free < 84*1024**3):
        raise RuntimeError('HF bulk cache and80GiB reserve plus4GiB allowance required')
    kwargs,cache,assets = verified_assets()
    proof = json.loads(args.identity_proof.read_text())
    if (proof['status'] != 'PASS_UB2_FOUR_CASE_IDENTITY_LATENTS_AND_VIDEO_BYTES'
            or proof['identity']['assets'] != assets or len(proof['completed']) != 4
            or {(v['cohort'],v['index']) for v in proof['completed']} != {(c,i) for c in ['formal20','disjoint20'] for i in [0,10]}):
        raise RuntimeError('Complete four fixed same-implementation identity controls required')
    for path,sha in proof['identity']['sources'].items():
        if digest(path) != sha:
            raise RuntimeError('Identity-proof implementation changed')
    expected_control = {**policy,'identity_control':True,'expected_residual_positions':160,'nominal_auxiliary_MAC_density':256/13824}
    for row in proof['completed']:
        if (digest(row['case_report']) != row['case_report_sha256'] or not row['four_latents_bitwise_equal']
                or not row['video_bytes_equal']):
            raise RuntimeError('Completed identity proof changed')
        case = json.loads(Path(row['case_report']).read_text())
        if case['hif4_runtime']['packed_reference']['policy'] != expected_control:
            raise RuntimeError('Identity proof used another parent/fit/implementation')
    flow = json.loads(args.terminal_flow_proof.read_text())
    if (flow['status'] != 'PASS_UB2_DENSE_TERMINAL_AND_UNCHANGED_STEPS_NATIVE_FLOW'
            or flow['identity']['policy_sha256'] != digest(args.policy)
            or flow['identity']['identity_control_proof_sha256'] != digest(args.identity_proof)
            or not flow['unchanged_steps0_2_bitwise'] or not flow['terminal_flow_post_bitwise_equals_DQ']):
        raise RuntimeError('Active UB2 requires exact terminal and unchanged-step flow controls')
    for path,sha in flow['identity']['sources'].items():
        if digest(path) != sha: raise RuntimeError('Terminal flow control implementation changed')
    reference = None
    if args.reference_suite:
        reference = json.loads(args.reference_suite.read_text())
        if (reference['status'] != 'PASS_FORMAL_GENERATION_SLICE_ONLY'
                or reference['identity']['packed_policy_sha256'] != digest(args.policy)
                or reference['identity']['assets'] != assets or reference['identity']['inputs_sha256'] != digest(inputs)
                or {v['index'] for v in reference['completed']} != set(range(20))):
            raise RuntimeError('Exact completed same-policy20-source reference required')
    from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
    native = [Path(adapter.__file__),ROOT/'.runtime/code/LightX2V/lightx2v/models/networks/wan/infer/transformer_infer.py',
              ROOT/'.runtime/code/LightX2V/lightx2v/common/ops/mm/mm_weight.py',
              ROOT/'.runtime/code/LightX2V/lightx2v/models/schedulers/wan/step_distill/scheduler.py',
              ROOT/'.runtime/code/LightX2V/lightx2v/models/schedulers/wan/scheduler.py']
    sources = [Path(__file__),ROOT/'tools/generate_formal_suite.py',ROOT/'tools/distill_assets.py',
               ROOT/'tools/resident_ffn_runtime.py',ROOT/'tools/terminal_dense_runtime.py',ROOT/'tools/terminal_dense_validation.py']
    identity = {'assets':assets,'inputs_sha256':digest(inputs),'input_recovery_sha256':digest(recovery_path),
        'indices':list(range(20)),'arm':'D-Q-PACKED-TRAINED-REFERENCE','packed_policy_sha256':digest(args.policy),
        'cohort':args.cohort,'identity_control_proof_sha256':digest(args.identity_proof),
        'terminal_flow_proof_sha256':digest(args.terminal_flow_proof),
        'candidate_protocol_sha256':policy['protocol']['sha256'],'runtime_sources':policy['runtime_sources'],
        'sources':{str(p):digest(p) for p in sources},'native_sources':{str(p):digest(p) for p in native},
        'versions':{n:importlib.metadata.version(n) for n in ['torch','torchvision','numpy','triton','safetensors','transformers','diffusers']},
        'seed_semantics':'scheduler.generator=None before every native request',
        'execution':'nonresident paired equivalence' if args.nonresident else 'resident active UB2',
        'reference_suite_sha256':digest(args.reference_suite) if reference else None}
    args.out.mkdir(parents=True,exist_ok=True)
    lock = (args.out/'JOB.lock').open('w')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    report = args.out/'RUN.json'
    completed = []
    if report.exists():
        previous = json.loads(report.read_text())
        if previous['identity'] != identity:
            raise RuntimeError('Active UB2 generation identity changed')
        completed = previous['completed']
    for row in completed:
        if digest(row['case_report']) != row['case_report_sha256']:
            raise RuntimeError('Completed active UB2 case changed')
    meta = {'status':'RUNNING','identity':identity,'completed':completed,'quality_or_speed_acceptance':False}
    atomic_json(report,meta)
    try:
        pending = [i for i in range(20) if i not in {v['index'] for v in completed}]
        if pending:
            if args.nonresident:
                pipe = adapter.build_lightx2v_distill_pipeline(**kwargs)
                audit = None
            else:
                with unused_raw_placeholders(adapter) as audit:
                    pipe = adapter.build_lightx2v_distill_pipeline(**kwargs)
                if len(audit['names']) != 400 or audit['calls'] < 800:
                    raise RuntimeError('All400 covered raw placeholders required')
            runtime = create_runtime(cache['path'],args.policy)
            if not args.nonresident:
                meta['residency'] = make_resident(runtime)
            adapter.activate_lightx2v_hooks(runtime=runtime)
            try:
                for index in pending:
                    path = generate_one(pipe,runtime,samples[index],index,args.payload_root.resolve(),identity)
                    case = json.loads(path.read_text())
                    validate_runtime(case['hif4_runtime']['packed_reference'],digest(args.policy),require_timing=False)
                    if audit and audit['fallback_calls']:
                        raise RuntimeError('A covered raw placeholder fallback occurred')
                    if reference:
                        refrow = next(v for v in reference['completed'] if v['index'] == index)
                        if digest(refrow['case_report']) != refrow['case_report_sha256']:
                            raise RuntimeError('Reference case changed')
                        ref = json.loads(Path(refrow['case_report']).read_text())
                        if ([v['raw_tensor_sha256'] for v in case['trajectory']] != [v['raw_tensor_sha256'] for v in ref['trajectory']]
                                or case['video']['sha256'] != ref['video']['sha256']):
                            raise RuntimeError('UB2 paired nonresident/resident equivalence failed')
                    shared = args.out/f'input{index:02d}_{path.parent.name}_INPUT.json'
                    shutil.copyfile(path,shared)
                    completed.append({'index':index,'case_report':str(shared),'case_report_sha256':digest(shared),
                                      'paired_latents_and_video_equal':True if reference else None})
                    atomic_json(report,meta)
                    print(f'UB2 {args.cohort} committed input{index:02d}',flush=True)
            finally:
                adapter.deactivate_lightx2v_hooks()
        if len(completed) != 20 or {v['index'] for v in completed} != set(range(20)):
            raise RuntimeError('Exactly20 completed independent sources required')
        meta['status'] = 'PASS_FORMAL_GENERATION_SLICE_ONLY'
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error))
        raise
    finally:
        atomic_json(report,meta)


if __name__ == '__main__':
    main()
