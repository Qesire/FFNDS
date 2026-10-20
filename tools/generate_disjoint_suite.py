"""Generate frozen supplemental sources with the already accepted resident runtime."""
import argparse
import fcntl
import importlib.metadata
import json
from pathlib import Path
import shutil

import torch

from ar2_checkpoint import atomic_json, digest
from distill_assets import ROOT, verified_assets
from generate_formal_suite import generate_one
from resident_ffn_runtime import make_resident, unused_raw_placeholders


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--indices', type=int, nargs='+', required=True)
    ap.add_argument('--policy', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--payload-root', type=Path, required=True)
    args = ap.parse_args()
    root = ROOT / 'runs/worker_restart_20261020'
    acceptance = json.loads((root / 'UB1_acceptance_controller/RUN.json').read_text())
    if acceptance['status'] != 'COMPLETED_JOINT_ENGINEERING_CHECKS' or not acceptance['joint_engineering_acceptance_pass']:
        raise RuntimeError('Finish and pass original same-policy joint engineering acceptance first')
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs_path = ROOT / '.runtime/inputs/disjoint20/canonical.json'
    recovery_path = inputs_path.with_name('recovery_report.json')
    protocol_path = ROOT / 'config/ub1_source_disjoint20.json'
    samples, recovery, protocol = [json.loads(p.read_text()) for p in [inputs_path, recovery_path, protocol_path]]
    if (recovery['status'] != 'PASS_FROZEN_SOURCE_DISJOINT20_INPUTS'
            or recovery['canonical_sha256'] != digest(inputs_path)
            or recovery['protocol_sha256'] != digest(protocol_path)
            or (args.policy and digest(args.policy) != protocol['frozen_candidate_policy_sha256'])
            or digest(ROOT / 'tools/step_budget_runtime.py') != protocol['frozen_runtime_source_sha256']
            or digest(ROOT / 'tools/trained_packed_runtime.py') != protocol['frozen_parent_runtime_source_sha256']):
        raise RuntimeError('Frozen supplement/implementation identity changed')
    for p, sha in recovery['exclusion_bindings'].items():
        if digest(p) != sha:
            raise RuntimeError('Source exclusion manifest changed')
    if (len(samples) != 20 or len(set(args.indices)) != len(args.indices)
            or not args.indices or any(i not in range(20) for i in args.indices)
            or len({v['source_group'] for v in samples}) != 20
            or len({v['input_provenance']['image_sha256'] for v in samples}) != 20):
        raise RuntimeError('Distinct20 sources and a valid disjoint slice required')
    for sample in samples:
        if sample['seed'] != 42 or digest(sample['image_path']) != sample['input_provenance']['image_sha256']:
            raise RuntimeError('Frozen seed/image changed')
    if (not args.payload_root.resolve().is_relative_to(Path('/var/tmp/ffnds-xiang/huggingface/ffnds-derived'))
            or shutil.disk_usage('/var/tmp').free < 84 * 1024 ** 3):
        raise RuntimeError('Bulk outputs require node HF cache with80GiB reserve plus4GiB allowance')
    kwargs, cache, assets = verified_assets()
    from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
    identity = {'assets': assets, 'inputs_sha256': digest(inputs_path), 'input_recovery_sha256': digest(recovery_path),
                'indices': args.indices, 'arm': 'D-Q-PACKED-TRAINED-REFERENCE' if args.policy else 'D-Q',
                'packed_policy_sha256': digest(args.policy) if args.policy else None,
                'supplement_protocol_sha256': digest(protocol_path),
                'primary_acceptance_sha256': digest(root / 'UB1_acceptance_controller/RUN.json'),
                'sources': {str(p): digest(p) for p in [Path(__file__), ROOT / 'tools/generate_formal_suite.py',
                    ROOT / 'tools/resident_ffn_runtime.py', ROOT / 'tools/distill_assets.py']},
                'native_sources': {str(p): digest(p) for p in [Path(adapter.__file__),
                    ROOT / '.runtime/code/LightX2V/lightx2v/models/networks/wan/infer/transformer_infer.py',
                    ROOT / '.runtime/code/LightX2V/lightx2v/models/schedulers/wan/step_distill/scheduler.py']},
                'versions': {n: importlib.metadata.version(n) for n in ['torch', 'torchvision', 'numpy', 'triton',
                    'safetensors', 'transformers', 'diffusers']},
                'seed_semantics': 'scheduler.generator=None before every native request',
                'execution': 'native resident bridge already proved byte equivalent on original formal20'}
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / 'JOB.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    report, completed = args.out / 'RUN.json', []
    if report.exists():
        previous = json.loads(report.read_text())
        if previous['identity'] != identity:
            raise RuntimeError('Supplement run identity changed')
        completed = previous['completed']
    for row in completed:
        if digest(row['case_report']) != row['case_report_sha256']:
            raise RuntimeError('Completed supplemental case changed')
    meta = {'status': 'RUNNING', 'identity': identity, 'completed': completed, 'quality_or_speed_acceptance': False}
    atomic_json(report, meta)
    try:
        pending = [i for i in args.indices if i not in {v['index'] for v in completed}]
        if pending:
            with unused_raw_placeholders(adapter) as audit:
                pipe = adapter.build_lightx2v_distill_pipeline(**kwargs)
            if len(audit['names']) != 400 or audit['calls'] < 800:
                raise RuntimeError('All covered raw matrices must be strict placeholders')
            if args.policy:
                from packed_reference_runtime import create_runtime
                runtime = create_runtime(cache['path'], args.policy)
            else:
                runtime = adapter.HIF4CacheRuntime(cache['path'])
            meta['residency'] = make_resident(runtime)
            adapter.activate_lightx2v_hooks(runtime=runtime)
            try:
                for index in pending:
                    case_path = generate_one(pipe, runtime, samples[index], index, args.payload_root.resolve(), identity)
                    if audit['fallback_calls']:
                        raise RuntimeError('Covered raw placeholder fallback occurred')
                    shared = args.out / f'input{index:02d}_{case_path.parent.name}_INPUT.json'
                    shutil.copyfile(case_path, shared)
                    completed.append({'index': index, 'case_report': str(shared), 'case_report_sha256': digest(shared)})
                    atomic_json(report, meta)
                    print(f'SUPPLEMENT committed input{index:02d}, packed={bool(args.policy)}', flush=True)
            finally:
                adapter.deactivate_lightx2v_hooks()
        if len(completed) != len(args.indices) or {v['index'] for v in completed} != set(args.indices):
            raise RuntimeError('Supplement slice incomplete')
        meta['status'] = 'PASS_FORMAL_GENERATION_SLICE_ONLY'
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE', error=str(error))
        raise
    finally:
        atomic_json(report, meta)


if __name__ == '__main__':
    main()
