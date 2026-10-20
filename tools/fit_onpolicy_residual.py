"""Fixed OP1 correction-only CV, then conditional all160 rank256/lambda1 refit."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import time

import torch

from ar2_checkpoint import CheckpointStore, atomic_json, digest
from evaluate_residual_rr1 import open_verified
from onpolicy_residual import check_old_replay, load_collections, position_cases, score_cases
from reduced_rank_residual import fit_rrr, native_parameters
from run_ar2_restart import ROOT, tensor_sha
from run_residual_rr1 import encode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--collections', type=Path, nargs=2, required=True)
    ap.add_argument('--student-policy', type=Path, required=True)
    ap.add_argument('--mode', choices=['cv', 'full'], required=True)
    ap.add_argument('--cv-report', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(8); torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    if not any(args.out.resolve().is_relative_to(p) for p in [
            Path('/var/tmp/ffnds-xiang/huggingface/ffnds-derived'), Path.home() / '.cache/huggingface/ffnds-derived']):
        raise RuntimeError('OP1 parameters must stay in HF derived cache')
    if shutil.disk_usage('/var/tmp').free < 83 * 1024**3:
        raise RuntimeError('Keep80GiB reserve plus3GiB fit allowance')
    args.out.mkdir(parents=True, exist_ok=True); args.report.parent.mkdir(parents=True, exist_ok=True)
    lock = (args.out / 'JOB.lock').open('w'); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    records, bindings, capture = load_collections(args.collections, ROOT)
    if capture['student_policy_sha256'] != digest(args.student_policy):
        raise RuntimeError('Must use the same frozen C24 collection student')
    policy = json.loads(args.student_policy.read_text()); bases = {}; base_bindings = []
    for expert in ['high_noise_model', 'low_noise_model']:
        binding = policy['fit_reports'][expert]; path = Path(binding['path'])
        if digest(path) != binding['sha256']:
            raise RuntimeError('Frozen C24 base report changed')
        model, report, identity = open_verified(path, 'COMPLETED_CALIBRATION_PARAMETER_FIT')
        if model.fingerprint != binding['fingerprint'] or len(report['positions']) != 80:
            raise RuntimeError('Incomplete frozen C24 base')
        bases[expert] = model
        base_bindings.append({'path': str(path.resolve()), 'sha256': digest(path), 'fingerprint': model.fingerprint})
    old_binding = policy['residual_fit']; old_path = Path(old_binding['path'])
    if digest(old_path) != old_binding['sha256']:
        raise RuntimeError('Frozen C24 residual report changed')
    old, old_report, old_identity = open_verified(old_path, 'COMPLETED_RR1_FULL160_CALIBRATION_FIT')
    if old.fingerprint != old_binding['fingerprint'] or old_identity['base_fits'] != base_bindings:
        raise RuntimeError('Frozen C24 residual/base identity differs')
    protocol_path = ROOT / 'config/onpolicy_residual_op1.json'
    source_paths = [Path(__file__), ROOT / 'tools/onpolicy_residual.py', ROOT / 'tools/reduced_rank_residual.py',
                    ROOT / 'tools/run_residual_rr1.py']
    identity = {'method': 'OP1_LOCAL_DENSE_TEACHER_ON_C24_STUDENT_STATES', 'protocol_sha256': digest(protocol_path),
        'training_collections': bindings, 'capture_identity': capture, 'student_policy_sha256': digest(args.student_policy),
        'base_fits': base_bindings, 'cache_manifest_sha256': policy['cache_manifest']['sha256'],
        'rank': 256, 'lambda': 1., 'torch': torch.__version__, 'TF32': False,
        'sources': {str(p): digest(p) for p in source_paths}}
    cv = None
    if args.mode == 'full':
        if args.cv_report is None: ap.error('--cv-report required for conditional full fit')
        cv, cv_report, cv_identity = open_verified(args.cv_report, 'COMPLETED_OP1_CV_GATE_PASSED')
        if cv_identity['method_identity'] != identity or len(cv_report['CV']) != 4 or not all(r['gate_pass'] for r in cv_report['CV']):
            raise RuntimeError('Same-method OP1 four-fold gate must pass first')
    store = CheckpointStore(args.out, {'method_identity': identity, **identity, 'mode': args.mode,
        'CV_report_sha256': digest(args.cv_report) if args.cv_report else None})
    meta = {'status': 'RUNNING', 'method': identity['method'], 'fingerprint': store.fingerprint,
        'checkpoint_root': str(args.out.resolve()), 'selected': {'rank': 256, 'lambda': 1., 'selection': 'OP1 preregistered fixed'},
        'CV': [], 'positions': [], 'correction_only_CV': True, 'student_base_used_all24_sources': True,
        'quality_or_speed_acceptance': False}
    atomic_json(args.report, meta); started = time.perf_counter()
    try:
        scopes = [(s, 20) for s in [2, 3]] if args.mode == 'cv' else [(s, b) for s in range(4) for b in range(40)]
        for step, block in scopes:
            expert = 'high_noise_model' if step < 2 else 'low_noise_model'; name = f'step{step}_block{block:02d}'
            base = bases[expert].load('positions', name); previous = old.load('positions', name)
            if base is None or previous is None: raise RuntimeError('Missing frozen base/residual position')
            old_parameters = {k: torch.from_numpy(v).to(device='cuda', dtype=torch.bfloat16) for k, v in previous['arrays'].items()}
            if {k: tensor_sha(v) for k, v in old_parameters.items()} != previous['metadata']['parameter_sha256']:
                raise RuntimeError('Frozen C24 residual parameter encoding differs')
            cases = position_cases(records, step, block, base['metadata'])
            check_old_replay(cases, old_parameters)
            for fold in [0, 1] if args.mode == 'cv' else [None]:
                stage = 'CV_models' if args.mode == 'cv' else 'positions'
                key = f'step{step}_fold{fold}' if args.mode == 'cv' else name
                part = store.load(stage, key)
                if part is None:
                    train = [c for c in cases if fold is None or c['index'] % 2 != fold]
                    validation = [c for c in cases if fold is None or c['index'] % 2 == fold]
                    x = torch.cat([c['V1'] for c in train])
                    y = torch.cat([(c['TeacherY'].float() - c['BaseY'].float()) * c['gate'].float() for c in train])
                    model = fit_rrr(x, y, 1., 256)
                    parameters = native_parameters(model, cases[0]['gate'], 256)
                    row = {'step': step, 'block': block, 'expert': expert, 'fold': fold, 'rank': 256, 'lambda': 1.,
                        'train_inputs': [c['index'] for c in train], 'validation_inputs': [c['index'] for c in validation],
                        'gate_sha256': base['metadata']['gate_sha256'], 'weight_sha256': base['metadata']['weight_sha256'],
                        'parameter_sha256': {k: tensor_sha(v) for k, v in parameters.items()}, **score_cases(validation, parameters)}
                    row['gate_pass'] = row['new_over_old'] <= .90
                    store.save(stage, key, encode(parameters), row); part = store.load(stage, key)
                    del x, y, model, parameters
                meta['CV' if args.mode == 'cv' else 'positions'].append(part['metadata']); atomic_json(args.report, meta)
                print('OP1 FIT ' + json.dumps(part['metadata']), flush=True)
            del cases, old_parameters
        if args.mode == 'cv':
            meta['CV_gate_pass'] = len(meta['CV']) == 4 and all(r['gate_pass'] for r in meta['CV'])
            meta['status'] = 'COMPLETED_OP1_CV_GATE_PASSED' if meta['CV_gate_pass'] else 'COMPLETED_OP1_CV_GATE_REJECTED'
        else:
            if len(meta['positions']) != 160: raise RuntimeError('OP1 requires all160 residual fits')
            # Existing runtime consumes the same two-linear RR parameter schema.
            meta['status'] = 'COMPLETED_RR1_FULL160_CALIBRATION_FIT'
        meta['elapsed_seconds'] = time.perf_counter() - started
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE', error=str(error)); raise
    finally:
        atomic_json(args.report, meta); atomic_json(args.out / 'run_meta.json', meta)


if __name__ == '__main__': main()
