"""Validate full-M dense FFN replay and derive two late-position Q2 caches."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import shutil
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ar2_checkpoint import CheckpointStore, atomic_json, digest
from dense_ffn_teacher import from_v1
from fit_cstatic_centered import load_training_reports
from onpolicy_residual import load_collections, same_bits
from run_ar2_restart import ROOT, tensor_sha


def replay(scope, w0, b0, w2, smooth2, quantizer, *, student):
    if digest(scope['path']) != scope['sha256']: raise RuntimeError('Native source operand checksum differs')
    names = ['V1', 'TeacherY', 'b2'] if student else ['V1', 'V2', 'Y', 'b2']
    with safe_open(scope['path'], framework='pt', device='cpu') as handle:
        values = {k: handle.get_tensor(k).cuda() for k in names}
    n = math.prod(scope['grid']); ids = torch.tensor(scope['selected_rows'], device='cuda')
    if any(v.dtype != torch.bfloat16 or not bool(torch.isfinite(v).all()) for v in values.values()):
        raise RuntimeError('Expected finite BF16 native operands')
    full = torch.zeros((n, 5120), device='cuda', dtype=torch.bfloat16)
    full.index_copy_(0, ids, values['V1'])
    y, v2 = from_v1(full, w0, b0, w2, values['b2'].reshape(-1), smooth2, quantizer)
    y, v2 = y.index_select(0, ids), v2.index_select(0, ids)
    reference = values['TeacherY' if student else 'Y']
    if not same_bits(y, reference): raise RuntimeError('Full-geometry replay Y does not match actual native output')
    if not student and not same_bits(v2, values['V2']):
        raise RuntimeError('Full-geometry replay V2 does not match actual saved dense V2')
    return v2.detach().cpu().contiguous(), {'full_rows': n, 'sample_rows': len(ids),
        'Y_bitwise_equals_actual': True, 'V2_bitwise_equals_actual_dense_capture': True if not student else None,
        'Y_sha256': tensor_sha(y), 'V2_sha256': tensor_sha(v2), 'extra_full_GEMMs': 2,
        'source_operand_sha256': scope['sha256']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--onpolicy', type=Path, nargs=2, required=True)
    ap.add_argument('--dense', type=Path, nargs=2, required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args(); torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
    from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq as quantizer
    if (not args.out.resolve().is_relative_to(Path('/var/tmp/ffnds-xiang/huggingface/ffnds-derived'))
            or shutil.disk_usage('/var/tmp').free < 81 * 1024**3):
        raise RuntimeError('Node HF control must preserve80GiB reserve plus1GiB allowance')
    args.out.mkdir(parents=True, exist_ok=True); args.report.parent.mkdir(parents=True, exist_ok=True)
    lock = (args.out / 'JOB.lock').open('w'); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    records, bindings, capture = load_collections(args.onpolicy, ROOT)
    dense, dense_bindings, dense_capture = load_training_reports(args.dense, expected_inputs=24, expected_rows=256)
    if capture['calibration_collections'] != dense_bindings or capture['assets'] != dense_capture['assets']:
        raise RuntimeError('Original dense/observed student calibration identity differs')
    cache_path = ROOT / '.runtime/hif4_rebuilt_cache.json'; cache = json.loads(cache_path.read_text())
    shard = next(s for s in cache['shards'] if s['expert'] == 'low_noise_model' and s['block'] == 20)
    if digest(shard['path']) != shard['sha256']: raise RuntimeError('Original effective cache shard changed')
    payload = torch.load(shard['path'], map_location='cpu', weights_only=True, mmap=True)['low_noise_model']
    w0 = payload['blocks.20.ffn.0']['weight_qdq'].cuda()
    w2 = payload['blocks.20.ffn.2']['weight_qdq'].cuda(); smooth2 = payload['blocks.20.ffn.2']['smooth_scale'].cuda()
    mirror = json.loads((ROOT / '.runtime/node_local_model_view.json').read_text())
    source = next(p for p in mirror['files'] if '/low_noise_model/' in p['view_path'] and p['filename'].endswith('.safetensors'))
    with safe_open(source['view_path'], framework='pt', device='cpu') as handle:
        b0 = handle.get_tensor('blocks.20.ffn.0.bias').to(device='cuda', dtype=torch.bfloat16)
    if any(s['original_b0_sha256'] != tensor_sha(b0) for r in records.values() for s in r['scopes'] if s['block'] == 20 and s['step'] in [2, 3]):
        raise RuntimeError('Original raw b0 differs from observations')
    protocol = ROOT / 'config/dense_v2_replay_control.json'
    sources = [Path(__file__), ROOT / 'tools/dense_ffn_teacher.py', ROOT / 'tools/onpolicy_residual.py',
        ROOT / '.runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/wan_i2v.py']
    store = CheckpointStore(args.out, {'protocol_sha256': digest(protocol), 'onpolicy_collections': bindings,
        'dense_control_collections': dense_bindings, 'capture_identity': capture,
        'cache_manifest_sha256': digest(cache_path), 'cache_shard_sha256': shard['sha256'],
        'raw_b0_sha256': tensor_sha(b0), 'torch': torch.__version__, 'TF32': False,
        'sources': {str(p): digest(p) for p in sources}})
    meta = {'status': 'RUNNING', 'fingerprint': store.fingerprint, 'checkpoint_root': str(args.out.resolve()),
        'dense_controls': [], 'derived': [], 'quality_or_speed_acceptance': False}
    atomic_json(args.report, meta); started = time.perf_counter()
    try:
        with torch.inference_mode():
            for index in [0, 1]:
                for step in [2, 3]:
                    scope = next(s for s in dense[index]['scopes'] if s['step'] == step and s['block'] == 20)
                    _, control = replay(scope, w0, b0, w2, smooth2, quantizer, student=False)
                    meta['dense_controls'].append({'index': index, 'step': step, **control}); atomic_json(args.report, meta)
            for index in range(24):
                for step in [2, 3]:
                    scope = next(s for s in records[index]['scopes'] if s['step'] == step and s['block'] == 20)
                    key = f'input{index:02d}_step{step}'; sidecar = args.out / (key + '.json')
                    if sidecar.exists():
                        previous = json.loads(sidecar.read_text())
                        if previous['fingerprint'] != store.fingerprint or digest(previous['path']) != previous['sha256']:
                            raise RuntimeError('Derived checkpoint identity/payload changed')
                        meta['derived'].append(previous); continue
                    v2, result = replay(scope, w0, b0, w2, smooth2, quantizer, student=True)
                    path = args.out / (key + '.safetensors'); temp = Path(str(path) + '.tmp')
                    save_file({'V2': v2}, str(temp)); temp.replace(path)
                    entry = {'fingerprint': store.fingerprint, 'index': index, 'step': step, 'block': 20,
                        'path': str(path.resolve()), 'sha256': digest(path), 'bytes': path.stat().st_size, **result}
                    atomic_json(sidecar, entry); meta['derived'].append(entry); atomic_json(args.report, meta)
                    print(f'DERIVED native V2 input{index:02d} step{step}: actual TeacherY bitwise PASS', flush=True)
        if len(meta['dense_controls']) != 4 or len(meta['derived']) != 48: raise RuntimeError('Incomplete replay control')
        meta.update(status='PASS_DENSE_FULL_GEOMETRY_V2_Y_CONTROLS_AND48_STUDENT_TEACHER_Y_REPLAYS',
            elapsed_seconds=time.perf_counter() - started,
            scope='Derived native Q2; original OP1 internal teacher Q2 was not saved, so no direct OP1 Q2 bit-identity claim')
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE', error=str(error)); raise
    finally:
        atomic_json(args.report, meta); atomic_json(args.out / 'run_meta.json', meta)


if __name__ == '__main__': main()
