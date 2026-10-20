"""Verified OP1 observations and residual replay with the captured native geometry."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from ar2_checkpoint import canonical_json, digest
from reduced_rank_residual import predict_y
from run_ar2_restart import tensor_sha


def same_bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(torch.int16), b.view(torch.int16))


def replay_rows(x, scope, parameters, *, fp32=False):
    """Restore M/layout before both linears; unused rows cannot affect sampled rows."""
    shape, stride = tuple(scope['native_V1_shape']), tuple(scope['native_V1_stride'])
    if (len(shape) not in [2, 3] or shape[-1] != x.shape[-1]
            or (len(shape) == 3 and shape[0] != 1)
            or math.prod(shape[:-1]) != scope['full_rows']
            or len(stride) != len(shape) or any(s <= 0 for s in stride)):
        raise RuntimeError('Unsupported captured native residual geometry')
    ids = torch.tensor(scope['selected_rows'], device=x.device, dtype=torch.long)
    if len(ids) != len(x) or len(ids.unique()) != len(ids) or bool((ids < 0).any()) or bool((ids >= scope['full_rows']).any()):
        raise RuntimeError('Invalid native sampled row coordinates')
    full = torch.empty_strided(shape, stride, device=x.device, dtype=x.dtype).zero_()
    rows = full if len(shape) == 2 else full[0]
    rows.index_copy_(0, ids, x)
    result = predict_y(full, parameters, fp32=fp32)
    return result.reshape(-1, result.shape[-1]).index_select(0, ids)


def native_sse(y, target, gate):
    return float(((y * gate).double() - (target * gate).double()).square().sum())


def load_collections(paths, root):
    records, bindings, common = {}, [], None
    for path in paths:
        run = json.loads(path.read_text())
        if run['status'] != 'PASS_ONPOLICY_CALIBRATION24_SHARD_AND_NOOP':
            raise RuntimeError('Both OP1 shards must finish before fitting')
        if hashlib.sha256(canonical_json(run['identity']).encode()).hexdigest() != run['fingerprint']:
            raise RuntimeError('OP1 collection identity fingerprint differs')
        identity = {k: v for k, v in run['identity'].items() if k != 'shard'}
        if common is not None and common != identity:
            raise RuntimeError('OP1 shard identities differ')
        common = identity
        bindings.append({'run': str(path.resolve()), 'sha256': digest(path), 'inputs': run['completed']})
        for entry in run['completed']:
            index = entry['index']
            if index in records or index % 2 != run['identity']['shard'] or digest(entry['input_report']) != entry['input_report_sha256']:
                raise RuntimeError('Duplicate, changed or wrong-shard OP1 input')
            value = json.loads(Path(entry['input_report']).read_text())
            if (value['status'] != 'PASS_NATIVE_ONPOLICY_FULL160_AND_NOOP'
                    or value['index'] != index or value['fingerprint'] != run['fingerprint']
                    or value['trajectories']['plain'] != value['trajectories']['observed']
                    or [p['step'] for p in value['trajectories']['plain']] != list(range(4))
                    or value['videos']['plain']['sha256'] != value['videos']['observed']['sha256']
                    or value['input']['calibration_seed'] != 42 + index
                    or value['offline_extra_full_teacher_GEMMs'] != 320):
                raise RuntimeError('OP1 input lacks complete native no-op/teacher evidence')
            if digest(value['input']['img_paths'][0]) != value['input']['input_provenance']['image_sha256']:
                raise RuntimeError('OP1 source image changed')
            for video in value['videos'].values():
                if digest(video['path']) != video['sha256']:
                    raise RuntimeError('OP1 no-op video changed')
            scopes = value['scopes']
            if len(scopes) != 160 or {(s['step'], s['block']) for s in scopes} != {(s, b) for s in range(4) for b in range(40)}:
                raise RuntimeError('OP1 all160 positions required')
            for scope in scopes:
                n = math.prod(scope['grid'])
                salt = f"FFNDS_CALIB24_UNIFORM_V1|{index}|{scope['step']}|{scope['block']}|{scope['grid']}"
                seed = int.from_bytes(hashlib.sha256(salt.encode()).digest()[:8], 'little')
                ids = np.sort(np.random.default_rng(seed).choice(n, 256, replace=False)).tolist()
                if (scope['rows'] != 256 or scope['full_rows'] != n or scope['sampling_seed'] != seed
                        or scope['selected_rows'] != ids or scope['inclusion_probability'] != 256 / n
                        or not scope['Q1_teacher_equals_actual']
                        or not scope['actual_native_BaseY_plus_old_residual_equals_StudentY']
                        or scope['expert'] != ('high_noise_model' if scope['step'] < 2 else 'low_noise_model')):
                    raise RuntimeError('OP1 sampling/native geometry/expert differs')
            records[index] = value
    if set(records) != set(range(24)):
        raise RuntimeError('All24 fixed OP1 inputs required')
    if (common['protocol_sha256'] != digest(root / 'config/onpolicy_residual_op1.json')
            or common['assets']['cache_manifest_sha256'] != digest(root / '.runtime/hif4_rebuilt_cache.json')):
        raise RuntimeError('OP1 protocol or native D-Q cache differs')
    for name, sha in common['sources'].items():
        if digest(name) != sha:
            raise RuntimeError('OP1 capture numerical source changed')
    return records, bindings, common


def position_cases(records, step, block, base_info, device='cuda'):
    cases = []
    for index in range(24):
        scope = next(s for s in records[index]['scopes'] if s['step'] == step and s['block'] == block)
        if (scope['gate_sha256'] != base_info['gate_sha256']
                or scope['original_b2_sha256'] != base_info['original_b2_sha256']
                or scope['effective_base_bias_sha256'] != base_info['policies']['C']['modes']['centered_gain_bias']['b2_sha256']
                or digest(scope['path']) != scope['sha256']):
            raise RuntimeError('OP1 native operands or frozen base changed')
        with safe_open(scope['path'], framework='pt', device='cpu') as handle:
            if set(handle.keys()) != {'V1', 'BaseY', 'TeacherY', 'StudentY', 'b2', 'gate'}:
                raise RuntimeError('Unexpected OP1 payload fields')
            values = {k: handle.get_tensor(k).to(device) for k in handle.keys()}
        if any(t.dtype != torch.bfloat16 or not bool(torch.isfinite(t).all()) for t in values.values()):
            raise RuntimeError('Invalid OP1 BF16 payload')
        if any(values[k].shape != (256, 5120) for k in ['V1', 'BaseY', 'TeacherY', 'StudentY']):
            raise RuntimeError('OP1 row geometry differs')
        values['gate'] = values['gate'].reshape(1, 5120)
        if tensor_sha(values['gate']) != scope['gate_sha256'] or tensor_sha(values['b2']) != scope['original_b2_sha256']:
            raise RuntimeError('OP1 native gate or raw b2 payload differs')
        cases.append({'index': index, 'scope': scope, **values})
    return cases


def check_old_replay(cases, parameters):
    for case in cases:
        replay = case['BaseY'] + replay_rows(case['V1'], case['scope'], parameters)
        if not same_bits(replay, case['StudentY']):
            raise RuntimeError(f"Captured frozen student residual did not replay bitwise: input{case['index']}")


def score_cases(cases, parameters):
    old, base, new, control = 0., 0., 0., 0.
    for case in cases:
        x, scope, gate, teacher = case['V1'], case['scope'], case['gate'], case['TeacherY']
        candidate = case['BaseY'] + replay_rows(x, scope, parameters)
        old += native_sse(case['StudentY'], teacher, gate)
        base += native_sse(case['BaseY'], teacher, gate)
        new += native_sse(candidate, teacher, gate)
        fp32 = (case['BaseY'].float() + replay_rows(x, scope, parameters, fp32=True)) * gate.float()
        control += float((fp32.double() - (teacher * gate).double()).square().sum())
    if old <= 0 or not all(math.isfinite(v) for v in [old, base, new, control]):
        raise RuntimeError('Degenerate OP1 risk denominator')
    return {'old_captured_native_SSE': old, 'base_native_SSE': base, 'new_native_SSE': new,
            'new_FP32_SSE': control, 'new_over_old': new / old, 'new_over_base': new / base,
            'old_BaseY_plus_residual_full_geometry_replay_bitwise': True}
