"""Strict original C24 state loader with explicit historical source verification.

The original validation body is retained. Only original source/protocol lookup
uses the registered private snapshot, never a blanket fingerprint bypass.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from ar2_checkpoint import canonical_json, digest


def load_original_collections(paths, root, history):
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
    history.verify(root / 'config/onpolicy_residual_op1.json', common['protocol_sha256'])
    if common['assets']['cache_manifest_sha256'] != digest(root / '.runtime/hif4_rebuilt_cache.json'):
        raise RuntimeError('OP1 protocol or native D-Q cache differs')
    for name, sha in common['sources'].items():
        history.verify(name, sha)
    return records, bindings, common
