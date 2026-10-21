"""Validate the completed OP2 UB1 collection before reading native payloads."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from ar2_checkpoint import canonical_json, digest


def load_ub1_collections(paths, root):
    records, bindings, common = {}, [], None
    for shard, path in enumerate(paths):
        report = json.loads(path.read_text())
        identity = report['identity']
        if (report['status'] != 'PASS_OP2_UB1_CALIBRATION_SHARD'
                or hashlib.sha256(canonical_json(identity).encode()).hexdigest() != report['fingerprint']
                or identity['stage'] != 'calibration' or identity['arm'] != 'UB1'
                or identity['indices'] != [i for i in range(24) if i % 2 == shard]
                or identity['protocol_sha256'] != digest(root / 'config/onpolicy_ub1_op2.json')):
            raise RuntimeError('Incomplete or mismatched native UB1 collection')
        for name, sha in identity['sources'].items():
            if digest(name) != sha:
                raise RuntimeError('Native UB1 collection source changed')
        shared = {k: v for k, v in identity.items() if k != 'indices'}
        if common is not None and common != shared:
            raise RuntimeError('Native UB1 shard identities differ')
        common = shared
        if {e['index'] for e in report['completed']} != set(identity['indices']):
            raise RuntimeError('Native UB1 shard source coverage differs')
        bindings.append({'path': str(path), 'sha256': digest(path)})
        for entry in report['completed']:
            index = entry['index']
            if index in records or digest(entry['input_report']) != entry['input_report_sha256']:
                raise RuntimeError('Duplicate or changed UB1 native input')
            case = json.loads(Path(entry['input_report']).read_text())
            if (case['status'] != 'PASS_OP2_NATIVE_INPUT_AND_NOOP' or case['fingerprint'] != report['fingerprint']
                    or case['arm'] != 'UB1' or case['index'] != index or case['input']['seed'] != 42 + index
                    or case['trajectories']['plain'] != case['trajectories']['observed']
                    or [p['step'] for p in case['trajectories']['plain']] != list(range(4))
                    or case['videos']['plain']['sha256'] != case['videos']['observed']['sha256']
                    or case['offline_extra_teacher_GEMMs'] != 320):
                raise RuntimeError('Native UB1 input no-op or teacher evidence differs')
            if digest(case['input']['image_path']) != case['input']['input_provenance']['image_sha256']:
                raise RuntimeError('Native UB1 image changed')
            for video in case['videos'].values():
                if digest(video['path']) != video['sha256']:
                    raise RuntimeError('Native UB1 video changed')
            scopes = case['scopes']
            if len(scopes) != 160 or {(s['step'], s['block']) for s in scopes} != {(s, b) for s in range(4) for b in range(40)}:
                raise RuntimeError('Native UB1 scope coverage differs')
            for scope in scopes:
                n = math.prod(scope['grid'])
                salt = f"FFNDS_CALIB24_UNIFORM_V1|{index}|{scope['step']}|{scope['block']}|{scope['grid']}"
                seed = int.from_bytes(hashlib.sha256(salt.encode()).digest()[:8], 'little')
                selected = np.sort(np.random.default_rng(seed).choice(n, 256, replace=False)).tolist()
                if (scope['full_rows'] != n or scope['rows'] != 256 or scope['sampling_seed'] != seed
                        or scope['selected_rows'] != selected or scope['inclusion_probability'] != 256 / n
                        or not scope['Q1_teacher_equals_actual']
                        or not scope['actual_native_BaseY_plus_old_residual_equals_StudentY']
                        or scope['declared_main_width'] != [3456, 6912, 6912, 10368][scope['step']]
                        or scope['expert'] != ('high_noise_model' if scope['step'] < 2 else 'low_noise_model')):
                    raise RuntimeError('Native UB1 sampling, width or original teacher control differs')
            records[index] = case
    if set(records) != set(range(24)):
        raise RuntimeError('All24 UB1 native source records are required')
    return records, bindings, common
