"""Narrow high04 recovery; caller must already hold reviewed compute scope.

Only the frozen trusted failed job's SHA-bound file is read. This is never an
unrestricted pickle load: TorchVersion is the sole temporary weights-only
allowlist entry. This module does not dispatch jobs or write the old stage.
"""
import hashlib
import json
from pathlib import Path

import torch
from icc_local_replay import FactorPair
from icc_stage3_assets import checked_path, save_factors, sha

OLD_SCOPE = '9d472dea3acddedebc5bc706b60f3d496c65adce43612a0e51dd53387bb56071'
OLD_FILE = '827a3e57a36e2c719cf4950ea2b58c30ad7477e153d36342391e8f8840ff4011'
OLD_SIZE = 106695389
EXPECTED_TENSOR_SHA256 = {
 'first_left':'beb4b534a83d336198f42df3a7f2325b5de930f442004418a73296667549aa21',
 'first_right':'5de9f19b34d18de4d8b5ccf5fec53f5e0bfa6c0302ced35a4178915804ffd1f7',
 'second_left':'7b5ac07dffefd50fb2729ee4bf17fc04f15ece7f85cd6fa0198836588adb2324',
 'second_right':'706427929d069e7a0525bfcdddfda698bf1f17bf4f50e90ed240d497479183ba'}
SHAPES = {'first_left':(13824,1408), 'first_right':(1408,5120),
          'second_left':(5120,1408), 'second_right':(1408,13824)}


def _read_json(spec): return json.loads(checked_path(spec).read_text(encoding='utf-8'))


def tensor_sha(value):
    # 128 rows bounds temporary host allocations; identity covers raw BF16 bytes.
    h = hashlib.sha256()
    for i in range(0, value.shape[0], 128):
        h.update(value[i:i+128].contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def recover(scope, output):
    spec = scope['recovery']
    if spec['original_scope']['sha256'] != OLD_SCOPE or spec['original_factor']['sha256'] != OLD_FILE:
        raise RuntimeError('unexpected historical artifact identity')
    old_path = checked_path(spec['original_scope'])
    old = json.loads(old_path.read_text(encoding='utf-8'))
    run, end = _read_json(spec['original_run']), _read_json(spec['original_exit'])
    log = checked_path(spec['original_log']).read_text(encoding='utf-8')
    if (old['mode'] != 'six_site_svd' or old['max_matrix_decompositions'] != 12
            or run['scope_sha256'] != OLD_SCOPE or run['slurm_job_id'] != '168881'
            or end['slurm_job_id'] != '168881' or end['exit_code'] != 1
            or run['wan_nfe'] != 0 or run['videos'] != 0
            or 'Unsupported global: GLOBAL torch.torch_version.TorchVersion' not in log
            or log.count('SVD_PAIR_START') != 1 or 'SVD_PAIR_SAVED' in log):
        raise RuntimeError('original execution evidence does not match failure')
    if old['svd_driver'] != scope['svd_driver'] or old['checkpoints'] != scope['checkpoints'] or old['torch_version'] != scope['torch_version']:
        raise RuntimeError('producer numeric/checkpoint scope differs')
    for name in ('bws/icc_decompose.py', 'bws/icc_ffn.py'):
        if old['bundle_sha256'][name] != scope['bundle_sha256'][name]:
            raise RuntimeError('factorization implementation changed')
    path = checked_path(spec['original_factor'])
    if path != old_path.parent / 'output' / 'SVD__high__04__r1408.pt' or path.stat().st_size != OLD_SIZE:
        raise RuntimeError('wrong historical factor path/size')
    before = path.stat()
    unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(path)
    if unsafe != ['torch.torch_version.TorchVersion']:
        raise RuntimeError(f'unexpected checkpoint globals: {unsafe}')
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        data = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    after = path.stat()
    if (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns) != (after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns) or sha(path) != OLD_FILE:
        raise RuntimeError('historical artifact changed during load')
    if set(data) != set(SHAPES) | {'identity'}:
        raise RuntimeError('unexpected factor checkpoint contents')
    identity = data['identity']
    if type(identity) is not dict or type(identity.get('torch_version')) is not torch.torch_version.TorchVersion:
        raise RuntimeError('unexpected TorchVersion metadata type')
    expected = {'family':'SVD','expert':'high','layer':4,'rank':1408,
                'checkpoint_sha256':scope['checkpoints']['high']['sha256'],
                'calibration_manifest_sha256':None,'scale_sha256':None,'svd_driver':'gesvd',
                'svd_full_matrices':False,'scale_epsilon':1e-6,'factor_split':'U*S, Vh',
                'torch_version':scope['torch_version'],'allow_tf32':False,'decomposition_dtype':'float32',
                'execution_factor_dtype':'bfloat16','prefix_rank512_is_same_decomposition':True}
    normalized = dict(identity); normalized['torch_version'] = str(normalized['torch_version'])
    if normalized != expected: raise RuntimeError('historical factor identity mismatch')
    for key, shape in SHAPES.items():
        tensor = data[key]
        if type(tensor) is not torch.Tensor or tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
            raise RuntimeError('historical tensor shape/dtype/layout mismatch: ' + key)
        if not torch.isfinite(tensor).all().item(): raise RuntimeError('non-finite factor: ' + key)
    hashes = {key:tensor_sha(data[key]) for key in SHAPES}
    if hashes != EXPECTED_TENSOR_SHA256: raise RuntimeError('historical tensor byte identities differ')
    pair1 = FactorPair(data['first_left'], data['first_right'])
    pair2 = FactorPair(data['second_left'], data['second_right'])
    target = Path(output) / 'SVD__high__04__r1408.pt'
    receipt = save_factors(target, pair1, pair2, normalized)
    restored = torch.load(target, map_location='cpu', mmap=True, weights_only=True)
    if any(tensor_sha(restored[k]) != hashes[k] for k in SHAPES) or restored['identity'] != normalized:
        raise RuntimeError('tensor bytes changed in normalized re-export')
    if sha(path) != OLD_FILE: raise RuntimeError('historical artifact changed during re-export')
    return {'source_job_id':'168881','source_scope_sha256':OLD_SCOPE,'source_factor_sha256':OLD_FILE,
            'source_factor_bytes':OLD_SIZE,'tensor_sha256':hashes,
            'legacy_load':'weights_only; temporary TorchVersion allowlist only',
            'metadata_change':'torch_version TorchVersion to equal str',
            'reused_matrix_decompositions':2}, receipt
