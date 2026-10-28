"""Validate a frozen original CPU head view and export explicit runtime layout."""
import hashlib
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def export(view_path, file_sha256, expert, expected):
    import torch
    from safetensors.torch import load_file
    if expert not in ('high_noise_model', 'low_noise_model') or sha(view_path) != file_sha256:
        raise ValueError('Original head view identity differs')
    tensors = load_file(str(view_path), device='cpu')
    prefix = expert + '.'
    names = {'weight': prefix + 'head.head.weight', 'bias': prefix + 'head.head.bias',
             'modulation': prefix + 'head.modulation'}
    shapes = {'weight': (64, 5120), 'bias': (64,), 'modulation': (1, 2, 5120)}
    result = {}
    for key, name in names.items():
        value = tensors[name]
        if value.dtype != torch.bfloat16 or tuple(value.shape) != shapes[key] or not bool(torch.isfinite(value).all()):
            raise ValueError('Original head tensor shape/dtype differs')
        raw = hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        if raw != expected[key + '_raw_sha256']: raise ValueError('Original head tensor bytes differ')
        result[key] = value
    return {'head_weight_runtime_KN': result['weight'].t().contiguous(),
            'head_bias': result['bias'].contiguous(), 'head_modulation': result['modulation'].contiguous()}
