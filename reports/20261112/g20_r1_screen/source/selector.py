"""Exact-identity CAL/DEV pair loader. Metadata and SHA checked before tensor use.

Adapted unchanged in substance from the reviewed G19/ICC paired_inputs loader:
flattened-ID join with THW verification, checkpoint/geometry/conditioning checks,
BF16 native U and pre-gate W2 output, per-channel current gate.
"""
import hashlib
import json
from pathlib import Path
import torch


def pair_rows(source, target, expected):
    for obj, step in ((source, expected['source_step']), (target, expected['target_step'])):
        meta = obj['metadata']
        for key in ('source', 'seed', 'layer', 'expert', 'checkpoint_sha256'):
            if meta[key] != expected[key]:
                raise ValueError('metadata mismatch: ' + key)
        if meta['step'] != step:
            raise ValueError('step mismatch')
        ids = obj['token_indices']
        coords = obj['token_thw']
        if ids.unique().numel() != ids.numel() or coords.shape != (ids.numel(), 3):
            raise ValueError('invalid coordinates')
    ids0 = source['token_indices']
    ids1 = target['token_indices']
    order = torch.argsort(ids1)
    pos = torch.searchsorted(ids1[order], ids0)
    valid = pos < ids1.numel()
    valid &= ids1[order[pos.clamp(max=ids1.numel() - 1)]] == ids0
    i = valid.nonzero().flatten()
    j = order[pos[valid]]
    if not i.numel() or not torch.equal(source['token_thw'][i], target['token_thw'][j]):
        raise ValueError('no matching coordinates')
    return i, j


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 ** 2), b''):
            h.update(b)
    return h.hexdigest()


def load_pair(assets, source, seed, expert, steps, checkpoint_sha, layer):
    objects = []
    for step in steps:
        hits = [x for x in assets['captures']
                if (x['metadata']['source'], x['metadata']['seed'],
                    x['metadata']['step'], x['metadata']['layer']) == (source, seed, step, layer)]
        if len(hits) != 1:
            raise ValueError('missing/duplicate capture')
        e = hits[0]
        m = e['metadata']
        if sha(e['path']) != e['sha256'] or sha(e['sidecar']) != e['sidecar_sha256'] \
                or json.loads(Path(e['sidecar']).read_text()) != m:
            raise ValueError('capture identity')
        data = torch.load(e['path'], map_location='cpu', weights_only=True)
        for k in ('source', 'seed', 'step', 'layer', 'expert', 'checkpoint_sha256'):
            if data['metadata'][k] != m[k]:
                raise ValueError('payload identity ' + k)
        if m['grid'] != [21, 29, 52] or m['checkpoint_sha256'] != checkpoint_sha or m['expert'] != expert:
            raise ValueError('checkpoint/geometry')
        ids = data['token_indices']
        if ids.dtype != torch.int64 or ids.shape != (2048,) or ids.min() < 0 or ids.max() >= 31668:
            raise ValueError('invalid token ids')
        expected = torch.stack((ids // 1508, (ids // 52) % 29, ids % 52), 1)
        if not torch.equal(data['token_thw'], expected):
            raise ValueError('token geometry')
        if any(data[k].shape != (2048, 5120) or data[k].dtype != torch.bfloat16
               or not torch.isfinite(data[k]).all() for k in ('z', 'w2_output_before_gate')):
            raise ValueError('tensor shape/dtype/finiteness')
        if not torch.isfinite(data['gate']).all():
            raise ValueError('nonfinite gate')
        objects.append(data)
    expected = {'source': source, 'seed': seed, 'source_step': steps[0],
                'target_step': steps[1], 'layer': layer, 'expert': expert,
                'checkpoint_sha256': checkpoint_sha}
    i, j = pair_rows(*objects, expected)
    a, b = objects
    for key in ('image_sha256', 'prompt_sha256', 'grid'):
        if a['metadata'][key] != b['metadata'][key]:
            raise ValueError('conditioning mismatch ' + key)

    def gate(x, idx):
        g = x['gate'].squeeze()
        if g.shape == (5120,):
            return g[None, :].expand(len(idx), -1).cuda().float()
        if g.shape == (31668, 5120):
            return g.index_select(0, x['token_indices'][idx]).cuda().float()
        raise ValueError('gate shape')

    return {'source': source, 'ids': a['token_indices'][i].tolist(),
            'u0': a['z'][i].cuda(), 'u1': b['z'][j].cuda(),
            'y0': a['w2_output_before_gate'][i].cuda(),
            'y1': b['w2_output_before_gate'][j].cuda(), 'g': gate(b, j)}