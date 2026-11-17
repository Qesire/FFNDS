"""Identity-checked Stage-3 input and factor operations.

Caller must separately verify ACTIVE/review/resource scope before using GPU.
No default paths or fallback to BWS CAL3 parameters or old rollout statistics.
"""
import hashlib
import json
import os
from pathlib import Path
import torch


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def checked_path(spec):
    p = Path(spec['path']).resolve()
    if not p.is_file() or sha(p) != spec['sha256']:
        raise ValueError(f'asset identity mismatch: {p}')
    return p


_CHECKPOINTS = {}

def _stat_identity(path):
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

def preflight_checkpoint(spec):
    """Full SHA once per job, then require unchanged inode/size/timestamps."""
    path = Path(spec['path']).resolve()
    key = (str(path), spec['sha256'])
    before = _stat_identity(path)
    if key not in _CHECKPOINTS:
        if sha(path) != spec['sha256'] or _stat_identity(path) != before:
            raise ValueError('checkpoint hash or identity changed during preflight')
        _CHECKPOINTS[key] = before
    elif _CHECKPOINTS[key] != before:
        raise ValueError('checkpoint stat identity changed after preflight')
    return path, before

def load_weights(checkpoint, layer, device):
    from safetensors import safe_open
    path, initial_stat = preflight_checkpoint(checkpoint)
    with safe_open(str(path), framework='pt', device='cpu') as f:
        tensors = [f.get_tensor(f'blocks.{layer}.ffn.{name}') for name in ('0.weight','0.bias','2.weight','2.bias')]
    if [tuple(x.shape) for x in tensors] != [(13824,5120),(13824,),(5120,13824),(5120,)]:
        raise ValueError('actual checkpoint FFN geometry mismatch')
    if any(x.dtype != torch.bfloat16 for x in tensors): raise ValueError('checkpoint must be BF16')
    values = [x.to(device, copy=True) for x in tensors]
    if _stat_identity(path) != initial_stat:
        raise ValueError('checkpoint changed during tensor materialization')
    return values


def load_endpoint(spec, *, source, step, layer):
    """Separate full-M captured Y from canonical local512 targets.

    New full-U and panel files come from the stage1 capture manifest; historical
    V2 files contain P512 U/Y/gate together. Each carries an explicitly frozen
    SHA and semantic site identity; caller compares coordinate identity endpoints.
    """
    if spec['source'] != source or spec['step'] != step or spec['layer'] != layer or spec['seed'] != 42:
        raise ValueError('input manifest site mismatch')
    if spec['kind'] == 'stage1_full_u':
        u_path, panel_path = checked_path(spec['full_u']), checked_path(spec['panel'])
        u = torch.load(str(u_path),map_location='cpu',mmap=True,weights_only=True)
        panel = torch.load(panel_path,map_location='cpu',weights_only=True)
        idx = panel['token_indices'].long()
        if u.shape != (31668,5120) or u.dtype != torch.bfloat16 or idx.shape != (512,):
            raise ValueError('full-U/panel shape')
        selected_u = u.index_select(0,idx)
        y = panel['Y_fullM_panel']
    elif spec['kind'] == 'v2_panel':
        file = checked_path(spec['panel'])
        panel = torch.load(file,map_location='cpu',weights_only=True)
        metadata = panel['metadata']
        if (metadata['source'],metadata['seed'],metadata['step'],metadata['layer']) != (source,42,step,layer):
            raise ValueError('historical site metadata mismatch')
        selected_u, y = panel['U'],panel['Y_pre_gate']
        idx = panel['token_indices'].long()
    else: raise ValueError('unsupported source identity')
    if selected_u.shape != (512,5120) or y.shape != (512,5120) or selected_u.dtype != torch.bfloat16 or y.dtype != torch.bfloat16:
        raise ValueError('P512 tensors must be BF16')
    if idx.shape != (512,) or idx.min().item() < 0 or idx.max().item() >= 31668 or torch.unique(idx).numel() != 512:
        raise ValueError('invalid P512 IDs')
    actual_sha = hashlib.sha256(idx.numpy().astype('<i8',copy=False).tobytes()).hexdigest()
    if actual_sha != spec['token_id_sha256_le_int64']:
        raise ValueError('panel coordinate identity mismatch')
    gate = panel['gate'].reshape(-1)
    if gate.numel() != 5120 or not torch.isfinite(gate).all().item(): raise ValueError('gate shape or values')
    return {'u':selected_u,'Y_fullM_panel':y,'gate':gate,'indices':idx,'identity':spec}


def load_pair(site, device):
    source, expert, layer = site['source'], site['expert'], site['layer']
    if expert not in ('high','low'): raise ValueError('invalid expert')
    a,h = (0,1) if expert == 'high' else (2,3)
    anchor = load_endpoint(site['anchor'],source=source,step=a,layer=layer)
    hit = load_endpoint(site['hit'],source=source,step=h,layer=layer)
    if anchor['identity']['input_identity'] != hit['identity']['input_identity']: raise ValueError('anchor/hit condition identity differs')
    if not torch.equal(anchor['indices'],hit['indices']): raise ValueError('anchor/hit rows differ')
    return {'source':source,'expert':expert,'layer':layer,
            'u0':anchor['u'].to(device),'u1':hit['u'].to(device),
            'gate':hit['gate'].to(device), 'full_y0_panel':anchor['Y_fullM_panel'].to(device),
            'full_y1_panel':hit['Y_fullM_panel'].to(device), 'indices':anchor['indices']}


def make_site_factors(factorize, w1, w2, family, scale_path, device, *, driver='gesvd'):
    """One decomposition per matrix at rank1408; rank512 is its prefix."""
    if family not in ('SVD','CA','CD'): raise ValueError('invalid family')
    if family == 'SVD':
        if scale_path is not None: raise ValueError('SVD must have no scales')
        scale = None
    else:
        if scale_path is None: raise ValueError('CA/CD must bind new clean statistics')
        scale = torch.load(checked_path(scale_path),map_location='cpu',weights_only=True)
        if set(scale) != {'u','z','h','y'}: raise ValueError('four source-equal scales required')
        scale = {k:scale[k]['capped'].to(device=device,dtype=torch.float32) for k in scale}
    f1 = factorize(w1,1408,left_scale=None if scale is None else scale['z'],right_scale=None if scale is None else scale['u'],driver=driver)
    f2 = factorize(w2,1408,left_scale=None if scale is None else scale['y'],right_scale=None if scale is None else scale['h'],driver=driver)
    return f1.to(device=device,dtype=torch.bfloat16), f2.to(device=device,dtype=torch.bfloat16)


def save_factors(path, first, second, identity):
    """Persist once and verify exact reload, preserving original decomposition."""
    path = Path(path)
    tensors = {'first_left':first.left.detach().cpu().contiguous(), 'first_right':first.right.detach().cpu().contiguous(),
               'second_left':second.left.detach().cpu().contiguous(), 'second_right':second.right.detach().cpu().contiguous()}
    with path.open('xb') as f:
        torch.save({'identity':identity, **tensors},f); f.flush(); os.fsync(f.fileno())
    loaded = torch.load(path,map_location='cpu',weights_only=True)
    if loaded['identity'] != identity or any(not torch.equal(loaded[k],v) for k,v in tensors.items()):
        raise RuntimeError('factor serialization mismatch')
    return {'path':str(path.resolve()),'sha256':sha(path),'identity':identity,'bytes':path.stat().st_size}


def verify_live_weights(capture_manifest, expert, layer, w1, b1, w2, b2):
    """Compare checkpoint values to actual stage1 point-of-use runtime arrays.

    LightX2V MMWeight layout is input x output, while checkpoint is output x
    input. Hash the logical transposed rows, not raw tensor storage order.
    """
    manifest = json.loads(checked_path(capture_manifest).read_text(encoding='utf8'))
    rows = [r for r in manifest['files'] if r['kind']=='live_w1_w2_identity' and r['expert']==expert and r['layer']==layer]
    if len(rows)!=1: raise ValueError('missing/duplicate stage1 live weight identity')
    records={r['name']:r for r in rows[0]['matrices']}
    if set(records)!={'W1','W2'}: raise ValueError('live weight pair incomplete')
    for name,weight,bias in (('W1',w1,b1),('W2',w2,b2)):
        record=records[name]; live=weight.T
        if list(live.shape)!=record['shape'] or str(live.dtype)!=record['dtype'] or list(bias.shape)!=record['bias_shape'] or str(bias.dtype)!=record['bias_dtype']:
            raise ValueError('live weight/bias metadata mismatch')
        h=hashlib.sha256()
        for i in range(0,live.shape[0],128):
            block=live[i:i+128].detach().to('cpu').contiguous()
            h.update(block.view(torch.uint8).numpy().tobytes())
        bias_bytes=bias.detach().to('cpu').contiguous().view(torch.uint8).numpy().tobytes()
        if h.hexdigest()!=record['sha256'] or hashlib.sha256(bias_bytes).hexdigest()!=record['bias_sha256']:
            raise ValueError('checkpoint versus point-of-use weight/bias differs')



