"""Fixed V4 CAL11 Q1 fitting and BF16 anchor/hit semantics.

No mixed-CAL3 parameters. Fitting is a separate reviewed CPU/GPU scope; this
module has no implicit data paths, GPU allocation or job dispatch.
"""
import hashlib
import math
import os
from pathlib import Path
import torch
from icc_stage3_assets import load_pair,load_weights,sha

CAL11 = tuple(sorted(('vbi_1005','vbi_1039','vbi_0895','vbi_0937','vbi_1094','vbi_1082',
                      'vbi_0966','vbi_0304','vbi_0648','vbi_0456','vbi_0032')))


def tensor_hash(t):
    t = t.detach().contiguous().view(torch.uint8).reshape(-1)
    h = hashlib.sha256()
    for i in range(0,t.numel(),8*1024*1024):
        h.update(t[i:i+8*1024*1024].to('cpu').numpy().tobytes())
    return h.hexdigest()


def validate_calibration_pairs(scope):
    entries = scope['calibration_pairs']
    index = {(x['source'],x['expert'],x['layer']):x for x in entries}
    expected = {(s,e,l) for s in CAL11 for e in ('high','low') for l in range(40)}
    if len(index) != len(entries) or set(index) != expected:
        raise ValueError('full 11 source x 80 unique sites required')
    for (source,expert,layer),spec in index.items():
        steps=(0,1) if expert=='high' else (2,3)
        a,h=spec['anchor'],spec['hit']
        for endpoint,step in ((a,steps[0]),(h,steps[1])):
            if (endpoint['source'],endpoint['expert'] if 'expert' in endpoint else expert,endpoint['layer'],endpoint['step'],endpoint['seed'],endpoint['kind']) != (source,expert,layer,step,42,'v2_panel'):
                raise ValueError('Q1 original V2 site identity mismatch')
            if endpoint['input_identity']['checkpoint_sha256'] != scope['checkpoints'][expert]['sha256']:
                raise ValueError('Q1 source/checkpoint identity mismatch')
            if len(endpoint['panel']['sha256'])!=64 or len(endpoint['token_id_sha256_le_int64'])!=64:
                raise ValueError('Q1 input identities missing')
        if a['input_identity']!=h['input_identity'] or a['token_id_sha256_le_int64']!=h['token_id_sha256_le_int64']:
            raise ValueError('Q1 CAL anchor/hit conditions or panel mismatch')
    return index
def fit_site(scope, expert, layer, *, common_replay, device, output, authorize):
    """V4 P512 original CAL11 full-M-derived Y targets; fit one site.

    Q0 alpha/A0 follow fixed BWS common source, which explicitly reproduces V4.
    Output is alpha_Q0, alpha_Q1 and F32 A_Q1, identity and PCG receipt. Caller
    must bind actual CAL11 data, original producer, source/panel IDs and weights.
    """
    authorize(scope,expert,layer)
    if expert not in ('high','low') or layer not in range(40): raise ValueError('invalid site')
    if scope['target_identity'] != 'V4_FULLM_Y_PRE_GATE_PANEL512': raise ValueError('wrong CAL target identity')
    if (scope['ridge'],scope['maxiter'],scope['tolerance']) != (1e-3,240,1e-5): raise ValueError('Q1 solver rule mismatch')
    if sha(common_replay.__file__) != scope['common_replay_sha256']:
        raise ValueError('Q1 producer code identity mismatch')
    if common_replay.Q1_RIDGE != 1e-3 or common_replay.Q1_MAXITER != 240 or common_replay.Q1_TOL != 1e-5:
        raise ValueError('underlying solver constants differ')
    index = validate_calibration_pairs(scope)

    w1,b1,w2,b2 = load_weights(scope['checkpoints'][expert],layer,device)
    cal=[]
    for source in CAL11:
        spec = index[(source,expert,layer)]
        if spec['anchor']['kind'] != 'v2_panel' or spec['hit']['kind'] != 'v2_panel':
            raise ValueError('Q1 fit must use original V2 CAL11 panel inputs')
        if spec['anchor']['input_identity']['checkpoint_sha256'] != scope['checkpoints'][expert]['sha256']:
            raise ValueError('Q1 source/checkpoint identity mismatch')
        pair = load_pair(spec,device)
        u0,u1 = pair['u0'].float(),pair['u1'].float()
        y0,y1 = pair['full_y0_panel'].float(),pair['full_y1_panel'].float()
        cal.append({'source':source,'u0':u0,'u1':u1,'X':u1-u0,'Y':y1-y0,'g':pair['gate'].float(),'n':512})
    alpha0,a0,z = common_replay.q0_prior(cal,w1,b1,w2)
    a_q1,alpha1,info = common_replay.fit_q1(cal,w1,w2,a0,alpha0,lam=1e-3,maxiter=240,tol=1e-5)
    if info['status'] != 'CONVERGED' or not math.isfinite(info['true_relative_residual']) or not 0 <= info['true_relative_residual'] <= 1e-5:
        raise RuntimeError('Q1 PCG did not meet frozen true residual')
    # BW's F32 export formula, followed by runtime BF16 cast, matches q1_predict.
    a_f32 = (w1.float().T * alpha1) @ w2.float().T
    if a_f32.dtype != torch.float32 or not torch.isfinite(a_f32).all().item():
        raise ValueError('invalid F32 export')
    identity = {'expert':expert,'layer':layer,'family':'Q1r2','calibration_sources':CAL11,
                'calibration_target':'V4_FULLM_Y_PRE_GATE_PANEL512', 'checkpoint_sha256':scope['checkpoints'][expert]['sha256'],
                'calibration_manifest_sha256':scope['calibration_manifest_sha256'],
                'common_replay_sha256':scope['common_replay_sha256'],'a0_sha256_f64':tensor_hash(a0),
                'alpha_q0_sha256_f32':tensor_hash(alpha0),'ridge':1e-3,'maxiter':240,'tolerance':1e-5,
                'runtime_numeric':'A_f32 cast BF16; BF16 GEMM; cache/subtract and add in FP32 then BF16'}
    data = {'identity':identity,'alpha_Q0':alpha0.detach().cpu().float().contiguous(),
            'alpha_Q1':alpha1.detach().cpu().float().contiguous(), 'A_q1_f32':a_f32.detach().cpu().contiguous(),
            'q1_fit':info,'z_norm':z}
    out = Path(output)
    with out.open('xb') as f:
        torch.save(data,f); f.flush(); os.fsync(f.fileno())
    reloaded = torch.load(out,map_location='cpu',weights_only=True)
    for name in ('alpha_Q0','alpha_Q1','A_q1_f32'):
        if not torch.equal(reloaded[name],data[name]): raise RuntimeError('Q1 export reload mismatch')
    if reloaded['identity'] != identity: raise RuntimeError('Q1 identity reload mismatch')
    return {'file':str(out.resolve()),'sha256':sha(out),'bytes':out.stat().st_size,
            'identity':identity,'q1_fit':info}


class Q1NativeCache:
    """One request/expert/layer. Returns pre-gate output; host applies gate once."""
    def __init__(self, matrix_f32):
        if matrix_f32.dtype != torch.float32 or matrix_f32.ndim != 2 or matrix_f32.shape[0] != matrix_f32.shape[1]:
            raise ValueError('Q1 export must be square F32')
        self.a = matrix_f32.to(torch.bfloat16)
        if not torch.isfinite(self.a).all().item(): raise ValueError('nonfinite BF16 matrix')
        self.clear()
    def clear(self): self.key=self.cache=None
    def anchor(self,key,u,y):
        if not isinstance(key,tuple) or len(key)!=3 or any(v is None for v in key): raise ValueError('request/expert/layer key required')
        if u.dtype != torch.bfloat16 or y.dtype != torch.bfloat16 or u.shape != y.shape or u.shape[-1] != self.a.shape[0]:
            raise ValueError('invalid BF16 anchor')
        q = u @ self.a
        self.key = key
        self.cache = (y.float()-q.float()).bfloat16().detach().clone()
        return y
    def hit(self,key,u):
        if self.cache is None or key != self.key: raise RuntimeError('missing/mismatched anchor')
        if u.dtype != torch.bfloat16 or u.shape != self.cache.shape or u.device != self.cache.device:
            raise ValueError('invalid hit shape/dtype/device')
        return ((u @ self.a).float()+self.cache.float()).bfloat16()





