"""Fixed orthogonal native-state bands; posthoc evidence, not semantic masks.

No model or GPU imports. Called only by a separately registered native scope.
"""
import json
import math
from pathlib import Path
import time

from ffnds_observation_common import sha256, write_json


def masks(shape, np, spatial_cutoff=8):
    if len(shape) != 4:
        raise ValueError('Channel/time/y/x state required')
    _, t, y, x = shape
    ft = np.rint(np.fft.fftfreq(t)*t).astype(int)[:, None, None]
    fy = np.rint(np.fft.fftfreq(y)*y).astype(int)[None, :, None]
    fx = np.rint(np.fft.fftfreq(x)*x).astype(int)[None, None, :]
    low = np.broadcast_to((abs(fy) <= spatial_cutoff) & (abs(fx) <= spatial_cutoff), (t,y,x))
    dc = (ft == 0) & (fy == 0) & (fx == 0)
    result = {'all_axis_DC': dc, 'temporal_DC_spatial_low_nonDC': (ft == 0) & low & ~dc,
              'temporal_nonDC_spatial_low': (ft != 0) & low, 'remaining_spatial_high': ~low}
    if not np.all(sum(m.astype(int) for m in result.values()) == 1):
        raise ValueError('Bands must partition every Fourier coordinate exactly once')
    return result


def describe(fields, np, spatial_cutoff=8):
    needed = ['z_dense', 'z_student', 'z_dense_after', 'z_student_after', 'z_dense_at_student_after']
    arrays = [fields[k] for k in needed]
    if len({a.shape for a in arrays}) != 1 or any(not np.isfinite(a).all() for a in arrays):
        raise ValueError('Finite paired native states with identical shapes required')
    zd, zs, ad, ass, ads = arrays
    values = {'input_delta': zs-zd, 'P': ads-ad, 'N': ass-ads, 'next_delta': ass-ad}
    fft = {k: np.fft.fftn(v, axes=(1,2,3), norm='ortho') for k,v in values.items()}
    whole = {k+'_SSE': float((v*v).sum()) for k,v in values.items()}
    whole['signed_cross'] = float(2*(values['P']*values['N']).sum())
    rows = []
    for name, mask in masks(zd.shape, np, spatial_cutoff).items():
        r = {'name': name, 'real_dimensions': int(mask.sum()*zd.shape[0])}
        for k,v in fft.items():
            r[k+'_SSE'] = float((np.abs(v[:,mask])**2).sum())
        r['signed_cross'] = float(2*(fft['P'][:,mask].conj()*fft['N'][:,mask]).real.sum())
        r['identity_residual'] = r['next_delta_SSE']-r['P_SSE']-r['N_SSE']-r['signed_cross']
        rows.append(r)
    closure = {}
    for k,v in whole.items():
        scale = max(2*math.sqrt(whole['P_SSE']*whole['N_SSE']),1.) if k == 'signed_cross' else max(abs(v),1.)
        tolerance = 1e-10*scale
        residual = sum(r[k] for r in rows)-v
        closure[k] = {'band_minus_time_domain': residual, 'absolute_tolerance': tolerance,
                      'natural_energy_scale': scale}
        if abs(residual) > tolerance:
            raise ValueError('Native signed band/Parseval closure failed: '+k)
    if any(abs(r['identity_residual']) > 1e-9*max(r['next_delta_SSE'],r['P_SSE'],r['N_SSE'],1.) for r in rows):
        raise ValueError('Native signed identity failed within a band')
    return {'shape': list(zd.shape), 'whole': whole, 'bands': rows, 'Parseval_closure': closure,
        'FFT_calls': 4, 'basis': 'UNITARY_PERIODIC_FOURIER_CHANNELWISE_TYX',
        'scope': 'OBSERVED_NATIVE_STATES_POSTHOC; NO_BAND_CAUSALITY_OR_SEMANTIC_MASK_OR_ONLINE_SELECTOR'}


def describe_saved_pair(directory):
    import numpy as np
    from safetensors import safe_open
    directory = Path(directory)
    result = {'status': 'PASS_NATIVE_DISJOINT_BAND_IDENTITIES_NOT_QUALITY', 'cases': [],
              'spatial_cutoff': 8, 'GPU_calls': 0, 'new_fits': 0}
    start = time.perf_counter()
    for step in range(4):
        p = directory / f'flow_s{step}.safetensors'
        meta = json.loads((directory/f'flow_s{step}.json').read_text())
        if sha256(p) != meta['saved_tensors']['sha256'] or meta['status'] != 'PASS_FLOW_AND_NATIVE_MAP_IDENTITIES':
            raise ValueError('Complete native state binding required before FFT')
        with safe_open(p, framework='pt', device='cpu') as f:
            fields = {k:f.get_tensor(k).double().numpy() for k in
                ['z_dense','z_student','z_dense_after','z_student_after','z_dense_at_student_after']}
        if fields['z_dense'].shape != (16,21,58,104):
            raise ValueError('Frozen complete Wan state geometry required')
        result['cases'].append({'step': step, 'state_sha256': sha256(p), **describe(fields,np)})
    result['CPU_elapsed_seconds'] = time.perf_counter()-start
    target = directory/'DISJOINT_BANDS.json'; write_json(target,result)
    return {'path':target.name, 'sha256':sha256(target), 'CPU_cases':4,'CPU_FFT_calls':16}
