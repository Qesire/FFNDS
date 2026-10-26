"""Bounded CPU Fourier description of saved finite state response, not a repair.

Orthogonal spectral subspaces explain geometry; they do not make teacher-derived
coefficients available online or identify semantic motion or causal importance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
ID = 'E0_SAVED_FLOW_SPECTRUM_20261026'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spectral_panels(p, n, temporal=(0, 1, 3, 10), spatial=(0, 1, 2, 4, 8)):
    """Full signed cross and energy inside conjugate-symmetric rectangular bands."""
    import torch
    if (p.shape != n.shape or p.ndim != 4 or p.device.type != 'cpu'
            or n.device.type != 'cpu' or p.is_complex() or n.is_complex()):
        raise ValueError('Equal CPU C,T,H,W operands required')
    p, n = p.double(), n.double()
    if not torch.isfinite(p).all() or not torch.isfinite(n).all():
        raise ValueError('Finite saved native differences required')
    a, b = [torch.fft.fftn(v, dim=(-3, -2, -1), norm='ortho') for v in (p, n)]
    ep, en, cross = a.abs().square(), b.abs().square(), 2 * (a.conj() * b).real
    actual_total = float((p + n).square().sum())
    direct = {'P_SSE': float(p.square().sum()), 'N_SSE': float(n.square().sum()),
              'signed_cross': float(2 * (p * n).sum()), 'next_SSE': actual_total}
    frequency = [torch.fft.fftfreq(size, d=1 / size).abs() for size in p.shape[1:]]
    panels = []
    for kt in temporal:
        for ks in spatial:
            mask = ((frequency[0] <= kt)[:, None, None]
                    & (frequency[1] <= ks)[None, :, None]
                    & (frequency[2] <= ks)[None, None, :])
            pe, ne, c = [float(v[:, mask].sum()) for v in (ep, en, cross)]
            next_energy = float(((a + b).abs().square())[:, mask].sum())
            panels.append({'temporal_max_cycles': kt, 'spatial_max_cycles_each_axis': ks,
                'real_subspace_dimension': int(mask.sum()) * p.shape[0],
                'P_SSE': pe, 'N_SSE': ne, 'signed_cross': c, 'next_SSE': next_energy,
                'next_energy_fraction': next_energy / actual_total if actual_total else None,
                'signed_energy_recomposition_error': next_energy - pe - ne - c})
    parseval = {k: float(v.sum()) - direct[k] for k, v in
                [('P_SSE', ep), ('N_SSE', en), ('signed_cross', cross)]}
    if any(abs(v) > 1e-10 * max(abs(direct[k]), 1.) for k, v in parseval.items()):
        raise ValueError('CPU Parseval/cross identity did not close')
    return {'direct': direct, 'parseval_errors': parseval, 'bands': panels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--execute-cpu', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.protocol.read_text())
    if (plan['id'] != ID or plan['temporal'] != [0, 1, 3, 10]
            or plan['spatial'] != [0, 1, 2, 4, 8] or plan['max_cases'] != 8
            or plan['max_cpu_threads'] != 4 or plan['max_output_bytes'] != 262144):
        raise ValueError('Unexpected bounded saved-state study')
    required = {plan['native_result'], 'tools/analyze_e0_flow_spectrum.py',
                'tests/test_e0_flow_spectrum.py'}
    if not required <= set(plan['bound_sources']):
        raise ValueError('Native result, analyzer and tests must be checksum bound')
    for name, expected in plan['bound_sources'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Bound source drift')
    native = json.loads((ROOT / plan['native_result']).read_text())
    if native['status'] != 'COMPLETE_FIXED_SCOPE_NOT_QUALITY':
        raise ValueError('Complete original native scope required')
    if ([(r['source_index'], r['seed']) for r in native['runs']] != [(0, 42), (12, 54)]
            or any([s['step'] for s in r['steps']] != [0, 1, 2, 3] for r in native['runs'])):
        raise ValueError('Exactly two original sources and eight ordered states required')
    print(json.dumps({'status': 'SOURCE_PREFLIGHT_ONLY', 'new_model_calls': 0}), flush=True)
    if not args.execute_cpu:
        return
    active = json.loads((ROOT / 'research/ACTIVE_RESEARCH.json').read_text())
    matches = [s for s in active['experiment_authority']['current_user_authorized_scopes']
               if s['id'] == ID and s['protocol_sha256'] == sha(args.protocol)
               and s['execution_status'] == 'AUTHORIZED_NOT_STARTED']
    if len(matches) != 1:
        raise RuntimeError('Exact CPU scope registration required')
    output = Path(plan['canonical_output'])
    output.mkdir(parents=True, exist_ok=False)
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(4)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU-only analysis required')
    result = {'id': ID, 'status': 'STARTED', 'protocol_sha256': sha(args.protocol),
              'new_model_calls': 0, 'GPU_calls': 0, 'fits': 0, 'quality_accepted': False,
              'interpretation': 'Orthogonal saved-state geometry; no online coefficients, semantic motion or video quality acceptance',
              'cases': []}
    started = time.monotonic()
    try:
        for run in native['runs']:
            for step in run['steps']:
                path = Path(plan['native_study']) / f"dc_source{run['source_index']:02d}" / f"flow_s{step['step']}.safetensors"
                if sha(path) != step['tensor_sha256']:
                    raise ValueError('Original full-state tensor changed')
                tensors = load_file(path, device='cpu')
                zd, zs, dc = [tensors[k].double() for k in
                              ['z_dense_after', 'z_student_after', 'z_dense_at_student_after']]
                if zd.shape != (16, 21, 58, 104) or zd.dtype != torch.float64:
                    raise ValueError('Unexpected original state geometry')
                item = spectral_panels(dc-zd, zs-dc)
                if abs(item['direct']['next_SSE'] - step['native']['actual_next_delta_SSE']) > 1e-8:
                    raise ValueError('Native direct SSE differs from verified result')
                item.update(source_index=run['source_index'], seed=run['seed'], step=step['step'],
                            tensor_sha256=step['tensor_sha256'])
                result['cases'].append(item)
        if len(result['cases']) != 8 or torch.cuda.is_initialized():
            raise ValueError('Fixed CPU-only eight-state coverage required')
        result.update(status='PASS_SAVED_SPECTRAL_DESCRIPTION_NOT_METHOD_OR_QUALITY',
                      elapsed_seconds=time.monotonic()-started, GPU_initialized=torch.cuda.is_initialized())
    except BaseException as error:
        result.update(status='FAILED_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        data = json.dumps(result, indent=2, allow_nan=False) + '\n'
        if len(data.encode()) > plan['max_output_bytes']:
            raise RuntimeError('Output budget exceeded')
        (output / 'RUN.json').write_text(data)


if __name__ == '__main__':
    main()
