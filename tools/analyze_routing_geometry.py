"""CPU geometry of actually saved full160 pre-W1 row masks; no model run."""
import hashlib
import os
from pathlib import Path

from rollout_contract import ROOT, atomic, read, sha


def geometry(mask):
    import numpy as np
    frames = mask.mean(axis=(-2, -1))
    flip = np.mean(mask[..., 1:, :, :] != mask[..., :-1, :, :], axis=(-3, -2, -1))
    independent = np.mean(frames[..., 1:] + frames[..., :-1] -
                          2 * frames[..., 1:] * frames[..., :-1], axis=-1)
    return {'latent_time_bin_dense_fraction': frames,
            'adjacent_latent_time_bin_mask_flip_fraction': flip,
            'independent_mask_flip_at_same_per_time_marginals': independent,
            'horizontal_neighbor_mask_flip_fraction': np.mean(mask[..., 1:] != mask[..., :-1], axis=(-3, -2, -1)),
            'vertical_neighbor_mask_flip_fraction': np.mean(mask[..., 1:, :] != mask[..., :-1, :], axis=(-3, -2, -1))}


def main():
    import numpy as np
    hf = Path(os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface')))
    root = hf / 'derived/ffnds/R17E0_fresh_quality'
    dynamic_path = root / 'warm_dynamic_residual/REQUEST.json'
    dynamic = read(dynamic_path)
    if dynamic['job'] != {'name': 'warm_dynamic_residual', 'phase': 'warmup', 'arm': 'dynamic_residual', 'source': -1}:
        raise RuntimeError('This diagnostic is explicitly the old exposed warmup source')
    trace = dynamic['routing_trace']
    trace_path = hf / trace['HF_relative']
    if sha(trace_path) != trace['sha256']:
        raise RuntimeError('Actual native routing trace bytes changed')
    with np.load(trace_path, allow_pickle=False) as data:
        grid = tuple(int(v) for v in data['grid'])
        raw = np.unpackbits(data['packed'], axis=-1, bitorder='little')
    rows = int(np.prod(grid))
    if grid != (21, 29, 52) or raw.shape[:2] != (4, 40) or raw[..., rows:].any():
        raise RuntimeError('Native layout or bitpacked padding differs')
    mask = raw[..., :rows].astype(np.bool_).reshape(4, 40, *grid)
    if not np.all(mask.sum(axis=(-3, -2, -1)) == rows // 4):
        raise RuntimeError('Actual native dense-row fraction differs')
    observations = {'dynamic_residual': geometry(mask)}
    binding = {'dynamic_request_sha256': sha(dynamic_path), 'actual_trace_sha256': sha(trace_path)}
    for arm in ('fixed_stride', 'fixed_hash'):
        path = root / ('warm_' + arm) / 'REQUEST.json'
        record = read(path)
        ids = np.arange(rows, dtype=np.int64)
        dense = ids[::4][:rows // 4] if arm == 'fixed_stride' else np.sort(
            np.argsort((ids * 1103515245 + 12345) % (1 << 31), kind='stable')[:rows // 4])
        fingerprint = hashlib.sha256(dense.tobytes()).hexdigest()
        if len(record['dispatch_events']) != 160 or any(e['dense_row_ids_sha256'] != fingerprint for e in record['dispatch_events']):
            raise RuntimeError('CPU fixed-mask formula is not the actually executed fixed control')
        fixed = np.zeros(rows, dtype=np.bool_)
        fixed[dense] = True
        observations[arm] = geometry(fixed.reshape(grid))
        binding[arm + '_request_sha256'] = sha(path)
        binding[arm + '_actual_dense_ids_sha256'] = fingerprint
    # Pure mathematical controls, separately labeled from actual masks.
    stable = np.array([[[1, 0]], [[1, 0]]], dtype=np.bool_)
    switching = np.array([[[1, 0]], [[0, 1]]], dtype=np.bool_)
    if (geometry(stable)['adjacent_latent_time_bin_mask_flip_fraction'] != 0 or
            geometry(switching)['adjacent_latent_time_bin_mask_flip_fraction'] != 1):
        raise RuntimeError('Synthetic geometry identity controls failed')
    def json_value(value):
        if isinstance(value, dict):
            return {k: json_value(v) for k, v in value.items()}
        return value.tolist() if hasattr(value, 'tolist') else value
    report = {'id': 'R17E0_OLD_SOURCE_NATIVE_MASK_GEOMETRY',
        'status': 'RECORDED_NATIVE_MASK_CPU_ANALYSIS_NOT_NEW_FORWARD_OR_QUALITY_RESULT',
        'source_sha256': sha(__file__), 'raw_binding': binding, 'grid': list(grid),
        'source_exposure': 'old_calibration24_source0_seed42_warmup', 'actual_dynamic_FFN_positions': 160,
        'dense_fraction_each_position': .25, 'observations': json_value(observations),
        'synthetic_controls': {'time_stable_flip_fraction': 0, 'time_switching_flip_fraction': 1,
                               'not_native_evidence': True},
        'interpretation': 'Adjacent mask differences are latent-token allocation geometry, not measured flicker or causal motion damage. A changing foreground can justify changing masks. VAE temporal bins are not81decoded video frames. No correlation alone identifies the cause of the local-to-free-trajectory mismatch.',
        'new_model_forwards': 0, 'parameter_or_policy_change': False}
    target = ROOT / 'research/20261023/R17E0_oldsource_routing_geometry.json'
    if target.exists():
        if read(target) != report:
            raise RuntimeError('Preserve the original recorded-mask readout')
    else:
        atomic(target, report)
    for arm, values in observations.items():
        time = values['latent_time_bin_dense_fraction']
        print(arm, 'mean_first_time_bin_dense', float(time[..., 0].mean()),
              'mean_other_time_bin_dense', float(time[..., 1:].mean()),
              'mean_adjacent_time_mask_flip', float(values['adjacent_latent_time_bin_mask_flip_fraction'].mean()), flush=True)


if __name__ == '__main__':
    main()
