"""Independent arithmetic audit of archived joint-action Grams; no model execution.

Every field is recomputed from the raw receipt, not the archived summary.
The archived summary is used only as a final comparison target.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read_bytes(path):
    value = path.read_bytes()
    return gzip.decompress(value) if path.suffix == '.gz' else value


def load(path):
    return json.loads(read_bytes(path))


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


class Audit:
    def __init__(self):
        self.grams = []
        self.summary_comparisons = []

    def gram(self, value, size, label):
        g = np.asarray(value, dtype=np.float64)
        assert g.shape == (size, size) and np.isfinite(g).all(), label
        asym = float(np.max(np.abs(g - g.T)))
        scale = float(max(np.max(np.abs(g)), 1e-30))
        assert asym <= 1e-12 * scale + 1e-12, label
        eigen = np.linalg.eigvalsh((g + g.T) / 2)
        assert eigen[0] >= -1e-10 * scale, (label, eigen)
        self.grams.append(dict(label=label, size=size, max_asymmetry=asym,
                               min_eigenvalue=float(eigen[0]), max_eigenvalue=float(eigen[-1]),
                               relative_min_eigenvalue=float(eigen[0] / scale)))
        return g

    def compare(self, actual, expected, label=''):
        if isinstance(expected, dict):
            assert actual.keys() == expected.keys(), (label, actual.keys(), expected.keys())
            for k in actual:
                self.compare(actual[k], expected[k], label + '/' + k)
        elif isinstance(expected, list):
            assert len(actual) == len(expected), label
            for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
                self.compare(a, b, label + '/' + str(i))
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            assert math.isclose(actual, expected, rel_tol=2e-12, abs_tol=1e-10), (label, actual, expected)
            self.summary_comparisons.append(dict(label=label, absolute_error=float(abs(actual - expected)),
                normalized_error=float(abs(actual - expected) / max(1., abs(expected)))))
        else:
            assert actual == expected, (label, actual, expected)


def two(g):
    rr, tt = map(float, g.diagonal())
    rt = float(g[0, 1])
    return dict(raw=rr, transport=tt, ratio=tt / rr,
                cosine=rt / math.sqrt(rr * tt), update_energy=rr + tt - 2 * rt,
                raw_dot_update=rt - rr)


def background(g):
    rr, tt, b = map(float, g.diagonal())
    raw = rr + b + 2 * float(g[0, 2])
    trans = tt + b + 2 * float(g[1, 2])
    wide = tt - rr
    cross = 2 * float(g[1, 2] - g[0, 2])
    assert math.isclose(trans - raw, wide + cross, rel_tol=1e-9, abs_tol=1e-9)
    return dict(raw=raw, transport=trans, ratio=trans / raw, background_energy=b,
                twice_background_raw_dot=2 * float(g[0, 2]),
                twice_background_transport_dot=2 * float(g[1, 2]),
                fixed_wide_delta=wide, background_cross_delta=cross, total_delta=trans - raw)


def innovation_stats(g):
    route, shape, scale = map(float, g.diagonal())
    transported = float(g[:2, :2].sum())
    raw = float(g.sum())
    return dict(route_energy=route, shape_energy=shape, scale_energy=scale,
                route_shape_cross=2 * float(g[0, 1]),
                route_shape_cosine=float(g[0, 1]) / math.sqrt(route * shape),
                scale_cross=2 * float(g[0, 2] + g[1, 2]),
                raw_energy=raw, transport_energy=transported, ratio=transported / raw)


def main(args):
    src = args.source
    receipt_path = src / 'JOINT_ACTION_201276.json.gz'
    data = load(receipt_path)
    archived = load(src / 'JOINT_ACTION_SUMMARY.json.gz')
    old = load(args.previous_transport) if args.previous_transport else None
    historical = load(args.historical_quality) if args.historical_quality else None
    assert data['status'] == 'complete' and data['job'] == '201276'
    assert data['plan_commit'] == '51b6e235964df2f1fb1370604387cb9c998f6591'
    assert set(data['sources']) == {'vbi_0392', 'vbi_1013'}
    assert data['layers'] == [0, 13, 26, 39]
    assert hashlib.sha256((src / 'attention_joint_action.py').read_bytes()).hexdigest() == data['source_sha256']
    audit = Audit()
    summary = dict(job=data['job'], plan_commit=data['plan_commit'],
                   receipt_sha256=hashlib.sha256(read_bytes(receipt_path)).hexdigest(), sources={})
    derived, traces, history = {}, {}, {}
    background_sign_flips = []
    for sid, track in data['sources'].items():
        assert track['trace_exact'] and len(track['sites']) == 8
        assert track['counters'] == dict(sites=16, anchors=8, peak_slots=4, hits=8, native_probe_queries=512)
        if historical:
            href = historical['control'][sid]['wide8192']
            traces[sid] = {k: track['track'][k] == href[k] for k in ('initial', 'flows', 'latents', 'video', 'experts')}
            assert all(traces[sid].values())
        summary['sources'][sid], derived[sid] = {}, {}
        for where, row in sorted(track['sites'].items(), key=lambda x: tuple(map(int, x[0].split(':')))):
            label = sid + '/' + where
            tail = row['wan_wide8192_tail']
            assert tail['shared_probe_ffn_input_exact'] and tail['reused_live_ffn_probe_teacher']
            assert tail['shared_probe_vs_live_relative_l2'] == 0
            assert tail['dense_native_ffn_extra_held_rows'] == 448 and tail['axes'] == [448, 5120]
            stages, stage_grams = {}, {}
            for name, val in tail['stage_gram_raw_transport'].items():
                g = audit.gram(val, 2, label + '/stage/' + name)
                stage_grams[name], stages[name] = g, two(g)
            assert len(stages) == 10
            for mode in ('raw', 'transport'):
                assert math.isclose(stages['attention_head'][mode], row['fit_energy'][mode], rel_tol=1e-12)
            backgrounds = {}
            for name, val in tail['stage_gram_raw_transport_background'].items():
                g = audit.gram(val, 3, label + '/background/' + name)
                assert np.allclose(g[:2, :2], stage_grams[name], rtol=1e-12, atol=1e-12)
                backgrounds[name] = background(g)
                if backgrounds[name]['fixed_wide_delta'] * backgrounds[name]['total_delta'] < 0:
                    background_sign_flips.append(dict(source=sid, site=where, stage=name, **backgrounds[name]))
            assert len(backgrounds) == 3
            info = row['innovation_decomposition']
            gn = audit.gram(info['route_shape_gram_standardized'], 2, label + '/innovation/standardized')
            g0 = audit.gram(info['route_shape_scale_gram_physical'], 3, label + '/innovation/before')
            g1 = audit.gram(info['route_shape_scale_post_fit_gram_physical'], 3, label + '/innovation/after')
            assert info['independent_queries'] == 512 and info['keys'] == 31668
            inv = dict(standardized_route_energy=float(gn[0, 0]), standardized_shape_energy=float(gn[1, 1]),
                standardized_route_shape_cosine=float(gn[0, 1] / math.sqrt(gn[0, 0] * gn[1, 1])),
                before=innovation_stats(g0), after=innovation_stats(g1),
                transported_remaining_after_fit=float(g1[:2, :2].sum() / g0[:2, :2].sum()),
                scale_remaining_after_fit=float(g1[2, 2] / g0[2, 2]),
                value_shape_relative_l2=info['normalized_value_shape_relative_l2'],
                current_to_anchor_scale_quantiles=info['current_to_anchor_scale_quantiles'], witness=info['witness'])
            site = dict(stages=stages, backgrounds=backgrounds, innovation=inv,
                live_relative_l2=tail['live_relative_l2'],
                shared_probe_vs_live_relative_l2=tail['shared_probe_vs_live_relative_l2'])
            if sid == 'vbi_0392' and old:
                oldsite = old['replay']['observations'][where]
                history[where] = {mode: row['fit_energy'][mode] / oldsite['fit_energy'][mode] for mode in ('raw', 'transport')}
                assert all(math.isclose(x, 1, rel_tol=1e-12) for x in history[where].values())
                if where.endswith(':39'):
                    oldtail = oldsite['wan_wide8192_tail']['held_block_energy']
                    site['historical_tail_ratios'] = {k: stages['block_exit'][k] / oldtail[k] for k in ('raw', 'transport')}
            elif sid == 'vbi_0392' and where.endswith(':39'):
                # Comparison deliberately skips historical fields when its raw receipt was not supplied.
                archived['sources'][sid][where].pop('historical_tail_ratios', None)
            summary['sources'][sid][where] = site
            post = inv['after']
            x2, s2 = post['transport_energy'], post['scale_energy']
            xs = post['scale_cross'] / 2
            ratio_norm = math.sqrt(s2 / x2)
            cosine_xs = xs / math.sqrt(x2 * s2)
            stage_extra = {}
            for name, st in stages.items():
                d2 = st['update_energy']
                stage_extra[name] = dict(update_to_raw_norm_ratio=math.sqrt(d2 / st['raw']),
                    cosine_raw_update=st['raw_dot_update'] / math.sqrt(st['raw'] * d2),
                    radial_coefficient=(st['raw_dot_update'] + st['raw']) / st['raw'],
                    orthogonal_transport_energy_fraction=1 - st['cosine'] ** 2)
            block = backgrounds['block_exit']
            block_g = np.asarray(tail['stage_gram_raw_transport_background']['block_exit'])
            bd = float(block_g[1, 2] - block_g[0, 2])
            derived[sid][where] = dict(scale_to_transport_norm_ratio=ratio_norm,
                cosine_transport_scale=cosine_xs, scale_benefit_cosine_threshold=-ratio_norm / 2,
                net_scale_penalty_fraction_raw=(s2 + 2 * xs) / post['raw_energy'],
                scale_self_fraction_raw=s2 / post['raw_energy'], scale_cross_fraction_raw=2 * xs / post['raw_energy'],
                post_route_shape_energy_ratio=post['route_energy'] / post['shape_energy'],
                post_route_shape_cross_fraction=post['route_shape_cross'] / (post['route_energy'] + post['shape_energy']),
                route_remaining_after_fit=float(g1[0, 0] / g0[0, 0]),
                shape_remaining_after_fit=float(g1[1, 1] / g0[1, 1]),
                independent_ratio_minus_runtime=post['ratio'] - stages['attention_head']['ratio'],
                block_background_to_raw_energy=block['background_energy'] / stages['block_exit']['raw'],
                block_background_cosine_update=bd / math.sqrt(block['background_energy'] * stages['block_exit']['update_energy']),
                block_background_raw_cosine=float(block_g[0, 2]) / math.sqrt(block_g[0, 0] * block_g[2, 2]),
                block_background_transport_cosine=float(block_g[1, 2]) / math.sqrt(block_g[1, 1] * block_g[2, 2]),
                stage_direction=stage_extra)
    audit.compare(summary, archived)
    rows = [(sid, where, site) for sid, ss in summary['sources'].items() for where, site in ss.items()]
    layer0 = [(sid, where, site) for sid, where, site in rows if where.endswith(':0')]
    others = [(sid, where, site) for sid, where, site in rows if not where.endswith(':0')]
    aggregates = dict(
        source_clusters=2, nested_sites=16, observed_layer0_sites=4,
        stage_gram_count=160, background_gram_count=48, innovation_gram_count=48,
        max_live_relative_l2=max(s['live_relative_l2'] for _, _, s in rows),
        max_shared_probe_vs_live_relative_l2=max(s['shared_probe_vs_live_relative_l2'] for _, _, s in rows),
        max_physical_witness_relative_l2=max(s['innovation']['witness'][k] for _, _, s in rows for k in ('anchor_relative_l2_physical', 'hit_relative_l2_physical')),
        max_layer0_independent_vs_runtime_ratio_absolute_difference=max(abs(derived[sid][where]['independent_ratio_minus_runtime']) for sid, where, _ in layer0),
        layer0_attention_energy_ratio_range=[min(s['stages']['attention_head']['ratio'] for _, _, s in layer0), max(s['stages']['attention_head']['ratio'] for _, _, s in layer0)],
        layer0_block_energy_ratio_range=[min(s['stages']['block_exit']['ratio'] for _, _, s in layer0), max(s['stages']['block_exit']['ratio'] for _, _, s in layer0)],
        layer0_dense_block_energy_ratio_range=[min(s['backgrounds']['block_exit']['ratio'] for _, _, s in layer0), max(s['backgrounds']['block_exit']['ratio'] for _, _, s in layer0)],
        layer0_scale_penalty_fraction_range=[min(derived[sid][where]['net_scale_penalty_fraction_raw'] for sid, where, _ in layer0), max(derived[sid][where]['net_scale_penalty_fraction_raw'] for sid, where, _ in layer0)],
        other_scale_penalty_fraction_range=[min(derived[sid][where]['net_scale_penalty_fraction_raw'] for sid, where, _ in others), max(derived[sid][where]['net_scale_penalty_fraction_raw'] for sid, where, _ in others)],
        layer0_remaining_energy_range=[min(s['innovation']['transported_remaining_after_fit'] for _, _, s in layer0), max(s['innovation']['transported_remaining_after_fit'] for _, _, s in layer0)],
        other_remaining_energy_range=[min(s['innovation']['transported_remaining_after_fit'] for _, _, s in others), max(s['innovation']['transported_remaining_after_fit'] for _, _, s in others)],
        block_wide_dense_sign_agreement=sum((s['stages']['block_exit']['ratio'] - 1) * (s['backgrounds']['block_exit']['ratio'] - 1) > 0 for _, _, s in rows),
        block_wide_improvement_count=sum(s['stages']['block_exit']['ratio'] < 1 for _, _, s in rows),
        min_gram_eigenvalue=min(g['min_eigenvalue'] for g in audit.grams),
        max_gram_asymmetry=max(g['max_asymmetry'] for g in audit.grams),
        summary_numeric_fields_compared=len(audit.summary_comparisons),
        summary_max_absolute_error=max(c['absolute_error'] for c in audit.summary_comparisons),
        summary_max_normalized_error=max(c['normalized_error'] for c in audit.summary_comparisons))
    result = dict(status='VERIFIED', evidence_commit='6baa2a44ffaac5cd90d9a4f4df81809a651a784d',
        raw_sha256=summary['receipt_sha256'], source_sha256=data['source_sha256'], method_sha256=data['method_sha256'],
        execution_scope='CPU arithmetic on archived real Wan measurements; no model or GPU executed',
        aggregates=aggregates, trace_exact_by_field=traces, historical_0392_fit_energy_ratios=history,
        background_sign_flips=background_sign_flips, summary=summary, derived=derived, gram_validation=audit.grams)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'joint_numeric_verified.json').write_text(json.dumps(plain(result), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(plain(dict(status='VERIFIED', aggregates=aggregates, background_sign_flips=background_sign_flips)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--previous-transport', type=Path)
    parser.add_argument('--historical-quality', type=Path)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
    main(parser.parse_args())
