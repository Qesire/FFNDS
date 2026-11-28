"""Recompute real Wan panel evidence; no new model runs or SparsePR execution."""
from pathlib import Path
from statistics import mean
import json
import math

ROOT = Path(__file__).resolve().parent
x = json.loads((ROOT / 'EVIDENCE_INPUT.json').read_text())
rows = x['panel_rows']
assert len(rows) == 192
dev = ['vbi_0392', 'vbi_0800', 'vbi_1013']
fields = ['source_id', 'expert', 'background', 'layer', 'fit_level']
key = lambda r: tuple(r[k] for k in fields)
lookup = {(*key(r), r['arm']): r for r in rows}
assert len(lookup) == len(rows)
contrasts = []
for background in ['Dense', 'wide8192']:
    for expert in ['high', 'low']:
        for level in ['P64_estimate', 'family_floor_head']:
            paired = []
            for r in rows:
                if (r['background'], r['expert'], r['fit_level'], r['arm']) == (
                        background, expert, level, 'OUT-D'):
                    paired.append((r, lookup[(*key(r), 'OUT-ABS')]))
            assert len(paired) == 12
            record = dict(background=background, expert=expert, level=level,
                          sources=3, sites=12, metrics={})
            for metric in (['head', 'point', 'spatial', 'temporal']
                           if level == 'P64_estimate' else ['head']):
                ratios = []
                source_means = {}
                for a, b in paired:
                    ratio = a[metric] / b[metric]
                    assert ratio > 0
                    if metric != 'head':
                        assert min(a[metric] / a[metric + '_native_noise'],
                                   b[metric] / b[metric + '_native_noise']) > 1
                    ratios.append(dict(source=a['source_id'], layer=a['layer'],
                                       squared_energy_ratio=ratio))
                for s in dev:
                    source_means[s] = math.exp(mean(math.log(r['squared_energy_ratio'])
                                                   for r in ratios if r['source'] == s))
                geom = math.exp(mean(math.log(v) for v in source_means.values()))
                record['metrics'][metric] = dict(
                    source_equal_geometric_squared_energy_ratio=geom,
                    corresponding_geometric_L2_ratio=math.sqrt(geom),
                    improved_sites=sum(r['squared_energy_ratio'] < 1 for r in ratios),
                    source_ratios=source_means,
                    per_site=ratios,
                    full_S_estimator_required_squared_error_reduction_vs_current_CAL32_to_tie=1-geom,
                    threshold_scope='Conditional threshold at exactly these old panels, route, probes and metric; actual full-S baseline not measured.')
            contrasts.append(record)

directions = []
for r in x['mass_direction_rows']:
    for target in ['attention', 'dense_block']:
        g = r['attention_gram'] if target == 'attention' else r['dense_block_gram']
        a, b, c = g[0][0], g[0][1], g[1][1]
        if target == 'dense_block':
            a += 2*g[0][2]+g[2][2]
            b += g[0][2]+g[1][2]+g[2][2]
            c += 2*g[1][2]+g[2][2]
        d2 = a+c-2*b
        ed = b-a
        eta = -ed/d2
        oracle = (a+2*eta*ed+eta*eta*d2)/a
        directions.append(dict(source=r['source'], layer=int(r['layer']), target=target,
            centered_to_raw_squared_energy_ratio=c/a,
            oracle_line_gain=eta,
            oracle_line_ratio=oracle,
            direction_antialignment_cosine=-ed/math.sqrt(a*d2),
            extra_raw_energy_fraction_saved_by_oracle_scaling=c/a-oracle,
            scope='Post-hoc interpolation of two observed response fields, not a new model run or realizable propagated trajectory.'))

wide_low = next(c for c in contrasts if c['background']=='wide8192'
                and c['expert']=='low' and c['level']=='P64_estimate')
assert abs(wide_low['metrics']['point']['source_equal_geometric_squared_energy_ratio']
           - 0.38349305352349666) < 1e-12
out = dict(
    status='Recomputed from real archived scores/Gram matrices; no full-SparsePR comparison',
    input_provenance={k:x[k] for k in ['head_checked','panel_source','mass_source']},
    aggregation='Per-site squared error energy ratios; mean log ratio within four layers for each source, then equal weight across three source means. Not total pooled energy or mean relative L2.',
    metric_semantics=dict(head='H448 pre-W_O attention output squared Frobenius error',
        point='H448 error after current W_O and self gate; before outer residual/FFN',
        spatial='224 held spatial-edge error differences, after W_O and self gate',
        temporal='224 held temporal-edge error differences, after W_O and self gate'),
    panel_contrasts=contrasts, centered_direction_diagnostics=directions,
    limitations=['OUT-ABS uses CAL32, not SparsePR v2 full-S features, current standardization or coverage weighting.',
        'Old OUT-D is raw anchored response plus temporal CAL32 features, not the newer centered implementation.',
        '11/24 panels and 201500 fixed C use different trajectories; their ratios must not be multiplied.',
        'Current archived Gram matrices and CAL projections do not recover the missing full S/A panel arrays.',
        'The original school light-panel producer stores native/selected outputs in full 128 channels, allowing estimator replay after the external files are located.'])
(ROOT/'EVIDENCE_VERIFIED.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({c['background']+'/'+c['expert']+'/'+c['level']:
    {k:v['source_equal_geometric_squared_energy_ratio'] for k,v in c['metrics'].items()}
    for c in contrasts},indent=2))
