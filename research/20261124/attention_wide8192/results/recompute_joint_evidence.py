#!/usr/bin/env python3
"""Reanalyse existing FFNDS panel energies; no model execution.

Inputs are the unpacked replay_evidence.zip frozen at b1e15c2d.
New contrasts are exploratory. Source-macro ratios preserve nested sites.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEV = ['vbi_0392', 'vbi_0800', 'vbi_1013']
KEYS = ['source_id', 'background', 'expert', 'layer']

def ratio(data, a, b, fields, level='P64_estimate', background=None,
          expert=None, layer=None):
    d = data[data.source_id.isin(DEV) & data.module.eq('self') & data.fit_level.eq(level)]
    for col, value in [('background', background), ('expert', expert), ('layer', layer)]:
        if value is not None:
            d = d[d[col].eq(value)]
    aa = d[d.arm.eq(a)].set_index(KEYS)
    bb = d[d.arm.eq(b)].set_index(KEYS)
    result = {}
    for field in fields:
        logs = np.log(aa[field] / bb[field])
        sources = logs.groupby('source_id').mean()
        result[field] = dict(ratio=float(np.exp(sources.mean())),
            source_ratios={s: float(np.exp(v)) for s, v in sources.items()},
            nested_sites=len(logs))
    return result

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--replay-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    data = pd.read_csv(args.replay_dir / 'site_metrics.csv')
    panel = json.loads((args.replay_dir / 'plan_assets/attention_panel_index.json').read_text())
    held = sorted(panel['held_rows'])
    edge_audit = {}
    for edge in ['spatial', 'temporal']:
        edges = panel[edge + '_edges_rows']
        count = Counter(i for pair in edges for i in pair)
        assert sorted(count) == held and set(count.values()) == {1}
        edge_audit[edge] = dict(edges=len(edges), held_nodes=len(held), degree=1)
        data[edge + '_difference'] = data[edge] / 2
        data[edge + '_common'] = data.point - data[edge] / 2
        assert (data[edge + '_common'] >= 0).all()

    keys = ['source_id', 'module', 'expert', 'layer', 'arm', 'fit_level']
    a = data[data.background.eq('Dense')].set_index(keys)
    b = data[data.background.eq('wide8192')].set_index(keys)
    numeric = [c for c in a.select_dtypes('number').columns
               if not c.endswith('_common') and not c.endswith('_difference')]
    equality = {}
    for module in ['self', 'text_cross']:
        x = a.xs((module, 'high', 0), level=['module', 'expert', 'layer'])
        y = b.xs((module, 'high', 0), level=['module', 'expert', 'layer'])
        eq = (x[numeric] == y[numeric]) | (x[numeric].isna() & y[numeric].isna())
        assert eq.all().all()
        equality[module] = dict(rows=len(x), numeric_columns=numeric,
                                all_metrics_equal=True)

    divergence = {}
    for module in ['self', 'text_cross']:
        for expert in ['high', 'low']:
            for layer in [0, 13, 26, 39]:
                x = a.xs((module, expert, layer), level=['module', 'expert', 'layer'])
                y = b.xs((module, expert, layer), level=['module', 'expert', 'layer'])
                eq = (x[numeric] == y[numeric]) | (x[numeric].isna() & y[numeric].isna())
                divergence[f'{module}/{expert}/layer{layer}'] = dict(
                    paired_rows=len(x), numeric_fields=len(numeric),
                    differing_rows=int((~eq.all(axis=1)).sum()),
                    differing_numeric_cells=int((~eq).to_numpy().sum()))

    out = dict(status='EXPLORATORY_REANALYSIS', evidence_commit='b1e15c2df4d1dad9a285798e552192ee4fca0bd8',
        new_model_executions=0, independent_DEV_sources=3,
        aggregation='Equal site log ratios within each source, then equal source log mean; exponentiate.',
        measured_boundary='40 concatenated head errors; real W_O and current self gate; no downstream cross/FFN or whole-block counterfactual output is measured.',
        high_layer0_background_identity=equality,
        background_metric_divergence_by_observed_layer=divergence,
        high_layer0_C0_over_OUT_ABS=ratio(data, 'FIN-C0', 'OUT-ABS',
            ['head','point','spatial','temporal'], background='Dense', expert='high', layer=0),
        wide_high_C0_over_OUT_ABS_by_fit={level:ratio(data,'FIN-C0','OUT-ABS',
            ['head','point','spatial','temporal'],level=level,background='wide8192',expert='high')
            for level in ['P64_estimate','all_H_estimator_reference','family_floor_head']},
        edge_partition_audit=edge_audit,
        error_pair_decomposition=dict(identity='E_difference=E_edge/2; E_common=E_point-E_edge/2. Pair basis is (e_i+e_j)/sqrt(2), (e_i-e_j)/sqrt(2).',
            scope='Orthogonal decomposition of the error field on the fixed disjoint edge matching. It is not a perceptual or semantic motion metric.',
            C0_over_OUT_ABS={bg+'/'+expert:ratio(data,'FIN-C0','OUT-ABS',
                ['point','spatial_common','spatial_difference','temporal_common','temporal_difference'],
                background=bg,expert=expert) for bg in ['Dense','wide8192'] for expert in ['high','low']}))
    hi=out['high_layer0_C0_over_OUT_ABS']
    out['high_layer0_relative_directional_energy_gain']=hi['point']['ratio']/hi['head']['ratio']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),background_identity_verified=True,
        directional_energy_gain=out['high_layer0_relative_directional_energy_gain'],
        orthogonal_edge_partitions_verified=True)))

if __name__ == '__main__':
    main()
