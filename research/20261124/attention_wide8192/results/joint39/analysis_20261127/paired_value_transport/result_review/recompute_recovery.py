"""Independent numeric review of sealed panel scores and four-arm error Grams.

Usage: python recompute_recovery.py /path/to/experiments [output.json]
No Wan execution or estimator fitting occurs in this script.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).with_name('RECOVERY_RECOMPUTED.json')
SOURCES = ['vbi_0392', 'vbi_0800', 'vbi_1013']
LAYERS = [0, 13, 26, 39]
ARMS = ['A_PFRR', 'B_diagonal', 'C_paired', 'D_shuffled']
counter = {'grams': 0, 'archived_numeric_fields': 0}


def read(name):
    return json.loads((ROOT / name).read_text())


def geometric(values):
    values = list(values)
    assert values and min(values) > 0
    return math.exp(math.fsum(math.log(x) for x in values) / len(values))


def compare(rows, numerator, denominator):
    ratios = {(s, l): values[numerator] / values[denominator] for (s, l), values in rows.items()}
    by_source = {s: geometric(ratios[s, l] for l in LAYERS) for s in SOURCES}
    return dict(ratio=geometric(by_source.values()), by_source=by_source,
                by_layer={str(l): geometric(ratios[s, l] for s in SOURCES) for l in LAYERS},
                winning_sites=sum(r < 1 for r in ratios.values()), sites=len(ratios))


def gram(raw):
    g = np.asarray(raw, dtype=np.float64)
    assert g.ndim == 2 and g.shape[0] == g.shape[1] and np.isfinite(g).all()
    factor = max(float(np.diag(g).max()), 1e-30)
    assert np.max(np.abs(g - g.T)) / factor < 1e-12
    assert np.linalg.eigvalsh((g + g.T) / (2 * factor)).min() > -1e-10
    counter['grams'] += 1
    return g


def check_fields(derived, archived):
    for key, expected in archived.items():
        if key not in derived:
            continue
        got = derived[key]
        if isinstance(expected, dict):
            check_fields(got, expected)
        elif isinstance(expected, (float, int)):
            assert math.isclose(got, expected, rel_tol=1e-10, abs_tol=1e-12), (key, got, expected)
            counter['archived_numeric_fields'] += 1


panel = read('PFRR_PANEL_204479.json')
four = read('FOURARM_204487.json')
archived = read('SUMMARY.json')
assert panel['status'] == four['status'] == 'complete'
assert four['arms'] == ARMS and set(four['records']) == set(SOURCES)
assert len(set(four['value_fit_rows'] + four['value_held_rows'])) == 320
assert len(four['value_fit_rows']) == 256 and len(four['value_held_rows']) == 64
assert sorted(four['shuffled_current_order']) == list(range(256))
assert all(i != j for i, j in enumerate(four['shuffled_current_order']))

panel_ratios = {}
for denominator in ['OUT-ABS', 'OUT-D']:
    panel_ratios[denominator] = {}
    for metric in ['head', 'point', 'spatial', 'temporal']:
        rows = {(r['source'], r['layer']): {a: v[metric] for a, v in r['scores'].items()} for r in panel['sites']}
        assert set(rows) == {(s, l) for s in SOURCES for l in LAYERS}
        panel_ratios[denominator][metric] = compare(rows, 'PFRR-S128', denominator)
check_fields(panel_ratios, archived['panel']['energy_ratios'])

metric_rows = {}
site_details = []


def add(metric, key, values, names=ARMS):
    metric_rows.setdefault(metric, {})[key] = dict(zip(names, map(float, values)))


for source, record in four['records'].items():
    assert record['trace_exact']
    assert sorted(map(int, record['sites'])) == LAYERS
    for layer, row in record['sites'].items():
        key = source, int(layer)
        tail = row['tail']
        assert row['fixed_attention_probes_exact'] and tail['shared_probe_ffn_input_exact']
        assert tail['reused_live_ffn_probe_teacher'] and tail['dense_native_ffn_extra_held_rows'] == 448
        value = gram(row['value_held_gram'])
        pre = gram(row['held_base_gram'])
        post = gram(row['held_attention_gram'])
        add('held_value', key, np.diag(value) / row['held_value_target_energy'], ARMS[1:])
        add('pre_recovery_attention', key, np.diag(pre))
        add('held_attention', key, np.diag(post))
        for name, raw in tail['stage_gram_raw_transport'].items():
            g = gram(raw)
            add(name + '_to_wide', key, np.diag(g))
        dense_blocks = {}
        for name, raw in tail['stage_gram_raw_transport_background'].items():
            g = gram(raw)
            # Each candidate-to-Dense error equals candidate-to-wide plus wide-to-Dense.
            dense_blocks[name] = g
            values = []
            for a in range(4):
                c = np.zeros(5); c[a] = 1; c[4] = 1
                values.append(float(c @ g @ c))
            add(name + '_to_dense', key, values)
        for name, raw in tail['block_edges'].items():
            g = gram(raw)
            values = []
            for a in range(4):
                c = np.zeros(5); c[a] = 1; c[4] = 1
                values.append(float(c @ g @ c))
            add(name + '_edge_to_dense', key, values)
        details = dict(source=source, layer=int(layer), ridge_max_condition=row['ridge_system_max_condition'])
        for name, g, b, c in [('value', value, 0, 1), ('pre', pre, 1, 2), ('post', post, 1, 2)]:
            update = g[c, c] + g[b, b] - 2 * g[b, c]
            alignment = g[b, b] - g[b, c]
            details[name] = dict(C_over_B=float(g[c, c] / g[b, b]), update_energy=float(update),
                                 target_alignment=float(alignment), oracle_eta=float(alignment / update),
                                 oracle_energy_ratio=float(1 - alignment**2 / (g[b, b] * update)))
        details['post_over_pre_update_energy'] = details['post']['update_energy'] / details['pre']['update_energy']
        g = dense_blocks['block_exit']
        b_dense = metric_rows['block_exit_to_dense'][key]['B_diagonal']
        details['dense_block'] = dict(
            C_over_B=metric_rows['block_exit_to_dense'][key]['C_paired'] / b_dense,
            wide_perturbation_difference_fraction=float((g[2, 2] - g[1, 1]) / b_dense),
            background_cross_difference_fraction=float(2 * (g[2, 4] - g[1, 4]) / b_dense))
        site_details.append(details)

four_ratios = {}
for metric, rows in metric_rows.items():
    four_ratios[metric] = {}
    for arm in ARMS:
        if arm == 'B_diagonal' or arm not in next(iter(rows.values())):
            continue
        if metric == 'pre_recovery_attention' and arm == 'A_PFRR':
            continue
        four_ratios[metric][arm] = compare(rows, arm, 'B_diagonal')
    four_ratios[metric]['C_over_D'] = compare(rows, 'C_paired', 'D_shuffled')
check_fields(four_ratios, archived['fourarm']['ratios'])

primary = ['held_value', 'held_attention', 'block_exit_to_dense']
gate = all(max(four_ratios[k][a]['by_source'].values()) < 1 for k in primary for a in ['C_paired', 'C_over_D'])
assert gate is False and archived['fourarm']['continuation_gate_passed'] is False

output = dict(evidence_commit='b26f644923646dcd9d853849cf58fdd122c55a54',
    input_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                  ['PFRR_PANEL_204479.json', 'FOURARM_204487.json', 'SUMMARY.json']},
    scope='Independent arithmetic over real sealed records; no new Wan run or estimator fit',
    metric_scope='Squared error energy ratios, source-equal geometric aggregation of four nested layers',
    panel=panel_ratios, fourarm=four_ratios, site_diagnostics=site_details,
    checks=counter, continuation_gate_passed=gate,
    oracle_scope='Post-hoc interpolation between recorded response fields using held-out labels; diagnosis only, not an online setting or propagated trajectory',
    head_metric_audit='Legacy metrics computes delta.reshape(512,5120) before W_O. P_v projects features only. Both old/new head metrics retain all 128 value channels, on different trajectories/method versions.')
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(output=str(OUT), checks=counter, gate=gate,
    panel_point={a: v['point']['ratio'] for a, v in panel_ratios.items()},
    fourarm_primary={m: {a: v['ratio'] for a, v in four_ratios[m].items()} for m in primary}), indent=2))
