"""Compare row sharing and hidden pruning using saved same-input FFN outputs.

This is a posthoc local diagnostic. Neither a router nor a video model runs.
Only middle steps have matched50% mainMAC; no output-flow splice is used.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def auc(scores, labels):
    positive, negative = sum(labels), len(labels) - sum(labels)
    if not positive or not negative:
        return None
    ordered = sorted(zip(scores, labels))
    rank_sum = 0.
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        rank_sum += (i + 1 + j) / 2 * sum(label for _, label in ordered[i:j])
        i = j
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def summarize(rows):
    a = math.fsum(r['hidden_pruning_pair_SSE'] for r in rows)
    b = math.fsum(r['row_sharing_pair_SSE'] for r in rows)
    oracle = math.fsum(min(r['hidden_pruning_pair_SSE'], r['row_sharing_pair_SSE']) for r in rows)
    return {'pairs': len(rows), 'hidden_pruning_total_SSE': a, 'row_sharing_total_SSE': b,
        'best_global_fixed_axis': 'hidden_pruning' if a <= b else 'row_sharing',
        'local_axis_oracle_SSE': oracle,
        'oracle_reduction_over_best_global_fixed_axis': 1 - oracle / min(a, b),
        'row_sharing_beneficial_pairs': sum(r['row_sharing_pair_SSE'] < r['hidden_pruning_pair_SSE'] for r in rows),
        'pre_W1_input_cosine_AUC_for_positive_sharing_value': auc(
            [r['input_cosine'] for r in rows], [r['row_sharing_pair_SSE'] < r['hidden_pruning_pair_SSE'] for r in rows]),
        'scope': 'Descriptive exposed-sample pair geometry; best global axis is weaker than an audited best fixed spatial mask/schedule. No deployed selector or downstream result.'}


def analyze(scalar_rows):
    grouped = defaultdict(list)
    for row in scalar_rows:
        if row['step'] in (1, 2) and row['graph'] == 'near' and row['family'] == 'full_output':
            key = row['index'], row['step'], *sorted((row['donor_native_row'], row['target_native_row']))
            grouped[key].append(row)
    result, used, excluded = [], defaultdict(set), 0
    for (index, step, left, right), rows in sorted(grouped.items()):
        if (len(rows) != 2 or {r['donor_native_row'] for r in rows} != {left, right}
                or {r['target_native_row'] for r in rows} != {left, right}
                or not math.isclose(rows[0]['BF16_copy_SSE'], rows[1]['BF16_copy_SSE'], rel_tol=1e-12)):
            raise ValueError('Pair lacks symmetric true-coordinate donor observations')
        occupied = used[index, step]
        if occupied.intersection((left, right)):
            excluded += 1
            continue
        occupied.update((left, right))
        old = math.fsum(r['UB1_SSE'] for r in rows)
        # One exact full anchor plus one copied row: one difference, not two.
        # The two possible anchor choices have identical local pair SSE when
        # the native gate is row-constant, though downstream effects can differ.
        shared = rows[0]['BF16_copy_SSE']
        result.append({'index': index, 'step': step, 'left_native_row': left, 'right_native_row': right,
            'hidden_pruning_pair_SSE': old, 'row_sharing_pair_SSE': shared,
            'sharing_advantage': old - shared, 'input_cosine': rows[0]['input_cosine'],
            'input_L2_squared': rows[0]['input_L2_squared']})
    return result, excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Preserve the previous saved-axis analysis')
    audit_path = ROOT / 'research/20261022/R13_spatial_residual_coherence.json'
    audit = json.loads(audit_path.read_text())
    binding = audit['paired_scalar_evidence']
    raw = ROOT / binding['locator']
    if sha(raw) != binding['sha256']:
        raise ValueError('Saved spatial scalar evidence changed')
    rows, excluded = analyze(json.loads(raw.read_text())['rows'])
    result = {'id': 'R13_SAVED_LOCAL_COMPRESSION_AXIS_VALUE',
        'status': 'COMPLETED_POSTHOC_LOCAL_SAVED_OUTPUT_AXIS_COMPARISON',
        'evidence_level': 'SAVED_NATIVE_FFN_OUTPUTS_LOCAL_GATED_PAIR_ONLY',
        'source_sha256': sha(Path(__file__)), 'spatial_audit_sha256': sha(audit_path),
        'paired_scalar_sha256': binding['sha256'], 'new_model_generations': 0, 'fitted_parameters': 0,
        'pairing': 'Greedy sorted actual near edges, output-label blind, no vertex overlap per source/step; partial observed graph, not a full native partition.',
        'excluded_overlapping_edges': excluded, 'rows': rows, 'pooled': summarize(rows),
        'per_step': {str(s): summarize([r for r in rows if r['step'] == s]) for s in (1, 2)},
        'per_source': [{'index': i, **summarize([r for r in rows if r['index'] == i])} for i in sorted({r['index'] for r in rows})],
        'actual_operator_comparison': {
            'hidden_pruning': 'Original two same-input UB1 half-hidden outputs including their frozen calibration/R256.',
            'row_sharing': 'Original full-width FFN on one original input row; copy its pre-gate Y to the other while retaining each original residual stream. Existing recorded fullM teacher Y only; reducedM numeric identity is untested.',
            'pair_main_MAC_in_full_dense_row_FFN_units': {'hidden_pruning': 1., 'row_sharing': 1.},
            'native_rounding_scope': 'BF16 CPU gate-product comparison on saved native Y. Complete x+gate*Y residual writeback and downstream are unmeasured.',
            'signal': 'Saved post-Q1 input cosine is available before W1 but requires charging full-row Q1, similarity/layout and any policy cost. AUC is a descriptive rank statistic; no threshold/policy fitted.'},
        'limits': 'Known24 sources, sampled coordinates, oneblock/two middle steps. No new native pair execution, full graph coverage, strongest fixed spatial policy, action-specific downstream flow, video quality or FFN timing. This does not reopen D0 or authorize a model experiment.'}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'pooled': result['pooled'], 'per_step': result['per_step']}, indent=2))


if __name__ == '__main__':
    main()
