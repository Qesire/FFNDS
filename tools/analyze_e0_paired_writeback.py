"""JSON-only native-writeback followthrough of the retained v3 paired study."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(result_path, hf_derived):
    result = json.loads(result_path.read_text())
    runs = []
    for binding in result['runs']:
        out = hf_derived / binding['raw']['HF_relative']
        for name in ['RUN.json', 'TRACE.json', 'OBSERVATION.json', 'STRICT_BUNDLE.json']:
            if sha(out / name) != binding['raw']['files_sha256'][name]:
                raise ValueError('Retained v3 evidence changed: ' + name)
        trace = json.loads((out / 'TRACE.json').read_text())
        if (len(trace['sites']) != 160 or {(s['step'], s['block']) for s in trace['sites']}
                != {(s, b) for s in range(4) for b in range(40)}):
            raise ValueError('Native 40x4 observations required')
        ratios = [s['native_writeback']['student_operator_SSE'] / s['native_writeback']['teacher_operator_SSE']
                  for s in trace['sites']]
        runs.append({'source_index': trace['source_index'], 'seed': trace['seed'],
            'TRACE_sha256': sha(out / 'TRACE.json'),
            'native_writeback_smaller_sites': sum(r < 1 for r in ratios),
            'native_writeback_equal_sites': [[s['step'], s['block']] for s, ratio in zip(trace['sites'], ratios) if ratio == 1],
            'native_writeback_larger_sites': sum(r > 1 for r in ratios),
            'student_to_teacher_native_writeback_operator_SSE_ratio_median': median(ratios),
            'per_step': [{'step': step,
                'native_writeback_operator_ratio_median': median(r for s, r in zip(trace['sites'], ratios) if s['step'] == step),
                'native_writeback_operator_to_prefix_SSE_ratio_median': median(s['native_writeback']['student_operator_SSE'] / s['native_writeback']['prefix_response_SSE']
                    for s in trace['sites'] if s['step'] == step and s['native_writeback']['prefix_response_SSE'] > 0),
                'native_writeback_operator_prefix_cosine_median': median(s['native_writeback']['operator_prefix_cosine']
                    for s in trace['sites'] if s['step'] == step and s['native_writeback']['operator_prefix_cosine'] is not None),
                'block39_relative_SSE': next(s['block_output']['relative_SSE'] for s in trace['sites'] if s['step'] == step and s['block'] == 39),
                'largest_block_output_relative_SSE': max(s['block_output']['relative_SSE'] for s in trace['sites'] if s['step'] == step)}
                for step in range(4)]})
    return {'id': 'E0_NATIVE_WRITEBACK_GEOMETRY_FOLLOWTHROUGH_20261026',
            'analysis_source_sha256': sha(Path(__file__)), 'paired_result_sha256': sha(result_path),
            'runs': runs, 'new_model_calls': 0, 'new_GPU_jobs': 0, 'new_videos': 0,
            'interpretation': 'Actual native writeback same-input operator error also decreases318/320 sites. Prefix term includes residual base and finite dense response. These are dependent posthoc decompositions, not causal percentages or pure G*FFN errors.',
            'quality_accepted': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired-result', type=Path, required=True)
    parser.add_argument('--hf-derived', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(analyze(args.paired_result, args.hf_derived), indent=2) + '\n')


if __name__ == '__main__':
    main()
