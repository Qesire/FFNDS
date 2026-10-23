"""Read exact installed evaluator source without importing any evaluator."""
import argparse
import ast
import os
from pathlib import Path

from rollout_contract import ROOT, atomic, read, sha


def audit(source_root):
    preflight = read(ROOT / 'research/20261023/R17_metric_import_preflight.json')
    evidence = []
    for record in preflight['functions']:
        module = source_root / (record['module'].replace('.', '/') + '.py')
        text = module.read_text()
        if sha(module) != record['module_sha256']:
            raise RuntimeError('Source differs from the exact successful CPU import preflight')
        functions = {node.name: node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}
        dimension = record['dimension']
        if dimension == 'aesthetic_quality':
            name = 'laion_aesthetic'
        elif dimension == 'imaging_quality':
            name = 'technical_quality'
        else:
            name = dimension
        node = functions[name]
        body = ast.get_source_segment(text, node)
        # These assertions bind this audit to the actually inspected original
        # version, rather than guessing how an arbitrary future scorer works.
        if dimension in ('subject_consistency', 'background_consistency'):
            required = ['cnt += 1', 'sim += video_sim', 'sim / cnt']
            aggregate = 'sum_video_frame_pair_similarity / sum_video_frame_pairs'
        elif dimension == 'imaging_quality':
            required = ['acc_score_video/len(images)', 'average_score = average_score / 100.']
            aggregate = 'mean(raw_per_video_MUSIQ) / 100'
        elif dimension == 'aesthetic_quality':
            required = ['aesthetic_scores / 10', 'aesthetic_avg /= num']
            aggregate = 'mean(per_video_mean_frame_predictor_score / 10)'
        else:
            required = ['np.mean(sim_list)'] if dimension.startswith('i2v_') else ['np.mean(sim)']
            aggregate = 'equal_video_mean'
        if dimension in ('i2v_subject', 'i2v_background'):
            required += ['max_weight = 0.4', 'mean_weight = 0.3', 'min_weight = 0.3',
                         'np.max(conformity_scores)', 'np.mean(consec_scores)', 'np.min(consec_scores)']
        if any(token not in body for token in required):
            raise RuntimeError('Inspected original aggregation/formula changed: ' + dimension)
        evidence.append({'dimension': dimension, 'module': record['module'],
                         'module_sha256': sha(module), 'function': name,
                         'source_first_line': node.lineno, 'source_last_line': node.end_lineno,
                         'original_aggregate': aggregate,
                         'downstream_detail_scale': .01 if dimension == 'imaging_quality' else 1,
                         'official_scoring_modified': False})
    return {'id': 'R17_OFFICIAL_NINE_AGGREGATION_SOURCE_AUDIT',
            'status': 'EXACT_ORIGINAL_SOURCE_RECONSTRUCTION_NO_EVALUATOR_EXECUTION',
            'audit_source_sha256': sha(__file__),
            'CPU_import_receipt_sha256': sha(ROOT / 'research/20261023/R17_metric_import_preflight.json'),
            'dimensions': evidence,
            'i2v_similarity_formula': '.4*max(input_frame_similarity) + .3*mean(consecutive_similarity) + .3*min(consecutive_similarity)',
            'consistency_weights': 'Actual decoded frame count minus one; equal-source statistical summaries reported separately from raw official frame-pair aggregation.',
            'MUSIQ_warning': 'Original per-video details are in raw score units; original aggregate divides their mean by100. Never compare these units directly.',
            'statistical_unit': 'Underlying conditioning image, with the original dimension-specific source subset.',
            'new_model_generations': 0, 'new_metric_model_forwards': 0,
            'actual_video_quality_scores': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    hf = Path(os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface')))
    parser.add_argument('--source-root', type=Path,
                        default=hf / 'derived/ffnds/R17_metric_assets/installed_source_view')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    report = audit(args.source_root)
    target = ROOT / 'research/20261023/R17_metric_aggregation_source_audit.json'
    if args.check:
        if read(target) != report:
            raise RuntimeError('Saved source audit differs from exact original scorer')
    else:
        if target.exists():
            raise RuntimeError('Preserve the first complete aggregation source audit')
        atomic(target, report)
    print(report['status'], 'dimensions=' + str(len(report['dimensions'])))


if __name__ == '__main__':
    main()
