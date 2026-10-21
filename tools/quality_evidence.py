"""Read original20-source official metric evidence with an explicit trace validator."""
import csv
import json
import math
from pathlib import Path

from ar2_checkpoint import digest

DIMS = ['aesthetic_quality','imaging_quality','motion_smoothness']


def load_evaluation(root,validate_runtime):
    meta = json.loads((root/'RUN.json').read_text())
    protocol = json.loads((root/'PROTOCOL.json').read_text())
    if (meta['status'] != 'PASS_THREE_METRIC_EXECUTION_ONLY' or meta['protocol_sha256'] != digest(root/'PROTOCOL.json')
            or set(meta['scores']) != set(DIMS) or protocol['unique_generation_count'] != 20
            or protocol['materialized_names'] != 100 or [r['dimension'] for r in meta['workers']] != DIMS):
        raise RuntimeError('Complete frozen20-source three-metric official evaluation required')
    cases = protocol['generation_cases']
    if len(cases) != 20 or len({v['source_group'] for v in cases}) != 20:
        raise RuntimeError('Exactly20 independent sources required')
    for binding in protocol['generation_suites']:
        if digest(binding['path']) != binding['sha256']:
            raise RuntimeError('Frozen generation suite changed')
        suite = json.loads(Path(binding['path']).read_text())
        for row in suite['completed']:
            if digest(row['case_report']) != row['case_report_sha256']:
                raise RuntimeError('Frozen native generation case changed')
            case = json.loads(Path(row['case_report']).read_text())
            if (case['status'] != 'PASS_FORMAL_INPUT_GENERATION_ONLY' or case['video']['frames'] != 81
                    or case['video']['fps'] != 16 or case['hif4_counter_delta']['cache_misses']
                    or digest(case['video']['path']) != case['video']['sha256']):
                raise RuntimeError('Complete unchanged native video/precision required')
            if protocol['generation_identity']['packed_policy_sha256']:
                validate_runtime(case['hif4_runtime']['packed_reference'],protocol['generation_identity']['packed_policy_sha256'],require_timing=False)
    per_metric = {}
    for worker in meta['workers']:
        if digest(worker['report']) != worker['report_sha256']:
            raise RuntimeError('Official worker report changed')
        report = json.loads(Path(worker['report']).read_text())
        dim = worker['dimension']
        if digest(report['result_path']) != worker['result_sha256'] or report['score'] != meta['scores'][dim]:
            raise RuntimeError('Official score identity changed')
        raw = json.loads(Path(report['result_path']).read_text())[dim]
        if len(raw) != 2 or raw[0] != meta['scores'][dim] or len(raw[1]) != 100:
            raise RuntimeError('All100 materialized filenames required')
        values = {Path(v['video_path']).name:v['video_results'] for v in raw[1]}
        expected = {f'{v["prompt"]}-{copy}.mp4' for v in cases for copy in range(5)}
        if len(values) != 100 or set(values) != expected:
            raise RuntimeError('Worker filenames differ from the fixed20-source protocol')
        scores = {}
        for case in cases:
            rows = [float(values[f'{case["prompt"]}-{copy}.mp4']) for copy in range(5)]
            if any(not math.isfinite(v) for v in rows):
                raise RuntimeError('Finite official per-video scores required')
            scores[case['source_group']] = sum(rows)/5/(100 if dim == 'imaging_quality' else 1)
        if not math.isclose(sum(scores.values())/20,meta['scores'][dim],rel_tol=1e-6,abs_tol=1e-8):
            raise RuntimeError('Paired source units do not reproduce the official aggregate')
        per_metric[dim] = scores
    return meta,protocol,per_metric
