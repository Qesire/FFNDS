"""Public offline checks for the actually completed finite R17 result packet."""
import math
from pathlib import Path
import re
import statistics

from fresh_quality_contract import ARMS, CAPS, CORE, METHOD, ROOT, generation_jobs, read, sha
from fresh_quality_statistics import normalized_video_score, verify_official_aggregate


def main():
    base = ROOT / 'research/20261023'
    result = read(base / 'R17E0_analysis.json')
    budget = read(base / 'R17E0_budget_debit.json')
    records = read(base / 'R17E0_generation_receipts.json')['records']
    official = read(base / 'R17E0_official_metric_records.json')['records']
    decode = read(base / 'R17E0_video_decode.json')
    resources = read(base / 'R17E0_completed_resources.json')
    need = lambda ok, message: None if ok else (_ for _ in ()).throw(AssertionError(message))
    need(result['id'] == METHOD and budget['method_id'] == METHOD, 'Actual method identity differs')
    need(result['actual_budget'] == budget['used'] == budget['caps'] == CAPS, 'Finite actual debit differs')
    need(len(budget['debits']) == len({r['name'] for r in budget['debits']}) == 190 and
         all(r['status'] == 'COMPLETED' for r in budget['debits']), 'Missing/failed/duplicate model work')
    need([r['job'] for r in records] == generation_jobs(), 'Actual145source/arm/order changed')
    need(not result['algorithm_changed_or_refit'] and not result['automatic_quality_or_Pareto_or_paper_acceptance'],
         'Unsupported fitted/quality/Pareto promotion')
    need(resources['max_concurrent_GPUs'] == 1 and resources['logical_experiment_output_bytes'] <= 4 << 30,
         'Fixed actual output/GPU scope exceeded')
    need(resources['peak_allocated_bytes'] <= 79 << 30, 'Actual allocated memory cap exceeded')
    decoded = {(s['source'], arm): v for s in decode['sources'] for arm, v in s['videos'].items()}
    need(len(decoded) == 115 and all(r['decoded_frames'] == 81 for r in decoded.values()), 'Incomplete full decode')
    sources = read(base / 'R17_fresh23_input_audit.json')['inputs']
    noise = {}
    for record in records:
        job = record['job']
        need(record['actual_native_counter_delta'] == {'linear_forward_calls': 1600, 'activation_qdq_calls': 1600,
             'cache_misses': 0, 'weight_transfers': 0}, 'Native logical runtime coverage differs')
        need(record['actual_FFN_dispatch_sites'] == (0 if job['arm'] == 'dense_DQ' else 160), 'Not actual full160')
        need([x['step'] for x in record['trajectory']] == list(range(4)), 'Incomplete free fourstep trajectory')
        source = job['source']
        if source in noise:
            need(noise[source] == record['initial_noise_sha256'], 'Paired initial noise differs')
        noise[source] = record['initial_noise_sha256']
        if job['phase'] == 'quality':
            need(record['source_image_sha256'] == sources[source]['input_provenance']['image_sha256'],
                 'Not the frozen genuinely new source image')
            need(record['video_sha256'] == decoded[source, job['arm']]['video_sha256'], 'Decode/media mismatch')
    need(len(official) == 45, 'Not all nine official dimensions/fivearms')
    seen = set()
    for record in official:
        dimension, arm = record['dimension'], record['arm']
        need((dimension, arm) not in seen, 'Duplicate official call')
        seen.add((dimension, arm))
        expected = {s['index'] for s in sources if dimension in s['core_dimensions']}
        values, counts = {}, {}
        for row in record['official_return'][-1]:
            match = re.match(r'^source(\d+)(?: |\.mp4)', Path(row['video_path']).name)
            need(match is not None, 'Invalid video/source identity')
            index = int(match[1]); need(index not in values, 'Pseudo-replicated source')
            values[index] = normalized_video_score(dimension, row)
            counts[row['video_path']] = decoded[index, arm]['decoded_frames']
        need(set(values) == expected, 'Incorrect original dimension population')
        verify_official_aggregate(dimension, record['official_return'], counts)
        reported = result['official_nine_quality'][dimension]
        need({str(k): v for k, v in values.items()} == reported['per_source_scores_in_official_aggregate_units'][arm],
             'Public normalized source score changed')
        need(math.isclose(statistics.fmean(values.values()),
             reported['equal_source_means_in_official_aggregate_units'][arm], rel_tol=1e-12, abs_tol=1e-12),
             'Equal-source mean changed')
    need(seen == {(d, a) for d in CORE for a in ARMS}, 'Missing required official cell')
    for receipt in resources['path_transform_receipts']:
        need(sha(ROOT / receipt['public_repository_path']) == receipt['public_sha256'] and
             receipt['numeric_values_unchanged'], 'Public raw-to-numeric export changed')
    need(METHOD not in read(ROOT / 'research/CURRENT_DECISION.json')['eligible_model_method_ids'],
         'Completed fixed budget cannot authorize a rerun')
    print('R17E0 actual145generations/45officialcalls,115decoded freshvideos,source units and public numeric identity CONSISTENT.')
    print('Small frozen source subsets and one-source technical timing; no automatic paper/Pareto acceptance.')


if __name__ == '__main__':
    main()
