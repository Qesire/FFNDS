"""Versioned LR2 quality comparison with explicit reuse of frozen D-Q evidence."""
import argparse
import csv
import json
from pathlib import Path
import math

from ar2_checkpoint import atomic_json,digest
from compare_vbench3 import metric_gate
from late_block_validation import validate_runtime

DIMS=['aesthetic_quality','imaging_quality','motion_smoothness']


def validate_generation_pair(di,ci,trusted,policy_sha,proof_sha,cohort,runtime_sources,common_paths):
    if (di['arm'] != 'D-Q' or di['packed_policy_sha256'] is not None
            or ci['packed_policy_sha256'] != policy_sha or ci['identity_control_proof_sha256'] != proof_sha
            or ci['cohort'] != cohort or ci['runtime_sources'] != runtime_sources):
        raise RuntimeError('Explicit D-Q reference and frozen active same-policy LR2 cohort required')
    for key in ['assets','inputs_sha256','input_recovery_sha256','versions','seed_semantics']:
        if di[key] != ci[key]:
            raise RuntimeError(f'Paired generation conditioning/assets/precision differ: {key}')
    if (ci['assets'] != trusted['assets'] or ci['native_sources'] != trusted['native_sources']
            or any(ci['native_sources'].get(p) != sha for p,sha in di['native_sources'].items())):
        raise RuntimeError('Native generation pipeline differs from the frozen D-Q reference')
    for path in common_paths:
        if (path not in trusted['sources'] or ci['sources'].get(path) != trusted['sources'][path]
                or di['sources'].get(path) != trusted['sources'][path]):
            raise RuntimeError('Common native generation/seed helper changed')


def load_evaluation(root):
    meta=json.loads((root/'RUN.json').read_text())
    protocol=json.loads((root/'PROTOCOL.json').read_text())
    if (meta['status']!='PASS_THREE_METRIC_EXECUTION_ONLY' or meta['protocol_sha256']!=digest(root/'PROTOCOL.json')
            or set(meta['scores'])!=set(DIMS) or protocol['unique_generation_count']!=20
            or protocol['materialized_names']!=100):
        raise RuntimeError('Complete frozen20-source three-metric evaluation required')
    if [r['dimension'] for r in meta['workers']]!=DIMS:
        raise RuntimeError('Expected all three official workers once')
    per_metric={}
    cases=protocol['generation_cases']
    if len(cases)!=20 or len({r['source_group'] for r in cases})!=20:
        raise RuntimeError('Independent source count differs')
    for binding in protocol['generation_suites']:
        if digest(binding['path'])!=binding['sha256']:
            raise RuntimeError('Frozen generation suite changed')
        suite=json.loads(Path(binding['path']).read_text())
        for row in suite['completed']:
            if digest(row['case_report'])!=row['case_report_sha256']:
                raise RuntimeError('Frozen formal generation case changed')
            case=json.loads(Path(row['case_report']).read_text())
            runtime=case['hif4_runtime']
            if (case['status']!='PASS_FORMAL_INPUT_GENERATION_ONLY' or case['video']['frames']!=81
                    or case['video']['fps']!=16 or case['hif4_counter_delta']['cache_misses']
                    or digest(case['video']['path'])!=case['video']['sha256']):
                raise RuntimeError('Incomplete or mismatched native formal video')
            if protocol['generation_identity']['packed_policy_sha256']:
                packed=runtime['packed_reference']
                if packed['actual_FFN_linear_calls']!=320:
                    raise RuntimeError('Candidate lacks complete same-policy main/auxiliary coverage')
                validate_runtime(packed,protocol['generation_identity']['packed_policy_sha256'],require_timing=False)
    for worker in meta['workers']:
        if digest(worker['report'])!=worker['report_sha256']:
            raise RuntimeError('Official worker report changed')
        report=json.loads(Path(worker['report']).read_text());dim=worker['dimension']
        if digest(report['result_path'])!=worker['result_sha256'] or report['score']!=meta['scores'][dim]:
            raise RuntimeError('Official score identity changed')
        raw=json.loads(Path(report['result_path']).read_text())[dim]
        if len(raw)!=2 or raw[0]!=meta['scores'][dim] or len(raw[1])!=100:
            raise RuntimeError('Official metric must contain all100 materialized names')
        rows={Path(row['video_path']).name:row['video_results'] for row in raw[1]}
        expected={f'{r["prompt"]}-{copy}.mp4' for r in cases for copy in range(5)}
        if len(rows)!=100 or set(rows)!=expected:
            raise RuntimeError('Metric coverage differs from the20-source five-copy protocol')
        source_scores={}
        for case in cases:
            values=[float(rows[f'{case["prompt"]}-{copy}.mp4']) for copy in range(5)]
            if any(not math.isfinite(v) for v in values):
                raise RuntimeError('Nonfinite per-video score')
            # Official imaging_quality.technical_quality retains raw MUSIQ
            # units per video, then divides only its aggregate by100. Match
            # aggregate units for paired source tables; official scores stay
            # exactly as returned by the original worker.
            divisor=100 if dim=='imaging_quality' else 1
            source_scores[case['source_group']]=sum(values)/5/divisor
        if not math.isclose(sum(source_scores.values())/20,meta['scores'][dim],rel_tol=1e-6,abs_tol=1e-8):
            raise RuntimeError('Official aggregate differs from20 equally weighted sources')
        per_metric[dim]=source_scores
    return meta,protocol,per_metric


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dense',type=Path,required=True)
    ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--policy',type=Path,required=True)
    ap.add_argument('--identity-proof',type=Path,required=True)
    ap.add_argument('--cohort',choices=['formal20','disjoint20'],required=True)
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    dense,dp,ds = load_evaluation(args.dense)
    candidate,cp,cs = load_evaluation(args.candidate)
    for key in ['metrics','unique_generation_count','materialized_names','seed','camera_excluded','historical_worker_sha256',
                'full_info_sha256','code','models_manifest_sha256','models','versions','python','device','local','read_frame','resolution','source_sha256']:
        if dp[key] != cp[key]:
            raise RuntimeError(f'Paired official metric mechanics differ: {key}')
    di,ci = dp['generation_identity'],cp['generation_identity']
    policy = json.loads(args.policy.read_text())
    if policy['identity_control']:
        raise RuntimeError('Identity controls cannot enter quality acceptance')
    root = Path(__file__).resolve().parents[1]
    reference = json.loads((root/'runs/worker_restart_20261020/formal20_dense/RUN.json').read_text())
    trusted = reference['identity']
    if reference['status'] != 'PASS_FORMAL_GENERATION_SLICE_ONLY':
        raise RuntimeError('Complete frozen original native D-Q generation provenance required')
    common_paths = [str(root/'tools'/name) for name in ['generate_formal_suite.py','distill_assets.py']]
    validate_generation_pair(di,ci,trusted,digest(args.policy),digest(args.identity_proof),args.cohort,policy['runtime_sources'],common_paths)
    for path in common_paths:
        if digest(path) != trusted['sources'][path]:
            raise RuntimeError('Common native generation/seed helper changed')
    proof = json.loads(args.identity_proof.read_text())
    if proof['status'] != 'PASS_LR2_FOUR_CASE_IDENTITY_LATENTS_AND_VIDEO_BYTES' or proof['identity']['assets'] != ci['assets']:
        raise RuntimeError('Frozen four-case wrapper identity proof required')
    for mapping in [proof['identity']['sources'],ci['sources'],policy['runtime_sources'],ci['native_sources']]:
        for path,sha in mapping.items():
            if digest(path) != sha:
                raise RuntimeError('Numerical source changed after generation/control')
    source_map = {r['source_group']:r['prompt'] for r in dp['generation_cases']}
    if (source_map != {r['source_group']:r['prompt'] for r in cp['generation_cases']}
            or len(source_map) != 20 or sum(s.startswith('i2v_subject:') for s in source_map) != 10
            or sum(s.startswith('i2v_background:') for s in source_map) != 10):
        raise RuntimeError('Same subject10/background10 paired sources required')
    gates = {dim:metric_gate(dense['scores'][dim],candidate['scores'][dim]) for dim in DIMS}
    paired = []
    for dim in DIMS:
        for source in sorted(source_map):
            paired.append({'dimension':dim,'source_group':source,'prompt':source_map[source],
                'D_Q':ds[dim][source],'candidate':cs[dim][source],
                'relative_drop':(ds[dim][source]-cs[dim][source])/ds[dim][source]})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    csv_path = args.output.with_suffix('.csv')
    with csv_path.open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    result = {'status':'COMPLETED_PAIRED_FORMAL20_THREE_METRIC_COMPARISON','cohort':args.cohort,'metrics':gates,
        'formal20_quality_pass':all(r['pass'] for r in gates.values()),'unique_source_count':20,'filename_copies_per_source':5,
        'candidate_policy_sha256':digest(args.policy),'dense_protocol_sha256':digest(args.dense/'PROTOCOL.json'),
        'candidate_protocol_sha256':digest(args.candidate/'PROTOCOL.json'),'paired_csv_sha256':digest(csv_path),
        'source_sha256':digest(Path(__file__)),'identity_control_proof_sha256':digest(args.identity_proof),
        'candidate_runtime_sources':{**policy['runtime_sources'],**ci['sources']},
        'baseline_reuse':'original D-Q videos and official scores reused with their original generation identities; native pipeline/assets/helper hashes checked, no rewritten or fabricated D-Q generation',
        'generation_wrapper_difference':'LR2-specific wrapper/rank validator; same generate_one and native pipeline, four-case UB1 identity proof and explicit candidate numerical source bindings',
        'per_source_units':'official units; raw imaging MUSIQ divided100 per source',
        'source_exposure':'both original20 and supplement20 informed development; no untouched-test claim',
        'full_FFN_speed_acceptance':False,'joint_project_acceptance':False}
    atomic_json(args.output,result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
