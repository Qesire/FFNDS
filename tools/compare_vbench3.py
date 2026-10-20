"""Compare all three frozen formal20 metrics, preserving20 independent sources."""
import argparse
import csv
import json
import math
from pathlib import Path

from ar2_checkpoint import atomic_json,digest
from compare_ffn_benchmark import validate_candidate_runtime

DIMS=['aesthetic_quality','imaging_quality','motion_smoothness']


def metric_gate(dense,candidate):
    if not math.isfinite(dense) or not math.isfinite(candidate) or dense<=0:
        raise ValueError('Finite scores and a positive paired baseline are required')
    return {'D_Q':dense,'candidate':candidate,'relative_drop':(dense-candidate)/dense,
        'max_relative_drop':.02,'pass':candidate>=.98*dense}


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
                    or case['video']['fps']!=16 or case['hif4_counter_delta']['cache_misses']):
                raise RuntimeError('Incomplete or mismatched native formal video')
            if protocol['generation_identity']['packed_policy_sha256']:
                packed=runtime['packed_reference']
                if packed['actual_FFN_linear_calls']!=320:
                    raise RuntimeError('Candidate lacks complete same-policy main/auxiliary coverage')
                validate_candidate_runtime(packed,protocol['generation_identity']['packed_policy_sha256'],require_timing=False)
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
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dense',type=Path,required=True)
    ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    dense,dp,ds=load_evaluation(args.dense);candidate,cp,cs=load_evaluation(args.candidate)
    for key in ['metrics','unique_generation_count','materialized_names','seed','camera_excluded','historical_worker_sha256',
            'full_info_sha256','code','models_manifest_sha256','models','versions','python','device','local','read_frame','resolution','source_sha256']:
        if dp[key]!=cp[key]:raise RuntimeError(f'Paired evaluation identity differs: {key}')
    di={k:v for k,v in dp['generation_identity'].items() if k not in ['arm','packed_policy_sha256']}
    ci={k:v for k,v in cp['generation_identity'].items() if k not in ['arm','packed_policy_sha256']}
    if (di!=ci or dp['generation_identity']['arm']!='D-Q' or dp['generation_identity']['packed_policy_sha256'] is not None
            or not cp['generation_identity']['packed_policy_sha256']):
        raise RuntimeError('Dense/candidate generation model, inputs or sources differ')
    source_map={r['source_group']:r['prompt'] for r in dp['generation_cases']}
    if source_map!={r['source_group']:r['prompt'] for r in cp['generation_cases']}:
        raise RuntimeError('Dense/candidate source identities differ')
    if sum(s.startswith('i2v_subject:') for s in source_map)!=10 or sum(s.startswith('i2v_background:') for s in source_map)!=10:
        raise RuntimeError('Expected subject10+background10, camera excluded')
    gates={dim:metric_gate(dense['scores'][dim],candidate['scores'][dim]) for dim in DIMS}
    paired=[]
    for dim in DIMS:
        for source in sorted(source_map):
            if source not in ds[dim] or source not in cs[dim]:raise RuntimeError('Missing paired source score')
            paired.append({'dimension':dim,'source_group':source,'prompt':source_map[source],
                'D_Q':ds[dim][source],'candidate':cs[dim][source],
                'relative_drop':(ds[dim][source]-cs[dim][source])/ds[dim][source]})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    csv_path=args.output.with_suffix('.csv')
    with csv_path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    result={'status':'COMPLETED_PAIRED_FORMAL20_THREE_METRIC_COMPARISON','metrics':gates,
        'formal20_quality_pass':all(r['pass'] for r in gates.values()),'unique_source_count':20,'filename_copies_per_source':5,
        'candidate_policy_sha256':cp['generation_identity']['packed_policy_sha256'],
        'dense_protocol_sha256':digest(args.dense/'PROTOCOL.json'),'candidate_protocol_sha256':digest(args.candidate/'PROTOCOL.json'),
        'paired_csv_sha256':digest(csv_path),'source_sha256':digest(Path(__file__)),
        'candidate_runtime_sources':{str(p.resolve()):digest(p) for p in [Path(__file__).with_name(n) for n in
            ['step_budget_runtime.py','trained_packed_runtime.py','packed_reference_runtime.py','reduced_rank_residual.py']]},
        'per_source_units':'official aggregate units; imaging raw per-video MUSIQ divided by100, other metrics unchanged',
        'full_FFN_speed_acceptance':False,'joint_project_acceptance':False}
    atomic_json(args.output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
