"""Versioned UB2 quality comparison with explicit reuse of frozen D-Q evidence."""
import argparse
import csv
import json
from pathlib import Path
import math

from ar2_checkpoint import atomic_json,digest
from compare_vbench3 import metric_gate
from terminal_dense_validation import validate_runtime
from quality_evidence import load_evaluation as load_validated_evaluation
from compare_late_block_quality import validate_generation_pair

DIMS=['aesthetic_quality','imaging_quality','motion_smoothness']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dense',type=Path,required=True)
    ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--policy',type=Path,required=True)
    ap.add_argument('--identity-proof',type=Path,required=True)
    ap.add_argument('--terminal-flow-proof',type=Path,required=True)
    ap.add_argument('--cohort',choices=['formal20','disjoint20'],required=True)
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    dense,dp,ds = load_validated_evaluation(args.dense,validate_runtime)
    candidate,cp,cs = load_validated_evaluation(args.candidate,validate_runtime)
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
    flow = json.loads(args.terminal_flow_proof.read_text())
    if (ci['terminal_flow_proof_sha256'] != digest(args.terminal_flow_proof)
            or flow['status'] != 'PASS_UB2_DENSE_TERMINAL_AND_UNCHANGED_STEPS_NATIVE_FLOW'
            or flow['identity']['policy_sha256'] != digest(args.policy)):
        raise RuntimeError('Exact same-policy terminal-flow controls required')
    proof = json.loads(args.identity_proof.read_text())
    if proof['status'] != 'PASS_UB2_FOUR_CASE_IDENTITY_LATENTS_AND_VIDEO_BYTES' or proof['identity']['assets'] != ci['assets']:
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
        'terminal_flow_proof_sha256':digest(args.terminal_flow_proof),
        'candidate_runtime_sources':{**policy['runtime_sources'],**ci['sources']},
        'baseline_reuse':'original D-Q videos and official scores reused with their original generation identities; native pipeline/assets/helper hashes checked, no rewritten or fabricated D-Q generation',
        'generation_wrapper_difference':'UB2-specific wrapper/rank validator; same generate_one and native pipeline, four-case UB1 identity proof and explicit candidate numerical source bindings',
        'per_source_units':'official units; raw imaging MUSIQ divided100 per source',
        'source_exposure':'both original20 and supplement20 informed development; no untouched-test claim',
        'full_FFN_speed_acceptance':False,'joint_project_acceptance':False}
    atomic_json(args.output,result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
