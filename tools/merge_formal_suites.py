"""Join disjoint completed formal slices with identical frozen execution identities."""
import argparse
import json
from pathlib import Path

from ar2_checkpoint import atomic_json,digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--suites',type=Path,nargs='+',required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    samples=json.loads((ROOT/'.runtime/inputs/formal20/canonical.json').read_text())
    common=None;completed={};bindings=[]
    for path in args.suites:
        source=json.loads(path.read_text())
        if source['status']!='PASS_FORMAL_GENERATION_SLICE_ONLY':raise RuntimeError('Every slice must complete')
        identity={k:v for k,v in source['identity'].items() if k!='indices'}
        if common is not None and common!=identity:raise RuntimeError('Slices use different methods/models/inputs/versions')
        common=identity
        if {r['index'] for r in source['completed']}!=set(source['identity']['indices']):raise RuntimeError('Slice index coverage differs')
        bindings.append({'path':str(path.resolve()),'sha256':digest(path)})
        for row in source['completed']:
            index=row['index']
            if index in completed or digest(row['case_report'])!=row['case_report_sha256']:
                raise RuntimeError('Duplicate index or changed case report')
            case=json.loads(Path(row['case_report']).read_text())
            if (case['status']!='PASS_FORMAL_INPUT_GENERATION_ONLY' or case['input']!=samples[index]
                    or case['model_assets']!=identity['assets'] or case['arm']!=identity['arm']
                    or digest(case['video']['path'])!=case['video']['sha256']
                    or case['hif4_counter_delta']['linear_forward_calls']!=1600
                    or case['hif4_counter_delta']['activation_qdq_calls']!=1600
                    or case['hif4_counter_delta']['cache_misses']):
                raise RuntimeError('Invalid formal case or incomplete native coverage')
            if identity['packed_policy_sha256'] and case['hif4_runtime']['packed_reference']['policy_sha256']!=identity['packed_policy_sha256']:
                raise RuntimeError('Candidate policy differs across cases')
            completed[index]=row
    if set(completed)!=set(range(20)):raise RuntimeError('Exactly all20 distinct formal indices required')
    result={'status':'PASS_FORMAL_GENERATION_SLICE_ONLY','identity':{**common,'indices':list(range(20))},
        'completed':[completed[i] for i in range(20)],'merged_suites':bindings,
        'merger_source_sha256':digest(Path(__file__)),'quality_or_speed_acceptance':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists() and json.loads(args.output.read_text())!=result:
        raise RuntimeError('Refusing to overwrite a different merged formal identity')
    atomic_json(args.output,result);print(json.dumps({'output':str(args.output),'source_count':20}))


if __name__=='__main__':main()
