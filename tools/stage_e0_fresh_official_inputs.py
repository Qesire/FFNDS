"""Stage four byte-bound official inputs in HF; no model or tensor work."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time

from ffnds_observation_common import sha256,write_json

ROOT=Path(__file__).resolve().parents[1]
ID='E0_FRESH_OFFICIAL_FOUR_INPUT_STAGING_20261026'


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path,required=True)
    ap.add_argument('--execute-staging',action='store_true');args=ap.parse_args()
    started=time.monotonic();p=json.loads(args.protocol.read_text())
    if p['id']!=ID or p['official_rows']!=[908,376,425,437] or p['automatic_retry'] is not False:
        raise ValueError('Exact four-source staging scope required')
    for name,digest in p['bound_sources'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Staging source changed: '+name)
    audit=json.loads((ROOT/p['audit']).read_text())
    if [s['source_index'] for s in audit['selected_sources']]!=p['official_rows']:
        raise ValueError('Independent source selection changed')
    archive=Path(p['archive']);metadata=Path(p['metadata'])
    if sha256(archive)!=audit['archive_sha256'] or sha256(metadata)!=audit['official_metadata_sha256']:
        raise ValueError('Original archive/metadata bytes changed')
    if not args.execute_staging:
        print('PASS_STAGING_IDENTITIES_ONLY_NO_EXTRACT_OR_MODEL');return
    a=json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text())
    if not any(s['id']==ID and s['protocol_sha256']==sha256(args.protocol)
        and s['execution_status']=='AUTHORIZED_NOT_STARTED' for s in a['experiment_authority']['current_user_authorized_scopes']):
        raise RuntimeError('One registered staging attempt required')
    out=Path(p['canonical_output']);out.mkdir(parents=True,exist_ok=False)
    run={'id':ID,'status':'STARTED','protocol_sha256':sha256(args.protocol),
         'model_calls':0,'GPU_calls':0,'tensor_analysis_calls':0,'new_asset_downloads':0,'completed':[]}
    try:
        originals=json.loads(metadata.read_text());wanted={s['archive_member']:s for s in audit['selected_sources']}
        canonical={}
        with tarfile.open(archive,mode='r|gz') as tar:
            for member in tar:
                if time.monotonic()-started>p['max_wall_seconds']:raise TimeoutError('Fixed staging wall budget')
                if member.name not in wanted:continue
                s=wanted.pop(member.name)
                if not member.isfile() or member.size!=s['bytes']:raise ValueError('Bound regular member size required')
                content=tar.extractfile(member).read(s['bytes']+1)
                if len(content)!=s['bytes'] or hashlib.sha256(content).hexdigest()!=s['image_sha256']:
                    raise ValueError('Selected original image bytes differ')
                original=originals[s['source_index']]
                if original['prompt_en']!=s['prompt'] or original['image_name']!=s['image_name'] or original['dimension']!=s['official_dimensions']:
                    raise ValueError('Actual official source row differs')
                destination=out/f"official{s['source_index']}"/'input.jpg';destination.parent.mkdir()
                destination.write_bytes(content)
                canonical[s['source_index']]={**s,'image_path':str(destination),'source_id':s['image_sha256'],
                    'cohort_id':'E0_FRESH_OFFICIAL4_20261026','official_metadata_row':s['source_index'],
                    'original_metadata_sha256':audit['official_metadata_sha256']}
                run['completed'].append({'official_metadata_row':s['source_index'],'image_sha256':sha256(destination),'bytes':len(content)})
                write_json(out/'RUN.json',run)
                if not wanted:break
        if wanted:raise ValueError('Missing bound original image member')
        cohort=[dict(canonical[row],local_index=i) for i,row in enumerate(p['official_rows'])]
        write_json(out/'canonical.json',cohort)
        run.update(status='PASS_FOUR_ORIGINAL_INPUTS_STAGED_NO_MODEL',canonical_sha256=sha256(out/'canonical.json'))
    except BaseException as error:
        run.update(status='FAILED_STAGING_PRESERVED_NO_RETRY',error_type=type(error).__name__,error=str(error));raise
    finally:
        run['elapsed_seconds']=time.monotonic()-started
        run['new_file_bytes']=sum(f.stat().st_size for f in out.rglob('*') if f.is_file())
        if run['new_file_bytes']>p['max_output_bytes']:run['status']='FAILED_OUTPUT_CAP_PRESERVED_NO_RETRY'
        write_json(out/'RUN.json',run)
    if not run['status'].startswith('PASS_'):raise RuntimeError(run['status'])
    print(json.dumps({k:run[k] for k in ['status','elapsed_seconds','new_file_bytes','canonical_sha256']}))


if __name__=='__main__':main()
