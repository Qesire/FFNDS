"""Read-only verified binding of V2/new full-U inputs for local Stage 3.

Writes only a new explicit manifest at --output; does not execute torch or model.
Requires actual on-disk original receipts and files, never synthesizes missing ones.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

DEV=('vbi_1013','vbi_1040','vbi_0886','vbi_0947','vbi_1109','vbi_1084','vbi_0950','vbi_0176','vbi_0800','vbi_0392','vbi_0056')
CAL=tuple(sorted(('vbi_1005','vbi_1039','vbi_0895','vbi_0937','vbi_1094','vbi_1082','vbi_0966','vbi_0304','vbi_0648','vbi_0456','vbi_0032')))
SITES=(('high',4),('high',19),('high',34),('low',4),('low',19),('low',34))
NEUTRALS=('vbi_1013','vbi_0800')

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def bound_json(path,expected):
    path=Path(path).resolve()
    if sha(path)!=expected: raise ValueError('manifest identity mismatch: '+str(path))
    return json.loads(path.read_text(encoding='utf8'))

def build(args):
    bindings=bound_json(args.bindings,args.bindings_sha256)
    bank_manifest=Path(bindings['bank']['manifest_path'])
    if sha(bank_manifest)!=bindings['bank']['manifest_sha256']: raise ValueError('V2 bank manifest changed')
    root=Path(bindings['bank']['bank_root']).resolve()
    capture_root=None; capture={}
    if args.mode=='dev':
        manifest_path=Path(args.capture_manifest).resolve()
        manifest=bound_json(manifest_path,args.capture_manifest_sha256)
        summary=bound_json(args.capture_summary,args.capture_summary_sha256)
        if manifest['schema']!='FFNDS_FULL_U_STAGE1_V1' or summary['status']!='COMPLETE' or summary['d4_completed']!=4:
            raise ValueError('new two-source capture is not complete')
        for source in NEUTRALS:
            row=summary['per_source'][source]
            if any(len(row[k]) != 4 for k in ('flows_equal','latents_equal','time_sigma_equal')):
                raise ValueError('incomplete neutral trajectory')
            if not (row['initial_noise_equal'] and row['decoded_equal'] and all(row['flows_equal']) and all(row['latents_equal']) and all(row['time_sigma_equal'])):
                raise ValueError('new capture failed neutral comparison')
        capture_root=manifest_path.parent
        for row in manifest['files']:
            if row['kind']=='full_u':
                key=(row['source'],row['step'],row['layer'],'full_u')
            elif row['kind']=='panel':
                e=row['event']; key=(e['source'],e['step'],e['layer'],'panel')
            else: continue
            if key in capture: raise ValueError('duplicate capture site')
            path=(capture_root/row['path']).resolve()
            if not path.is_relative_to(capture_root) or sha(path)!=row['sha256']:
                raise ValueError('new capture file identity mismatch')
            capture[key]=row
        if len(capture)!=48: raise ValueError('24 U + 24 panel files required')
    def endpoint(source,expert,step,layer):
        base={'source':source,'seed':42,'step':step,'layer':layer}
        expected_panel=bindings['panel']['per_source'][source]['token_id_sha256_le_int64']
        if source in NEUTRALS and args.mode=='dev':
            ur=capture[(source,step,layer,'full_u')]; pr=capture[(source,step,layer,'panel')]
            event=pr['event']; token=event['token_id_sha256_le_int64']
            if event['expert']!=expert or pr['u_panel_readback_equal'] is not True:
                raise ValueError('new panel expert/readback mismatch')
            # Keep actual new panel coordinates; never substitute old rows.
            identity={k:event[k] for k in ('image_sha256','prompt_sha256','source_cluster_id','checkpoint_sha256')}
            return {**base,'kind':'stage1_full_u','token_id_sha256_le_int64':token,'input_identity':identity,
                    'full_u':{'path':str(capture_root/ur['path']),'sha256':ur['sha256']},
                    'panel':{'path':str(capture_root/pr['path']),'sha256':pr['sha256']}}
        name=f'{source}__seed42__s{step:02d}__b{layer:02d}.pt'
        path=root/source/name; side=json.loads(Path(str(path)+'.json').read_text(encoding='utf8'))
        if (side['source'],side['seed'],side['step'],side['layer'],side['expert'])!=(source,42,step,layer,expert):
            raise ValueError('V2 sidecar site mismatch')
        if side['token_id_sha256_le_int64'] != expected_panel or side['file']!=name or sha(path)!=side['file_sha256']:
            raise ValueError('V2 panel identity mismatch')
        identity={k:side[k] for k in ('image_sha256','prompt_sha256','source_cluster_id','checkpoint_sha256')}
        return {**base,'kind':'v2_panel','token_id_sha256_le_int64':expected_panel,'input_identity':identity,
                'panel':{'path':str(path),'sha256':side['file_sha256']}}
    sources=DEV if args.mode=='dev' else CAL
    sites=SITES if args.mode=='dev' else tuple((e,l) for e in ('high','low') for l in range(40))
    pairs=[]
    for source in sources:
        for expert,layer in sites:
            a,h=(0,1) if expert=='high' else (2,3)
            first=endpoint(source,expert,a,layer); second=endpoint(source,expert,h,layer)
            if first['input_identity'] != second['input_identity']: raise ValueError('anchor/hit condition or checkpoint differs')
            if first['token_id_sha256_le_int64']!=second['token_id_sha256_le_int64']:
                raise ValueError('anchor/hit panel differs')
            pairs.append({'source':source,'expert':expert,'layer':layer,'anchor':first,'hit':second})
    return {'schema':'FFNDS_STAGE3_INPUT_PAIRS_V1','mode':args.mode,'sources':sources,'sites':sites,'input_pairs':pairs,
            'bindings_sha256':args.bindings_sha256,'v2_bank_manifest_sha256':bindings['bank']['manifest_sha256'],
            'capture_manifest_sha256':args.capture_manifest_sha256 if args.mode=='dev' else None,
            'capture_summary_sha256':args.capture_summary_sha256 if args.mode=='dev' else None}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=('dev','cal'),required=True)
    p.add_argument('--bindings',required=True); p.add_argument('--bindings-sha256',required=True)
    p.add_argument('--capture-manifest'); p.add_argument('--capture-manifest-sha256')
    p.add_argument('--capture-summary'); p.add_argument('--capture-summary-sha256'); p.add_argument('--output',required=True)
    args=p.parse_args()
    if args.mode=='dev' and not all((args.capture_manifest,args.capture_manifest_sha256,args.capture_summary,args.capture_summary_sha256)):
        p.error('dev requires capture manifest and summary identities')
    result=build(args)
    with Path(args.output).open('x',encoding='utf8') as f:
        json.dump(result,f,indent=2,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    print(json.dumps({'output':str(Path(args.output).resolve()),'sha256':sha(args.output),'pairs':len(result['input_pairs'])}))



