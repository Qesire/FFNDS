"""Selected ICC r1408/4S explanation from saved candidate values.

Requires completed reviewed 792-cell grid and its exact source/factor identities.
Two F64 B2 actions per site/source (132 total); no F64 dense W2 or new NFE.
"""
import math
from pathlib import Path
import json
import torch
from icc_local_replay import DEV11,SITES,FactorPair,delta_alignment,error_decomposition,gated_risk
from icc_stage3_assets import checked_path,load_pair,sha
from icc_stage3_grid import exclusive_json,validate_scope


def run(scope,grid_dir,*,device,authorize,check_resources):
    """No mutation of cached grid artifacts; write a separate output directory."""
    pairs,factors = validate_scope(scope); authorize(scope)
    grid=Path(grid_dir); selection=json.loads((grid/'selection.json').read_text(encoding='utf8'))
    if sha(grid/'selection.json') != scope['selection_sha256']: raise ValueError('selection identity mismatch')
    family=selection['family']
    if family not in ('SVD','CA','CD'): raise ValueError('unknown selection')
    cache_manifest=json.loads((grid/'cache_manifest.json').read_text(encoding='utf8'))
    if sha(grid/'cache_manifest.json') != scope['cache_manifest_sha256']: raise ValueError('cache manifest identity mismatch')
    receipts={r['file']:r for r in cache_manifest['files']}
    if len(receipts)!=len(cache_manifest['files']): raise ValueError('duplicate cache identity')
    floor=scope['decomposition_norm_floor']
    if not isinstance(floor,(int,float)) or floor<0 or not math.isfinite(floor): raise ValueError('need frozen norm floor')
    output=Path(scope['decomposition_output']); output.mkdir(parents=True,exist_ok=False)
    rows=[]; actions=0
    with torch.no_grad():
        for expert,layer in SITES:
            factor_spec=factors[(family,expert,layer)]
            artifact=torch.load(checked_path(factor_spec),map_location='cpu',weights_only=True)
            if artifact['identity'] != factor_spec['identity']: raise ValueError('factor provenance mismatch')
            f2=FactorPair(artifact['second_left'].to(device),artifact['second_right'].to(device))
            for source in DEV11:
                check_resources()
                native_name=f'native__{source}__{expert}__{layer:02d}.pt'
                pred_name=f'pred__{source}__{expert}__{layer:02d}__{family}.pt'
                for name in (native_name,pred_name):
                    if name not in receipts or sha(grid/'cache'/name) != receipts[name]['sha256']:
                        raise ValueError('cached value identity mismatch')
                native=torch.load(grid/'cache'/native_name,map_location=device,weights_only=True)
                pred=torch.load(grid/'cache'/pred_name,map_location=device,weights_only=True)
                pair=load_pair(pairs[(source,expert,layer)],device)
                terms=error_decomposition(native,pred['h'],pred['y'],f2,pair['gate']); actions+=2
                row={'source':source,'expert':expert,'layer':layer,'family':family,'rank':1408,'mode':'4S',
                     'local_risk':gated_risk(pred['y'],native['y1'],pair['gate']),
                     'delta_alignment':delta_alignment(native['y0'],native['y1'],pred['y'],pair['gate'],floor),
                     'terms':terms,'native_cache_sha256':receipts[native_name]['sha256'],
                     'prediction_cache_sha256':receipts[pred_name]['sha256'],'factor_sha256':factor_spec['sha256']}
                rows.append(row)
                del native,pred,pair
            del f2,artifact
    if len(rows)!=66 or actions!=132: raise RuntimeError('decomposition budget mismatch')
    exclusive_json(output/'decomposition.json',{'selection_sha256':scope['selection_sha256'],
        'factor_actions':actions,'rows':rows,'status':'DECOMPOSITION_COMPLETE'})
    return {'rows':len(rows),'factor_actions':actions,'family':family}

