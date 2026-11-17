"""Bounded exact-SVD factor production; invoke only through independent guard.

Six sites x three families = 18 factor pairs, 36 matrix decompositions. The
ordinary SVD can be built independently of clean statistic aggregation.
"""
from pathlib import Path
import json
import os
import torch
from icc_local_replay import SITES,FAMILIES
from icc_stage3_assets import checked_path,load_weights,make_site_factors,save_factors,verify_live_weights
from icc_stage3_grid import exclusive_json


def execute(scope, *, factorize, device, authorize, check_resources):
    authorize(scope)
    six_site_families = {
        'six_site_three_family': FAMILIES,
        'six_site_svd': ('SVD',),
        'six_site_calibrated': ('CA', 'CD'),
    }
    if scope['mode'] in six_site_families:
        work=[(f,e,l) for f in six_site_families[scope['mode']] for e,l in SITES]
        if scope['max_matrix_decompositions'] != 2 * len(work): raise ValueError('six-site decomposition budget')
    elif scope['mode'] == 'selected_remaining_74':
        select_path=checked_path(scope['selection'])
        selected=json.loads(select_path.read_text(encoding='utf8'))['family']
        if selected not in FAMILIES: raise ValueError('invalid selected family')
        work=[(selected,e,l) for e in ('high','low') for l in range(40) if (e,l) not in SITES]
        if len(work)!=74 or scope['max_matrix_decompositions'] != 148: raise ValueError('remaining decomposition budget')
    else: raise ValueError('unsupported factor-build scope')
    entries=scope['scales']; index={(r['family'],r['expert'],r['layer']):r for r in entries}
    if len(index)!=len(entries): raise ValueError('duplicate scale identity')
    output=Path(scope['output']); output.mkdir(parents=True,exist_ok=False)
    receipts=[]
    for family,expert,layer in work:
        check_resources()
        print(json.dumps({'event': 'SVD_PAIR_START', 'family': family, 'expert': expert, 'layer': layer}), flush=True)
        w1,b1,w2,b2=load_weights(scope['checkpoints'][expert],layer,device)
        if (expert,layer) in SITES:
            verify_live_weights(scope['live_weight_manifest'],expert,layer,w1,b1,w2,b2)
        scale = None
        if family in ('CA','CD'):
            record=index.get((family,expert,layer))
            if record is None or record['calibration_manifest_sha256'] != scope['clean_cal11_manifest_sha256']:
                raise ValueError('missing new clean calibration scale identity')
            scale=record['file']
        first,second=make_site_factors(factorize,w1,w2,family,scale,device,driver=scope['svd_driver'])
        identity={'family':family,'expert':expert,'layer':layer,'rank':1408,
                  'checkpoint_sha256':scope['checkpoints'][expert]['sha256'],
                  'calibration_manifest_sha256':scope['clean_cal11_manifest_sha256'] if scale else None,
                  'scale_sha256':scale['sha256'] if scale else None,
                  'svd_driver':scope['svd_driver'],'svd_full_matrices':False,'scale_epsilon':1e-6,
                  'factor_split':'U*S, Vh','torch_version':torch.__version__,
                  'allow_tf32':torch.backends.cuda.matmul.allow_tf32,
                  'decomposition_dtype':'float32','execution_factor_dtype':'bfloat16',
                  'prefix_rank512_is_same_decomposition':True}
        name=f'{family}__{expert}__{layer:02d}__r1408.pt'
        receipt=save_factors(output/name,first,second,identity)
        receipts.append(receipt)
        exclusive_json(output/f'{name}.json', receipt)
        print(json.dumps({'event': 'SVD_PAIR_SAVED', 'family': family, 'expert': expert,
                          'layer': layer, 'sha256': receipt['sha256']}), flush=True)
        check_resources()
        del w1,w2,first,second
        torch.cuda.empty_cache()
    exclusive_json(output/'FACTOR_MANIFEST.json',{'mode':scope['mode'],'matrix_decompositions':2*len(receipts),
                                                 'factor_pairs':receipts,'status':'FACTORS_WRITTEN_PENDING_ORACLE'})
    return {'factor_pairs':len(receipts),'matrix_decompositions':2*len(receipts)}

