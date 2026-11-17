"""Bounded stage-3 792-cell local ICC grid. Invoked by a reviewed external guard.

This is P512 FFN replay, not video generation. The independent scope binds all
input files, checkpoints, factors, code/numerics and output limits. Q1/GP64 and
selected-family decomposition/80-site export are separately identified work.
"""
import json
import os
from pathlib import Path
import torch
from icc_local_replay import (DEV11,SITES,FAMILIES,RANKS,MODES,FactorPair,
                              canonical_endpoints,icc_prediction,gated_risk,select_family)
from icc_stage3_assets import sha,checked_path,load_weights,load_pair,verify_live_weights


def validate_scope(scope, *, require_factors=True):
    if tuple(scope['sources']) != DEV11 or [tuple(x) for x in scope['sites']] != list(SITES):
        raise ValueError('fixed DEV11 and site order required')
    pairs = scope['input_pairs']; factors = scope.get('factors',[])
    pair_keys = [(x['source'],x['expert'],x['layer']) for x in pairs]
    factor_keys = [(x['family'],x['expert'],x['layer']) for x in factors]
    for pair in pairs:
        expected_kind = 'stage1_full_u' if pair['source'] in ('vbi_1013','vbi_0800') else 'v2_panel'
        if pair['anchor']['kind'] != expected_kind or pair['hit']['kind'] != expected_kind:
            raise ValueError('two neutral sources must use new capture, remaining nine original V2')
    if len(pair_keys) != 66 or set(pair_keys) != {(s,e,l) for s in DEV11 for e,l in SITES}:
        raise ValueError('66 unique source/site inputs required')
    if require_factors and (len(factor_keys) != 18 or set(factor_keys) != {(f,e,l) for f in FAMILIES for e,l in SITES}):
        raise ValueError('18 unique r1408 factor-pair assets required')
    if scope['max_cells'] != 792 or scope['temp_cache_bytes'] > 7*1024**3:
        raise ValueError('local cell or cache limit mismatch')
    if scope['cal22_enabled'] is not False: raise ValueError('CAL22 disabled for this scope')
    if set(scope['checkpoints']) != {'high','low'}: raise ValueError('both expert checkpoints required')
    return {(x['source'],x['expert'],x['layer']):x for x in pairs}, {(x['family'],x['expert'],x['layer']):x for x in factors}


def exclusive_json(path, value):
    with Path(path).open('x',encoding='utf8') as f:
        json.dump(value,f,indent=2,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())


def execute(scope, *, scope_sha256, native_ffn, device, authorize, check_resources):
    """No automatic retries. Only a separately reviewed guard supplies authorize."""
    pairs, factors = validate_scope(scope)
    authorize(scope)
    for path, expected in scope['implementation_sha256'].items():
        if sha(path) != expected: raise ValueError('implementation identity mismatch')
    if len(scope['implementation_sha256']) < 4: raise ValueError('unbound code identities')
    output = Path(scope['output']); output.mkdir(parents=True,exist_ok=False)
    cache = output/'cache'; cache.mkdir()
    rows, cache_receipts, used = [], [], 0
    journal = output/'cells.jsonl'
    def save_payload(name, data):
        nonlocal used
        path = cache/name
        cpu = {k:v.detach().to('cpu').contiguous() for k,v in data.items()}
        logical = sum(x.numel()*x.element_size() for x in cpu.values())
        if used + logical + 1024**2 > scope['temp_cache_bytes']:
            raise RuntimeError('temporary tensor cache budget')
        with path.open('xb') as f:
            torch.save(cpu,f); f.flush(); os.fsync(f.fileno())
        loaded=torch.load(path,map_location='cpu',weights_only=True)
        if set(loaded)!=set(cpu) or any(not torch.equal(cpu[k],loaded[k]) for k in cpu):
            raise RuntimeError('cached candidate readback mismatch')
        used += path.stat().st_size
        if used > scope['temp_cache_bytes']: raise RuntimeError('temporary cache size exceeded')
        rec = {'file':name,'sha256':sha(path),'bytes':path.stat().st_size}
        cache_receipts.append(rec)
        return rec
    with torch.no_grad():
        for expert, layer in SITES:
            check_resources()
            w1,b1,w2,b2 = load_weights(scope['checkpoints'][expert],layer,device)
            verify_live_weights(scope['live_weight_manifest'],expert,layer,w1,b1,w2,b2)
            candidate_factors = {}
            for family in FAMILIES:
                spec = factors[(family,expert,layer)]
                artifact = torch.load(checked_path(spec),map_location='cpu',weights_only=True)
                if artifact['identity'] != spec['identity']:
                    raise ValueError('factor provenance mismatch')
                identity = artifact['identity']
                if identity['family'] != family or identity['expert'] != expert or identity['layer'] != layer or identity['rank'] != 1408:
                    raise ValueError('factor site/family/rank mismatch')
                if identity['checkpoint_sha256'] != scope['checkpoints'][expert]['sha256']:
                    raise ValueError('factor checkpoint mismatch')
                if family in ('CA','CD') and identity['calibration_manifest_sha256'] != scope['clean_cal11_manifest_sha256']:
                    raise ValueError('clean CAL11 calibration identity mismatch')
                candidate_factors[family] = (FactorPair(artifact['first_left'].to(device),artifact['first_right'].to(device)),
                                             FactorPair(artifact['second_left'].to(device),artifact['second_right'].to(device)))
            for source in DEV11:
                check_resources()
                input_spec = pairs[(source,expert,layer)]
                if input_spec['anchor']['input_identity']['checkpoint_sha256'] != scope['checkpoints'][expert]['sha256']:
                    raise ValueError('input/checkpoint identity mismatch')
                pair = load_pair(input_spec,device)
                c = canonical_endpoints(native_ffn,pair['u0'],pair['u1'],w1,b1,w2,b2)
                name = f'native__{source}__{expert}__{layer:02d}.pt'
                keep = {k:c[k] for k in ('h0','y0','h1','y1')}
                if source in ('vbi_1013','vbi_0800') and layer == 19: keep['z0'] = c['z0']
                save_payload(name,keep)
                for family in FAMILIES:
                    f1,f2 = candidate_factors[family]
                    for rank in RANKS:
                        for mode in MODES:
                            h,y = icc_prediction(c,f1,f2,rank,mode)
                            row = {'source':source,'expert':expert,'layer':layer,'family':family,'rank':rank,'mode':mode,
                                   'risk':gated_risk(y,c['y1'],pair['gate']), 'scope_sha256':scope_sha256,
                                   'factor_sha256':factors[(family,expert,layer)]['sha256']}
                            if source in ('vbi_1013','vbi_0800'):
                                row['risk_vs_Y_fullM_panel'] = gated_risk(y,pair['full_y1_panel'],pair['gate'])
                                row['dense_local512_vs_fullM_panel'] = gated_risk(c['y1'],pair['full_y1_panel'],pair['gate'])
                            if rank == 1408 and mode == '4S':
                                rec = save_payload(f'pred__{source}__{expert}__{layer:02d}__{family}.pt',{'h':h,'y':y})
                                row['prediction_cache'] = rec
                            with journal.open('a',encoding='utf8') as f:
                                f.write(json.dumps(row,allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
                            rows.append(row)
                            if len(rows)>792: raise RuntimeError('local cell budget exceeded')
                del pair,c
            del w1,b1,w2,b2,candidate_factors
    selection = select_family(rows)
    exclusive_json(output/'scores.json',{'scope_sha256':scope_sha256,'cell_count':len(rows),'rows':rows})
    exclusive_json(output/'selection.json',{'scope_sha256':scope_sha256,**selection})
    exclusive_json(output/'cache_manifest.json',{'scope_sha256':scope_sha256,'bytes':used,'files':cache_receipts,
        'status':'PRESERVED_PENDING_SELECTED_FAMILY_ERROR_DECOMPOSITION'})
    return {'status':'GRID_COMPLETE_PENDING_Q1_GP64_AND_ERROR_DECOMPOSITION','count':len(rows),'selection':selection,
            'scores_sha256':sha(output/'scores.json'),'selection_sha256':sha(output/'selection.json'),
            'cache_manifest_sha256':sha(output/'cache_manifest.json')}






