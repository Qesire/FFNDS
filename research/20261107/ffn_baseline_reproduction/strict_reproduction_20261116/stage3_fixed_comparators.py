"""Q1 and Global-P64 66+66 fixed local P512 cells, no new videos.

Q1 uses the bound complete CAL11 producer; Global-P64 receives the original 64
real token inputs for the nine V2 sources and newly captured full-U for neutrals.
"""
import json
import math
from pathlib import Path
import torch
from icc_local_replay import DEV11,SITES,delta_alignment,gated_risk
from icc_stage3_assets import checked_path,load_pair,load_weights,verify_live_weights,sha
from icc_stage3_grid import exclusive_json,validate_scope


def run(scope,*,common,native_ffn,device,authorize,check_resources):
    pairs,_=validate_scope(scope,require_factors=False); authorize(scope)
    floor = scope['decomposition_norm_floor']
    if not math.isfinite(floor) or floor < 0:
        raise ValueError('need frozen norm floor')
    q1_file=checked_path(scope['q1_manifest'])
    q1=json.loads(q1_file.read_text(encoding='utf8'))
    if q1['status'] not in ('ALL_80_CONVERGED','ALL_80_FINITE_FIXED_ITERATES') or q1['target_identity']!='V4_FULLM_Y_PRE_GATE_PANEL512':
        raise ValueError('Q1 complete V4 CAL11 identity required')
    qsite={(r['expert'],r['layer']):r['artifact'] for r in q1['sites']}
    if len(qsite)!=80 or set(qsite)!={(e,l) for e in ('high','low') for l in range(40)}:
        raise ValueError('Q1 manifest must contain all 80 distinct sites')
    p64_file=checked_path(scope['historical_p64_manifest'])
    p64=json.loads(p64_file.read_text(encoding='utf8'))
    fixed=common.chip_fixed64().long()
    if fixed.tolist()!=p64['fixed64_token_ids'] or p64['selected_k']!=5120 or p64['rows_are_panel_first64']:
        raise ValueError('fixed64/selector identity differs')
    cache_path=checked_path(scope['cache_manifest'])
    cache_manifest=json.loads(cache_path.read_text(encoding='utf8'))
    cache_root=cache_path.parent/'cache'
    cache_receipts={x['file']:x for x in cache_manifest['files']}
    if len(cache_receipts)!=len(cache_manifest['files']): raise ValueError('duplicate shared cache identity')
    output=Path(scope['comparator_output']); output.mkdir(parents=True,exist_ok=False)
    cells=[]
    with torch.no_grad():
        for expert,layer in SITES:
            check_resources(); w1,b1,w2,b2=load_weights(scope['checkpoints'][expert],layer,device)
            verify_live_weights(scope['live_weight_manifest'],expert,layer,w1,b1,w2,b2)
            qspec=qsite[(expert,layer)]
            # A reused or new file is accepted only if its identity was validated in Q1_MANIFEST.
            qpath=checked_path({'path':qspec.get('path',qspec.get('file')),'sha256':qspec['sha256']})
            qobj=torch.load(qpath,map_location='cpu',weights_only=True)
            if qobj['A_q1_f32'].shape!=(5120,5120) or qobj['A_q1_f32'].dtype!=torch.float32:
                raise ValueError('invalid Q1 F32 export')
            identity=qobj['identity']
            if identity['expert']!=expert or identity['layer']!=layer or identity['checkpoint_sha256']!=scope['checkpoints'][expert]['sha256']:
                raise ValueError('Q1 site/weight identity mismatch')
            if identity['calibration_target']!='V4_FULLM_Y_PRE_GATE_PANEL512' or identity['calibration_manifest_sha256']!=scope['q1_calibration_manifest_sha256']:
                raise ValueError('Q1 CAL11 ancestry mismatch')
            a=qobj['A_q1_f32'].to(device)
            for source in DEV11:
                check_resources(); spec=pairs[(source,expert,layer)]; pair=load_pair(spec,device)
                native_name=f'native__{source}__{expert}__{layer:02d}.pt'
                if native_name not in cache_receipts or sha(cache_root/native_name)!=cache_receipts[native_name]['sha256']:
                    raise ValueError('shared native endpoint cache mismatch')
                c=torch.load(cache_root/native_name,map_location=device,weights_only=True)
                bpack={'u0_bf16':pair['u0'],'u1_bf16':pair['u1'],'h0':c['h0'],
                       'y0':c['y0'],'y1':c['y1'],'g':pair['gate'].float(),'n':512}
                if source in ('vbi_1013','vbi_0800'):
                    u64=[]
                    for endpoint in ('anchor','hit'):
                        full=torch.load(str(checked_path(spec[endpoint]['full_u'])),map_location='cpu',mmap=True,weights_only=True)
                        u64.append(full.index_select(0,fixed).to(device))
                else:
                    src=p64['sources'][source]
                    identity=spec['anchor']['input_identity']
                    for field in ('image_sha256','prompt_sha256','source_cluster_id'):
                        if src[field] != identity[field]: raise ValueError('historical fixed64 condition mismatch: '+field)
                    if src['panel_token_id_sha256_le_int64'] != spec['anchor']['token_id_sha256_le_int64']:
                        raise ValueError('historical fixed64/panel mismatch')
                    rec=src['sites'][f'{expert}{layer:02d}']
                    if not rec['chip_cell_unblocked'] or not all(rec['same_state'].values()):
                        raise ValueError('historical fixed64 same-state receipt incomplete')
                    u64=[]
                    for endpoint in ('anchor','hit'):
                        x=rec[endpoint]
                        t=torch.load(checked_path({'path':x['path'],'sha256':x['file_sha256']}),map_location='cpu',weights_only=True)
                        u64.append(t['U_fixed64'].to(device))
                if any(x.shape!=(64,5120) or x.dtype!=torch.bfloat16 for x in u64): raise ValueError('invalid real fixed64 rows')
                qpred=common.q1_predict(bpack,a)
                ppred,indices=common.chip_predict(bpack,w1,b1,w2,u64[0],u64[1],fixed)
                for method,pred in (('Q1',qpred),('Global-P64',ppred)):
                    cell={'source':source,'expert':expert,'layer':layer,'method':method,
                          'risk':gated_risk(pred,c['y1'],pair['gate']),'target':'Y_local512',
                          'delta_alignment':delta_alignment(c['y0'], c['y1'], pred, pair['gate'], floor),
                          'input_kind':spec['anchor']['kind']}
                    if source in ('vbi_1013','vbi_0800'):
                        cell['risk_vs_Y_fullM_panel']=gated_risk(pred,pair['full_y1_panel'],pair['gate'])
                    cells.append(cell)
                if indices.numel()!=5120 or torch.unique(indices).numel()!=5120: raise ValueError('Global-P64 exact K mismatch')
            del w1,b1,w2,b2,a
    if len(cells)!=132: raise RuntimeError('fixed comparator count mismatch')
    exclusive_json(output/'comparators.json',{'cells':cells,'count':132,'q1_manifest_sha256':sha(q1_file),
        'historical_p64_manifest_sha256':sha(p64_file),'native_cache_manifest_sha256':sha(cache_path),
        'status':'LOCAL_COMPARATORS_COMPLETE'})
    return {'q1_cells':66,'global_p64_cells':66}




