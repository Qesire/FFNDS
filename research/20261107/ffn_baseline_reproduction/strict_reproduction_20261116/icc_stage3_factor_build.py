"""Bounded exact-SVD factor production; invoke only through independent guard.

Six sites x three families = 18 factor pairs, 36 matrix decompositions. The
ordinary SVD can be built independently of clean statistic aggregation.
"""
from pathlib import Path
import json
import os
from time import monotonic
import torch
from icc_local_replay import SITES,FAMILIES
from icc_stage3_assets import checked_path,load_weights,make_site_factors,save_factors,verify_live_weights
from icc_stage3_grid import exclusive_json


def validate_calibrated_scales(scope):
    proof = scope['calibrated_stats_prerequisite']
    stage = Path(proof['stage'])
    evidence = {}
    for name, path in (
        ('scope', 'scope.json'), ('run', 'output/RUN.json'), ('exit', 'output/EXIT.json'),
        ('launcher_run', 'launch/RUN.json'), ('launcher_exit', 'launch/EXIT.json'),
        ('summary', 'output/SUMMARY.json'), ('scales', 'output/SCALES.json'),
    ):
        checked = checked_path({'path': str(stage / path), 'sha256': proof[f'{name}_sha256']})
        evidence[name] = json.loads(checked.read_bytes())
    if (evidence['run']['scope_sha256'] != proof['scope_sha256']
            or evidence['launcher_run']['scope_sha256'] != proof['scope_sha256']
            or evidence['exit']['exit_code'] != 0 or evidence['launcher_exit']['exit_code'] != 0):
        raise ValueError('calibrated factors require a successful exact scale producer')
    summary = evidence['summary']
    if (summary['status'], summary['source_count'], summary['main_nfe'], summary['diagnostic_nfe'],
            summary['scale_records'], summary['scales_sha256']) != ('PASS', 11, 66, 27, 160, proof['scales_sha256']):
        raise ValueError('incomplete clean CAL11 scale provenance')
    bank = evidence['scales']
    expected = [row for row in bank['scales'] if (row['expert'], row['layer']) in SITES]
    if (bank['calibration_manifest_sha256'] != scope['clean_cal11_manifest_sha256']
            or len(bank['sources']) != 11 or len(set(bank['sources'])) != 11
            or len(expected) != 12 or expected != scope['scales']):
        raise ValueError('scoped CA/CD scale records differ from calibrated source')


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
        if scope['mode'] == 'six_site_calibrated':
            validate_calibrated_scales(scope)
    elif scope['mode'] == 'six_site_svd_recover_high04':
        if scope['max_matrix_decompositions'] != 10 or scope['recovered_matrix_decompositions'] != 2:
            raise ValueError('recovery must budget five new pairs and one recovered pair')
        work=[('SVD', e, l) for e, l in SITES if (e, l) != ('high', 4)]
        if len(work) != 5: raise ValueError('unexpected recovery site count')
    elif scope['mode'] == 'selected_shard':
        from icc_stage3_selected_shards import build, validate_shard, sha
        binder = Path(__file__).resolve().with_name('icc_stage3_selected_shards.py')
        if scope['implementation_sha256'].get(str(binder)) != sha(binder):
            raise ValueError('selected-shard binder not bound by scope')
        plan_path = checked_path(scope['selected_shard_plan'])
        plan = json.loads(plan_path.read_text(encoding='utf8'))
        # Repeat the metadata audit in the allocated job before any SVD.
        if build(plan) != plan:
            raise ValueError('selected-shard proof changed since review')
        work = validate_shard(plan, scope)
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
    recovery_proof = None
    if scope['mode'] == 'six_site_svd_recover_high04':
        from icc_stage3_assets import sha
        recovery_path = Path(__file__).resolve().with_name('icc_svd_high04_recovery.py')
        if scope['implementation_sha256'].get(str(recovery_path)) != sha(recovery_path):
            raise RuntimeError('recovery source not bound')
        from icc_svd_high04_recovery import recover
        recovery_proof, receipt = recover(scope, output)
        receipts.append(receipt)
        exclusive_json(output / 'SVD__high__04__r1408.pt.json', receipt)
    for family,expert,layer in work:
        check_resources()
        started = monotonic()
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
                  'factor_split':'U*S, Vh','torch_version':str(torch.__version__),
                  'allow_tf32':torch.backends.cuda.matmul.allow_tf32,
                  'decomposition_dtype':'float32','execution_factor_dtype':'bfloat16',
                  'prefix_rank512_is_same_decomposition':True}
        name=f'{family}__{expert}__{layer:02d}__r1408.pt'
        receipt=save_factors(output/name,first,second,identity)
        receipts.append(receipt)
        exclusive_json(output/f'{name}.json', receipt)
        print(json.dumps({'event': 'SVD_PAIR_SAVED', 'family': family, 'expert': expert,
                          'layer': layer, 'sha256': receipt['sha256'],
                          'elapsed_seconds': monotonic() - started}, allow_nan=False), flush=True)
        check_resources()
        del w1,w2,first,second
        torch.cuda.empty_cache()
    manifest={'mode':scope['mode'],'matrix_decompositions':2*len(receipts),
              'factor_pairs':receipts,'status':'FACTORS_WRITTEN_PENDING_ORACLE'}
    if scope['mode']=='six_site_svd_recover_high04':
        if len(receipts) != 6 or len(work) != 5: raise RuntimeError('incomplete recovery coverage')
        manifest.update(mode='six_site_svd', new_matrix_decompositions=10,
                        recovered_matrix_decompositions=2, recovery_provenance=recovery_proof)
    if scope['mode']=='selected_shard':
        manifest.update(shard_id=scope['shard_id'], selected_shard_plan_sha256=scope['selected_shard_plan']['sha256'],
                        selected_family=scope['selected_family'])
    exclusive_json(output/'FACTOR_MANIFEST.json',manifest)
    return {'factor_pairs':len(receipts),'matrix_decompositions':2*len(receipts),
            'new_matrix_decompositions':2*len(work), 'recovered_matrix_decompositions':2 if recovery_proof else 0}

