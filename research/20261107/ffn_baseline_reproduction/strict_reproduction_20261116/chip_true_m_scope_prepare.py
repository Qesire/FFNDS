"""Prepare a hash-bound true-M scope from completed producers; no jobs or GPU.

Input is a hash-bound JSON recipe with a partially filled `template` (all
non-grid guard fields), `grid` producer and `factor_producers`. Optional
`q1` producer and `selected_icc` manifest are audited for generation readiness,
not prerequisites for the independent true-M computation. Never invent missing
receipts, outputs, budget, reviewer approval, or factor values.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath

from icc_stage3_selected_shards import DEV, FAMILIES, SIX, select_grid

NEUTRAL = ('vbi_1013','vbi_0800')


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def spec(path):
    p = Path(path).resolve()
    if not p.is_file(): raise ValueError(f'missing file: {p}')
    return {'path':str(p), 'sha256':sha(p)}


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def checked(item):
    actual = spec(item['path'])
    if actual['sha256'] != item['sha256']: raise ValueError('changed recipe asset')
    return read(actual['path'])


def complete(item, payload_name):
    stage = Path(item['stage']).resolve()
    scope_path, run_path, exit_path = stage/'scope.json',stage/'receipt/RUN.json',stage/'receipt/EXIT.json'
    scope_spec = spec(scope_path); run_spec=spec(run_path); exit_spec=spec(exit_path)
    cfg, run, end = read(scope_path),read(run_path),read(exit_path)
    if (scope_spec['sha256'] != item['scope_sha256'] or run['scope_sha256'] != item['scope_sha256']
            or str(run['slurm_job_id']) != str(item['job_id']) or str(end['slurm_job_id']) != str(item['job_id'])
            or end['exit_code'] != 0): raise ValueError('producer scope/job/exit mismatch')
    review_path=stage/'REVIEW.json'; review_spec=spec(review_path); review=read(review_path)
    if review['scope_sha256'] != item['scope_sha256'] or review['recheck_disposition'] != 'APPROVE':
        raise ValueError('producer missing independent approval')
    payload=stage/'output'/payload_name
    if not payload.resolve().is_relative_to((stage/'output').resolve()): raise ValueError('payload outside producer')
    return cfg,read(payload),{'scope':scope_spec,'run':run_spec,'exit':exit_spec,'review':review_spec,'payload':spec(payload)}


def verify_stage_or_external(path, expected, template, preview_root=None):
    """Preview only exact lexically safe paths under the reviewed final stage.

    External inputs never fall back. If final stage bytes exist they must match
    as well; preview does not weaken the runtime guard's final path binding.
    """
    value=PurePosixPath(path); stage=PurePosixPath(template['remote_stage'])
    if (not value.is_absolute() or str(value)!=path or '..' in value.parts
            or not stage.is_absolute() or str(stage)!=template['remote_stage'] or '..' in stage.parts
            or template['output']!=str(stage/'output')):
        raise ValueError('invalid absolute scope/stage path')
    if value.is_relative_to(stage):
        if path not in template['implementation_sha256'] or template['implementation_sha256'][path]!=expected:
            raise ValueError('stage asset not bound in implementation map')
        relative=value.relative_to(stage)
        if not relative.parts: raise ValueError('stage asset is directory')
        final=Path(path)
        if final.exists() and (not final.is_file() or sha(final)!=expected):
            raise ValueError('existing final stage bytes differ')
        if preview_root is not None:
            root=Path(preview_root).resolve(strict=True)
            preview=(root.joinpath(*relative.parts)).resolve(strict=True)
            if not preview.is_relative_to(root) or not preview.is_file() or sha(preview)!=expected:
                raise ValueError('preview asset missing, escaped, or changed')
        elif not final.is_file(): raise ValueError('stage asset missing and no preview root')
    else:
        if spec(path)['sha256']!=expected: raise ValueError('external bound asset changed')


def validate_template(template, preview_root=None):
    if (template['operation']!='chip_true_m' or template['selector_seed']!=42 or
            (template['max_predictions'],template['max_arithmetic_pairs'],template['num_videos'],template['num_model_nfe'])!=(24,4,0,0)):
        raise ValueError('scope changes true-M task budget')
    resources=template['resources']
    if (not 0<resources['gpu_memory_gib']<=78 or not 0<resources['host_memory_gib']<=160
            or not 0<resources['wall_seconds']<=10800 or not 0<resources['soft_seconds']<resources['wall_seconds']
            or not 0<resources['output_bytes']<=2*1024**3 or resources['min_free_disk_gib']<=0):
        raise ValueError('scope resource limits invalid')
    pinned_gp64 = {'gp64_proxy_source':'f46ba6feee6905b3e875a3d4e3118ae97259a978ce6ba55413ed6d2a0a5544be',
                   'gp64_action_source':'9a3e789e15564a0b2979418f9b69e26737abc057a0afcb8132f934dadde2160b'}
    for field, identity in pinned_gp64.items():
        if template[field]['sha256'] != identity: raise ValueError('GP64 differs from pinned reference: '+field)
    for field in ('runtime_guard','gp64_proxy_source','gp64_action_source','tail_candidate_source','voxel'):
        item=template[field]
        verify_stage_or_external(item['path'], item['sha256'], template, preview_root)
    if spec(template['python_executable'])['sha256']!=template['python_executable_sha256']:
        raise ValueError('Python executable identity mismatch')
    for path, expected in template['implementation_sha256'].items():
        verify_stage_or_external(path, expected, template, preview_root)
    provider=template['provider']; root=Path(provider['pinned_src']).resolve()
    expected_names={'modules/mlp.py','util/layer_counter.py','util/config.py',
                    'util/storage/layer_storage.py','util/storage/offloaded_tensor.py',
                    'ops/mlp.py','ops/indexed_io.py','triton/mlp_csp_mm1_bf16.py',
                    'triton/mlp_csp_mm2_bf16.py'}
    if set(provider['source_sha256']) != expected_names: raise ValueError('incomplete author source identity')
    for name, expected in provider['source_sha256'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or spec(path)['sha256']!=expected: raise ValueError('author source mismatch')
    if spec(provider['library'])['sha256']!=provider['library_sha256']: raise ValueError('safe selector binary mismatch')
    prior=template['prerequisites']
    module_run,module_exit,module=(checked(prior[k]) for k in ('module_run','module_exit','module_result'))
    selector_run,selector_exit,selector=(checked(prior[k]) for k in ('selector_run','selector_exit','selector_result'))
    if (str(module_run['slurm_job_id'])!='168787' or str(module_exit['slurm_job_id'])!='168787'
            or module_exit['exit_code']!=0 or module['status']!='PASS' or module['source_sha256']!=provider['source_sha256']):
        raise ValueError('bounded author module prerequisite mismatch')
    if (str(selector_run['slurm_job_id'])!='168713' or str(selector_exit['slurm_job_id'])!='168713'
            or selector_exit['exit_code']!=0 or selector['variant']!='CHIPMUNK_AUTHOR_MLP_SAFE_SELECTOR_V1'
            or selector['loaded_extension_sha256']!=provider['library_sha256']
            or selector['adversarial_wan_width']!={'count':13824,'unique':13824,'unwritten_slots':0,'copy_exact':True}):
        raise ValueError('safe selector prerequisite mismatch')


def build(recipe):
    if not recipe.get('grid'): raise ValueError('completed grid SUBMITTED proof is required')
    template = dict(recipe['template'])
    validate_template(template, recipe.get('preview_root'))
    grid, summary, proof = complete(recipe['grid'], 'SUMMARY.json')
    output=Path(recipe['grid']['stage']).resolve()/'output'
    scores_path, selection_path, cache_path = output/'scores.json',output/'selection.json',output/'cache_manifest.json'
    scores,selection,cache=read(scores_path),read(selection_path),read(cache_path)
    gid=proof['scope']['sha256']
    if (summary['scope_sha256']!=gid or summary['status']!='GRID_COMPLETE_PENDING_Q1_GP64_AND_ERROR_DECOMPOSITION'
            or summary['count']!=792 or summary['scores_sha256']!=sha(scores_path)
            or summary['selection_sha256']!=sha(selection_path) or summary['cache_manifest_sha256']!=sha(cache_path)
            or scores['scope_sha256']!=gid or scores['cell_count']!=792 or cache['scope_sha256']!=gid
            or cache['status']!='PRESERVED_PENDING_SELECTED_FAMILY_ERROR_DECOMPOSITION'):
        raise ValueError('grid completion and shared cache do not agree')
    selected, means, score_hashes=select_grid(scores['rows'],gid)
    if (selection['scope_sha256']!=gid or selection['family']!=selected or selection['risk_by_family']!=means
            or summary['selection']!={k:v for k,v in selection.items() if k!='scope_sha256'}):
        raise ValueError('grid selection mismatch')
    expected={(family,e,l) for family in FAMILIES for e,l in SIX}
    grid_factors={(v['family'],v['expert'],v['layer']):v for v in grid['factors']}
    if len(grid['factors'])!=18 or set(grid_factors)!=expected: raise ValueError('grid factor identities incomplete')
    supplied={}; producer_proofs=[]
    for source in recipe['factor_producers']:
        _cfg, manifest, evidence=complete(source,'FACTOR_MANIFEST.json')
        if manifest['status']!='FACTORS_WRITTEN_PENDING_ORACLE' or manifest['matrix_decompositions']!=2*len(manifest['factor_pairs']):
            raise ValueError('factor production ledger mismatch')
        stage=Path(source['stage']).resolve(); producer_proofs.append(evidence)
        for row in manifest['factor_pairs']:
            identity=row['identity']; key=(identity['family'],identity['expert'],identity['layer'])
            if key not in expected or key in supplied: raise ValueError('unexpected or duplicate six-site factor')
            path=Path(row['path']).resolve(); side=Path(str(path)+'.json')
            if (not path.is_relative_to(stage/'output') or not path.is_file() or path.stat().st_size!=row['bytes']
                    or sha(path)!=row['sha256'] or read(side)!=row or row['sha256']!=score_hashes[key]
                    or row['sha256']!=grid_factors[key]['sha256'] or identity!=grid_factors[key]['identity']):
                raise ValueError('factor producer/grid bytes or identity mismatch')
            if (identity['rank']!=1408 or identity['checkpoint_sha256']!=grid['checkpoints'][key[1]]['sha256']
                    or identity['calibration_manifest_sha256']!=(None if key[0]=='SVD' else grid['clean_cal11_manifest_sha256'])):
                raise ValueError('factor calibration or weight ancestry mismatch')
            supplied[key]=row
    if set(supplied)!=expected: raise ValueError('three families x six sites not complete')
    entries=cache['files']; cache_index={row['file']:row for row in entries}
    if len(entries)!=len(cache_index): raise ValueError('duplicate native cache receipt')
    for source in DEV:
        for expert,layer in SIX:
            name=f'native__{source}__{expert}__{layer:02d}.pt'
            row=cache_index[name]; path=output/'cache'/name
            if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
                raise ValueError('shared canonical cache changed')
    pairs=checked(template['dev_pairs'])
    if pairs['input_pairs']!=grid['input_pairs']: raise ValueError('grid and true-M input manifests differ')
    capture=checked(template['capture_manifest'])
    ids=capture['fixed64_token_ids']
    if len(ids)!=64 or len(set(ids))!=64 or any(type(x) is not int or not 0<=x<31668 for x in ids):
        raise ValueError('invalid frozen fixed64')
    if template['checkpoints']!=grid['checkpoints']: raise ValueError('grid checkpoint mismatch')
    floor=grid['decomposition_norm_floor']
    if not isinstance(floor,(int,float)) or not math.isfinite(floor) or floor!=1e-12: raise ValueError('grid must bind frozen 1e-12 norm floor')
    template['decomposition_norm_floor']=floor; template['fixed64_token_ids']=ids
    template['canonical_grid_scope']=proof['scope']; template['canonical_cache_manifest']=spec(cache_path)
    template['prerequisites']={**template['prerequisites'],'grid_run':proof['run'],'grid_exit':proof['exit'],'grid_scores':spec(scores_path)}
    readiness={'grid':proof,'factor_producers':producer_proofs,'selected_family':selected,
               'q1':'NOT_BOUND', 'selected_icc_80':'NOT_BOUND'}
    if recipe.get('q1'):
        _qcfg,q1,qproof=complete(recipe['q1'],'Q1_MANIFEST.json')
        sites={(r['expert'],r['layer']) for r in q1['sites']}
        if (q1['status']!='ALL_80_FINITE_FIXED_ITERATES' or q1['target_identity']!='V4_FULLM_Y_PRE_GATE_PANEL512'
                or len(q1['sites'])!=80 or sites!={(e,l) for e in ('high','low') for l in range(40)}
                or sum(q1['fit_status_counts'].values())!=80 or set(q1['fit_status_counts'])-{'CONVERGED','FINITE_ITERATE_NOT_OPTIMUM'}): raise ValueError('Q1 fixed-iterate coverage incomplete')
        for row in q1['sites']:
            item=row['artifact']; path=item.get('path') or item.get('file')
            if not path or spec(path)['sha256']!=item['sha256']: raise ValueError('Q1 artifact identity mismatch')
        readiness['q1']=qproof
    if recipe.get('selected_icc'):
        chosen=checked(recipe['selected_icc'])
        if (chosen['status']!='ALL_80_FROZEN' or chosen['family']!=selected or chosen['rank']!=1408
                or chosen['scores_sha256']!=sha(scores_path) or chosen['selection_sha256']!=sha(selection_path)
                or len(chosen['factor_pairs'])!=80 or {(r['identity']['expert'],r['identity']['layer']) for r in chosen['factor_pairs']}!={(e,l) for e in ('high','low') for l in range(40)}):
            raise ValueError('frozen selected ICC parameters do not bind grid')
        frozen_rows={}
        for producer in chosen['factor_producers']:
            _cfg, manifest, evidence=complete(producer,'FACTOR_MANIFEST.json')
            if (evidence['run']['sha256']!=producer['run_sha256'] or evidence['exit']['sha256']!=producer['exit_sha256']
                    or evidence['payload']['sha256']!=producer['factor_manifest_sha256']
                    or manifest['status']!='FACTORS_WRITTEN_PENDING_ORACLE'
                    or manifest['matrix_decompositions']!=2*len(manifest['factor_pairs'])):
                raise ValueError('selected ICC producer evidence mismatch')
            stage=Path(producer['stage']).resolve()
            for row in manifest['factor_pairs']:
                if row['identity']['family']!=selected: continue
                key=(row['identity']['expert'],row['identity']['layer'])
                path=Path(row['path']).resolve()
                if key in frozen_rows or not path.is_relative_to(stage/'output') or read(str(path)+'.json')!=row:
                    raise ValueError('selected ICC producer pair identity mismatch')
                frozen_rows[key]=row
        if len(frozen_rows)!=80: raise ValueError('selected ICC producer coverage incomplete')
        for row in chosen['factor_pairs']:
            identity=row['identity']; key=(identity['expert'],identity['layer']); path=Path(row['path']).resolve()
            if (row!=frozen_rows[key] or not path.is_file() or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']
                    or identity['checkpoint_sha256']!=grid['checkpoints'][identity['expert']]['sha256']):
                raise ValueError('selected ICC artifact identity mismatch')
            if key in SIX and row['sha256']!=score_hashes[(selected,*key)]:
                raise ValueError('selected ICC six-site factor differs from grid')
        readiness['selected_icc_80']=recipe['selected_icc']
    return {'scope':template, 'readiness':readiness, 'status':'CANDIDATE_REQUIRES_INDEPENDENT_REVIEW'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('recipe');p.add_argument('recipe_sha256');p.add_argument('output');p.add_argument('audit');a=p.parse_args()
    if Path(a.output).exists() or Path(a.audit).exists(): raise ValueError('output already exists')
    recipe=checked({'path':a.recipe,'sha256':a.recipe_sha256}); result=build(recipe)
    with Path(a.output).open('x',encoding='utf-8') as f:
        json.dump(result['scope'],f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    result['readiness']['scope_sha256']=sha(a.output)
    with Path(a.audit).open('x',encoding='utf-8') as f:
        json.dump({'status':result['status'],**result['readiness']},f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
