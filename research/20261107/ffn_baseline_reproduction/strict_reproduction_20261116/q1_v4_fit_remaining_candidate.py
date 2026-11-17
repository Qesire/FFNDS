"""Bounded remaining Q1 CAL11 sites using the exact V4 fixed-iterate solver."""
from collections import Counter
import json
import os
from pathlib import Path
import time

import torch
from icc_stage3_assets import checked_path, preflight_checkpoint, sha
from icc_stage3_grid import exclusive_json
from q1_v4_fixed_iterate_candidate import CAL11, fit_site, validate_calibration_pairs
from q1_v4_reconstruct_six import SITES


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def same_q1_identity(artifact_identity, manifest_identity):
    """Compare exact identity after the JSON list/torch tuple boundary only."""
    if not isinstance(artifact_identity, dict) or not isinstance(manifest_identity, dict):
        return False
    a, b = dict(artifact_identity), dict(manifest_identity)
    for value in (a, b):
        sources = value.get('calibration_sources')
        if not isinstance(sources, (tuple, list)) or not all(isinstance(x, str) for x in sources):
            return False
        value['calibration_sources'] = tuple(sources)
    return a == b


def admitted_reconstructions(scope):
    receipt = scope['reconstruction']
    stage = Path(receipt['stage'])
    scope_path, run_path, exit_path = stage / 'scope.json', stage / 'receipt/RUN.json', stage / 'receipt/EXIT.json'
    original_scope = read(scope_path)
    run, exit_record = read(run_path), read(exit_path)
    if (sha(scope_path) != receipt['scope_sha256'] or run['scope_sha256'] != receipt['scope_sha256']
            or exit_record['exit_code'] != 0 or run['slurm_job_id'] != receipt['job_id']
            or exit_record['slurm_job_id'] != receipt['job_id'] or original_scope['max_new_pcg_fits'] != 0
            or original_scope['checkpoints'] != scope['checkpoints']
            or original_scope['calibration_manifest_sha256'] != scope['calibration_manifest_sha256']):
        raise ValueError('six-site reconstruction authority differs')
    manifest_path = stage / 'output/RECONSTRUCTED_Q1_SIX.json'
    manifest = read(manifest_path)
    entries = manifest['sites']
    keys = {(r['identity']['expert'], r['identity']['layer']) for r in entries}
    if manifest['status'] != 'RECONSTRUCTED_SIX_NO_PCG' or keys != set(SITES) or len(entries) != 6:
        raise ValueError('missing six distinct eligible Q1 reconstructions')
    for row in entries:
        path = checked_path(row)
        data = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
        identity, fit = row['identity'], row['q1_fit']
        if (not path.is_relative_to(stage / 'output') or not same_q1_identity(data['identity'], identity) or data['q1_fit'] != fit
                or data['A_q1_f32'].dtype != torch.float32 or data['A_q1_f32'].shape != (5120, 5120)
                or not torch.isfinite(data['A_q1_f32']).all().item()
                or fit['status'] != 'CONVERGED' or not 0 <= fit['true_relative_residual'] <= 1e-5
                or identity['checkpoint_sha256'] != scope['checkpoints'][identity['expert']]['sha256']
                or tuple(identity['calibration_sources']) != CAL11
                or identity['calibration_manifest_sha256'] != scope['calibration_manifest_sha256']
                or identity['calibration_target'] != scope['target_identity']
                or identity['common_replay_sha256'] != scope['common_replay_sha256']
                or identity['export_origin'] != 'V4_SAVED_CONVERGED_ALPHA_RECONSTRUCTED_NO_PCG'):
            raise ValueError('reconstructed artifact differs from frozen Q1 contract')
    proof = {'scope_sha256': sha(scope_path), 'run_sha256': sha(run_path), 'exit_sha256': sha(exit_path),
             'manifest_sha256': sha(manifest_path), 'job_id': receipt['job_id']}
    return entries, proof


def run(scope, *, common_replay, device, authorize, check_resources, deadline):
    authorize(scope)
    old = scope['prior_fit']
    prior_journal = checked_path(old['journal'])
    prior_exit = read(checked_path(old['exit']))
    attempts = [json.loads(line) for line in prior_journal.read_text(encoding='utf-8').splitlines()]
    if (attempts != [{'event': 'FIT_START', 'expert': 'high', 'layer': 0}]
            or prior_exit['slurm_job_id'] != old['job_id'] or prior_exit['exit_code'] != 1
            or scope['max_new_fits'] != 74 or scope['max_previous_and_new_fits'] != 75
            or scope['target_identity'] != 'V4_FULLM_Y_PRE_GATE_PANEL512'
            or scope['acceptance_mode'] != 'V4_FIXED_ITERATE_STATUS_PRESERVED'
            or scope['remaining_time_multiplier'] != 1.25):
        raise ValueError('Q1 cumulative attempt ledger or fixed target differs')
    validate_calibration_pairs(scope)
    if read(checked_path(scope['calibration_input_manifest']))['input_pairs'] != scope['calibration_pairs']:
        raise ValueError('Q1 embedded panel identities differ from original manifest')
    reuse, reuse_proof = admitted_reconstructions(scope)
    first = [('high',0), ('high',1), ('low',0), ('low',1)]
    all_sites = first + [(e, layer) for e in ('high', 'low') for layer in range(40) if (e, layer) not in first]
    targets = [site for site in all_sites if site not in SITES]
    if [list(x) for x in targets] != scope['target_sites']:
        raise ValueError('Q1 new fit order differs from the frozen 74')
    checkpoints = {}
    for expert, spec in scope['checkpoints'].items():
        path, identity = preflight_checkpoint(spec)
        checkpoints[expert] = {'path': str(path), 'sha256': spec['sha256'], 'stat_identity': identity}
    out = Path(scope['output'])
    out.mkdir()
    exclusive_json(out / 'CHECKPOINT_PREFLIGHT.json', checkpoints)
    exclusive_json(out / 'RECONSTRUCTION_PROOF.json', reuse_proof)
    sites = [{'expert': row['identity']['expert'], 'layer': row['identity']['layer'],
              'status': 'RECONSTRUCTED_FROM_V4', 'artifact': row} for row in reuse]
    durations = []
    journal = out / 'FIT_EVENTS.jsonl'

    def event(record):
        with journal.open('a', encoding='utf-8') as file:
            file.write(json.dumps(record, allow_nan=False) + '\n')
            file.flush()
            os.fsync(file.fileno())

    for i, (expert, layer) in enumerate(targets, 1):
        check_resources()
        start = time.monotonic()
        event({'event': 'FIT_START', 'expert': expert, 'layer': layer, 'global_attempt': 1 + i})
        dest = out / f'Q1__{expert}__{layer:02d}.pt'
        try:
            rec = fit_site(scope, expert, layer, common_replay=common_replay, device=device,
                           output=dest, authorize=lambda *_:None)
        except Exception as exc:
            event({'event': 'FIT_FAILED', 'expert': expert, 'layer': layer, 'error_type': type(exc).__name__,
                   'error': str(exc), 'telemetry': str(dest.with_suffix('.pcg.json'))})
            raise
        rec['path'] = rec.pop('file')
        elapsed = time.monotonic() - start
        row = {'expert': expert, 'layer': layer, 'status': 'FIT_' + rec['q1_fit']['status'],
               'artifact': rec, 'elapsed_seconds': elapsed}
        sites.append(row)
        exclusive_json(out / f'Q1__{expert}__{layer:02d}.receipt.json', row)
        event({'event': 'FIT_COMPLETE', **row})
        durations.append(elapsed)
        torch.cuda.empty_cache()
        if i == 4:
            forecast = max(durations) * (len(targets) - i) * scope['remaining_time_multiplier']
            remaining = deadline - time.monotonic()
            event({'event': 'FOUR_FIT_RESOURCE_FORECAST', 'first_four_seconds': durations,
                   'remaining_soft_seconds': remaining, 'forecast_remaining_seconds': forecast})
            if forecast >= remaining:
                raise TimeoutError('remaining Q1 fit workload exceeds the frozen wall budget')
        check_resources()
    counts = Counter(row['artifact']['q1_fit']['status'] for row in sites)
    if len(sites) != 80 or {(r['expert'],r['layer']) for r in sites} != set(all_sites):
        raise RuntimeError('incomplete Q1 coverage')
    manifest = {'status': 'ALL_80_FINITE_FIXED_ITERATES', 'acceptance_mode': 'V4_FIXED_ITERATE_STATUS_PRESERVED',
                'target_identity': scope['target_identity'],
                'num_sites': len(sites), 'prior_attempts': 1, 'new_fits': len(targets), 'reconstructed_fits': len(reuse),
                'fit_status_counts': dict(counts), 'sites': sites, 'reconstruction_proof': reuse_proof,
                'numerical_optimum_claim': 'NOT_CLAIMED_FOR_FINITE_ITERATE_NOT_OPTIMUM'}
    exclusive_json(out / 'Q1_MANIFEST.json', manifest)
    return {key: manifest[key] for key in ('status','new_fits','reconstructed_fits','fit_status_counts')}

