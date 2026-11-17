"""Fixed CAL11 Q1 fitting across all 80 expert/layer sites.

Review/ACTIVE/device authorization provided by external guard. No job submission,
no source/seed/hyperparameter search, no reuse of BWS mixed-CAL3 parameters.
"""
from pathlib import Path
import math
import json
import os
import resource
import time
import torch
from q1_cal11_reproduction_telemetry import fit_site,CAL11,validate_calibration_pairs
from icc_stage3_assets import checked_path,preflight_checkpoint
from icc_stage3_grid import exclusive_json


def run(scope,*,common_replay,device,authorize,check_resources,deadline):
    authorize(scope)
    multiplier=scope['remaining_time_multiplier']
    if not isinstance(multiplier,(int,float)) or not math.isfinite(multiplier) or multiplier < 1:
        raise ValueError('invalid predeclared timing margin')
    if not isinstance(deadline,(int,float)) or not math.isfinite(deadline): raise ValueError('missing shared soft deadline')
    if not isinstance(scope['prior_fit_attempts'],int) or not 0 <= scope['prior_fit_attempts'] <= 80:
        raise ValueError('invalid prior Q1 fit count')
    if not isinstance(scope['max_new_fits'],int) or not 0 <= scope['max_new_fits'] <= 80-scope['prior_fit_attempts'] or scope['target_identity'] != 'V4_FULLM_Y_PRE_GATE_PANEL512':
        raise ValueError('Q1 fit budget or target identity mismatch')
    validate_calibration_pairs(scope)
    # Hash both full checkpoints before the first fit; subsequent reads check stat identity.
    checkpoints={}
    for expert in ('high','low'):
        path,stat=preflight_checkpoint(scope['checkpoints'][expert])
        checkpoints[expert]={'path':str(path),'sha256':scope['checkpoints'][expert]['sha256'],
                             'stat_dev_ino_size_mtime_ns_ctime_ns':stat}
    output=Path(scope['output']); output.mkdir(parents=True,exist_ok=False)
    exclusive_json(output/'CHECKPOINT_PREFLIGHT.json',checkpoints)
    reuse={}
    for entry in scope.get('compatible_existing',[]):
        key=(entry['identity']['expert'],entry['identity']['layer'])
        identity=entry['identity']
        if key in reuse or key[0] not in ('high','low') or key[1] not in range(40): raise ValueError('duplicate/bad preexisting Q1')
        if identity['calibration_target'] != scope['target_identity'] or identity['calibration_manifest_sha256'] != scope['calibration_manifest_sha256']:
            raise ValueError('preexisting Q1 calibration mismatch')
        if (tuple(identity['calibration_sources']) != CAL11 or identity['ridge'] != 1e-3 or identity['maxiter'] != 240
            or identity['tolerance'] != 1e-5 or identity['common_replay_sha256'] != scope['common_replay_sha256']):
            raise ValueError('preexisting Q1 source/solver ancestry mismatch')
        if identity['checkpoint_sha256'] != scope['checkpoints'][key[0]]['sha256']:
            raise ValueError('preexisting Q1 checkpoint mismatch')
        artifact=torch.load(checked_path(entry),map_location='cpu',weights_only=True)
        if artifact.get('identity') != identity or artifact['A_q1_f32'].dtype != torch.float32 or artifact['A_q1_f32'].shape != (5120,5120):
            raise ValueError('preexisting Q1 identity or dtype mismatch')
        if not torch.isfinite(artifact['A_q1_f32']).all().item(): raise ValueError('nonfinite Q1 matrix')
        if artifact['alpha_Q1'].shape != (13824,) or artifact['alpha_Q1'].dtype != torch.float32 or not torch.isfinite(artifact['alpha_Q1']).all().item():
            raise ValueError('preexisting Q1 alpha mismatch')
        fit=artifact['q1_fit']
        if fit['status']!='CONVERGED' or not math.isfinite(fit['true_relative_residual']) or not 0 <= fit['true_relative_residual'] <= 1e-5: raise ValueError('preexisting fit not converged')
        reuse[key]=entry
    if 80-len(reuse) > scope['max_new_fits']: raise ValueError('remaining sites exceed available fit budget')
    records=[]; new_fits=0; durations=[]; started=time.monotonic()
    journal=output/'FIT_EVENTS.jsonl'
    def event(obj):
        with journal.open('a',encoding='utf8') as f:
            f.write(json.dumps(obj,allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
    first = [('high',0),('high',1),('low',0),('low',1)]
    order=first+[(e,l) for e in ('high','low') for l in range(40) if (e,l) not in first]
    for expert,layer in order:
            check_resources(); key=(expert,layer)
            if key in reuse:
                records.append({'expert':expert,'layer':layer,'status':'REUSED_EXACT_IDENTITY','artifact':reuse[key]})
                continue
            if new_fits>=scope['max_new_fits']: raise RuntimeError('Q1 fit budget exceeded')
            event({'event':'FIT_START','expert':expert,'layer':layer})
            fit_started=time.monotonic()
            try:
                rec=fit_site(scope,expert,layer,common_replay=common_replay,device=device,
                             output=output/f'Q1__{expert}__{layer:02d}.pt',authorize=lambda *_:None)
            except BaseException as exc:
                event({'event':'FIT_FAILED','expert':expert,'layer':layer,'error_type':type(exc).__name__,
                       'error':str(exc),'pcg_telemetry_file':f'Q1__{expert}__{layer:02d}.pcg.json'})
                raise
            elapsed=time.monotonic()-fit_started; durations.append(elapsed)
            new_fits+=1; row={'expert':expert,'layer':layer,'status':'FIT_CONVERGED','artifact':rec,'elapsed_seconds':elapsed}
            records.append(row)
            exclusive_json(output/f'Q1__{expert}__{layer:02d}.receipt.json',row)
            event({'event':'FIT_COMPLETE',**row})
            torch.cuda.empty_cache()
            if new_fits == 4:
                elapsed_total=time.monotonic()-started
                remaining=deadline-time.monotonic()
                forecast=max(durations)*(80-len(reuse)-new_fits)*multiplier
                report={'event':'FOUR_FIT_RESOURCE_FORECAST','first_four_seconds':durations,
                        'remaining_soft_seconds':remaining,'forecast_remaining_seconds':forecast,
                        'multiplier':multiplier,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                        'peak_gpu_reserved':torch.cuda.max_memory_reserved(),'continue':forecast < remaining}
                event(report)
                if forecast >= remaining:
                    raise TimeoutError('remaining work exceeds predeclared conservative time budget; four valid fits retained')
    if len(records)!=80: raise RuntimeError('Q1 coverage incomplete')
    exclusive_json(output/'Q1_MANIFEST.json',{'target_identity':scope['target_identity'],'new_fits':new_fits,
                                              'reused_fits':len(reuse),'sites':records,'status':'ALL_80_CONVERGED'})
    return {'new_fits':new_fits,'reused_fits':len(reuse),'sites':len(records)}









