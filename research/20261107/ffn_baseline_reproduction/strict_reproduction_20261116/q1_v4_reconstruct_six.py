"""Recover six V4 CAL11/512 Q1 exports from original converged coefficients.

No PCG invocation. Original V4 stored alpha_Q0/Q1 but not a reloadable A. This
reconstructs new F32 A from preserved alpha after strict same-data/weight/Q0/
normal-equation checks. It does not claim a bitwise comparison to the unsaved A.
Requires separately reviewed scope and compute allocation.
"""
import json,math,os
from pathlib import Path
import torch
from icc_stage3_assets import checked_path,load_pair,load_weights,sha,preflight_checkpoint
from icc_stage3_grid import exclusive_json
from q1_cal11_reproduction import CAL11,tensor_hash,validate_calibration_pairs

SITES=(('high',4),('high',19),('high',34),('low',4),('low',19),('low',34))
V4_REPR='8ead06e5caac5beea46800d0b3dff2ea3c4c17a2996371591cee8570ecb55257'
V4_RUNNER='9c6d8940524aa7feb11a1f047ee94824e24e7698a64de8cf98da856298739e35'
Q1_CORE='b76a0efafae4b1b9b08a6c5cef6ade11108abed574136e082468c58c3f3a6028'


def finite_close(actual,expected,rtol,atol):
    return math.isfinite(actual) and math.isfinite(expected) and abs(actual-expected)<=atol+rtol*abs(expected)


def run(scope,*,common_replay,Problem,device,authorize,check_resources):
    authorize(scope)
    if scope['max_reconstructions']!=6 or scope['max_new_pcg_fits']!=0:
        raise ValueError('six reconstruction / zero PCG scope required')
    # Fixed before any result: not an adaptive choice based on observed mismatch.
    if scope['q0_scalar_rtol']!=1e-12 or scope['q0_scalar_atol']!=1e-12 or scope['residual_rtol']!=0.005 or scope['residual_atol']!=1e-10:
        raise ValueError('reconstruction validation tolerances differ from scope')
    index=validate_calibration_pairs(scope)
    vspec=scope['v4_scope'];fspec=scope['v4_fits'];espec=scope['v4_exit']
    vscope_path=checked_path(vspec);fits_path=checked_path(fspec);exit_path=checked_path(espec)
    vscope=json.loads(vscope_path.read_text(encoding='utf8'))
    original_stage=Path(vscope['remote_stage']).resolve()
    if vscope_path!=original_stage/'scope.json' or fits_path!=original_stage/'output'/'VB_LOCAL_FITS.json' or exit_path!=original_stage/'EXIT.json':
        raise ValueError('V4 provenance paths do not belong to original stage')
    bound_cal_path=checked_path(scope['calibration_input_manifest'])
    if sha(bound_cal_path)!=scope['calibration_manifest_sha256']:raise ValueError('CAL11 input manifest SHA mismatch')
    bound_cal=json.loads(bound_cal_path.read_text(encoding='utf8'))
    if bound_cal['mode']!='cal' or bound_cal['input_pairs']!=scope['calibration_pairs'] or bound_cal['v2_bank_manifest_sha256']!=vscope['bank_manifest_sha256']:
        raise ValueError('CAL11 pairs do not match original V2 bank binding')
    fits=json.loads(fits_path.read_text(encoding='utf8'))['fits']
    ex=json.loads(exit_path.read_text(encoding='utf8'))
    if ex.get('exit_code')!=0 or ex.get('scope_sha256')!=vspec['sha256'] or ex.get('result',{}).get('status')!='V4_REPR_DENSITY_COMPLETE':
        raise ValueError('original V4 run did not complete')
    if vscope['bank_manifest_sha256']!=scope['original_v2_bank_manifest_sha256'] or vscope['checkpoints']!=scope['checkpoints']:
        raise ValueError('V4 CAL11 bank/checkpoints mismatch')
    if (vscope['code_sha256']['v4_repr.py'],vscope['code_sha256']['v4_runner.py'],vscope['code_sha256']['q1_fit_core.py'])!=(V4_REPR,V4_RUNNER,Q1_CORE):
        raise ValueError('V4 producer source identity mismatch')
    if sha(common_replay.__file__)!=scope['common_replay_sha256'] or sha(Problem.__module__ and __import__(Problem.__module__).__file__)!=Q1_CORE:
        raise ValueError('reconstruction source identity mismatch')
    for expert in ('high','low'):preflight_checkpoint(scope['checkpoints'][expert])
    output=Path(scope['output']);output.mkdir(parents=True,exist_ok=False)
    records=[]
    with torch.inference_mode():
        for expert,layer in SITES:
            check_resources(); old=fits[f'512|{expert}{layer:02d}'];info=old['q1_fit']
            if info['status']!='CONVERGED' or not math.isfinite(info['true_relative_residual']) or not 0<=info['true_relative_residual']<=1e-5:
                raise ValueError('historical V4 fit is not converged')
            w1,b1,w2,b2=load_weights(scope['checkpoints'][expert],layer,device)
            cal=[]
            for source in CAL11:
                pair=load_pair(index[(source,expert,layer)],device)
                u0,u1=pair['u0'].float(),pair['u1'].float();y0,y1=pair['full_y0_panel'].float(),pair['full_y1_panel'].float()
                cal.append({'source':source,'u0':u0,'u1':u1,'X':u1-u0,'Y':y1-y0,'g':pair['gate'].float(),'n':512})
            alpha0,a0,z=common_replay.q0_prior(cal,w1,b1,w2)
            old_alpha0=torch.tensor(old['alpha_Q0'],dtype=torch.float32,device=device)
            alpha1=torch.tensor(old['alpha_Q1'],dtype=torch.float32,device=device)
            if old_alpha0.shape!=(13824,) or alpha1.shape!=(13824,) or not torch.isfinite(alpha1).all().item():
                raise ValueError('historical alpha shape or finite mismatch')
            if not torch.equal(alpha0,old_alpha0): raise ValueError('original Q0 alpha differs from rebuilt CAL11 Q0')
            a0_fro2=float(a0.square().sum())
            if not finite_close(float(z),old['Z'],1e-12,1e-12) or not finite_close(a0_fro2,old['A0_fro2'],1e-12,1e-12):
                raise ValueError('original V4 Z/A0 Frobenius mismatch')
            sources=[{'x':c['X'].float(),'y':c['Y'].float(),'g':c['g'].float()} for c in cal]
            prob=Problem(w1,w2,a0.float(),sources,ridge=1e-3)
            rhs=prob.ridge*prob.k0/prob.za
            for item in prob.sources:
                h=item['h'];target=(item['y']*item['g'].square())@prob.w2
                rhs=rhs+(h*target).sum(0)/(len(prob.sources)*h.shape[0]*prob.z)
            initial=float(torch.linalg.vector_norm(rhs-prob.normal(alpha0)))
            final=float(torch.linalg.vector_norm(rhs-prob.normal(alpha1)))
            relative=final/max(initial,1e-30)
            if not (finite_close(initial,info['initial_residual'],.005,1e-10) and finite_close(final,info['true_normal_residual'],.005,1e-10)):
                raise ValueError('historical and current normal residual disagree')
            if not math.isfinite(relative) or not 0<=relative<=1e-5:
                raise ValueError('reconstructed alpha fails fixed convergence threshold')
            a_f32=prob.matrix(alpha1).double().float()
            if not torch.isfinite(a_f32).all().item():raise ValueError('nonfinite reconstructed matrix')
            identity={'expert':expert,'layer':layer,'family':'Q1r2','calibration_sources':CAL11,
                      'calibration_target':'V4_FULLM_Y_PRE_GATE_PANEL512','checkpoint_sha256':scope['checkpoints'][expert]['sha256'],
                      'calibration_manifest_sha256':scope['calibration_manifest_sha256'],'common_replay_sha256':scope['common_replay_sha256'],
                      'a0_sha256_f64':tensor_hash(a0),'alpha_q0_sha256_f32':tensor_hash(alpha0),
                      'ridge':1e-3,'maxiter':240,'tolerance':1e-5,
                      'runtime_numeric':'A_f32 cast BF16; BF16 GEMM; cache/subtract and add in FP32 then BF16',
                      'export_origin':'V4_SAVED_CONVERGED_ALPHA_RECONSTRUCTED_NO_PCG',
                      'original_v4_fits_sha256':fspec['sha256'],'original_A_bitwise_comparison':'UNAVAILABLE_NOT_SAVED'}
            data={'identity':identity,'alpha_Q0':alpha0.detach().cpu().contiguous(),
                  'alpha_Q1':alpha1.detach().cpu().contiguous(),'A_q1_f32':a_f32.detach().cpu().contiguous(),
                  'q1_fit':info,'z_norm':z,'reconstruction_checks':{'alpha_Q0_bitwise':True,'Z':z,'A0_fro2':a0_fro2,
                  'initial_residual_recomputed':initial,'final_residual_recomputed':final,'relative_residual_recomputed':relative}}
            path=output/f'Q1__{expert}__{layer:02d}.pt'
            with path.open('xb') as f:torch.save(data,f);f.flush();os.fsync(f.fileno())
            back=torch.load(path,map_location='cpu',weights_only=True)
            if back['identity']!=identity or any(not torch.equal(data[k],back[k]) for k in ('alpha_Q0','alpha_Q1','A_q1_f32')):
                raise RuntimeError('reconstructed Q1 reload mismatch')
            row={'path':str(path.resolve()),'sha256':sha(path),'bytes':path.stat().st_size,'identity':identity,
                 'q1_fit':info,'reconstruction_checks':data['reconstruction_checks']}
            records.append(row);exclusive_json(output/f'Q1__{expert}__{layer:02d}.receipt.json',row)
            del w1,b1,w2,b2,a0,a_f32,prob,cal;torch.cuda.empty_cache()
    if len(records)!=6:raise RuntimeError('six-site reconstruction incomplete')
    exclusive_json(output/'RECONSTRUCTED_Q1_SIX.json',{'status':'RECONSTRUCTED_SIX_NO_PCG','sites':records,
        'note':'original unexported A not available for byte-for-byte comparison'})
    return {'status':'RECONSTRUCTED_SIX_NO_PCG','sites':len(records),'new_pcg_fits':0}

