#!/usr/bin/env python3
"""Small-receipt-only summary, requires both actual 79-artifact closures."""
import json, math, statistics
from pathlib import Path
import e0_chipmunk_input_video_results as base
import chipmunk_p64_sync_abba_contract as contract
ROOT=base.ROOT; HF=base.HF

def summary(phases,part):
 labels=contract.schedule(part)
 if len(phases)!=7 or tuple(p['sync_arm'] for p in phases)!=labels:raise ValueError('seven-phase order differs')
 raw={}
 for p in phases:
  infer=[z for z in p['stages'] if z['name']=='infer']
  if len(infer)!=4 or {x['step'] for x in infer}!={0,1,2,3}:raise ValueError('infer stage coverage')
  v={'FFN_boundary_CUDA_seconds':p['FFN_boundary_CUDA_seconds'],'phase_wall_seconds':p['wall_seconds_including_observers'],'infer_four_CUDA_seconds':sum(x['CUDA_seconds'] for x in infer),'infer_four_wall_seconds':sum(x['wall_seconds'] for x in infer)}
  if not all(isinstance(x,(int,float)) and math.isfinite(x) and x>=0 for x in v.values()):raise ValueError('nonfinite timing')
  raw[p['sync_arm']]=v
 measures={}
 for key in next(iter(raw.values())):
  a=[raw[n][key] for n in ('A1','A2')];b=[raw[n][key] for n in ('B1','B2')];am=statistics.fmean(a);bm=statistics.fmean(b)
  measures[key]={'A_measured':a,'B_measured':b,'A_mean':am,'B_mean':bm,'B_minus_A':bm-am,'B_over_A':bm/am if am else None}
 return {'raw_phase_timings':raw,'measured_excluding_warmups':measures}

def one(part):
 cfgpath=ROOT/f'config/e0_chipmunk_sync_abba_20261030_part{part}_pending_review_v1.json';cfg,cid=base.load(cfgpath)
 proof,proofid=base.load(ROOT/f'.runtime/chipmunk_sync_abba_20261030/part{part}/CLOSED_VERIFICATION_FINAL.json')
 if proof.get('status')!='PASS_ACTUAL_EXIT_CLOSED_CHIPMUNK_SYNC_ABBA' or proof.get('id')!=cfg['id']:raise ValueError('missing actual closure')
 out=HF/cfg['output_HF_relative'];run,rr=base.receipt(out/'RUN.json');exit,er=base.receipt(out.parent/'EXIT_RECEIPT.json');base.exit_ok(exit,rr,cid['sha256'])
 if run.get('id')!=cfg['id'] or run.get('status')!='COMPLETE_CHIPMUNK_SYNC_ABBA_EXIT_PENDING' or run.get('job_spec_sha256')!=exit.get('spec_sha256') or run.get('automatic_retry') is not False:raise ValueError('GPU RUN identity')
 vr,ve=base.bound(proof['verification_RUN']),base.bound(proof['verification_exit']);base.exit_ok(ve,proof['verification_RUN'],cid['sha256'])
 if vr.get('status')!='PASS_CLOSED_CHIPMUNK_SYNC_ABBA' or vr.get('artifact_count')!=79 or vr.get('RUN')!=rr or vr.get('exit')!=er or vr.get('job_spec_sha256')!=ve.get('spec_sha256') or vr.get('predebited_bytes')!=vr.get('completed_bytes') or vr.get('inflight')!='NONE':raise ValueError('independent verifier mismatch')
 cpu=run['CPU_preflight'];cr,ce=base.bound(cpu['RUN']),base.bound(cpu['exit']);base.exit_ok(ce,cpu['RUN'],cid['sha256'])
 if cr.get('status')!='PASS_CHIPMUNK_SYNC_ABBA_BYTE_PREFLIGHT' or cr.get('job_spec_sha256')!=ce.get('spec_sha256'):raise ValueError('CPU preflight mismatch')
 if run.get('historical_reference_matches') is not True or len(run['prefix_byte_controls'])!=70 or not all(z['equal'] is True for z in run['prefix_byte_controls']) or run['neutral_video_checks']!={x:True for x in contract.schedule(part)}:raise ValueError('state/video gates mismatch')
 led=run['operation_ledger'];
 if any(led[k].get('deferred_completion_barriers')!=120 for k in ('predebited','observed')):raise ValueError('barrier ledger mismatch')
 phases=run['phases'];timings=summary(phases,part)
 public_phases=[{k:v for k,v in p.items() if k not in ('states','video')} for p in phases]
 return {'source_ordinal':run['source_ordinal'],'seed':run['seed'],'scope_id':run['id'],'protocol':{'path':str(cfgpath.relative_to(ROOT)),**cid},'closure_proof_sha256':proofid['sha256'],'CPU':cpu,'GPU':{'RUN':rr,'exit':er},'VERIFY':{'RUN':proof['verification_RUN'],'exit':proof['verification_exit'],'artifact_count':79,'bytes_read':vr['completed_bytes']},'phases':public_phases,**timings,'state_byte_checks':run['prefix_byte_controls'],'video_byte_checks':run['neutral_video_checks'],'operation_ledger':led,'peak_GPU_allocated_bytes':run['peak_GPU_allocated_bytes'],'peak_GPU_reserved_bytes':run['peak_GPU_reserved_bytes'],'peak_RSS_bytes':run['peak_RSS_bytes'],'wall_seconds':run['wall_seconds'],'raw_model_setup_seconds':run['raw_model_setup_seconds'],'memory_observations':run.get('memory_observations',[]),'physical_bus_bytes':'UNKNOWN'}

def source_equal(rows):
 if len(rows)!=2 or len({r['source_ordinal'] for r in rows})!=2:raise ValueError('two distinct sources required')
 keys=tuple(rows[0]['measured_excluding_warmups'])
 if any(set(r['measured_excluding_warmups'])!=set(keys) for r in rows):raise ValueError('metric coverage differs')
 out={}
 for key in keys:
  samples=[r['measured_excluding_warmups'][key] for r in rows]
  ratios=[x['B_over_A'] for x in samples]
  out[key]={'A_source_equal_mean':statistics.fmean(x['A_mean'] for x in samples),'B_source_equal_mean':statistics.fmean(x['B_mean'] for x in samples),'source_equal_mean_B_minus_A':statistics.fmean(x['B_minus_A'] for x in samples),'mean_of_within_source_B_over_A_ratios':statistics.fmean(ratios) if all(x is not None for x in ratios) else None,'source_count':2,'measured_repetitions_per_arm_per_source':2}
 return out

def main():
 rows=[one(0),one(1)]
 if [r['source_ordinal'] for r in rows]!=[898,979] or any(r['seed']!=42 for r in rows):raise ValueError('source identity differs')
 data={'status':'TWO_ACTUAL_GPU_AND_VERIFY_CLOSED','sources':rows,'source_equal':source_equal(rows),'interpretation':'Two exposed sources, one seed each. Warmups excluded from two measured repetitions per arm. A/B differences are conditional instrumented execution observations; cold NONE is not a repeated SYS control. Source 979 quality failure is not evidence of recovery. Physical bus traffic UNKNOWN.'}
 target=ROOT/'research/20261030/CHIPMUNK_SYNC_ABBA_RESULTS_20261030.json';target.write_text(json.dumps(data,indent=2)+'\n')
 lines=['# P64 same-stream completion: actual closed results','',data['interpretation'],'','| Source | A FFN s | B FFN s | B / A | A phase wall s | B phase wall s |','|---|---:|---:|---:|---:|---:|']
 for r in rows:
  m=r['measured_excluding_warmups'];f=m['FFN_boundary_CUDA_seconds'];w=m['phase_wall_seconds'];ratio='undefined' if f['B_over_A'] is None else f"{f['B_over_A']:.4f}";lines.append(f"| {r['source_ordinal']} | {f['A_mean']:.4f} | {f['B_mean']:.4f} | {ratio} | {w['A_mean']:.4f} | {w['B_mean']:.4f} |")
 lines+=['','Two source-equal means and the mean of within-source B/A ratios are in the JSON; these are two sources with two nested measurements per arm, not four independent sources. Two measurements per arm exclude warmups; raw phase events, four-step infer CUDA and wall sums, per-source differences, receipts, gates, ledger and resources are in the JSON.','']
 target.with_suffix('.md').write_text('\n'.join(lines));print(target)
if __name__=='__main__':main()
