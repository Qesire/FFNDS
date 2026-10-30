#!/usr/bin/env python3
"""Aggregate verified student observer metadata; no tensor or video reads."""
import json,math,statistics
from pathlib import Path
import e0_chipmunk_input_video_results as b
ROOT=b.ROOT;HF=b.HF
KEYS=('count','pre_squared','gated_squared','write_squared','ideal_total','ideal_diagonal','ideal_cross','ideal_vs_actual_gated_squared','dense_pre_squared','dense_gated_squared','dense_write_squared','signed_pre_sum','signed_gated_sum','signed_write_sum')

def group_stats(p):
 if set(p)!=set(KEYS) or any(len(p[k])!=84 for k in KEYS):raise ValueError('84-group coverage differs')
 if any(not isinstance(x,(int,float)) or not math.isfinite(x) for k in KEYS for x in p[k]):raise ValueError('nonfinite statistics')
 def ratio(a,z):
  if a<0 or z<0:raise ValueError('negative squared energy')
  return math.sqrt(a/z) if z>0 else None
 rows=[]
 for i in range(84):
  row={k:p[k][i] for k in KEYS};row.update(tau=i//4,spatial_quadrant=i%4,pre_relative_L2=ratio(row['pre_squared'],row['dense_pre_squared']),gated_relative_L2=ratio(row['gated_squared'],row['dense_gated_squared']),write_relative_L2=ratio(row['write_squared'],row['dense_write_squared']))
  rows.append(row)
 totals={k:sum(p[k]) for k in KEYS};totals.update(pre_relative_L2=ratio(totals['pre_squared'],totals['dense_pre_squared']),gated_relative_L2=ratio(totals['gated_squared'],totals['dense_gated_squared']),write_relative_L2=ratio(totals['write_squared'],totals['dense_write_squared']))
 return {'groups':rows,'total':totals}

def one(i):
 cp=ROOT/f'config/e0_chipmunk_student_observation_20261030_part{i}_pending_review_v2.json';cfg,cid=b.load(cp)
 proof,pid=b.load(ROOT/f'.runtime/chipmunk_student_observation_20261030/part{i}/CLOSED_VERIFICATION_FINAL.json')
 if proof.get('status')!='PASS_ACTUAL_EXIT_CLOSED_CHIPMUNK_STUDENT_OBSERVATION' or proof.get('id')!=cfg['id']:raise ValueError('actual closure missing')
 out=HF/cfg['output_HF_relative'];run,rr=b.receipt(out/'RUN.json');exit,er=b.receipt(out.parent/'EXIT_RECEIPT.json');b.exit_ok(exit,rr,cid['sha256'])
 if run.get('id')!=cfg['id'] or run.get('status')!='COMPLETE_CHIPMUNK_STUDENT_OBSERVATION_EXIT_PENDING' or run.get('job_spec_sha256')!=exit.get('spec_sha256') or run.get('automatic_retry') is not False:raise ValueError('producer identity differs')
 vr,ve=b.bound(proof['verification_RUN']),b.bound(proof['verification_exit']);b.exit_ok(ve,proof['verification_RUN'],cid['sha256'])
 if vr.get('status')!='PASS_CLOSED_CHIPMUNK_STUDENT_OBSERVATION' or vr.get('artifact_count')!=310 or vr.get('RUN')!=rr or vr.get('exit')!=er or vr.get('job_spec_sha256')!=ve.get('spec_sha256') or vr.get('predebited_bytes')!=vr.get('completed_bytes') or vr.get('inflight')!='NONE':raise ValueError('independent verification differs')
 cpu=run['CPU_preflight'];cr,ce=b.bound(cpu['RUN']),b.bound(cpu['exit']);b.exit_ok(ce,cpu['RUN'],cid['sha256'])
 if cr.get('status')!='PASS_CHIPMUNK_STUDENT_OBSERVATION_BYTE_PREFLIGHT' or cr.get('job_spec_sha256')!=ce.get('spec_sha256'):raise ValueError('CPU spec differs')
 if run.get('historical_reference_matches') is not True or len(run['prefix_byte_controls'])!=20 or not all(x['equal'] is True for x in run['prefix_byte_controls']) or not all(run['neutral_video_checks'].values()):raise ValueError('byte gate differs')
 phases=run['phases'];
 if [x['mode'] for x in phases]!=['h_cache_dense','high_h_readonly_p64']:raise ValueError('phase identity')
 manifests={x['mode']:x['observer_manifest'] for x in phases};sites=manifests['high_h_readonly_p64']['records'];blocks=[]
 if set(map(int,sites))!=set(range(40)):raise ValueError('40 blocks required')
 metadata={}; compact_manifests={}
 mask_shas={str(x['block']):x['mask_indices_sha256'] for x in run['mask_observations']}
 if set(mask_shas)!=set(str(n) for n in range(40)):raise ValueError('mask coverage')
 shared_sampling=None
 for mode,manifest in manifests.items():
  if manifest['sites']!=40 or set(map(int,manifest['records']))!=set(range(40)):raise ValueError('manifest coverage')
  compact_manifests[mode]={}
  for k,record in manifest['records'].items():
   compact_manifests[mode][k]={key:value for key,value in record.items() if key!='partitions'}
  first=b.bound(manifest['records']['0']['metadata'])
  sample={'selector_rows':first['selector_rows'],'evaluation_rows':first['evaluation_rows']}
  if shared_sampling is None:shared_sampling=sample
  elif shared_sampling!=sample:raise ValueError('sampling identities differ')
  if first['gate_shape']!=[1,1,5120] or first['gate_stride']!=[30720,5120,1] or first['gate_dtype']!='bfloat16':raise ValueError('gate metadata differs')
  metadata[mode]={'gate_shape':first['gate_shape'],'gate_stride':first['gate_stride'],'gate_dtype':first['gate_dtype'],'metadata_files':{k:v['metadata'] for k,v in manifest['records'].items()}}
 for key,rec in sorted(sites.items(),key=lambda x:int(x[0])):
  if rec['chunks']!=124:raise ValueError('chunk count differs')
  blocks.append({'block':int(key),'partition_stats':group_stats(rec['partitions'])})
 source_partition={k:[sum(rec['partition_stats']['groups'][j][k] for rec in blocks) for j in range(84)] for k in KEYS}
 source_totals=group_stats(source_partition)
 return {'source_partition_stats':source_totals,'source_ordinal':run['source_ordinal'],'seed':run['seed'],'scope_id':run['id'],'protocol':{'path':str(cp.relative_to(ROOT)),**cid},'closure_proof_sha256':pid['sha256'],'CPU':cpu,'GPU':{'RUN':rr,'exit':er},'VERIFY':{'RUN':proof['verification_RUN'],'exit':proof['verification_exit'],'artifact_count':vr['artifact_count'],'predebited_bytes':vr['predebited_bytes'],'completed_bytes':vr['completed_bytes']},'blocks':blocks,'observer_manifests':compact_manifests,'metadata':metadata,'sampling':shared_sampling,'student_mask_sha256_by_block':mask_shas,'source_bindings':cfg['repo_sources'],'state_byte_checks':run['prefix_byte_controls'],'video_byte_checks':run['neutral_video_checks'],'operation_ledger':run['operation_ledger'],'resources':{k:run.get(k) for k in ('peak_GPU_allocated_bytes','peak_GPU_reserved_bytes','peak_RSS_bytes','wall_seconds','output_accounting','raw_model_setup_seconds')},'payload_budget':cfg['payload_budget']}

def main():
 rows=[one(0),one(1)]
 if {r['source_ordinal'] for r in rows}!={898,979} or any(r['seed']!=42 for r in rows):raise ValueError('source identity differs')
 data={'status':'TWO_ACTUAL_GPU_AND_INDEPENDENT_VERIFY_CLOSED','sources':rows,'limitations':['Two exposed sources, one seed each; rows, sites and temporal groups are nested observations, not independent sources.','Write denominator contains residual and is not FFN motion or output energy alone.','Ideal omission diagonal/cross and actual BF16 error are distinct; negative cross indicates cancellation, not causal attribution.','84 additive groups do not measure object coverage or tail quantiles and cannot identify downstream H3 causality.','Byte-identical parent videos imply no new quality measurement or quality recovery claim.']}
 out=ROOT/'research/20261030/CHIPMUNK_STUDENT_OBSERVATION_RESULTS_20261030.json';out.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
 lines=['# Student observation: actual closed results','','Two exposed sources at seed 42; 40 sites and 84 groups per site are nested observations.','', '| Source | Pooled pre rel L2 | Pooled gated rel L2 | Pooled write rel L2 | Gated site mean (range) | Ideal total / diagonal / cross | Actual vs ideal SSE |','|---|---:|---:|---:|---:|---:|---:|']
 def fmt(x):return 'undefined' if x is None else f'{x:.6g}'
 details=[]
 for r in rows:
  t=r['source_partition_stats']['total'];vals=[v['partition_stats']['total']['gated_relative_L2'] for v in r['blocks']];valid=[x for x in vals if x is not None];desc=(f"{statistics.fmean(valid):.6g} ({min(valid):.6g}–{max(valid):.6g})" if valid else 'undefined');lines.append(f"| {r['source_ordinal']} | {fmt(t['pre_relative_L2'])} | {fmt(t['gated_relative_L2'])} | {fmt(t['write_relative_L2'])} | {desc} | {t['ideal_total']:.6g} / {t['ideal_diagonal']:.6g} / {t['ideal_cross']:.6g} | {t['ideal_vs_actual_gated_squared']:.6g} |")
  details += ['',f"Source {r['source_ordinal']} selected block gated relative L2: "+', '.join(f"{k}: {fmt(r['blocks'][k]['partition_stats']['total']['gated_relative_L2'])}" for k in (0,13,26,39))+'.']
  groups=r['source_partition_stats']['groups'];valid=[x for x in groups if x['gated_relative_L2'] is not None]
  if valid:
   lo=min(valid,key=lambda x:x['gated_relative_L2']);hi=max(valid,key=lambda x:x['gated_relative_L2']);details.append(f"Across 84 pooled temporal/quadrant cells, gated relative L2 ranges {lo['gated_relative_L2']:.6g} (τ {lo['tau']}, quadrant {lo['spatial_quadrant']}) to {hi['gated_relative_L2']:.6g} (τ {hi['tau']}, quadrant {hi['spatial_quadrant']}).")
 lines+=details+['', 'Pooled energies sum over sites; the site mean treats sites as nested descriptions, not independent samples. Write denominator includes residual; do not interpret as motion.','',*data['limitations'],''];out.with_suffix('.md').write_text('\n'.join(lines));print(out)
if __name__=='__main__':main()
