#!/usr/bin/env python3
"""Actual-exit-only summary of precomputed local/prefix CPU metadata."""
import hashlib,json,math,statistics
from pathlib import Path
from types import SimpleNamespace
import e0_chipmunk_p64_local_prefix_v2_owner as owner
import e0_chipmunk_input_video_results as base
ROOT=base.ROOT;HF=base.HF
ENERGIES=('L2','P2','T2','cross2','dense2','shadow2','student2','gate_L2','gate_P2','gate_T2','gate_cross2','gate_dense2','gate_shadow2','gate_student2')
SNAP=(0,13,26,39)

def run_metadata(binding):
 path=(HF/binding['path']).resolve()
 if not path.is_relative_to(HF.resolve()) or not 0<binding['bytes']<16<<20:raise ValueError('RUN metadata confinement/registered size')
 before=path.stat();data=path.read_bytes();after=path.stat()
 if before!=after or len(data)!=binding['bytes'] or hashlib.sha256(data).hexdigest()!=binding['sha256']:raise ValueError('RUN metadata identity')
 return json.loads(data)

def aggregate(groups):
 if not groups:raise ValueError('empty groups')
 if any(not g.get('identity_within_tolerance') or not g.get('gate_identity_within_tolerance') or g.get('vector_tolerance_excess_max',1)>0 for g in groups):raise ValueError('numeric identity failed')
 total={k:sum(g[k] for g in groups) for k in ENERGIES}
 if any(not math.isfinite(x) for x in total.values()):raise ValueError('nonfinite energy')
 out={'sum_energies':total,'row_observations':sum(g['count'] for g in groups),'weight_sum':sum(g['weight_sum'] for g in groups),'max_identity_tolerance_excess':max(g['vector_tolerance_excess_max'] for g in groups)}
 for pre in ('','gate_'):
  den=total[pre+'dense2'];out[pre+'relative_L2']={k:(math.sqrt(max(0,total[pre+k])/den) if den>0 and total[pre+k]>=0 else None) for k in ('L2','P2','T2')};out[pre+'cross_over_dense2']=total[pre+'cross2']/den if den>0 else None
  shadow=total[pre+'shadow2'];out[pre+'local_relative_L2_over_student_shadow']=math.sqrt(total[pre+'L2']/shadow) if shadow>0 else None
  out[pre+'shadow_energy_over_dense_energy']=shadow/den if den>0 else None
 return out

def collection(sites,label):
 chosen=[s for s in sites if label in s['groups']]
 if len(chosen)!=(4 if label=='full_M' else 40):raise ValueError('site coverage differs')
 if label=='full_M' and tuple(s['block'] for s in chosen)!=SNAP:raise ValueError('snapshot identity differs')
 per=[]
 for site in chosen:
  cells=site['groups'][label]
  if len(cells)!=84:raise ValueError('cell coverage differs')
  entry={'block':site['block'],'pooled':aggregate(cells),'gated_cross_signs':{'positive':sum(z['gate_cross2']>0 for z in cells if z['count']),'negative':sum(z['gate_cross2']<0 for z in cells if z['count'])}}
  if label=='full_M':
   entry['cells']=[{'tau':i//4,'quadrant':i%4,**{k:z[k] for k in ('count','weight_sum','gate_L2','gate_P2','gate_T2','gate_cross2','gate_dense2','gate_shadow2','gate_student2')}} for i,z in enumerate(cells)]
  per.append(entry)
 allrows=[z for site in chosen for z in site['groups'][label]];pool=aggregate(allrows)
 means={}
 for prefix in ('','gate_'):
  for k in ('L2','P2','T2'):
   vals=[x['pooled'][prefix+'relative_L2'][k] for x in per];vals=[v for v in vals if v is not None]
   means[prefix+k]={'mean':statistics.fmean(vals) if vals else None,'min':min(vals) if vals else None,'max':max(vals) if vals else None,'finite_sites':len(vals)}
 cross=[z['gate_cross2'] for z in allrows if z['count']>0]
 pooled_cells=[{'tau':i//4,'quadrant':i%4,'pooled':aggregate([site['groups'][label][i] for site in chosen])} for i in range(84)]
 return {'site_count':len(chosen),'pooled':pool,'site_relative_L2_summary':means,'gated_cell_cross_signs':{'positive':sum(x>0 for x in cross),'negative':sum(x<0 for x in cross),'zero':sum(x==0 for x in cross)},'sites':per,'pooled_spatiotemporal_cells':pooled_cells,'detail_location':'All per-site/per-cell ratios and numeric residuals remain in the SHA-bound HF RUN; public summary retains pooled84 cells, each site pool and full-M gated cell energies.'}

def one(i):
 cp=ROOT/f'config/e0_chipmunk_p64_local_prefix_v2_20261030_part{i}_pending_review_v1.json';p,pid=base.load(cp)
 proof=owner.inspect_exit(SimpleNamespace(protocol=cp));
 if proof.get('status')!='PASS_LOCAL_PREFIX_ACTUAL_EXIT':raise ValueError('actual EXIT required')
 r=run_metadata(proof['RUN']);e=base.bound(proof['EXIT']);base.exit_ok(e,proof['RUN'],pid['sha256'])
 if r.get('id')!=p['id'] or r.get('protocol_sha256')!=pid['sha256'] or r.get('job_spec_sha256')!=e.get('spec_sha256') or r.get('automatic_retry') is not False or len(r['sites'])!=40 or [x['block'] for x in r['sites']]!=list(range(40)):raise ValueError('scope/coverage differs')
 parent={k:v for k,v in p['parent'].items() if k!='sites'}
 return {'source_ordinal':p['source_ordinal'],'seed':p['seed'],'scope_id':p['id'],'protocol':{'path':str(cp.relative_to(ROOT)),**pid},'RUN':proof['RUN'],'EXIT':proof['EXIT'],'parent':parent,'selector64':collection(r['sites'],'selector'),'eval_weighted_excluding_selector64':collection(r['sites'],'eval'),'full_M_four_snapshots':collection(r['sites'],'full_M'),'ledger':{k:r.get(k) for k in ('predebited_bytes','completed_bytes','by_kind','calls_by_kind','metadata_predebited_bytes','metadata_completed_bytes','metadata_calls','CPU_predebited','CPU_completed')},'resources':{k:r.get(k) for k in ('wall_seconds','peak_RSS_bytes')}}

def main():
 rows=[one(0),one(1)]
 if {x['source_ordinal'] for x in rows}!={898,979} or any(x['seed']!=42 for x in rows):raise ValueError('source identity')
 result={'status':'TWO_ACTUAL_CPU_EXITS_CLOSED','summary_implementation_authors':['/root/teal_full_llm_calibration_build_terra','/root'],'scientific_author':'/root','previous_failed_scope':'research/20261030/CHIPMUNK_P64_LOCAL_PREFIX_FAILURE_20261030.json','sources':rows,'limitations':['Two exposed sources and one seed each; rows, blocks and 84 cells are nested, not independent source samples.','Selector64 is a decision sample; weighted evaluation estimates the eligible finite population excluding selector64. Four full-M blocks are separately reported.','L=candidate−student-shadow, P=student-shadow−dense, T=candidate−dense; cross is signed energy interaction, not a causal percentage.','The output decomposition excludes write/residual dynamics and does not identify downstream H3, quality acceptance or free Dense execution.']}
 out=ROOT/'research/20261030/CHIPMUNK_P64_LOCAL_PREFIX_V2_RESULTS_20261030.json';out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 text=['# P64 local and prefix decomposition','','L = candidate − Dense shadow on student input; P = student-shadow − free-Dense; T = candidate − free-Dense. All three columns in the main table use the free-Dense energy denominator. The separate local/shadow column uses the same-state reference and must not be confused with L/free-Dense.','','| Source | subset | gated L / P / T relative to free-Dense | local L / student-shadow | signed cross / free-Dense energy |','|---|---|---|---:|---:|']
 def fmt(v):return 'undefined' if v is None else f'{v:.6g}'
 for r in rows:
  for name in ('selector64','eval_weighted_excluding_selector64','full_M_four_snapshots'):
   s=r[name]['pooled'];q=s['gate_relative_L2'];text.append(f"| {r['source_ordinal']} | {name} | {fmt(q['L2'])} / {fmt(q['P2'])} / {fmt(q['T2'])} | {fmt(s['gate_local_relative_L2_over_student_shadow'])} | {fmt(s['gate_cross_over_dense2'])} |")
 text+=['','Each ratio pools energies before taking a square root; site mean/ranges, pooled84 cells and four full-M gated cell energies are in the compact JSON. Complete per-site cell ratios and numeric residuals remain in the SHA-bound HF RUN.','',*result['limitations'],''];out.with_suffix('.md').write_text('\n'.join(text));print(out)
if __name__=='__main__':main()
