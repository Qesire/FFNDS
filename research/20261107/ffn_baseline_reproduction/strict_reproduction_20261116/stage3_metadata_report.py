"""Read-only verified Stage3 local evidence aggregator; no tensor or model work."""
import argparse,hashlib,json,math
from pathlib import Path
from itertools import product
DEV=('vbi_1013','vbi_1040','vbi_0886','vbi_0947','vbi_1109','vbi_1084','vbi_0950','vbi_0176','vbi_0800','vbi_0392','vbi_0056')
TWO=('vbi_1013','vbi_0800');SITES=tuple((e,l) for e in ('high','low') for l in (4,19,34));FAMILIES=('SVD','CA','CD')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def need(v,msg):
 if not v:raise ValueError(msg)
def bound(path,digest):
 need(sha(path)==digest,'file identity mismatch '+str(path));return read(path)
def num(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>=0
def producer(spec):
 p=Path(spec['stage']).resolve();scope=bound(p/'scope.json',spec['scope_sha256']);r=bound(p/'receipt/RUN.json',spec['run_sha256']);e=bound(p/'receipt/EXIT.json',spec['exit_sha256'])
 need(r['scope_sha256']==spec['scope_sha256'] and str(r['slurm_job_id'])==str(e['slurm_job_id'])==str(spec['job_id']) and e['exit_code']==0,'producer did not complete')
 log=p/'receipt/runner.log';need(sha(log)==spec['runner_log_sha256'],'runner log mismatch');lines=[x for x in log.read_text(encoding='utf8').splitlines() if x.strip()]
 return p,scope,lines

def keyed(rows,key,expected):
 d={}
 for r in rows:
  k=tuple(r[x] for x in key);need(k in expected and k not in d,'duplicate/unexpected result cell');d[k]=r
 need(set(d)==expected,'incomplete result cells');return d

def aggregate(grid,selection,cache,decomp,numeric,comp,truem,scope):
 expected={(s,e,l,f,r,m) for s in DEV for e,l in SITES for f in FAMILIES for r in (512,1408) for m in ('4S','2R')}
 g=keyed(grid['rows'],('source','expert','layer','family','rank','mode'),expected);need(grid['cell_count']==792 and selection['family']=='CD','grid not fixed selected CD')
 for row in g.values():need(num(row['risk']),'grid risk invalid')
 ce={(s,e,l,m) for s in DEV for e,l in SITES for m in ('Q1','Global-P64')};c=keyed(comp['cells'],('source','expert','layer','method'),ce);need(comp['status']=='LOCAL_COMPARATORS_COMPLETE' and comp['count']==132,'comparator incomplete')
 for row in c.values():need(num(row['risk']) and row['target']=='Y_local512','comparator target/risk invalid')
 de={(s,e,l) for s in DEV for e,l in SITES};d=keyed(decomp['rows'],('source','expert','layer'),de);need(decomp['status']=='DECOMPOSITION_COMPLETE' and decomp['factor_actions']==132,'decomposition incomplete')
 ne={(s,e,l,m,t) for s in TWO for e,l in (('high',19),('low',19)) for m in ('4S','2R') for t in (False,True)};n=keyed(numeric['rows'],('source','expert','layer','mode','tf32_allowed'),ne);need(numeric['status']=='NUMERIC_CELLS_COMPLETE' and numeric['count']==16 and all(num(x['risk']) for x in n.values()),'numeric incomplete')
 te={(s,e,l) for s in TWO for e,l in SITES};t=keyed(truem['rows'],('source','expert','layer'),te);need(truem['status']=='COMPLETE' and truem['prediction_count']==24 and truem['arithmetic_pairs']==4,'true-M incomplete')
 caches={x['file']:x for x in cache['files']};need(len(caches)==len(cache['files']),'duplicate canonical cache')
 for (source,e,l,mode,tf32),row in n.items():
  name=f'native__{source}__{e}__{l:02d}.pt';need(row['factor_sha256']==g[(source,e,l,'CD',1408,'4S')]['factor_sha256'] and row['native_anchor_sha256']==caches[name]['sha256'],'numeric factor/cache identity mismatch')
 for k,row in d.items():
  source,e,l=k;gr=g[(source,e,l,'CD',1408,'4S')];name=f'native__{source}__{e}__{l:02d}.pt'
  need(row['family']=='CD' and row['rank']==1408 and row['mode']=='4S' and row['factor_sha256']==gr['factor_sha256'] and row['local_risk']==gr['risk'] and row['native_cache_sha256']==caches[name]['sha256'],'decomposition does not match grid')
  terms=row['terms'];need(all(num(terms['energy'][z]) for z in ('read','first','num')) and all(isinstance(terms['cross'][z],(int,float)) and math.isfinite(terms['cross'][z]) for z in ('read_first','read_num','first_num')) and num(terms['total_energy']),'invalid decomposition')
 for (source,e,l),row in t.items():
  name=f'native__{source}__{e}__{l:02d}.pt';need(row['canonical_sha256']==caches[name]['sha256'] and num(row['author_risk']) and num(row['gp64_risk']),'true-M canonical target mismatch')
 table=[]
 for source,e,l in sorted(te):
  icc=g[(source,e,l,'CD',1408,'4S')]['risk'];risks={'ICC':icc,'Q1':c[(source,e,l,'Q1')]['risk'],'Chipmunk Author Safe':t[(source,e,l)]['author_risk'],'Global-P64 true-M':t[(source,e,l)]['gp64_risk']}
  table.append({'source':source,'expert':e,'layer':l,'absolute_risk':risks,'ratio_to_icc':{k:(v/icc if icc>0 else None) for k,v in risks.items()},'icc_zero_denominator':icc==0,'canonical_sha256':t[(source,e,l)]['canonical_sha256']})
 summaries={}
 for method in table[0]['absolute_risk']:
  per={s:math.fsum(x['absolute_risk'][method] for x in table if x['source']==s)/6 for s in TWO}
  summaries[method]={'per_source_mean':per,'source_equal_mean':math.fsum(per.values())/2}
 def components(sources):
  out={}
  for typ,names in (('energy',('read','first','num')),('cross',('read_first','read_num','first_num'))):
   out[typ]={}
   for name in names:
    per={s:math.fsum(d[(s,e,l)]['terms'][typ][name] for e,l in SITES)/6 for s in sources}
    out[typ][name]={'per_source_mean':per,'source_equal_mean':math.fsum(per.values())/len(sources)}
  return out
 return {'status':'LOCAL_METADATA_AGGREGATE_COMPLETE','selected_family':'CD','grid_cell_count':792,'q1_local_cells':66,'global_p64_local_cells':66,'selected_decomposition_cells':66,'numeric_cells':16,'true_m_source_site_count':12,'true_m_prediction_count':24,'paired_two_source_table':table,'paired_two_source_source_equal_means':summaries,'decomposition_components_dev11':components(DEV),'decomposition_components_two_source':components(TWO),'grid_rows':grid['rows'],'comparator_rows':comp['cells'],'decomposition_rows':decomp['rows'],'numeric_rows':list(n.values()),'true_m_rows':truem['rows'],'ratio_policy':'risk divided by ICC risk at same source/site; NA only when ICC risk is exactly zero','target':'identical grid canonical Y_local512 and gate; own-W BF16; F64 gated MSE','execution':{'ICC':'P512 local','Q1':'P512 local','Chipmunk Author Safe':'true-M 31668 then P512 scoring','Global-P64 true-M':'true-M 31668 then P512 scoring'},'interpretation':'local paired diagnostics; no video quality or speed claim; final generated videos are independent evidence'}

def assemble(spec):
 stages={k:producer(spec[k]) for k in ('grid','decomposition','numeric','comparators','true_m')};grid_stage,gs,_=stages['grid'];gsp=spec['grid'];out=grid_stage/'output'
 grid=bound(out/'scores.json',gsp['scores_sha256']);selection=bound(out/'selection.json',gsp['selection_sha256']);cache=bound(out/'cache_manifest.json',gsp['cache_manifest_sha256']);summary=bound(out/'SUMMARY.json',gsp['summary_sha256'])
 need(summary['scores_sha256']==gsp['scores_sha256'] and summary['selection_sha256']==gsp['selection_sha256'] and summary['cache_manifest_sha256']==gsp['cache_manifest_sha256'] and grid['scope_sha256']==selection['scope_sha256']==cache['scope_sha256']==gsp['scope_sha256'],'grid lineage mismatch')
 payload={}
 for k,filename in (('decomposition','decomposition.json'),('numeric','numeric_diagnostic.json'),('comparators','comparators.json'),('true_m','SUMMARY.json')):
  p,scope,lines=stages[k];item=spec[k];payload[k]=bound(p/'output'/filename,item['result_sha256'])
  if k!='true_m':
   need(scope['input_pairs']==gs['input_pairs'] and scope['factors']==gs['factors'] and scope['checkpoints']==gs['checkpoints'] and scope['decomposition_norm_floor']==gs['decomposition_norm_floor']==1e-12,'local replay input lineage mismatch')
   need(lines and isinstance(json.loads(lines[-1]).get('result'),dict),'missing final runner result')
   final=json.loads(lines[-1])['result'];expected={'decomposition':{'rows':66,'factor_actions':132,'family':'CD'},'numeric':{'cells':16},'comparators':{'q1_cells':66,'global_p64_cells':66}}[k]
   need(all(final.get(name)==v for name,v in expected.items()),'runner result counts mismatch')
  else:
   result=bound(p/'receipt/RESULT.json',item['receipt_result_sha256']);need(result['status']=='COMPLETE' and result['scope_sha256']==item['scope_sha256'] and result['summary_sha256']==item['result_sha256'],'true-M RESULT mismatch')
   need(scope['canonical_grid_scope']['sha256']==gsp['scope_sha256'] and scope['canonical_cache_manifest']['sha256']==gsp['cache_manifest_sha256'] and scope['checkpoints']==gs['checkpoints'],'true-M canonical source mismatch')
  if k in ('decomposition','numeric'):need(payload[k]['selection_sha256']==gsp['selection_sha256'],'selection provenance mismatch')
  if k=='numeric':need(payload[k]['cache_manifest_sha256']==gsp['cache_manifest_sha256'],'numeric cache provenance')
  if k=='comparators':need(payload[k]['native_cache_manifest_sha256']==gsp['cache_manifest_sha256'],'comparator cache provenance')
 shards=spec['factor_shards'];need(len(shards)==2,'two selected factor shards required');seen=set();ids=set();plan_hash=None
 for item in shards:
  p,sc,lines=producer(item);manifest=bound(p/'output/FACTOR_MANIFEST.json',item['result_sha256']);shard=sc['shard_id']
  need(shard in ('high','low') and shard not in ids and sc['selected_family']=='CD' and sc['mode']=='selected_shard' and sc['max_matrix_decompositions']==74,'selected shard scope mismatch');ids.add(shard)
  plan=sc['selected_shard_plan']['sha256'];need(plan_hash in (None,plan) and manifest['selected_shard_plan_sha256']==plan and manifest['shard_id']==shard,'selected shard plan mismatch');plan_hash=plan
  rows=manifest['factor_pairs'];keys={(r['identity']['expert'],r['identity']['layer']) for r in rows};expected={(shard,l) for l in range(40) if l not in (4,19,34)}
  need(manifest['status']=='FACTORS_WRITTEN_PENDING_ORACLE' and manifest['matrix_decompositions']==74 and len(rows)==37 and keys==expected and not (seen&keys) and all(r['identity']['family']=='CD' for r in rows),'selected shard coverage mismatch');seen.update(keys)
  need(lines and json.loads(lines[-1])['result']['factor_pairs']==37 and json.loads(lines[-1])['result']['matrix_decompositions']==74,'selected shard runner result mismatch')
 need(len(seen)==74 and ids=={'high','low'},'selected shard pair incomplete')
 result=aggregate(grid,selection,cache,payload['decomposition'],payload['numeric'],payload['comparators'],payload['true_m'],gs);result['producers']={k:spec[k] for k in stages};result['factor_shard_producers']=shards;result['factor_shard_note']='manifest metadata verified; full factor bytes verified by separate whole80 freeze/reload';return result

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('input');p.add_argument('input_sha256');p.add_argument('output');a=p.parse_args();need(sha(a.input)==a.input_sha256,'input identity mismatch');result=assemble(read(a.input))
 with Path(a.output).open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
