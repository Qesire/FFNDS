"""Freeze a completed 792-cell ICC selection and divide the remaining 74 sites.

Metadata-only; hashes original factors, never runs SVD or submits a job. The
input JSON specifies grid {stage,scope_sha256,job_id,summary_sha256}, six-site
factor_producers [{stage,scope_sha256,job_id,factor_manifest_sha256}], and a
nonempty shards list of {id,sites:[[expert,layer],...]}. Output is immutable
review input and must itself be bound by each shard scope's SHA256.
"""
import argparse,hashlib,json,math,os
from pathlib import Path

FAMILIES=('SVD','CA','CD')
DEV=('vbi_1013','vbi_1040','vbi_0886','vbi_0947','vbi_1109','vbi_1084','vbi_0950','vbi_0176','vbi_0800','vbi_0392','vbi_0056')
SIX=tuple((e,l) for e in ('high','low') for l in (4,19,34))
FULL={(e,l) for e in ('high','low') for l in range(40)}
REMAINING=FULL-set(SIX)

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def require(ok,msg):
 if not ok:raise ValueError(msg)
def passed(spec):
 stage=Path(spec['stage']).resolve();r=read(stage/'receipt/RUN.json');x=read(stage/'receipt/EXIT.json')
 require(sha(stage/'scope.json')==r['scope_sha256']==spec['scope_sha256'] and x['exit_code']==0 and str(r['slurm_job_id'])==str(x['slurm_job_id'])==str(spec['job_id']),'producer job identity or exit')
 return stage

def select_grid(rows,scope_sha):
 expected={(s,e,l,f,r,m) for s in DEV for e,l in SIX for f in FAMILIES for r in (512,1408) for m in ('4S','2R')};grid={};digests={}
 for row in rows:
  k=tuple(row[x] for x in ('source','expert','layer','family','rank','mode'));v=row['risk']
  require(k in expected and k not in grid and row['scope_sha256']==scope_sha and isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>=0,'grid row invalid or duplicate')
  grid[k]=float(v);key=(row['family'],row['expert'],row['layer']);digest=row['factor_sha256']
  require(isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest),'bad grid factor SHA')
  require(key not in digests or digests[key]==digest,'grid factor changed within site');digests[key]=digest
 require(set(grid)==expected,'incomplete 792 grid')
 scores={}
 for family in FAMILIES:
  source_vals=[]
  for source in DEV:
   means=[math.fsum(grid[(source,e,l,family,1408,'4S')] for ee,l in SIX if ee==e)/3 for e in ('high','low')]
   source_vals.append(math.fsum(means)/2)
  scores[family]=math.fsum(source_vals)/len(DEV)
 return min(FAMILIES,key=lambda f:(scores[f],FAMILIES.index(f))),scores,digests

def shards_check(shards):
 require(isinstance(shards,list) and len(shards)>0,'no factor shards');seen=set();names=set()
 for shard in shards:
  name=shard['id'];require(isinstance(name,str) and name and name not in names and all(c.isascii() and (c.isalnum() or c in '_-') for c in name),'bad shard name');names.add(name)
  sites=[tuple(x) for x in shard['sites']];require(sites and len(sites)==len(set(sites)) and set(sites)<=REMAINING and not (seen & set(sites)),'invalid, duplicate, overlapping, or original-six site');seen.update(sites)
 require(seen==REMAINING,'shards do not cover all 74 remaining sites')

def build(spec):
 shards_check(spec['shards']);g=passed(spec['grid']);summary_path=g/'output/SUMMARY.json';require(sha(summary_path)==spec['grid']['summary_sha256'],'grid summary identity')
 summary=read(summary_path);scores_path=g/'output/scores.json';sel_path=g/'output/selection.json';scores=read(scores_path);sel=read(sel_path)
 require(summary['scope_sha256']==spec['grid']['scope_sha256'] and summary['status']=='GRID_COMPLETE_PENDING_Q1_GP64_AND_ERROR_DECOMPOSITION' and summary['count']==792 and summary['scores_sha256']==sha(scores_path) and summary['selection_sha256']==sha(sel_path),'grid summary incomplete')
 require(scores['scope_sha256']==sel['scope_sha256']==spec['grid']['scope_sha256'] and scores['cell_count']==len(scores['rows'])==792,'grid scope/count mismatch')
 family,means,digests=select_grid(scores['rows'],spec['grid']['scope_sha256']);require(sel['family']==family and sel['risk_by_family']==means and summary['selection']=={k:v for k,v in sel.items() if k!='scope_sha256'},'selected family/risk mismatch')
 originals={};proofs=[]
 for producer in spec['factor_producers']:
  stage=passed(producer);manifest=stage/'output/FACTOR_MANIFEST.json';require(sha(manifest)==producer['factor_manifest_sha256'],'six-site manifest changed');m=read(manifest)
  require(m['status']=='FACTORS_WRITTEN_PENDING_ORACLE' and m['matrix_decompositions']==2*len(m['factor_pairs']),'six-site producer accounting')
  producer_scope=read(stage/'scope.json')
  if producer_scope.get('mode')=='six_site_svd_recover_high04':
   recovery=m['recovery_provenance']
   require(m['mode']=='six_site_svd' and m['matrix_decompositions']==12 and m['new_matrix_decompositions']==producer_scope['max_matrix_decompositions']==10 and m['recovered_matrix_decompositions']==producer_scope['recovered_matrix_decompositions']==2,'recovery decomposition accounting')
   require(recovery['source_job_id']=='168881' and recovery['source_scope_sha256']=='9d472dea3acddedebc5bc706b60f3d496c65adce43612a0e51dd53387bb56071' and recovery['source_factor_sha256']=='827a3e57a36e2c719cf4950ea2b58c30ad7477e153d36342391e8f8840ff4011' and recovery['source_factor_bytes']==106695389 and recovery['tensor_sha256']=={'first_left': 'beb4b534a83d336198f42df3a7f2325b5de930f442004418a73296667549aa21', 'first_right': '5de9f19b34d18de4d8b5ccf5fec53f5e0bfa6c0302ced35a4178915804ffd1f7', 'second_left': '7b5ac07dffefd50fb2729ee4bf17fc04f15ece7f85cd6fa0198836588adb2324', 'second_right': '706427929d069e7a0525bfcdddfda698bf1f17bf4f50e90ed240d497479183ba'} and recovery['reused_matrix_decompositions']==2,'recovery provenance mismatch')

  proofs.append(producer)
  for row in m['factor_pairs']:
   ident=row['identity'];site=(ident['expert'],ident['layer'])
   if ident['family']!=family:continue
   require(site in SIX and site not in originals and ident['rank']==1408 and row['sha256']==digests[(family,*site)],'selected original factor identity')
   path=Path(row['path']).resolve();require(path.is_relative_to(stage/'output') and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'selected original factor bytes')
   side=Path(str(path)+'.json');require(read(side)==row,'selected original factor per-pair receipt')
   originals[site]=row
 require(set(originals)==set(SIX),'original selected factor coverage')
 return {'schema':'ICC_SELECTED_FACTOR_SHARDS_V1','status':'FROZEN_FROM_GRID_PENDING_REVIEW','family':family,'grid':spec['grid'],'selection_sha256':sha(sel_path),'scores_sha256':sha(scores_path),'factor_producers':proofs,'original_six':[originals[s] for s in SIX], 'shards':spec['shards'],'remaining_site_count':74,'max_matrix_decompositions':148,'no_retry':True}

def validate_shard(plan,scope):
 require(plan['schema']=='ICC_SELECTED_FACTOR_SHARDS_V1' and plan['status']=='FROZEN_FROM_GRID_PENDING_REVIEW' and plan['family'] in FAMILIES and plan['remaining_site_count']==74 and plan['max_matrix_decompositions']==148 and plan['no_retry'] is True,'invalid frozen factor plan')
 shards_check(plan['shards']);matches=[x for x in plan['shards'] if x['id']==scope['shard_id']];require(len(matches)==1,'missing reviewed shard')
 sites=[tuple(x) for x in matches[0]['sites']];require(scope['sites']==matches[0]['sites'] and scope['max_matrix_decompositions']==2*len(sites) and scope['selected_family']==plan['family'],'shard scope differs from plan')
 require(len(plan['original_six'])==6 and {(x['identity']['expert'],x['identity']['layer']) for x in plan['original_six']}==set(SIX) and all(x['identity']['family']==plan['family'] for x in plan['original_six']),'invalid original six')
 for row in plan['original_six']:
  i=row['identity'];cal=None if plan['family']=='SVD' else scope['clean_cal11_manifest_sha256']
  require(i['rank']==1408 and i['checkpoint_sha256']==scope['checkpoints'][i['expert']]['sha256'] and i['calibration_manifest_sha256']==cal and i['svd_driver']==scope['svd_driver']=='gesvd' and i['factor_split']=='U*S, Vh' and i['decomposition_dtype']=='float32' and i['execution_factor_dtype']=='bfloat16' and i['svd_full_matrices'] is False and i['prefix_rank512_is_same_decomposition'] is True,'new shard differs from original numeric/calibration contract')
 return [(plan['family'],e,l) for e,l in sites]

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('input');p.add_argument('input_sha256');p.add_argument('output');a=p.parse_args()
 require(sha(a.input)==a.input_sha256,'builder input identity');result=build(read(a.input))
 with Path(a.output).open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
