"""CPU-only provenance audit of complete cells in failed full80 attempts."""
import argparse,hashlib,json,math,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ID='AUDIT_COMPACT6016_PARTIAL_BANK_20261028'
LIMITS={'failed_parts':3,'complete_cells':40,'complete_functions':20,'export_bytes_hashed_max':8<<30,'GPU':0,'model':0,'forward':0,'GEMM':0,'QDQ':0,'QR':0,'fit':0,'CPU_threads':2,'wall_seconds':600,'outer_seconds':900,'output_bytes':8<<20}

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()

def bind(root,x):
 p=(root/x['path']).resolve()
 if not p.is_relative_to(root.resolve()) or not p.is_file() or p.stat().st_size!=x['bytes'] or sha(p)!=x['sha256']:raise ValueError('Bound file differs')
 return p

def atomic(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)

def authorize(p,digest,active):
 if p.get('id')!=ID or p.get('assets_finalized') is not True or p.get('limits')!=LIMITS or p.get('automatic_retry') is not False:raise ValueError('Unfrozen audit scope')
 u=active.get('user_continuation_authorization_20261027',{})
 if u.get('id')!='USER_RESUME_OFFLINE_IFC_20261027' or u.get('status')!='ACTIVE_CURRENT_USER_DIRECTED_SESSION' or u.get('new_scope_registration_allowed') is not True:raise ValueError('Inactive user authorization')
 if not any(x.get('id')==ID and x.get('protocol_sha256')==digest and x.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') for x in active['experiment_authority']['current_user_authorized_scopes']):raise ValueError('Unregistered exact audit')

def audit_cell(hf,entry,charge):
 if entry.get('status')!='COMPLETE_256_UPDATES_AND_INITIAL_FINAL_48_SOURCE_STEP_DIAGNOSTICS':raise ValueError('Incomplete cell')
 result=entry['result'];p=bind(hf,result);v=json.loads(p.read_text());export=entry['export']
 for key in ('run_id','expert','block','initialization','status'):
  if v[key]!=entry[key]:raise ValueError('Cell identity mismatch')
 if v['export']!=export or export.get('reloaded_all_tensor_bytes_equal') is not True or len(v['epoch_source_step_loss_means'])!=8:raise ValueError('Export or update receipt differs')
 gates=v['GPU_zero_LoRA_and_native_STE_byte_checks'];grads=v['GPU_gate_gradient_norms'];losses=v['epoch_source_step_loss_means']
 if set(gates)!={'post_GELU','pre_Q2','post_Q2','pre_gate_output'} or not all(x is True for x in gates.values()) or set(grads)!={'lora_A1','lora_B1','lora_A2','lora_B2'} or not all(math.isfinite(x) and x>=0 for x in grads.values()) or any(grads[k]<=0 for k in ('lora_B1','lora_B2')) or not all(math.isfinite(x) for x in losses):raise ValueError('Actual GPU gate or training epoch evidence differs')
 if v['expert']!='high_noise_model' or v['steps']!=[0,1] or v['initialization'] not in ('energy94','random94'):raise ValueError('Unexpected function identity')
 summaries={}
 for label in ('initial','final_folded_BF16'):
  rows=v['diagnostics'][label]
  if len(rows)!=48 or {(x['index'],x['step']) for x in rows}!={(i,s) for i in range(24) for s in (0,1)}:raise ValueError('Diagnostic source-step coverage differs')
  for x in rows:
   if x['role']!=('TRAIN' if x['index']<16 else 'HELDOUT_EXPOSED_DIAGNOSTIC'):raise ValueError('Diagnostic role mismatch')
  sources=v['source_cluster_means'][label]['actual_BF16_gate_relative_SSE']
  if len(sources)!=24 or {x['index'] for x in sources}!=set(range(24)):raise ValueError('Source cluster coverage differs')
  held=[]
  for i in range(16,24):
   values=[x['actual_BF16_gate_relative_SSE'] for x in rows if x['index']==i];record=next(x for x in sources if x['index']==i)['value']
   if len(values)!=2 or not all(math.isfinite(x) for x in values) or not math.isclose(record,sum(values)/2,rel_tol=1e-12,abs_tol=1e-15):raise ValueError('Source cluster mean mismatch')
   held.append({'index':i,'value':record})
  summaries[label]={'heldout_source_values':held,'equal_source_mean':sum(x['value'] for x in held)/8}
 charge(export['bytes']);bind(hf,export)
 return {'run_id':v['run_id'],'expert':v['expert'],'block':v['block'],'initialization':v['initialization'],'steps':v['steps'],'GPU_zero_LoRA_and_native_STE_byte_checks':gates,'GPU_gate_gradient_norms':grads,'zero_merge_base_weight_bitwise':v['zero_merge_base_weight_bitwise'],'epoch_source_step_loss_means':losses,'result':result,'export':{k:export[k] for k in ('path','sha256','bytes')},'updates_from_completion_contract':256,'per_cell_update_counter':'NOT_STORED_SEPARATELY','heldout_independent_sources':8,'diagnostic_summary':summaries}

def audit_parts(hf,parts,protocol_sha,charge):
 cells=[]
 if len(parts)!=3 or {x['part'] for x in parts}!={0,1,2}:raise ValueError('Three fixed failed original parts required')
 for part in parts:
  run=json.loads(bind(hf,part['run']).read_text());ex=json.loads(bind(hf,part['exit']).read_text())
  if run['id']!='E0_COMPACT6016_FULL80_BANK_20261028' or ex.get('id')!='E0_COMPACT6016_FULL80_BANK_20261028' or len(run['cells'])!={0:12,1:12,2:16}[part['part']] or run['status']!='FAILED_NO_RETRY' or run['part']!=part['part'] or run['protocol_sha256']!=protocol_sha or ex.get('launcher_returncode')!=1 or ex.get('process_exited') is not True or ex.get('protocol_sha256')!=protocol_sha:raise ValueError('Failed attempt identity differs')
  cells.extend(audit_cell(hf,c,charge) for c in run['cells'])
 if len(cells)!=40 or len({x['run_id'] for x in cells})!=40:raise ValueError('Expected 40 distinct complete cells')
 pairs={}
 for c in cells:pairs.setdefault((c['expert'],c['block']),set()).add(c['initialization'])
 if set(pairs)!={('high_noise_model',b) for b in (*range(0,6),*range(10,16),*range(20,28))} or any(v!={'energy94','random94'} for v in pairs.values()):raise ValueError('Expected 20 paired complete functions')
 return cells

def run(args):
 p=json.loads(args.protocol.read_text());digest=sha(args.protocol);authorize(p,digest,json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text()));hf=args.hf_root.resolve();out=(hf/p['output_HF_relative']).resolve()
 if out==hf or not out.is_relative_to(hf):raise ValueError('Output outside HF')
 out.mkdir(parents=True,exist_ok=True)
 with (out/'ATTEMPT.json').open('x') as f:json.dump({'id':ID,'protocol_sha256':digest},f)
 report={'id':ID,'status':'STARTED','protocol_sha256':digest,'export_bytes_predebited':0,'export_bytes_verified':0};start=time.monotonic();old=signal.getsignal(signal.SIGALRM)
 def timeout(*_):raise TimeoutError('Audit wall exceeded')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(LIMITS['wall_seconds'])
 try:
  for x in p['repo_sources']:bind(ROOT,x)
  os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
  def charge(n):
   if type(n)!=int or n<0 or report['export_bytes_predebited']+n>LIMITS['export_bytes_hashed_max']:raise ValueError('Byte budget exceeded')
   report['export_bytes_predebited']+=n;atomic(out/'RUN.json',report)
  cells=audit_parts(hf,p['parts'],p['original_protocol_sha256'],charge);report['export_bytes_verified']=sum(x['export']['bytes'] for x in cells)
  manifest={'schema':'ffnds.compact6016.partial_completed_cells.v1','status':'AUDITED_COMPLETE_CELLS_FROM_FAILED_PARTS','source_scope':'historically exposed heldout8; high expert prefix only; not full80/video','parts':p['parts'],'original_protocol_sha256':p['original_protocol_sha256'],'cells':cells}
  atomic(out/'MANIFEST.json',manifest)
  if (out/'MANIFEST.json').stat().st_size>LIMITS['output_bytes']:raise ValueError('Output exceeds budget')
  report.update(status='COMPLETE_PARTIAL_ASSET_AUDIT',manifest_sha256=sha(out/'MANIFEST.json'),cells=40,functions=20,elapsed_seconds=time.monotonic()-start)
 except BaseException as e:report.update(status='FAILED_NO_RETRY',error_type=type(e).__name__,error=str(e));raise
 finally:signal.alarm(0);signal.signal(signal.SIGALRM,old);atomic(out/'RUN.json',report)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--protocol',type=Path,required=True);a.add_argument('--hf-root',type=Path,required=True);run(a.parse_args())
