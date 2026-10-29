"""Metadata-only selection for the SharQ paired-source asset producer.

This plan never reads source tensors or weights. The future CPU asset producer
must separately pin master receipts, range-read/hashes, budgets and worker,
then receive independent review before producing an actual bundle.
"""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BRIDGE=ROOT/'research/20261029/S2_WHOLE_FFN_NATIVE_BOUNDARY_BRIDGE_RESULTS_20261029.json'
RAW=ROOT/'config/e0_s2_new12_sm120_fixed_depth_part0_20261029_v1.json'
NV=ROOT/'research/20261029/NVFP4_CHECKPOINT_BYTES_RESULTS_20261029.json'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def select(bridge,raw,nv):
 if bridge.get('status')!='COMPLETE_NATIVE_WHOLE_FFN_BOUNDARY_BRIDGE' or bridge.get('closed_verification',{}).get('status')!='PASS_CLOSED_NATIVE_BOUNDARY_BRIDGE':raise ValueError('bridge parent is not closed')
 selected={}
 for step,expert in ((1,'high'),(2,'low')):
  rows=[p for p in bridge['packets'] if p['run_id']=='whole_ffn_bridge_ord911_seed42_dense' and p['step']==step and p['block']==20]
  if len(rows)!=1:raise ValueError('paired source gate missing')
  packet=rows[0]
  if set(('V1','a','h','Y','gate','rows'))-set(packet['tensor_metadata']):raise ValueError('native boundary metadata missing')
  rawmodels=[m for m in raw['models'] if m['kind']==expert]
  nvmodels=[m for m in nv['files'] if f'_{expert}.' in m['filename']]
  if len(rawmodels)!=1 or len(nvmodels)!=1:raise ValueError('expert checkpoint missing')
  selected[expert]={'source':911,'step':step,'block':20,'rows':32,'packet':{'payload':packet['payload'],'metadata':packet['metadata'],'tensor_metadata':{k:packet['tensor_metadata'][k] for k in ('V1','a','h','Y','gate','rows')}},'original_bf16':rawmodels[0],'nvfp4_checkpoint':nvmodels[0]}
 return selected

def plan():
 bridge,raw,nv=read(BRIDGE),read(RAW),read(NV)
 if nv.get('status')!='ACTUAL_EXIT0_CLOSED':
  # Historical status uses its exact recorded spelling, not a generic claim.
  if not (nv.get('actual_exit')==0 and nv.get('GPU_calls')==0 and nv.get('model_calls')==0):raise ValueError('NVFP4 byte proof incomplete')
 return {'status':'METADATA_ONLY_NOT_REGISTERED_NOT_ASSETS','selected':select(bridge,raw,nv),'inputs':{str(p.relative_to(ROOT)):digest(p) for p in (BRIDGE,RAW,NV)},'bundle_sha256':None,'source_ranges_sha256':'NOT_BOUND_UNTIL_ACTUAL_CPU_ASSET_PRODUCER'}
if __name__=='__main__':print(json.dumps(plan(),indent=2))
