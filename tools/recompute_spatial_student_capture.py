"""Rebuild the closed spatial capture inventory from actual immutable receipts.

Metadata-only audit: producer tensor validation is reported with its identity;
this tool does not pretend to re-read all72GB of tensor values.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/'tools') not in sys.path:sys.path.insert(0,str(ROOT/'tools'))
import e0_s2_spatial_prefix_replay as replay
native,core=replay.native,replay.core

def rebuild(hf):
 parents=[];parts=[]
 for i in (0,1):
  path=ROOT/f'config/e0_s2_spatial_student_collect_part{i}_20261029_v2.json';plan=native.read(path);out=hf/plan['output_HF_relative'];run=native.read(out/'RUN.json');ex=native.read(out.parent/'EXIT_RECEIPT.json');proof={'protocol':native.binding(path,ROOT),'RUN':native.binding(out/'RUN.json',hf),'exit':native.binding(out.parent/'EXIT_RECEIPT.json',hf)}
  core.vm.exited(ex,proof['RUN'],proof['protocol']['sha256'],plan['id']);parents.append(proof)
  original=native.read(ROOT/plan['migration']['cancelled_V1_protocol']['path']);migration=plan['migration'];cancel=native.read(native.bound(hf,migration['queue_cancellation']));cancel_exit=native.read(native.bound(hf,migration['actual_queue_exit']))
  native.need(cancel['model_process_started'] is False and cancel_exit['child_returncode']==-15 and cancel_exit['run_sha256'] is None,'Original idle-queue cancellation changed')
  phase_rows=[]
  for phase in run['phases']:
   phase_rows.append({k:phase[k] for k in ('mode','source_state','status','completed_native_flow_steps','capture_wall_seconds','FFN_boundary_CUDA_seconds','peak_cuda_reserved_bytes','all_ten_prior_video_state_bytes_equal','observation_receipt')})
  parts.append({'part':i,'source':plan['cases'][0]['native_source_ordinal'],'source_id':plan['cases'][0]['source_id'],'image_sha256':plan['cases'][0]['image']['sha256'],'prompt_sha256':plan['cases'][0]['prompt_sha256'],'producer':proof,'status':run['status'],'actual_child_exit':ex['child_returncode'],'actual_outer_seconds':ex['outer_seconds'],'actual_worker_seconds':run['wall_seconds'],'geometry':{k:v for k,v in plan['geometry'].items() if k!='selected_rows'},'row_order_sha256':native.canonical(plan['geometry']['selected_rows']),'actual_packets':run['actual_native_packets'],'actual_tensor_payload_bytes':run['actual_packet_tensor_payload_bytes'],'phases':phase_rows,'actual_operations':run['operation_ledger']['observed'],'resources_no_violations':not run['assigned_GPU_resources']['violations'],'runtime_precision':run['runtime_precision'],'output_accounting':run['output_accounting'],'migration':migration,'cancelled_prior_scope_id':original['id']})
 # Independently reconstruct all160 row/source/header identities from closed
 # receipts. This is deliberately metadata, not an unregistered tensor scan.
 indices=[replay.collect_index(parents,hf,i) for i in (0,1)]
 native.need(sum(len(a) for ix in indices for c in ix['cases'] for a in c['arms'].values())==960,'Full960 unique expert-part packets required')
 return {'status':'COMPLETE_TWO_SOURCE_THREE_ARM_FULL160_SPATIAL_CAPTURE_ACTUAL_EXIT0','evidence':'NATIVE_PASSIVE_CAPTURE_INVENTORY_NOT_METHOD_ACCEPTANCE','independent_images':2,'exposure':'PREVIOUSLY_EXPOSED_EXTREME_DENSE_MOTION_DIAGNOSTICS_NOT_CONFIRMATION','arms':['dense','value10','dcac10'],'seed':42,'all160_positions_per_source_arm':True,'latent_time_slots':21,'spatial_anchors_per_source':52,'rows_per_packet':1092,'actual_packets':960,'actual_tensor_payload_bytes':sum(p['actual_tensor_payload_bytes'] for p in parts),'producer_tensor_SHA_checks':8640,'producer_old_state_byte_neutrals':60,'producer_full_native_prefixes_no_decode':6,'producer_full_model_flows':24,'producer_FFN_sites':960,'new_VAE_decodes':0,'new_videos':0,'new_fits':0,'new_official_quality_calls':0,'new_quality_or_speed_acceptance':False,'tensor_values_re_read_by_this_metadata_audit':0,'metadata_index_sha256_by_expert':[native.canonical(ix) for ix in indices],'not_captured':['PRE_GELU_A','ACTUAL_ATTENTION_CONSUMER_QKV_OR_RESPONSE','LATENT_MOTION_CORRESPONDENCE','SUBJECT_OR_OCCLUSION_MASK'],'parts':parts}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--hf-root',type=Path,default=Path('/home/xiang/.cache/huggingface'));p.add_argument('--out',type=Path,required=True);a=p.parse_args();result=rebuild(a.hf_root);native.write(a.out,result);print(json.dumps({k:v for k,v in result.items() if k!='parts'},ensure_ascii=False))
if __name__=='__main__':main()
