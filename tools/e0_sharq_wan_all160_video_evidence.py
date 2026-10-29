"""Strict record-level closure predicates for a future registered video owner.

No I/O and no authorization here. The owner must separately bind actual RUN,
EXIT, protocol, source, state/video files and exact supervisor SHA, and must
actually obtain the evidence represented by these fields.
"""
from e0_sharq_wan_all160_video_driver import STATE_NAMES

def need(x,msg):
 if not x:raise ValueError(msg)

def verify_record(plan,run,receipt,run_sha,protocol_sha):
 need(plan['id']==run['id']==receipt['id'] and run['protocol_sha256']==receipt['protocol_sha256']==protocol_sha and receipt['run_sha256']==run_sha and receipt['process_exited'] is True and receipt['child_returncode']==0 and receipt['timeout'] is False and receipt['automatic_retry'] is False and receipt['spec_sha256']==run['job_spec_sha256'],'actual EXIT0/protocol/supervisor identity differs')
 need(run['status']=='COMPLETE_SHARQ_WAN_ALL160_EXIT_PENDING' and run['source_ordinal']==plan['case']['native_source_ordinal'] and run['seed']==42 and run['quality_acceptance'] is False,'result identity differs')
 phases=run['phases'];need(len(phases)==4 and [p['mode'] for p in phases]==['bf16','bf16_cache','dense32','sharq'] and len({p['run_id'] for p in phases})==4,'phase identities differ')
 def ident(x):return tuple(x[k] for k in ('sha256','bytes','raw_tensor_sha256','shape','dtype'))
 for p in phases:
  need(p['actual_full_FFN_sites']==160 and p['actual_FFN_linear_dispatches']==320 and set(p['states'])==STATE_NAMES and p['original_prefetch']['original_FFN_prefetch_observations']==162 and p['original_prefetch']['original_FFN_prefetch_skips']==0,'site/prefetch/state proof differs')
  video=p['video'];need(video['frames']==81 and video['fps']==16 and video['size']==[832,464] and video['bytes']>0 and len(video['sha256'])==64,'video identity differs')
 need(all(ident(phases[0]['states'][k])==ident(phases[1]['states'][k]) for k in STATE_NAMES) and all(phases[0]['video'][k]==phases[1]['video'][k] for k in ('bytes','sha256')),'same-device BF16 neutral differs')
 need(all(ident(p['states'][k])==ident(phases[1]['states'][k]) for p in phases[2:] for k in ('raw_noise_FP32','initial_noise')),'candidate noise differs')
 need([p['weight_cache']['cache_entries'] for p in phases]==[0,160,160,160],'same-master packed cache identity differs')
 expected={'video_generations':4,'original_FFN_prefetch_observations':648,'sharq_static_W_materializations':160,'sharq_static_W_quantizations':160,'dense32_prepare':320,'dense32_gemm':320,'sharq_prepare':320,'sparse_gemm':320,'dense_residual_gemm':320,'output_scale':640,'bias_add':640}
 for side in ('predebited','observed'):
  need(all(run['operation_ledger'][side].get(k)==v for k,v in expected.items()),'operation ledger incomplete')
 need(run['GPU_monitor']['sampled_isolation_pass'] is True and not run['GPU_monitor']['violations'],'GPU monitor differs')
 return {'status':'RECORD_PREDICATES_PASS_REQUIRES_ARTIFACT_BINDINGS','id':plan['id'],'quality_acceptance':False}
