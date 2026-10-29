"""Five-phase ABBA driver; injected callbacks must predebit all real work.

No entry point, registration or execution authority. Numeric data and GPU
operations are only possible via a separately reviewed and registered owner.
"""
MODES=('bf16_cache','dense32','dense32','dense32','dense32')
ARMS=('warmup','legacy_A1','resident_B1','resident_B2','legacy_A2')

def need(x,msg):
 if not x:raise ValueError(msg)

def run(*,runtime,phase,validate,checkpoint,prior,progress):
 """phase(index, mode, arm) must do one complete supervised generation.

validate(arm, artifact, expected) must pay actual full byte comparisons.
checkpoint() must synchronize, verify actual cache identity, and return a
stable immutable binding; changing cache between phases fails closed.
"""
 results=[];cache=None
 for i,(mode,arm) in enumerate(zip(MODES,ARMS)):
  resident=arm.startswith('resident')
  if i:
   current=checkpoint();need(current==cache,'cache changed before phase')
   runtime.set_next_resident(resident)
  progress('phase_started',{'index':i,'arm':arm,'mode':mode})
  result=phase(i,mode,arm);need(result['mode']==mode and result['arm']==arm and result['ffn_sites']==160 and result['linear_sites']==320,'phase coverage')
  progress('phase_evidence_complete',result);results.append(result)
  if i==0:
   cache=checkpoint();need(cache['entries']==160,'incomplete weight cache')
   validate(arm,result,prior['bf16'])
  else:
   need(checkpoint()==cache,'cache changed during phase')
   validate(arm,result,prior['dense32'])
 return {'phases':results,'cache':cache,'timing_scope':'ABBA_ONLY_WARMUP_AND_STATIC_PACK_SEPARATELY_PAID','quality_acceptance':False}

def native_phase(*,torch,pipe,guard,runtime,vm,rid,mode,arm,image_path,prompt,output_path,save_state,check,authorize,emit,charge_block,PrefetchAudit,phase_synchronize):
 """One actual native phase when invoked by an independently authorized owner.

All callbacks are owner-supplied and must enforce exact finite budgets. No
execution occurs until this function is called by such an owner.
"""
 import time
 from types import SimpleNamespace
 authorize();check();runtime.begin(rid,mode)
 tap=vm.LightTap(pipe,guard,rid,save_state,check,runtime=SimpleNamespace(mode='dense'))
 entries=[(buf,e) for e,model in enumerate(pipe.runner.model.model) for buf in model.transformer_infer.offload_manager.cuda_buffers]
 audit=PrefetchAudit(entries,charge_block,emit);pipe.runner.scheduler.generator=None;started=time.perf_counter()
 def generate():
  with tap:
   with audit:
    with torch.inference_mode():pipe.generate(seed=42,image_path=str(image_path),prompt=prompt,negative_prompt='',save_result_path=str(output_path))
  phase_synchronize();tap.finish()
 guard.ledger.call(rid,{'video_generations':1},generate)
 load=audit.finish();dispatch=runtime.finish()
 need(len(tap.events)==160 and len(tap.transfers)==162,'native event/prefetch coverage')
 return {'mode':mode,'arm':arm,'ffn_sites':160,'linear_sites':320,'wall_seconds_including_observers':time.perf_counter()-started,
  'FFN_boundary_CUDA_seconds':sum(a.elapsed_time(b) for a,b in tap.events)/1000,'block_prefetch':load,'weight_cache':dispatch,
  'legacy_transfer_inspector':'SOURCE_REQUESTED_METADATA_ONLY_NOT_PROOF_OF_W_COPY','physical_bus_bytes':'UNKNOWN'}
