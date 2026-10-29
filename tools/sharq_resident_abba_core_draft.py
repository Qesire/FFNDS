"""ABBA generation core for a separately authorized durable owner.

No CLI or execution authorization. The caller must provide ACTIVE-verified
worker, finite prepaid ledger, state/video writing and GPU lifecycle controls.
"""
from sharq_resident_dense32_driver_draft import run,native_phase,need
from sharq_resident_cache_checkpoint_draft import cache_checkpoint
from sharq_resident_prefetch_audit_draft import PrefetchAudit
from sharq_resident_abba_coverage_draft import Coverage
from sharq_resident_reference_gate_draft import ReferenceGate,STATES
from pathlib import Path

def execute(*,torch,pipe,guard,runtime,vm,run_ids,image_path,prompt,output_paths,hf,prior,save_factory,inspect_video,check,authorize,charge_block,charge_compare,persist,emit,synchronize,phase_synchronize):
 need(len(run_ids)==len(output_paths)==5 and len(set(run_ids))==len(set(map(str,output_paths)))==5,'five distinct video phases required')
 reference=ReferenceGate(hf,prior,charge_compare,lambda state:persist('reference_gate',state));coverage=Coverage(lambda state:persist('prefetch_coverage',state))
 def event(row):coverage.event(row);emit(row)
 def do_phase(index,mode,arm):
  rid=run_ids[index];coverage.begin(arm);save,states=save_factory(rid)
  result=native_phase(torch=torch,pipe=pipe,guard=guard,runtime=runtime,vm=vm,rid=rid,mode=mode,arm=arm,image_path=image_path,prompt=prompt,output_path=output_paths[index],save_state=save,check=check,authorize=lambda:authorize(rid),emit=event,charge_block=charge_block,PrefetchAudit=PrefetchAudit,phase_synchronize=phase_synchronize)
  coverage.finish();need(set(states)==STATES,'state set differs');video=inspect_video(rid,output_paths[index]);need(video.get('frames')==81 and video.get('fps')==16 and video.get('size')==[832,464],'video geometry differs')
  expected_dir=Path(output_paths[index]).resolve().parent
  need(Path(video['path']).resolve()==Path(output_paths[index]).resolve(),'video output path differs')
  paths=[(Path(hf)/b['path']).resolve() for b in states.values()]
  need(all((Path(hf)/b['path']).resolve()==expected_dir/(name+'.safetensors') for name,b in states.items()),'state path differs')
  ids={(p.stat().st_dev,p.stat().st_ino) for p in paths};vpath=Path(output_paths[index]).resolve();vstat=vpath.stat();need(len(ids)==len(paths)==len(STATES) and (vstat.st_dev,vstat.st_ino) not in ids,'state/video alias')
  result.update(run_id=rid,states=states,video=video);return result
 def validate(arm,result,expected):
  for name in sorted(STATES):reference.check(arm,name,result['states'][name])
  reference.check(arm,'video',result['video'])
 def checkpoint():synchronize();return cache_checkpoint(runtime.cache)
 previous_observer=runtime._selector.observe;runtime._selector.observe=event
 try:
  result=run(runtime=runtime,phase=do_phase,validate=validate,checkpoint=checkpoint,prior={'bf16':prior['bf16'],'dense32':prior['dense32']},progress=lambda event,data:persist(event,data))
  result['reference_gate']=reference.finish();result['prefetch_coverage']=coverage.snapshot();return result
 finally:runtime._selector.observe=previous_observer
