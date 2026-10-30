"""Five actual-video phases; no authority or model execution at import."""
import time
from types import SimpleNamespace
from chipmunk_student_observation_tap import make_tap,need,memory_snapshot
STATE_NAMES=frozenset(('raw_noise_FP32','initial_noise',*(f'flow_step{i}' for i in range(4)),*(f'latent_step{i}' for i in range(4))))
MODES=('h_cache_dense','high_h_readonly_p64')
def execute(*,observer_factory,torch,pipe,guard,vm,native,run_ids,image_path,prompt,output_paths,capture_factory,inspect_video,check,authorize,gate,progress,kernels,emit):
 need(len(run_ids)==len(output_paths)==2 and len(set(run_ids))==len(set(map(str,output_paths)))==2,'two distinct runs required')
 def make_rows():
  cpu=torch.randperm(31668,generator=torch.Generator(device='cpu').manual_seed(42))[:64];selector_list=cpu.tolist();rows=cpu.to('cuda');torch.cuda.synchronize();return rows,selector_list
 rows,selector_list=guard.ledger.call('setup',{'p64_row_seed_and_H2D':1},make_rows)
 phases=[]
 for i,mode in enumerate(MODES):
  rid=run_ids[i];authorize(rid);check();emit({'event':'phase_memory_before','run_id':rid,'mode':mode,'memory':memory_snapshot(torch)});progress('phase_started',{'run_id':rid,'mode':mode});save,states=capture_factory(rid)
  observer=observer_factory(mode,rows,selector_list);tap_cls=make_tap(vm,native,mode=mode,rows=rows,kernels=kernels,emit=emit,observer=observer)
  tap=tap_cls(pipe,guard,rid,save,check,runtime=SimpleNamespace(mode='dense'));pipe.runner.scheduler.generator=None;started=time.perf_counter()
  def generate():
   with tap,torch.inference_mode():pipe.generate(seed=42,image_path=str(image_path),prompt=prompt,negative_prompt='',save_result_path=str(output_paths[i]))
   torch.cuda.synchronize();tap.finish()
  guard.ledger.call(rid,{'video_generations':1},generate)
  need(len(tap.events)==160 and len(tap.transfers)==162 and set(states)==STATE_NAMES,'missing FFN/prefetch/states');video=inspect_video(rid,output_paths[i]);need(video.get('frames')==81 and video.get('fps')==16 and video.get('size')==[832,464] and len(video.get('sha256',''))==64,'video geometry')
  phase={'run_id':rid,'mode':mode,'actual_full_FFN_sites':160,'logical_FFN_linear_dispatches':320,'states':states,'video':video,'wall_seconds_including_observers':time.perf_counter()-started,'FFN_boundary_CUDA_seconds':sum(a.elapsed_time(b) for a,b in tap.events)/1000,'native_transfer_observations':len(tap.transfers),'actual_dense_FFN_GEMMs':240 if mode=='high_h_readonly_p64' else 320,'actual_selected_W1_calls':40 if mode=='high_h_readonly_p64' else 0,'actual_selected_W2_calls':40 if mode=='high_h_readonly_p64' else 0,'actual_proxy_W1_calls':40 if mode=='high_h_readonly_p64' else 0,'observer_packed_W1_calls':0,'observer_packed_W2_calls':0,'actual_extra_old_W1_calls':40 if mode=='high_x_recompute_p64' else 0,'cumulative_peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated(),'cumulative_peak_GPU_reserved_bytes':torch.cuda.max_memory_reserved()}
  phase['original_prefetch']={'calls':len(tap.transfers),'requested_H2D_tensor_bytes':sum(x['requested_H2D_tensor_bytes'] for x in tap.transfers),'requested_D2D_tensor_bytes':sum(x['requested_D2D_tensor_bytes'] for x in tap.transfers),'original_W2_H2D_bytes':sum(x['original_W2_H2D_bytes'] for x in tap.transfers),'enqueue_wall_seconds':sum(x['enqueue_wall_seconds'] for x in tap.transfers),'metadata_observer_wall_seconds':sum(x['metadata_observer_wall_seconds'] for x in tap.transfers),'actual_physical_bus_bytes':'UNKNOWN'}
  phase['observer_manifest']=observer.sink.finalize()
  phase['stages']=[{'name':x['stage'],'step':x['step'],'wall_seconds':x['wall_seconds'],'CUDA_seconds':x['start'].elapsed_time(x['end'])/1000} for x in tap.stages]
  phase.update(cache_peak_logical_bytes=tap.book.peak,cache_creations=tap.cache_puts,cache_releases=tap.cache_releases,sparse_sites=tap.sparse_sites)
  emit({'event':'phase_memory_after','run_id':rid,'mode':mode,'memory':memory_snapshot(torch)});phases.append(phase);progress('phase_evidence_complete',phase)
 return {'phases':phases,'quality_acceptance':False}
