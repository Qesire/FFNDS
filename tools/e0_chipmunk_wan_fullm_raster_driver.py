"""One source, two same-device original-BF16 videos; one passive raster tap."""
from types import SimpleNamespace
import time
from e0_chipmunk_wan_fullm_raster_capture import make_tap_class,need
STATE_NAMES=frozenset(('raw_noise_FP32','initial_noise',*(f'flow_step{i}' for i in range(4)),*(f'latent_step{i}' for i in range(4))))
def execute(*,torch,pipe,guard,vm,native,run_ids,image_path,prompt,output_paths,capture_factory,transaction,inspect_video,check,authorize,gate,progress):
 need(len(run_ids)==len(output_paths)==2 and len(set(run_ids))==2 and len(set(map(str,output_paths)))==2,'two fresh videos required');phases=[]
 for i in range(2):
  rid=run_ids[i];mode=('original_dense','passive_raster')[i];authorize(rid);check();progress('phase_started',{'run_id':rid,'mode':mode});save,states=capture_factory(rid);tap_cls=vm.LightTap if i==0 else make_tap_class(vm,native);args={'runtime':SimpleNamespace(mode='dense')}
  if i:args['capture']=transaction(rid)
  tap=tap_cls(pipe,guard,rid,save,check,**args);pipe.runner.scheduler.generator=None;started=time.perf_counter()
  def generate():
   with tap,torch.inference_mode():pipe.generate(seed=42,image_path=str(image_path),prompt=prompt,negative_prompt='',save_result_path=str(output_paths[i]))
   torch.cuda.synchronize();tap.finish()
  guard.ledger.call(rid,{'video_generations':1},generate);need(len(tap.events)==160 and len(tap.transfers)==162 and set(states)==STATE_NAMES,'full depth or states missing');video=inspect_video(rid,output_paths[i]);need(video.get('frames')==81 and video.get('fps')==16 and video.get('size')==[832,464] and len(video.get('sha256',''))==64,'video differs')
  phase={'run_id':rid,'mode':mode,'actual_full_FFN_sites':160,'actual_FFN_linear_dispatches':320,'states':states,'video':video,'wall_seconds_including_observers':time.perf_counter()-started,'FFN_boundary_CUDA_seconds':sum(a.elapsed_time(b) for a,b in tap.events)/1000,'native_transfer_observations':len(tap.transfers)};phases.append(phase);progress('phase_evidence_complete',phase)
  if i:
   def neutral():
    def identity(x):return tuple(x[k] for k in ('sha256','bytes','raw_tensor_sha256','shape','dtype'))
    need(all(identity(phases[0]['states'][k])==identity(states[k]) for k in STATE_NAMES) and phases[0]['video']['sha256']==video['sha256'] and phases[0]['video']['bytes']==video['bytes'],'same-device passive neutral differs')
   gate(rid,{'neutral_state_checks':10,'neutral_video_checks':1},neutral)
 return {'phases':phases,'same_device_BF16_neutral':True,'quality_acceptance':False}
