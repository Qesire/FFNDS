"""One registered native-video phase for the SharQ full-depth runtime.

The caller is the separately reviewed owner of source/checkpoint proof,
ACTIVE authorization, worker/job isolation, complete ledger, persisted
states/video receipt, time and disk budgets. This library is deliberately not
an executable CLI and cannot authorize itself. It does perform the real
pipeline call when invoked by such a registered owner.
"""
import time
from contextlib import ExitStack
from types import SimpleNamespace
from e0_sharq_wan_all160_runtime_v2 import OriginalFFNPrefetchAudit,need


def run_clip(*,torch,pipe,guard,runtime,vm,rid,mode,image_path,prompt,output_path,
             save_state,check,authorize,emit):
    need(mode in ('bf16','bf16_cache','dense32','sharq') and rid and callable(authorize),'phase identity differs')
    authorize();check();runtime.begin(rid,mode)
    # LightTap's legacy transfer inspector uses mode='dense' to inspect the
    # retained original W2 copy. The actual compute mode remains in runtime.
    tap=vm.LightTap(pipe,guard,rid,save_state,check,runtime=SimpleNamespace(mode='dense'))
    audit=OriginalFFNPrefetchAudit(pipe,torch,guard.ledger,rid,emit=emit)
    pipe.runner.scheduler.generator=None;started=time.perf_counter()
    def generate():
        with tap:
            with audit:
                with torch.inference_mode():
                    pipe.generate(seed=42,image_path=str(image_path),prompt=prompt,negative_prompt='',save_result_path=str(output_path))
        torch.cuda.synchronize();tap.finish()
    try:
        guard.ledger.call(rid,{'video_generations':1},generate)
        prefetch=audit.finish();dispatch=runtime.finish()
        need(len(tap.events)==160 and len(tap.transfers)==162,'native complete coverage missing')
        return {'run_id':rid,'mode':mode,'actual_full_FFN_sites':160,'actual_FFN_linear_dispatches':320,
                'wall_seconds_including_observers':time.perf_counter()-started,
                'FFN_boundary_CUDA_seconds':sum(a.elapsed_time(b) for a,b in tap.events)/1000,
                'original_prefetch':prefetch,'weight_cache':dispatch,
                'full_depth_video_status':'GENERATED_NOT_YET_CLOSED_OR_QUALITY_ACCEPTED'}
    except BaseException:
        # The outer owner persists the failure ledger and partial output; do
        # not reset/retry or declare any partial video successful here.
        emit({'event':'phase_exception','run_id':rid,'mode':mode,'wall_seconds':time.perf_counter()-started,
              'runtime_phase_still_active':runtime.rid==rid,'entered_FFN_linear_sites':len(runtime.seen),
              'completed_static_weight_packs':len(runtime.cache),'completed_static_cache_bytes':runtime.cache_bytes,
              'inflight_pack_or_GPU_completion':'UNKNOWN; use predebit/return journal'})
        raise


STATE_NAMES=frozenset(('raw_noise_FP32','initial_noise',*(f'flow_step{i}' for i in range(4)),*(f'latent_step{i}' for i in range(4))))

def execute_part(*,torch,pipe,guard,runtime,vm,run_ids,image_path,prompt,output_paths,
                 capture_factory,inspect_video,check,authorize,emit,gate=None,progress=lambda *_:None):
    """Run paired old/new BF16, dense32, SharQ on one registered source.

    The owner must journal and persist each state and inspect actual video bytes
    in the supplied callbacks. All callbacks are budgeted by that owner. This
    function refuses to advance past the two-video neutral until it matches.
    """
    modes=('bf16','bf16_cache','dense32','sharq')
    need(len(run_ids)==len(output_paths)==4 and len(set(run_ids))==4 and len(set(map(str,output_paths)))==4,'four distinct registered outputs required')
    results=[]
    if gate is None:gate=lambda _rid,_cost,fn:fn()
    def identity(item):return tuple(item[k] for k in ('sha256','bytes','raw_tensor_sha256','shape','dtype'))
    with ExitStack() as stack:
        for index,(rid,mode,path) in enumerate(zip(run_ids,modes,output_paths)):
            if index==1:stack.enter_context(runtime)
            progress('phase_started',{'run_id':rid,'mode':mode});capture,states=capture_factory(rid)
            phase=run_clip(torch=torch,pipe=pipe,guard=guard,runtime=runtime,vm=vm,rid=rid,mode=mode,image_path=image_path,prompt=prompt,output_path=path,save_state=capture,check=check,authorize=authorize,emit=emit)
            need(set(states)==STATE_NAMES,'ten actual state identities required')
            video=inspect_video(rid,path);need(video.get('frames')==81 and video.get('fps')==16 and video.get('size')==[832,464] and len(video.get('sha256',''))==64,'actual video geometry/identity differs')
            phase['states']=states;phase['video']=video;results.append(phase);progress('phase_evidence_complete',phase);emit({'event':'phase_evidence_complete','run_id':rid,'mode':mode})
            if index==1:
                def neutral():need(all(identity(results[0]['states'][k])==identity(states[k]) for k in STATE_NAMES) and results[0]['video']['sha256']==video['sha256'] and results[0]['video']['bytes']==video['bytes'],'same-device BF16 neutral differs')
                gate(rid,{'neutral_state_checks':10,'neutral_video_checks':1},neutral)
            if index>=2:
                def paired():need(all(identity(results[1]['states'][k])==identity(states[k]) for k in ('raw_noise_FP32','initial_noise')),'paired candidate noise differs')
                gate(rid,{'noise_pair_checks':2},paired)
    return {'status':'FOUR_VIDEOS_COMPLETE_OWNER_CLOSURE_PENDING','phases':results,'same_device_BF16_neutral':True,'paired_noise':True,'quality_acceptance':False}
