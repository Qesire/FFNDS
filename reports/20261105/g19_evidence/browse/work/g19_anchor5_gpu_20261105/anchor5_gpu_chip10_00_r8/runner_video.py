"""Original-model 40-step Dense/neutral/anchored video producer; reviewed scope only."""
import argparse,hashlib,json,sys,time,subprocess,socket
from pathlib import Path
from runtime_guard import require_compute_allocation,verify_frozen_scope,RunReceipt,sha256_file,process_peak_rss_bytes

def tensor_digest(t,torch):
    t=t.detach().contiguous()
    if not torch.isfinite(t).all().item(): raise RuntimeError('nonfinite tensor')
    return {'sha256':hashlib.sha256(t.view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),'shape':list(t.shape),'dtype':str(t.dtype)}

def video_digest(path, geometry):
    import imageio_ffmpeg
    reader=imageio_ffmpeg.read_frames(str(path),pix_fmt='rgb24'); meta=next(reader); w,h=meta['size']; fps=float(meta['fps'])
    digest=hashlib.sha256(); count=0
    try:
        for frame in reader:
            if len(frame)!=w*h*3: raise RuntimeError('bad decoded frame')
            digest.update(frame); count+=1
    finally: reader.close()
    if [w,h,count]!=[geometry['width'],geometry['height'],geometry['frames']] or abs(fps-geometry['fps'])>0.01: raise RuntimeError('video geometry')
    return {'frames':count,'width':w,'height':h,'fps':fps,'decoded_rgb24_sha256':digest.hexdigest(),'mp4_sha256':sha256_file(path)}

def main():
    p=argparse.ArgumentParser()
    for opt in ('scope','scope-sha','run-id','output'): p.add_argument('--'+opt,required=True)
    a=p.parse_args(); root=Path(__file__).resolve().parent; allocation=require_compute_allocation(); scope=verify_frozen_scope(root,a.scope,a.scope_sha)
    if scope['phase']!='ORIGINAL40_GPU_ANCHOR5' or scope['mode']!='ANCHOR5_GPU_THREE_PASS' or scope['candidate_modes']!=['dense','gpu_bf16_anchor5','gpu_fp32_anchor5']: raise RuntimeError('scope mode')
    if scope.get('fixed_node')!='anode22' or socket.gethostname().split('.')[0]!='anode22': raise RuntimeError('wrong compute node')
    if scope['seed']!=43 or scope['run_ids']!=[a.run_id] or a.run_id!='chip10_00' or Path(a.output).resolve()!=Path(scope['output_dir']).resolve(): raise RuntimeError('scope input identity')
    if not 0<scope['budget']['cache_gpu_bytes']<=60*1024**3: raise RuntimeError('GPU cache limit')
    if scope['matrix_manifest_file'] not in scope.get('frozen_inputs_sha256',{}): raise RuntimeError('matrix manifest not bound to scope')
    if Path(sys.executable).resolve()!=Path(scope['python_executable']).resolve() or allocation['memory_mib']>scope['budget']['host_memory_gib']*1024: raise RuntimeError('runtime allocation mismatch')
    for tree in scope['python_source_roots']:
        d=Path(tree).resolve(); actual={str(p.resolve()) for p in d.rglob('*.py') if p.is_file()}; expected={str(Path(p).resolve()) for p in scope['external_sources_sha256'] if p.endswith('.py') and Path(p).resolve().is_relative_to(d)}
        if not actual or actual!=expected: raise RuntimeError('source tree inventory mismatch')
    with RunReceipt(a.output,scope,allocation) as receipt:
        from resource_limits import Limits
        limits=Limits(receipt.path,host_soft=176*1024**3,output_cap=scope['budget']['total_output_bytes'],soft_seconds=scope['budget']['soft_seconds']); limits.check()
        def queue_check():
            raw=subprocess.run(['squeue','-h','-u',allocation['user'],'-o','%i %T'],capture_output=True,text=True,encoding='utf8',errors='replace',check=True,timeout=30).stdout
            ids=[line.split()[0] for line in raw.splitlines() if line.split()]
            if ids!=[allocation['job_id']]:raise RuntimeError('unexpected user job queue: '+raw)
            return raw
        queue_start=queue_check()
        rows=json.loads((root/scope['input_manifest_file']).read_text(encoding='utf8'))
        if len(rows)!=10 or len({r['image_sha256'] for r in rows})!=10 or len({r['image_cluster'] for r in rows})!=10: raise RuntimeError('source cohort')
        matches=[r for r in rows if r['run_id']==a.run_id]
        if len(matches)!=1 or matches[0]['seed']!=43: raise RuntimeError('bad run identity')
        row=matches[0]
        if sha256_file(row['image_path'])!=row['image_sha256'] or hashlib.sha256(row['prompt'].encode()).hexdigest()!=row['prompt_sha256']: raise RuntimeError('input changed')
        base=Path(scope['base_model_dir']).resolve(); identities={}
        for rel,spec in scope['base_assets_sha256'].items():
            path=(base/rel).resolve(); before=path.stat()
            if not path.is_relative_to(base) or before.st_size!=spec['size'] or sha256_file(path)!=spec['sha256']: raise RuntimeError('base asset identity')
            after=path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns): raise RuntimeError('base asset changed')
            identities[str(path)]=(after.st_size,after.st_mtime_ns); receipt.check_deadline()
        expected={str((base/f'{expert}_noise_model'/f'diffusion_pytorch_model-{i:05d}-of-00006.safetensors').resolve()) for expert in ('high','low') for i in range(1,7)}
        if {str(Path(x['path']).resolve()) for x in scope['original_weight_files']}!=expected: raise RuntimeError('shard inventory')
        for spec in scope['original_weight_files']:
            path=Path(spec['path']); before=path.stat()
            if before.st_size!=spec['size'] or sha256_file(path)!=spec['sha256']: raise RuntimeError('original shard payload differs from CAL')
            after=path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns): raise RuntimeError('shard changed')
            identities[str(path.resolve())]=(after.st_size,after.st_mtime_ns); receipt.check_deadline()
        for path in reversed(scope['python_import_paths']): sys.path.insert(0,path)
        import torch,numpy,imageio_ffmpeg
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or torch.cuda.get_device_name(0)!='NVIDIA A100-SXM4-80GB': raise RuntimeError('GPU identity')
        v=scope['runtime_versions']
        if (torch.__version__,numpy.__version__,imageio_ffmpeg.__version__)!=(v['torch'],v['numpy'],v['imageio_ffmpeg']) or sha256_file(imageio_ffmpeg.get_ffmpeg_exe())!=scope['ffmpeg_sha256']: raise RuntimeError('runtime identity')
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
        frac=scope['budget']['gpu_memory_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory
        if not 0<frac<1: raise RuntimeError('GPU budget')
        torch.cuda.set_per_process_memory_fraction(frac)
        from normal_pipeline import build_normal_pipeline
        from anchored_hook import AnchoredHook
        from matrix_loader import MatrixLoader
        from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
        matrices=json.loads((root/scope['matrix_manifest_file']).read_text(encoding='utf8'))
        setup_start=time.monotonic(); loader=MatrixLoader(matrices,scope['matrix_provenance']); pipe,config=build_normal_pipeline(str(base),scope['required_runtime_config']); setup_seconds=time.monotonic()-setup_start
        from preflight_cache import run as cache_preflight
        preflights={}
        (receipt.path/'EFFECTIVE_CONFIG.json').write_text(json.dumps(dict(config),indent=2,default=str),encoding='utf8')
        scheduler=pipe.runner.scheduler; dispatch=pipe.runner.model; all_results=[]; progress=receipt.path/'PROGRESS.jsonl'
        def expert():
            i=int(dispatch.cur_model_index)
            if i not in (0,1): raise RuntimeError('expert unavailable')
            return ('high','low')[i]
        for mode in scope['candidate_modes']:
            receipt.check_deadline(); limits.check(); queue_before=queue_check()
            if mode!='dense':
                limits.check(host_reserve=8*1024**3); spec=scope['preflight_field']; sampleA=loader.load('high',0,torch.device('cuda',torch.cuda.current_device())); dtype=torch.bfloat16 if mode=='gpu_bf16_anchor5' else torch.float32
                preflight=cache_preflight(torch,spec['path'],spec,sha256_file,sampleA,dtype); preflights[mode]=preflight; del sampleA; loader.release(); torch.cuda.empty_cache(); limits.check(); (receipt.path/('PREFLIGHT_'+mode+'.json')).write_text(json.dumps(preflight,indent=2,allow_nan=False),encoding='utf8')
            arm_total_start=time.monotonic(); torch.cuda.reset_peak_memory_stats(); loader.release(); torch.cuda.synchronize(); torch.cuda.empty_cache(); baseline={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()}; loader.reset_load_stats(); latents=[]; noise=[]; model_latent=[]; calls=[]; events=[]; final_cost=[]; final_latents=[]; digest_seconds=[]; denoise=[]; video=receipt.path/(mode+'.mp4'); arm_setup_start=arm_total_start
            for native_model in dispatch.model:
                native_model.transformer_infer.offload_manager.need_init_first_buffer=True
            def emit(event):
                event['arm']=mode; events.append(event)
            old_prepare_latents=scheduler.prepare_latents
            def prepare_tap(*args,**kwargs):
                result=old_prepare_latents(*args,**kwargs); noise.append(tensor_digest(scheduler.latents,torch))
                if all_results and noise[0]!=all_results[0]['initial_noise']: raise RuntimeError('initial random latent mismatch')
                return result
            scheduler.prepare_latents=prepare_tap
            old_dispatch=dispatch.get_current_model_index
            def dispatch_tap():
                limits.check(); old_dispatch(); step=int(scheduler.step_index); print('ANCHOR5_STEP',mode,step,flush=True)
                if step==0:
                    if len(scheduler.timesteps)!=40 or not torch.isfinite(scheduler.timesteps).all(): raise RuntimeError('timetable')
                    model_latent.append(tensor_digest(scheduler.latents,torch))
                    if all_results and model_latent[0]!=all_results[0]['initial_model_latent']: raise RuntimeError('initial model latent mismatch before forward')
                    tap.timetable=tuple(float(x) for x in scheduler.timesteps.detach().cpu().tolist()); emit({'event':'TIMESTEPS','values':list(tap.timetable)}); torch.cuda.synchronize(); denoise.append(time.monotonic())
                calls.append({'step':step,'expert':expert(),'timestep':tap.timetable[step]})
            old_post=scheduler.step_post
            def post_tap():
                result=old_post(); step=int(scheduler.step_index)
                if step==39:
                    torch.cuda.synchronize(); denoise.append(time.monotonic()); before={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'a_entries':len(loader.cache)}; cost=tap.finish([x['expert'] for x in calls]); loader.release(); torch.cuda.synchronize(); after={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'a_entries':len(loader.cache)}; cost.update(gpu_before_release=before,gpu_after_release=after); final_cost.append(cost); diagnostic_start=time.monotonic(); final_latents.append(tensor_digest(scheduler.latents,torch)); digest_seconds.append(time.monotonic()-diagnostic_start); torch.cuda.empty_cache()
                return result
            dispatch.get_current_model_index=dispatch_tap; scheduler.step_post=post_tap
            tap=AnchoredHook(expert,lambda: dispatch.model[dispatch.cur_model_index].pre_infer.grid_sizes,loader.load,emit,mode=mode,cap_bytes=scope['budget']['cache_gpu_bytes'],shadow=False,seed=43)
            tap.install(WanTransformerInfer); scheduler.generator=None
            emit({'event':'GENERATOR_RESET','seed':43})
            try:
                limits.reserve_video(1024**3); torch.cuda.synchronize(); generate_start=time.monotonic()
                with torch.no_grad(): pipe.generate(seed=43,image_path=row['image_path'],prompt=row['prompt'],negative_prompt=row.get('negative_prompt',''),save_result_path=str(video))
                torch.cuda.synchronize(); generate_end=time.monotonic(); limits.release_video(); limits.check()
                if len(noise)!=1 or len(model_latent)!=1 or [x['step'] for x in calls]!=list(range(40)): raise RuntimeError('scheduler coverage')
                if tuple(float(x) for x in scheduler.timesteps.detach().cpu().tolist())!=tap.timetable: raise RuntimeError('timetable changed')
                if len(final_cost)!=1 or len(denoise)!=2 or len(final_latents)!=1: raise RuntimeError('final release/timing missing')
                cost=final_cost[0]
            finally: tap.uninstall(); dispatch.get_current_model_index=old_dispatch; scheduler.step_post=old_post; scheduler.prepare_latents=old_prepare_latents
            if sum(f.stat().st_size for f in receipt.path.iterdir() if f.is_file()) > scope['budget']['total_output_bytes']: raise RuntimeError('output budget exceeded')
            result={'arm':mode,'initial_noise':noise[0],'initial_noise_stage':'prepare_latents_after_draw','initial_model_latent':model_latent[0],'final_latent':final_latents[0],'latent_digests':latents,'expert_dispatch':calls,'ffn_site_count':len(tap.seen),'cache_accounting':cost,'process_peak_rss_bytes':process_peak_rss_bytes(),'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'gpu_arm_baseline':baseline,'resource_after':limits.snapshot(),'queue_before':queue_before,'queue_after':queue_check(),'wall_seconds':generate_end-generate_start,'arm_total_seconds':generate_end-arm_total_start,'arm_setup_seconds':generate_start-arm_setup_start,'wall_scope':'RAW_GENERATE_INCLUDES_FINAL_LATENT_DIAGNOSTIC','final_latent_digest_seconds':digest_seconds[0],'denoise_seconds':denoise[1]-denoise[0],'a_load_wall_seconds_included':loader.load_seconds,'a_load_count':loader.load_count,'step_hash_policy':'NO_STEP_HASH_IN_TIMED_PASS','timing_scope':'PILOT_WALL_WITH_NATIVE_FULL_PREFETCH_CLONE_GELU_AND_LAZY_A_LOAD',**video_digest(video,scope['decoded_video_geometry'])}
            limits.check()
            if loader.load_count!=(0 if mode=='dense' else 80): raise RuntimeError('A load count mismatch')
            if mode=='dense' and result['decoded_rgb24_sha256']!=scope['historical_dense_rgb_sha256']: raise RuntimeError('Dense differs from independently accepted historical 81-frame RGB')
            if all_results:
                if result['initial_noise']!=all_results[0]['initial_noise'] or result['expert_dispatch']!=all_results[0]['expert_dispatch']: raise RuntimeError('pair noise or scheduler mismatch')
            all_results.append(result); emit({'event':'PASS_COMPLETE',**result})
            with progress.open('a',encoding='utf8') as f:
                for event in events:f.write(json.dumps(event,allow_nan=False)+'\n')
        loader.verify_unchanged()
        for name,identity in identities.items():
            st=Path(name).stat()
            if (st.st_size,st.st_mtime_ns)!=identity: raise RuntimeError('base dependency changed')
        receipt.result={'source_id':a.run_id,'queue_start':queue_start,'shared_setup_seconds':setup_seconds,'preflight':{'status':'PASS','per_arm':preflights},'videos':all_results,'progress_sha256':sha256_file(progress),'gpu_peak_allocated_bytes':max(v['gpu_peak_allocated_bytes'] for v in all_results)}
if __name__=='__main__': main()
