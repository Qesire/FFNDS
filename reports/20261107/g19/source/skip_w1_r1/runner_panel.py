"""Canoe exploratory video: native Dense, G19 baseline and fused residual.

Execution requires independent review, exact ACTIVE registration and a live Slurm
allocation. This file does not submit jobs or alter the registry.
"""
import argparse, importlib, json, os, re, sys, time, shutil
from pathlib import Path
from runtime_guard import require_compute_allocation, RunReceipt, sha256_file
from legacy_helpers import verify_inputs, validate_effective_config, video_digest, finite_tensor_sha, reset_scheduler_generator


def arms_for_seed(seed):
    if seed == 43: return ('dense','packed_anchor_skip_w1')
    if seed == 44: return ('packed_anchor_skip_w1','dense')
    raise ValueError('unregistered seed')


RUN_PLAN = [
    {'kind':'warmup','arm':'packed_anchor'}, {'kind':'warmup','arm':'packed_anchor_skip_w1'},
    {'kind':'measure','arm':'packed_anchor'}, {'kind':'measure','arm':'packed_anchor_skip_w1'}, {'kind':'measure','arm':'packed_anchor_skip_w1'}, {'kind':'measure','arm':'packed_anchor'}, {'kind':'measure','arm':'packed_anchor'}, {'kind':'measure','arm':'packed_anchor_skip_w1'}, {'kind':'measure','arm':'packed_anchor_skip_w1'}, {'kind':'measure','arm':'packed_anchor'}
]

def guard(root, scope_path, digest):
    path = Path(scope_path).resolve()
    if not path.is_relative_to(root) or sha256_file(path) != digest:
        raise RuntimeError('scope path or SHA mismatch')
    scope = json.loads(path.read_text(encoding='utf8'))
    if scope['phase'] != 'G19_SKIP_W1_PILOT' or scope['mode'] != 'SKIP_W1_PILOT' or scope['status'] != 'REVIEW_CANDIDATE':
        raise RuntimeError('invalid scope identity')
    review = json.loads((root/'INDEPENDENT_REVIEW.json').read_text(encoding='utf8'))
    if (review.get('scope_sha256') != digest or review.get('disposition') != 'APPROVE'
        or review.get('recheck_disposition') != 'APPROVE' or not review.get('reviewer_agent_id')
        or review['reviewer_agent_id'] in scope['authors']
        or review.get('root_self_audit_sha256') != sha256_file(root/'ROOT_SELF_AUDIT.md')):
        raise RuntimeError('missing exact independent review/recheck')
    active = json.loads((root/'ACTIVE_RESEARCH.json').read_text(encoding='utf8'))
    item = active.get('scopes', {}).get(scope['scope_id'], {})
    if (item.get('scope_sha256') != digest or item.get('state') != 'AUTHORIZED_FOR_ONE_JOB' or item.get('max_jobs') != 1
        or item.get('review_sha256') != sha256_file(root/'INDEPENDENT_REVIEW.json') or active.get('session_max_gpus') != 2):
        raise RuntimeError('scope not ACTIVE')
    required = {'runner_panel.py', 'runtime_guard.py', 'legacy_helpers.py', 'direct.py', 'binding.py', 'resident.py', 'selective.py', 'fused_residual.py', 'fused_post.py', 'job_panel.sh', 'packed.py'}
    if not required.issubset(scope['code_sha256']): raise RuntimeError('missing bound code')
    for name, expected in {**scope['code_sha256'], **scope['frozen_inputs_sha256']}.items():
        p = (root/name).resolve()
        if not p.is_relative_to(root) or sha256_file(p) != expected: raise RuntimeError('bundle hash mismatch '+name)
    for name, expected in scope['external_sources_sha256'].items():
        if not Path(name).is_absolute() or sha256_file(name) != expected: raise RuntimeError('external hash mismatch '+name)
    if scope['budget']['gpus_per_job'] != 1 or scope['budget']['cpus'] > 4 or scope['budget']['host_memory_gib'] > 192 or scope['budget']['wall_seconds'] > 7200:
        raise RuntimeError('scope budget exceeded')
    if len(scope.get('pair_keys',[])) != 1 or scope.get('run_plan') != RUN_PLAN or not {'old_references.json','matrix_manifest.json',scope['input_manifest_file']}.issubset(scope['frozen_inputs_sha256']):
        raise RuntimeError('unexpected pilot cohort or arms')
    scope['_verified_scope_sha256'] = digest
    return scope


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scope', required=True); p.add_argument('--scope-sha', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); root = Path(__file__).resolve().parent
    allocation = require_compute_allocation(); scope = guard(root, a.scope, a.scope_sha)
    if allocation['hostname'].lower().split('.')[0] != scope['requested_node']:
        raise RuntimeError('allocated host differs from reviewed scope')
    if Path(sys.executable).resolve() != Path(scope['python_executable']).resolve() or Path(a.output).resolve() != Path(scope['output_dir']).resolve():
        raise RuntimeError('python/output mismatch')
    record = dict(re.findall(r'(\w+)=([^\s]+)', allocation['scheduler_job_record']))
    if int(record['NumCPUs']) > scope['budget']['cpus'] or allocation['memory_mib'] > scope['budget']['host_memory_gib']*1024:
        raise RuntimeError('allocation exceeds scope')
    match = re.fullmatch(r'(?:(\d+)-)?(\d+):(\d+):(\d+)', record['TimeLimit'])
    if not match: raise RuntimeError('unknown time limit')
    days,hours,minutes,seconds = (int(v or 0) for v in match.groups())
    if days*86400+hours*3600+minutes*60+seconds > scope['budget']['wall_seconds']: raise RuntimeError('wall allocation exceeds scope')
    with RunReceipt(a.output, scope, allocation) as receipt:
        if shutil.disk_usage(receipt.path).free < scope['budget']['min_free_disk_gib']*1024**3:raise RuntimeError('insufficient free disk before model load')
        def check_output_budget():
            size=sum(p.stat().st_size for p in receipt.path.rglob('*') if p.is_file())
            if size>scope['budget']['total_output_bytes']:raise RuntimeError('OUTPUT_BUDGET_EXCEEDED')
            receipt.result['output_bytes_checked']=size
            return size
        rows, identities = verify_inputs(scope, root)
        references=json.loads((root/'old_references.json').read_text(encoding='utf8'))
        refmap={(r['source_id'],r['seed']):r for r in references}
        if len(refmap)!=1 or set(refmap)!={(r['run_id'],r['seed']) for r in rows}:raise RuntimeError('reference coverage')
        if rows[0]['seed'] != 43: raise RuntimeError('pilot requires seed 43')
        for r in rows:
            ref=refmap[(r['run_id'],r['seed'])]
            if (ref['image_sha256'],ref['prompt_sha256'])!=(r['image_sha256'],r['prompt_sha256']):raise RuntimeError('reference source identity')
            for key in ('dense','g19'):
                entry=ref[key]
                if sha256_file(entry['path'])!=entry['sha256']:raise RuntimeError('old video identity')
        for path in reversed(scope['python_import_paths']): sys.path.insert(0, path)
        import torch
        import numpy, imageio_ffmpeg
        if (torch.__version__ != scope['runtime_versions']['torch'] or numpy.__version__ != scope['runtime_versions']['numpy']
            or imageio_ffmpeg.__version__ != scope['runtime_versions']['imageio_ffmpeg']
            or sha256_file(imageio_ffmpeg.get_ffmpeg_exe()) != scope['ffmpeg_sha256']):
            raise RuntimeError('runtime identity mismatch')
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != 'NVIDIA A100-SXM4-80GB':
            raise RuntimeError('expected single A100 80GB')
        torch.cuda.set_per_process_memory_fraction(scope['budget']['gpu_memory_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        from binding import NativeBinding
        from resident import ResidentMatrices
        import triton
        if triton.__version__ != scope['triton_version']: raise RuntimeError('Triton version mismatch')
        from fused_residual import run as fused_run, validate_synthetic
        synthetic_start = time.monotonic(); synthetic = validate_synthetic(); synthetic_seconds = time.monotonic()-synthetic_start
        from fused_post import run as post_run, validate_synthetic as validate_post
        post_synthetic = validate_post()
        from selective import SelectivePrefetch
        adapter = importlib.import_module('HiFloat4.ptq.adapters.lightx2v_wan_distill')
        if getattr(adapter,'_PATCHED',False) or getattr(adapter,'_ACTIVE_RUNTIME',None) is not None or getattr(adapter,'_ACTIVE_OBSERVER',None) is not None:
            raise RuntimeError('unexpected preexisting adapter patch')
        from lightx2v.utils.utils import find_torch_model_path
        original_builder = adapter.build_effective_lightx2v_pipeline_config
        def checked(pipeline):
            return validate_effective_config(original_builder(pipeline), scope, find_torch_model_path)
        adapter.build_effective_lightx2v_pipeline_config = checked
        setup_start = time.monotonic()
        try:
            pipe = adapter.build_lightx2v_distill_pipeline(base_model_dir=scope['base_model_dir'], high_noise_ckpt=scope['checkpoints']['high']['path'], low_noise_ckpt=scope['checkpoints']['low']['path'])
        finally: adapter.build_effective_lightx2v_pipeline_config = original_builder
        setup_seconds = time.monotonic()-setup_start
        if any(pipe.runner.config.get(k) != v for k,v in scope['required_runtime_config'].items()): raise RuntimeError('effective config mismatch')
        from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
        from lightx2v.models.runners.wan.wan_distill_runner import MultiDistillModelStruct
        from lightx2v.common.ops.mm.mm_weight import MMWeight
        def verify_phase(phase, skip_w1):
            objects = ((phase.ffn_2,(13824,5120)),) if skip_w1 else ((phase.ffn_0,(5120,13824)),(phase.ffn_2,(13824,5120)))
            for obj, shape in objects:
                if type(obj) is not MMWeight or obj.bias is None or getattr(obj,'has_lora_branch',True) or hasattr(obj,'weight_diff') or hasattr(obj,'bias_diff'):
                    raise RuntimeError('unsupported FFN weight')
                if tuple(obj.bias.shape) != (shape[1],) or obj.bias.device.type != 'cuda':
                    raise RuntimeError('FFN bias mismatch')
                weight = obj._get_actual_weight(); buf = getattr(obj,'weight_cuda_buffer',None)
                if buf is None or weight.device.type != 'cuda' or tuple(weight.shape) != shape or weight.data_ptr() != buf.data_ptr():
                    raise RuntimeError('FFN buffer mismatch')
        entries = json.loads((root/'matrix_manifest.json').read_text(encoding='utf8'))
        if set(entries) != {f'{e}/{l}' for e in ('high','low') for l in range(40)}: raise RuntimeError('matrix coverage')
        matrices = ResidentMatrices(entries, sha256_file, torch.device('cuda:0'))
        from packed import PackedMatrices
        packed = PackedMatrices(matrices, pipe.runner.model.model, torch.device('cuda:0'))
        results = []; receipt.result = {'status':'IN_PROGRESS','asset_identities':identities,
            'pipeline_setup_seconds':setup_seconds,'matrix_setup_seconds_synchronized':matrices.setup_seconds,
            'matrix_payload_bytes':matrices.payload_bytes, 'packed_setup_seconds_synchronized':packed.setup_seconds, 'packed_payload_bytes':packed.payload_bytes, 'packed_setup_allocated_delta':packed.setup_allocated_delta, 'packed_setup_peak_allocated':packed.setup_peak_allocated,'synthetic':synthetic,'post_synthetic':post_synthetic}
        for pair_index, row in enumerate(rows):
            pair_results = []
            ref=refmap[(row['run_id'],row['seed'])]
            old_video_digest = video_digest(ref['g19']['path'])
            for run_index, item in enumerate(scope['run_plan']):
                arm = item['arm']; is_warmup = item['kind'] == 'warmup'
                receipt.check_deadline(); dense = arm == 'dense'
                selective = None if dense else SelectivePrefetch(skip_hits=True, skip_anchor_w1=(arm == 'packed_anchor_skip_w1'))
                model_list = pipe.runner.model.model
                if len(model_list) != 2: raise RuntimeError('unexpected expert count')
                for native_model in model_list: native_model.transformer_infer.offload_manager.need_init_first_buffer = True
                if selective: selective.install({'high':model_list[0].transformer_infer,'low':model_list[1].transformer_infer})
                def on_site(model,phase,step,expert,layer):
                    selective.verify_current(model,phase,step,expert,layer)
                    if step in (0,2): verify_phase(phase, arm == 'packed_anchor_skip_w1')
                binding = NativeBinding('g19',packed,expected_shape=(31668,5120),on_site=on_site,residual_fn=fused_run,execution=arm,post_fn=post_run)
                binding.core.probe_enabled = is_warmup and arm == 'packed_anchor'
                binding.core.record_timings = is_warmup
                old_dispatch = MultiDistillModelStruct.get_current_model_index
                old_block = WanTransformerInfer.infer_block
                dispatch=[]; sites=[]; initial=[]
                def dispatch_call(model):
                    value=old_dispatch(model); step=int(model.scheduler.step_index)
                    dispatch.append((step,int(model.cur_model_index)))
                    if step==0:
                        initial.append(model.scheduler.latents.detach().clone())
                        if finite_tensor_sha(initial[-1],torch)!=ref['initial_noise']:raise RuntimeError('reference initial noise mismatch before forward')
                    if binding: binding.dispatch=int(model.cur_model_index)
                    if selective: selective.dispatch='high' if model.cur_model_index==0 else 'low' if model.cur_model_index==1 else None
                    return value
                def block_call(model,*args,**kwargs):
                    sites.append((int(model.scheduler.step_index),int(model.block_idx)))
                    return old_block(model,*args,**kwargs)
                scheduler=pipe.runner.scheduler
                MultiDistillModelStruct.get_current_model_index=dispatch_call
                WanTransformerInfer.infer_block=block_call
                if binding: binding.install(WanTransformerInfer)
                stem=f"{row['run_id']}_seed{row['seed']}_{run_index:02d}_{item['kind']}_{arm}"
                video=receipt.path/(stem+'.mp4')
                reset_scheduler_generator(scheduler,pipe.runner.model)
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.monotonic()
                try:
                    with torch.no_grad(): pipe.generate(seed=row['seed'],image_path=row['image_path'],prompt=row['prompt'],negative_prompt=row.get('negative_prompt',''),save_result_path=str(video))
                    torch.cuda.synchronize(); elapsed=time.monotonic()-start
                    latent=finite_tensor_sha(scheduler.latents,torch)
                    counts=binding.finish() if binding else {'native_ffn':len(sites)}
                    layer_timings = [{'step':st,'layer':ly,'action':ac,'milliseconds':ev0.elapsed_time(ev1)} for st,ly,ac,ev0,ev1 in binding.core.layer_events]
                    probe_result = binding.core.probe_result
                    gemm_counts = dict(binding.core.gemm_counts)
                    expected_gemm = {'native_w1':0,'native_w2':80,'anchor_a':0,'packed_wide':80,'hit_a':80,'probe_extra':6 if is_warmup and arm == 'packed_anchor' else 0}
                    if gemm_counts != expected_gemm or len(layer_timings) != (160 if is_warmup else 0): raise RuntimeError('GEMM/timing coverage mismatch')
                    if binding.core.probe_enabled and [(x['step'],x['layer']) for x in probe_result] != [(0,0),(2,0)]: raise RuntimeError('probe coverage mismatch')
                    transfer=selective.finish() if selective else {'status':'NATIVE_FULL_PREFETCH','physical_transfer_measurement':'NOT_MEASURED'}
                finally:
                    if binding: binding.uninstall()
                    if selective: selective.uninstall()
                    WanTransformerInfer.infer_block=old_block
                    MultiDistillModelStruct.get_current_model_index=old_dispatch
                if dispatch != [(i,0 if i<2 else 1) for i in range(4)] or sites != [(s,l) for s in range(4) for l in range(40)] or len(initial)!=1: raise RuntimeError('dispatch/site coverage')
                expected={'native_ffn':160} if dense else {'dense':0,'anchor':80,'hit':80,'native_ffn':80,'predictions':160}
                if counts != expected: raise RuntimeError('FFN action count')
                noise=finite_tensor_sha(initial[0],torch); del initial[:]
                if pair_results and noise != pair_results[0]['initial_noise']: raise RuntimeError('within-pair initial noise differs')
                result={'source_id':row['run_id'],'seed':row['seed'],'image_path':row['image_path'],'image_sha256':row['image_sha256'],
                    'prompt':row['prompt'],'prompt_sha256':row['prompt_sha256'],'arm':arm,'run_index':run_index,'kind':item['kind'],'video_path':str(video),'generation_wall_seconds':elapsed,
                    'timing_note':'warmups excluded from measurement; measured order AB BA AB BA; initial-noise clone and SHA included in every wall; CUDA layer events and numeric probe enabled only for warmups; packed warmup probe recomputes and synchronizes',
                    'first_cold_run':run_index==0,'initial_noise':noise,'matrix_manifest_sha256':sha256_file(root/'matrix_manifest.json'),'scope_sha256':scope['_verified_scope_sha256'],'checkpoint_sha256':{k:scope['checkpoints'][k]['sha256'] for k in ('high','low')},'reference_dense_sha256':ref['dense']['sha256'],'reference_g19_sha256':ref['g19']['sha256'],'reference_dense_path':ref['dense']['path'],'reference_g19_path':ref['g19']['path'],'cohort_sha256':sha256_file(root/scope['input_manifest_file']),'old_references_sha256':sha256_file(root/'old_references.json'),'code_sha256':scope['code_sha256'],'final_latent':latent,'counts':counts,
                    'video':video_digest(video),'old_g19_video':old_video_digest,'transfer':transfer,'logical_gemms':gemm_counts,'layer_timings':layer_timings,'numeric_probe':probe_result,'probe_in_generation_wall':binding.core.probe_enabled,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved()}
                with (receipt.path/(stem+'_PASS.json')).open('x',encoding='utf8') as stream:
                    json.dump(result,stream,indent=2,allow_nan=False); stream.flush(); os.fsync(stream.fileno())
                pair_results.append(result); results.append(result); receipt.result['videos']=list(results);check_output_budget()
            if len(pair_results)!=10: raise RuntimeError('incomplete pilot video plan')
            measured = [r for r in pair_results if r['kind']=='measure']
            summary = {}
            for arm in ('packed_anchor_skip_w1','packed_anchor'):
                selected = [r for r in measured if r['arm']==arm]
                first = selected[0]
                summary[arm] = {'run_seconds':[r['generation_wall_seconds'] for r in selected], 'mean_seconds':sum(r['generation_wall_seconds'] for r in selected)/len(selected), 'first_measured_video':first['video_path'], 'rgb24_matches_first':[r['video']['decoded_rgb24_sha256']==first['video']['decoded_rgb24_sha256'] for r in selected], 'mp4_matches_first':[r['video']['mp4_sha256']==first['video']['mp4_sha256'] for r in selected], 'latent_matches_first':[r['final_latent']==first['final_latent'] for r in selected], 'rgb24_matches_old_g19':[r['video']['decoded_rgb24_sha256']==old_video_digest['decoded_rgb24_sha256'] for r in selected], 'mp4_matches_old_g19':[r['video']['mp4_sha256']==old_video_digest['mp4_sha256'] for r in selected]}
            af = next(r for r in measured if r['arm']=='packed_anchor_skip_w1')
            bp = next(r for r in measured if r['arm']=='packed_anchor')
            summary['first_measured_cross_arm'] = {'latent_equal':af['final_latent']==bp['final_latent'], 'rgb24_equal':af['video']['decoded_rgb24_sha256']==bp['video']['decoded_rgb24_sha256'], 'mp4_equal':af['video']['mp4_sha256']==bp['video']['mp4_sha256'], 'old_g19_latent':'NOT_AVAILABLE_IN_REFERENCE'}
            summary['speed_ratio_full_over_skip'] = summary['packed_anchor']['mean_seconds']/summary['packed_anchor_skip_w1']['mean_seconds']
            summary['note'] = 'Four measured runs per arm after two warmups. Fixed seed; GPU event instrumentation and probe disabled on measured runs. Shared packed weights remain resident for both arms; setup reported separately.'
            receipt.result['steady_state_summary']=summary
        receipt.result.update(status='COMPLETE_REQUIRES_REVIEW',source_seed_count=len(rows),video_count=len(results),
                              q_label='SKIP_W1_PILOT',video_quality='NOT_MEASURED',cache_variant='BF16_GPU',physical_transfer_measurement='NOT_MEASURED')
        check_output_budget()
        packed.close()

if __name__ == '__main__': main()
