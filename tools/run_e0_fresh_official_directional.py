"""Four genuinely new official sources, with two crossed seeds; no implicit run.

Reuse the immutable native computational path, paired video gate and band tool.
The first dense controls for new sources have no historical clip to reproduce.
"""
import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
from ffnds_observation_common import sha256, write_json
from run_e0_paired_full_depth import ROOT, object_sha, persist_observation, collect_failed_video_evidence, MANDATORY_DRIVER_SOURCES, MANDATORY_NATIVE_SOURCES
from run_e0_multisource_directional import reserve_attempt,close_attempt,evaluate_current_pair,PASS
ID='E0_FRESH_OFFICIAL_DIRECTIONAL_NATIVE_20261026_V1'
PAIRS=[(908,42),(908,54),(376,42),(425,42),(425,54),(437,42)]


def validate_cohort(samples,audit):
    expected=[908,376,425,437]
    if ([s['source_index'] for s in samples]!=expected
            or len({s['source_id'] for s in samples})!=4):
        raise ValueError('Four unique original official sources required')
    for sample,original in zip(samples,audit['selected_sources']):
        for k in ['source_index','source_group','image_sha256','prompt','prompt_sha256',
                  'image_name','image_type','official_dimensions','camera_instruction']:
            if sample[k]!=original[k]:raise ValueError('Original audited source differs: '+k)
        if (sample['source_id']!=original['image_sha256']
                or sample['source_id'] in audit['prior_image_SHA']
                or sample['prompt_sha256'] in audit['prior_prompt_SHA']
                or sample['source_role']!='NEW_TO_ALL_FROZEN_PROJECT_INPUT_COHORTS_BEFORE_ANY_NEW_MODEL_RESULT'):
            raise ValueError('Fresh project-cohort identity/exposure contract failed')


def verify_phase1_controls(p,hf_root):
    from run_e0_multisource_directional import original_source
    from ffnds_paired_video_strict_gate import inspect
    prior=ROOT/p['phase1_protocol']
    if sha256(prior)!=p['phase1_protocol_sha256']:
        raise ValueError('Original phase1 protocol changed')
    frozen=json.loads(prior.read_text())
    for name in ['e0_paired_wan_controller.py','e0_paired_ffn_observer.py',
                 'e0_paired_weight_residency.py','e0_full_depth_trace_contract.py','e0_whole_flow_counterfactual.py']:
        key='tools/'+name
        if frozen['driver_sources_sha256'][key]!=p['driver_sources_sha256'][key]:
            raise ValueError('Previously controlled native mathematics changed')
    controls=p['completed_phase1_controls']
    if [r['source_index'] for r in controls]!=[4,8,16,20]:
        raise ValueError('Four completed native-controller controls required')
    for item in controls:
        receipt=(hf_root/'ffnds-derived'/item['RUN_HF_relative']).resolve()
        if not receipt.is_relative_to((hf_root/'ffnds-derived').resolve()) or sha256(receipt)!=item['RUN_sha256']:
            raise ValueError('Prior native control receipt binding failed')
        r=json.loads(receipt.read_text())
        pair=next(x for x in frozen['pairs'] if x['source_index']==item['source_index'])
        if (r['status']!=PASS or r['pair']!=pair or r['seed']!=pair['seed']
                or r['source_id']!=pair['source_id'] or r['protocol_sha256']!=p['phase1_protocol_sha256']
                or r['bank_RUN_sha256']!=p['bank_RUN_sha256']
                or object_sha(r['assets'])!=p['reference_model_assets_digest']):
            raise ValueError('Prior control protocol/bank/assets/source identity differs')
        _,_,old_phase=original_source(frozen,hf_root,pair)
        identity=[{k:v[k] for k in ['step','sha256','shape']} for v in r['trajectory']['teacher']]
        if (identity!=old_phase['trajectory'] or r['videos']['teacher']['sha256']!=old_phase['video_sha256']
                or inspect(receipt.parent/'STRICT_BUNDLE.json')):
            raise ValueError('Prior actual dense trajectory/video or strict bundle failed')


def preflight(protocol,bank,hf_root):
    p=json.loads(protocol.read_text())
    needed=MANDATORY_DRIVER_SOURCES|{'tools/e0_whole_flow_counterfactual.py',
        'tools/run_e0_multisource_directional.py','tools/run_e0_fresh_official_directional.py',
        'tools/e0_disjoint_native_bands.py'}
    if (p['id']!=ID or not needed<=set(p['driver_sources_sha256'])
            or not MANDATORY_NATIVE_SOURCES<=set(p['native_source_sha256'])
            or [(x['source_index'],x['seed']) for x in p['pairs']]!=PAIRS
            or [x['id'] for x in p['pairs']]!=[f'dc_official{i}_seed{s}' for i,s in PAIRS]
            or any(x['modes']!=['dense','candidate'] or x['save_vectors'] for x in p['pairs'])
            or p['max_full_generations_total']!=12 or p['max_HF_output_gib']!=1.5
            or p['planned_simultaneous_GPU_cards']!=1 or p['max_gpu_cards']!=4
            or p['counterfactual_steps']!=[0,1,2,3] or p['render_videos'] is not True
            or p['automatic_retry'] is not False
            or p['flow_discriminator_budget']!={'whole_latent_trajectories':12,'additional_complete_dense_flows':24,
                'new_videos':12,'new_weight_fits':0,'training_updates':0,'official_quality_calls':0}):
        raise ValueError('Exact four-fresh-source/six-pair scope required')
    for group in ['driver_sources_sha256','native_source_sha256']:
        for name,digest in p[group].items():
            if sha256(ROOT/name)!=digest:raise ValueError('Frozen source drift: '+name)
    for name,digest in [(p['input_default'],p['input_default_sha256']),
        ('.runtime/model_view.json',p['model_view_sha256']),
        ('.runtime/node_local_model_view.json',p['local_model_view_sha256']),
        ('.runtime/hif4_rebuilt_cache.json',p['cache_manifest_sha256']),
        (p['source_audit'],p['source_audit_sha256'])]:
        if sha256(ROOT/name)!=digest:raise ValueError('Frozen identity changed: '+name)
    audit=json.loads((ROOT/p['source_audit']).read_text())
    for row in audit['exposure_inventory']:
        if sha256(ROOT/row['path'])!=row['sha256']:raise ValueError('Historical exposure index changed')
    samples=json.loads((ROOT/p['input_default']).read_text());validate_cohort(samples,audit)
    for pair in p['pairs']:
        s=next(s for s in samples if s['source_index']==pair['source_index'])
        if (pair['source_id']!=s['source_id'] or sha256(Path(s['image_path']))!=s['source_id']
                or pair['prompt_digest']!=object_sha(s['prompt']) or pair['role']!=s['source_role']):
            raise ValueError('Actual fresh source/image/prompt identity differs')
    local=json.loads((ROOT/'.runtime/node_local_model_view.json').read_text())
    vae=[r for r in local['files'] if r['filename']=='Wan2.1_VAE.pth']
    if (len(vae)!=1 or vae[0]['sha256']!=p['VAE_sha256']
            or sha256(Path(vae[0]['view_path']))!=p['VAE_sha256']
            or Path(vae[0]['view_path']).stat().st_size!=p['VAE_bytes']):
        raise ValueError('Original VAE checkpoint changed')
    verify_phase1_controls(p,hf_root)
    if sha256(bank/'RUN.json')!=p['bank_RUN_sha256']:raise ValueError('Original bank receipt changed')
    packs=[x for x in json.loads((bank/'RUN.json').read_text())['packs'] if x['family']=='affine_residual']
    if len(packs)!=80 or len({(x['expert'],x['block']) for x in packs})!=80:
        raise ValueError('Exactly80 original fixed bank files required')
    for pack in packs:
        if sha256(bank/pack['path'])!=pack['sha256']:raise ValueError('Original fixed bank changed')
    scopes=json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text())['experiment_authority']['current_user_authorized_scopes']
    authorized=any(s['id']==ID and s['protocol_sha256']==sha256(protocol)
        and s.get('execution_status') in ['AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE'] for s in scopes)
    return p,packs,{'status':'STATIC_FRESH_SOURCE_AND_NATIVE_CONTROL_BINDINGS_PASS_NOT_NEW_RUN',
        'model_scope_authorized':authorized,'fresh_source_and_native_controls_verified':True,
        'new_videos':0,'GPU_initialized':False}

def execute(args, plan, manifests, status):
    if not status['model_scope_authorized']:
        raise RuntimeError('Exact fixed model scope requires separate explicit authorization registered in ACTIVE_RESEARCH')
    if args.study_root.resolve() != Path(plan['canonical_study_root']).resolve():
        raise ValueError('One canonical ledger; output root cannot reset attempts')
    if not status.get('fresh_source_and_native_controls_verified'):
        raise ValueError('Fresh original source and prior native-controller controls required')
    from worker_identity import private_config_path, require_worker
    if sha256(private_config_path()) != plan['private_worker_config_sha256']:
        raise ValueError('Privately selected executor binding changed')
    require_worker()
    pair = next(p for p in plan['pairs'] if p['id'] == args.pair)
    protocol_sha = sha256(args.protocol)
    reserve_attempt(args.study_root, plan, protocol_sha, pair)
    out = args.study_root / pair['id']
    out.mkdir(exist_ok=False)
    run = {'status': 'STARTED_NOT_ACCEPTED', 'id': plan['id'], 'protocol_sha256': protocol_sha,
           'pair': pair, 'latent_trajectories_started': 0, 'new_videos': 0,
           'new_weight_fits': 0, 'training_updates': 0, 'official_quality_calls': 0,
           'quality_accepted': False, 'automatic_retry': False}
    write_json(out / 'RUN.json', run)
    monitor = None
    try:
        for name in ['HiF4_Sp', 'LightX2V', 'Wan2.2-I2V-A14B-W4A4']:
            sys.path.insert(0, str(ROOT / '.runtime/code' / name))
        for name in ['HiFloat4', 'HiFloat4/hif4_gpu']:
            sys.path.insert(0, str(ROOT / '.runtime/code/HiF4_Sp' / name))
        import torch
        from safetensors.torch import load_file, save_file
        from historical_source_snapshot import HistoricalSources
        from restored_assets import verified_restored_assets
        from startup_gpu_reservation import reserve_before_assets
        from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
        from e0_paired_weight_residency import PairedWeightResidency
        from e0_paired_ffn_observer import PairedFFNObserver
        from e0_whole_flow_counterfactual import WholeFlowCounterfactualController
        from e0_full_depth_trace_contract import FullDepthTrace
        host = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        if int(host['MemAvailable'].split()[0]) < plan['minimum_available_host_gib'] * (1 << 20):
            raise RuntimeError('Host reserve unavailable; consumed attempt, no retry')
        reservation, monitor, receipt = reserve_before_assets(torch)
        if 'A100' not in torch.cuda.get_device_name(0):
            raise RuntimeError('One idle A100 required')
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = plan['cudnn_allow_tf32']
        run['environment'] = {'torch': torch.__version__, 'CUDA_runtime': torch.version.cuda,
            'GPU': torch.cuda.get_device_name(0), 'python': sys.version, 'TF32': False,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'BF16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
        setup_started = time.perf_counter()
        snapshot = out / 'EXECUTED_SOURCES'
        snapshot.mkdir()
        for name in {**plan['driver_sources_sha256'], **plan['native_source_sha256']}:
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        shutil.copyfile(args.protocol, snapshot / args.protocol.name)
        kwargs, cache, assets = verified_restored_assets(HistoricalSources())
        if object_sha(assets)!=plan['reference_model_assets_digest']:
            raise ValueError('Actual model assets differ from native controller controls')
        pipe = adapter.build_lightx2v_distill_pipeline(**kwargs)
        adapter.validate_effective_pipeline_config_against_contract(pipe.runner.config)
        run['VAE_sha256'] = plan['VAE_sha256']
        runtime = adapter.HIF4CacheRuntime(cache['path'])
        adapter.activate_lightx2v_hooks(runtime=runtime)
        packs = {(p['expert'], p['block']): load_file(str(args.bank / p['path']), device='cuda') for p in manifests}
        coefficients = {(p['expert'], p['block']): p['sha256'] for p in manifests}
        sample = next(s for s in json.loads((ROOT / plan['input_default']).read_text()) if s['source_index'] == pair['source_index'])
        image = Path(sample.get('image_path') or sample['img_paths'][0])
        seed = pair['seed']  # Explicit crossed-seed design, not a source identity.
        if sha256(image) != pair['source_id'] or sample['source_index'] != pair['source_index']:
            raise ValueError('Actual fresh source identity drift')
        run.update(source_id=pair['source_id'], seed=seed, assets=assets, CUDA_reservation=receipt,
            source_role=pair['role'], bank_RUN_sha256=plan['bank_RUN_sha256'],
            official_metadata_row=sample['source_index'], cohort_id=sample['cohort_id'],
            original_dense_receipt=None,
            dense_control_scope='FIRST_DENSE_TRAJECTORY_FOR_THIS_NEW_SOURCE_SEED; FROZEN_NATIVE_CONTROLLER_AND_STEP0_NEUTRALITY; NO_PRIOR_CLIP_IDENTITY_CLAIM')

        def budget(step=None, block=None):
            monitor.check()
            if shutil.disk_usage(out).free < plan['disk_reserve_gib'] * (1 << 30):
                raise RuntimeError('Disk reserve reached')
            if sum(p.stat().st_size for p in args.study_root.rglob('*') if p.is_file()) > plan['max_HF_output_gib'] * (1 << 30):
                raise RuntimeError('Fixed complete-study output cap reached')

        def save_counterfactual(step, tensors, record):
            path = out / f'flow_s{step}.safetensors'
            if tensors is not None:
                save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(path))
                binding = {'path': path.name, 'sha256': sha256(path),
                           'tensors': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in tensors.items()}}
            else:
                binding = record['saved_tensors']
            record['saved_tensors'] = binding
            write_json(out / f'flow_s{step}.json', record)
            budget()
            return binding

        provenance = {'protocol_sha256': protocol_sha, 'native_source_sha256': object_sha(plan['native_source_sha256']),
                      'model_digest': object_sha(assets), 'bank_RUN_sha256': plan['bank_RUN_sha256']}
        factory = lambda shape, state_shape: FullDepthTrace(shape, pair['source_index'], seed, provenance, state_shape=state_shape)
        with torch.inference_mode(), PairedWeightResidency(runtime,
                max_non_ffn_bytes=plan['max_non_ffn_resident_bytes'], max_block_ffn_bytes=plan['max_block_ffn_resident_bytes']) as policy:
            for expert, records in runtime.records.items():
                for name, record in records.items():
                    if not name.endswith(('.ffn.0', '.ffn.2')):
                        runtime._weight(expert=expert, block_index=int(name.split('.')[1]), module_name=name,
                                        record=record, device=torch.device('cuda', 0), dtype=torch.bfloat16)
            torch.cuda.synchronize()
            budget()
            run['setup_seconds'] = time.perf_counter() - setup_started
            with PairedFFNObserver(runtime, packs, adapter, None, coefficients, timing_events=True,
                                   numerical_rows=32, vector_rows=8, save_vectors=None) as observer:
                with WholeFlowCounterfactualController(pipe.runner, observer, policy, adapter, factory,
                        modes=pair['modes'], progress=budget, save_counterfactual=save_counterfactual,
                        counterfactual_steps=plan['counterfactual_steps']) as controller:
                    pipe.runner.scheduler.generator = None
                    run['latent_trajectories_started'] = 2
                    write_json(out / 'RUN.json', run)
                    before = dict(runtime.counters)
                    started = time.perf_counter()
                    pipe.generate(seed=seed, image_path=str(image), prompt=sample['prompt'], negative_prompt='',
                                  save_result_path=str(out / 'student.mp4'))
                    torch.cuda.synchronize()
                    run['observation'] = persist_observation(out, controller, observer, runtime, before, policy)
                    # Preserve completed trace before the second VAE/media step.
                    write_json(out / 'RUN.json', run)
                    info, original_info = controller.input_info, pipe.runner.input_info
                    try:
                        pipe.runner.input_info = info; info.save_result_path = str(out / 'teacher.mp4')
                        pipe.runner.gen_video_final = pipe.runner.run_vae_decoder(controller.teacher_latents)
                        pipe.runner.process_images_after_vae_decoder()
                    finally:
                        pipe.runner.input_info = original_info
                    torch.cuda.synchronize()
                    run['observation_wall_seconds'] = time.perf_counter() - started
                    timings = observer.timing_report()
                    run.update(trajectory=controller.trajectory, counterfactual_records=controller.counterfactual_records,
                        observer_counts=dict(observer.counts), native_runtime_counts={k: runtime.counters[k] - before[k] for k in before},
                        residency=policy.snapshot(), FFN_primary_plus_shadow_timings=timings,
                        trace_sha256=sha256(out / 'TRACE.json'),
                        cost={'generation_wall_seconds': run['observation_wall_seconds'],
                              'ffn_boundary_seconds': sum(t['primary_plus_shadow_ms'] for t in timings)/1000,
                              'setup_seconds': run['setup_seconds'],
                              'peak_gpu_memory_mib': monitor.report()['peak_physical_used_memory_MiB']},
                        cost_scope='OFFLINE_PAIRED_2_TRAJECTORIES_4_EXTRA_DENSE_FLOWS_LOCAL_SHADOWS_2_DECODES_NOT_DEPLOYMENT_LATENCY',
                        cost_deduplication_key=protocol_sha + '/' + pair['id'],
                        videos={b: {'path': b+'.mp4', 'sha256': sha256(out/(b+'.mp4'))} for b in ['teacher','student']})
                    write_json(out / 'RUN.json', run)
                    counts = run['native_runtime_counts']
                    if (len(controller.counterfactual_records) != 4 or observer.counts['completed_sites'] != 160
                            or counts['linear_forward_calls'] != 4800 or counts['activation_qdq_calls'] != 4800
                            or counts['cache_misses'] or counts['weight_transfers'] != 640
                            or policy.snapshot()['ffn_transfers'] != 640
                            or any(r['native_runtime_counts']['linear_forward_calls'] != 400
                                   or r['native_runtime_counts']['activation_qdq_calls'] != 400
                                   for r in controller.counterfactual_records)):
                        raise RuntimeError('Whole-flow native coverage/call/residency ledger mismatch')
                    expected = {'dense_main_GEMM': 320, 'dense_shadow_GEMM': 320, 'candidate_main_GEMM': 480,
                                'candidate_shadow_GEMM': 480, 'native_QDQ': 1280, 'continuous_GEMM': 960}
                    if any(observer.counts[k] != v for k, v in expected.items()):
                        raise RuntimeError('Original paired observer ledger changed')
                    # Both actual clips are new here; the unchanged native dense
                    # controller is controlled by the completed four-source byte-identity study.
                    # Each current step0 also checks exact full-flow/shadow neutrality.
                    if len(controller.trajectory['teacher']) != 4 or len(timings) != 320:
                        raise RuntimeError('Missing complete native dense control trajectory')
                    budget()
                    run['status'] = 'PASS_NATIVE_ACCOUNTING_REQUIRES_CPU_BANDS_AND_VIDEO_METRICS'
        adapter.deactivate_lightx2v_hooks()
        del pipe, packs, runtime, controller, observer, policy, reservation
        gc.collect(); torch.cuda.empty_cache()
        run['GPU_allocated_bytes_before_CPU_metrics'] = torch.cuda.memory_allocated()
        monitor.close(); monitor.check(); run['assigned_GPU_resources'] = monitor.report(); monitor = None
        run['cost']['peak_gpu_memory_mib'] = run['assigned_GPU_resources']['peak_physical_used_memory_MiB']
        write_json(out / 'NATIVE_RUN.json', run)
        evaluate_current_pair(out, run, pair, sample, protocol_sha)
        from e0_disjoint_native_bands import describe_saved_pair
        run['disjoint_bands'] = describe_saved_pair(out)
        if sum(p.stat().st_size for p in args.study_root.rglob('*') if p.is_file()) > plan['max_HF_output_gib']*(1<<30):
            raise RuntimeError('Fixed output cap exceeded after metrics')
        run.update(status=PASS, new_videos=2, native_RUN_sha256=sha256(out/'NATIVE_RUN.json'))
    except BaseException as error:
        run.update(status='FAILED_PRESERVED_NO_RETRY', error_type=type(error).__name__,
                   error=str(error), traceback=traceback.format_exc())
        costs = run.setdefault('cost', {})
        if 'started' in locals():
            costs.setdefault('generation_wall_seconds', time.perf_counter()-started)
        if 'setup_seconds' in run:
            costs.setdefault('setup_seconds', run['setup_seconds'])
        elif 'setup_started' in locals():
            run['partial_setup_elapsed_seconds'] = time.perf_counter()-setup_started
        if monitor is not None:
            costs.setdefault('peak_gpu_memory_mib', monitor.report().get('peak_physical_used_memory_MiB'))
        run.setdefault('cost_scope', 'FAILED_PARTIAL_OFFLINE_OBSERVATION; MISSING_BOUNDARIES_REMAIN_UNMEASURED')
        if 'controller' in locals():
            run['counterfactual_records'] = controller.counterfactual_records
            try:
                run['observation'] = persist_observation(out, controller, locals().get('observer'),
                    locals().get('runtime'), locals().get('before'), locals().get('policy'))
            except BaseException as preservation_error:
                run['observation_preservation_error'] = repr(preservation_error)
        try:
            run['failed_video_evidence'] = collect_failed_video_evidence(out, run, pair, locals().get('sample', {}), protocol_sha)
        except BaseException as evidence_error:
            run['failed_video_evidence_error'] = repr(evidence_error)
        raise
    finally:
        original_error = sys.exc_info()[1]
        finalization_errors = []
        if 'adapter' in locals():
            try:
                adapter.deactivate_lightx2v_hooks()
            except BaseException as error:
                finalization_errors.append(error)
        if monitor is not None:
            try:
                monitor.close()
                run['assigned_GPU_resources'] = monitor.report()
                monitor.check()
            except BaseException as error:
                finalization_errors.append(error)
        if finalization_errors:
            run.update(status='FAILED_PRESERVED_NO_RETRY', finalization_errors=[repr(e) for e in finalization_errors])
        write_json(out / 'RUN.json', run)
        close_attempt(args.study_root, pair['id'], out / 'RUN.json')
        if finalization_errors and original_error is None:
            raise RuntimeError('Native finalization/isolation failed; attempt preserved') from finalization_errors[0]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['protocol','bank','hf-root','study-root']:
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--pair',choices=[f'dc_official{i}_seed{s}' for i,s in PAIRS],required=True)
    ap.add_argument('--execute-native',action='store_true');args=ap.parse_args()
    plan,packs,status=preflight(args.protocol,args.bank,args.hf_root)
    print(json.dumps(status),flush=True)
    if args.execute_native:execute(args,plan,packs,status)

if __name__=='__main__':main()
