"""G20 B_CHIP_STATS conditional fixed64 U-input production engine.

Route VBench-restart-V4 / TWO_CORE rev5 task B_CHIP_STATS. Invoked ONLY by
guarded_runner_chip after allocation/review/ACTIVE/hash verification. Produces
B_CHIP_P64_INPUTS: the real CHIPMUNK_GLOBAL_P64 fixed64 U0/U1 rows at six sites x
two endpoints for the 11 existing DEV sources, captured from the SAME W_D D4 native
state as the landed VB_DENSE22 bank. Reuses the V2 rev2 pipeline binding VERBATIM
(hash-gated E1 LightX2V adapter, base assets, dense unfused runtime config, W_D
checkpoints). At most 11 new D4 full executions (one per DEV source); 0 new O40,
0 new source/image/prompt, 0 video decode/scoring, 0 full-M block stats, 0 full-M F1.

Same-state identity: each target capture also saves the PANEL512 U cross-check and
bit-compares it (torch.equal) against the landed bank U at the exact matched
(source,step,layer). A mismatch blocks only that source's Chip cells; it is never
silently spliced into the BW main table. The completed BW Q1/ICC fits/results are
reused downstream; this scope produces inputs only, not another method comparison.

The per-source capture, same-state check, record build, aggregation and summary are
module-level functions so the author smoke exercises the EXACT formal integration
path (capture -> same-state -> record -> aggregate -> serialize) on one real source.
"""
import hashlib
import importlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

from runtime_guard import sha256_file as sha
import panel512
from b_capture_inputs import (NativeTapFixed64, Fixed64Sink, regenerate_fixed64,
                              EXPECTED_GRID, FULL_M, FIXED64_COUNT, PANEL_COUNT,
                              SITE_OF, TARGET_SITES, SITES)

DIM = 5120


def finite_tensor_sha(t, torch):
    t = t.detach().contiguous()
    if not torch.isfinite(t).all().item():
        raise RuntimeError('nonfinite initial noise or scheduler latent')
    return {'sha256': hashlib.sha256(t.view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
            'shape': list(t.shape), 'dtype': str(t.dtype)}


def reset_scheduler_generator(scheduler, model):
    if getattr(model, 'scheduler', None) is not scheduler:
        raise RuntimeError('MODEL_SCHEDULER_IDENTITY_MISMATCH')
    scheduler.generator = None


def exclusive_json(path, data):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def verify_chip_conditions(conds, dev11):
    if len(conds) != 11:
        raise RuntimeError('exact 11 DEV capture conditions required')
    if sorted(c['record_id'] for c in conds) != sorted(dev11):
        raise RuntimeError('chip conditions must be exactly the frozen DEV11')
    for c in conds:
        if c['cycle_role'] != 'DEV':
            raise RuntimeError('chip capture must be DEV only ' + c['record_id'])
        if c['seed'] != 42 or c['orient'] != 'landscape':
            raise RuntimeError('condition seed/orient ' + c['record_id'])
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,96}', c['record_id']):
            raise RuntimeError('bad record_id ' + c['record_id'])
        if hashlib.sha256(c['prompt'].encode('utf8')).hexdigest() != c['prompt_sha256']:
            raise RuntimeError('prompt hash mismatch ' + c['record_id'])
        if sha(c['image_path']) != c['image_sha256']:
            raise RuntimeError('source image changed ' + c['record_id'])
        if c['panel_source'] != c['record_id']:
            raise RuntimeError('panel source must be record_id ' + c['record_id'])
    if len({c['source_cluster_id'] for c in conds}) != 11:
        raise RuntimeError('capture clusters not separated')


def verify_pipeline_binding(scope):
    """Hash-gate the full reused E1 pipeline tree + base assets + checkpoints."""
    base = Path(scope['base_model_dir']).resolve()
    for name in ('config.json', 'vae/config.json'):
        if (base / name).exists():
            raise RuntimeError('undeclared priority config ' + name)
    for relative, spec in scope['base_assets_sha256'].items():
        path = (base / relative).resolve()
        if not path.is_relative_to(base) or not path.is_file() or path.stat().st_size != spec['size']:
            raise RuntimeError('base asset path/size mismatch ' + relative)
        if sha(path) != spec['sha256']:
            raise RuntimeError('base asset SHA mismatch ' + relative)
    for model in ('high', 'low'):
        spec = scope['checkpoints'][model]
        if sha(spec['path']) != spec['sha256']:
            raise RuntimeError('checkpoint changed ' + model)
    expected = scope['external_sources_sha256']
    for tree in scope['python_source_roots']:
        paths = {str(p.resolve()) for p in Path(tree).rglob('*.py') if p.is_file()}
        expected_paths = {str(Path(p).resolve()) for p in expected
                          if Path(p).resolve().is_relative_to(Path(tree).resolve()) and p.endswith('.py')}
        if not paths or paths != expected_paths:
            raise RuntimeError('source tree inventory mismatch ' + tree)
    for p, d in expected.items():
        if sha(p) != d:
            raise RuntimeError('pipeline source SHA mismatch ' + p)


def validate_effective_config(config, scope, find_model_path):
    base = Path(scope['base_model_dir']).resolve()
    for key, expected in scope['required_runtime_config'].items():
        if config.get(key) != expected:
            raise RuntimeError('preload config mismatch: ' + key)
    if Path(config.get('model_path', '')).resolve() != base:
        raise RuntimeError('model path differs')
    for model in ('high', 'low'):
        val = config.get(model + '_noise_original_ckpt')
        if not val or Path(val).resolve() != Path(scope['checkpoints'][model]['path']).resolve():
            raise RuntimeError('checkpoint path mismatch ' + model)
    for key in ('dit_original_ckpt', 'adapter_model_path', 'dit_quantized_ckpt', 'high_noise_quantized_ckpt',
                'low_noise_quantized_ckpt', 't5_quantized_ckpt', 'clip_quantized_ckpt', 'lora_configs', 'lora_path', 'config_json'):
        if config.get(key) not in (None, False, '', []):
            raise RuntimeError('forbidden alternate model path: ' + key)
    for key in ('dit_quantized', 't5_quantized', 'clip_quantized', 'use_lightvae', 'dummy_model', 'lazy_load',
                'unload_modules', 'compile', 'use_compile', 'seq_parallel', 'parallel', 'mxfp8_fuse_enable',
                'video_frame_interpolation', 'video_super_resolution', 'use_image_encoder'):
        if config.get(key) not in (None, False, ''):
            raise RuntimeError('forbidden model mode: ' + key)
    if config.get('quant_method') not in (None, ''):
        raise RuntimeError('quant_method forbidden')
    if config.get('feature_caching') not in (None, 'NoCaching'):
        raise RuntimeError('feature cache forbidden')
    for key, filename in (('t5_original_ckpt', 'models_t5_umt5-xxl-enc-bf16.pth'), ('vae_path', 'Wan2.1_VAE.pth')):
        if Path(find_model_path(config, key, filename)).resolve() != (base / filename).resolve():
            raise RuntimeError('actual asset selection mismatch: ' + key)
    if config.get('vae_name') not in (None, 'Wan2.1_VAE.pth'):
        raise RuntimeError('alternate VAE name forbidden')
    return config


def build_checked_pipeline(scope, out):
    """Build the dense unfused pipeline through the hash-gated adapter, validating
    the effective config and writing EFFECTIVE_CONFIG.json. Shared by run() and the
    author smoke so the exact checked_builder path is exercised before dispatch."""
    adapter = importlib.import_module('HiFloat4.ptq.adapters.lightx2v_wan_distill')
    if str(Path(adapter.__file__).resolve()) not in scope['external_sources_sha256'] or getattr(adapter, '_PATCHED', False):
        raise RuntimeError('adapter provenance or inherited hook')
    from lightx2v.utils.utils import find_torch_model_path
    original_builder = adapter.build_effective_lightx2v_pipeline_config

    def checked_builder(pipeline):
        cfg = original_builder(pipeline)
        validate_effective_config(cfg, scope, find_torch_model_path)
        exclusive_json(out / 'EFFECTIVE_CONFIG.json', dict(cfg))
        return cfg

    adapter.build_effective_lightx2v_pipeline_config = checked_builder
    try:
        pipe = adapter.build_lightx2v_distill_pipeline(
            base_model_dir=scope['base_model_dir'],
            high_noise_ckpt=scope['checkpoints']['high']['path'],
            low_noise_ckpt=scope['checkpoints']['low']['path'])
    finally:
        adapter.build_effective_lightx2v_pipeline_config = original_builder
    config = pipe.runner.config
    for k, v in scope['required_runtime_config'].items():
        if config.get(k) != v:
            raise RuntimeError('runtime config differs: ' + k)
    if getattr(adapter, '_ACTIVE_RUNTIME', None) is not None or getattr(adapter, '_ACTIVE_OBSERVER', None) is not None \
            or getattr(adapter, '_PATCHED', False):
        raise RuntimeError('unexpected HIF4 patch')
    return pipe


def capture_source(cond, scope, scope_sha, chip_root, byproduct, fixed64_ids, budget,
                   check_resources, pipe, scheduler):
    """Run one W_D D4 trajectory for a DEV source; capture fixed64 U + PANEL512
    cross-check at the 12 target endpoint sites via the formal NativeTapFixed64."""
    import torch
    from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
    from lightx2v.models.runners.wan.wan_distill_runner import MultiDistillModelStruct
    from lightx2v.common.ops.mm.mm_weight import MMWeight
    if MMWeight.apply.__module__ != 'lightx2v.common.ops.mm.mm_weight' or MMWeight.apply.__qualname__ != 'MMWeight.apply':
        raise RuntimeError('dense MMWeight patched')
    check_resources()
    src = cond['record_id']
    video = byproduct / ('%s__chip.mp4' % src)
    cap_dir = chip_root / src
    sink = Fixed64Sink(torch, cap_dir, budget['capture_bytes_per_source'])

    def guarded_sink(event, payload):
        check_resources()
        event.update(image_sha256=cond['image_sha256'], prompt_sha256=cond['prompt_sha256'],
                     cycle_role=cond['cycle_role'], source_cluster_id=cond['source_cluster_id'],
                     scope_sha256=scope_sha, checkpoint_sha256=scope['checkpoints'][event['expert']]['sha256'])
        sink(event, payload)
        check_resources()

    tap = NativeTapFixed64(torch, guarded_sink, fixed64_ids, MMWeight, EXPECTED_GRID)
    tap.begin_source(src, cond['seed'])
    tap.install(WanTransformerInfer)
    old_dispatch = MultiDistillModelStruct.get_current_model_index
    dispatch_rows = []
    noise = []

    def dispatched(model):
        value = old_dispatch(model)
        step = int(model.scheduler.step_index)
        if step == 0 and not noise:
            noise.append(finite_tensor_sha(model.scheduler.latents, torch))
        dispatch_rows.append({'step': step, 'expert': int(model.cur_model_index)})
        tap.dispatch = int(model.cur_model_index)
        return value

    MultiDistillModelStruct.get_current_model_index = dispatched
    start = time.monotonic()
    reset_scheduler_generator(scheduler, pipe.runner.model)
    try:
        with torch.no_grad():
            pipe.generate(seed=cond['seed'], image_path=cond['image_path'], prompt=cond['prompt'],
                          negative_prompt='', save_result_path=str(video))
        tap.finish_source()
    finally:
        tap.uninstall()
        MultiDistillModelStruct.get_current_model_index = old_dispatch
    elapsed = time.monotonic() - start
    # Byproduct mp4 is required by pipe.generate but NEVER decoded/scored here.
    if not video.is_file() or not video.stat().st_size:
        raise RuntimeError('trajectory byproduct video absent ' + src)
    if dispatch_rows != [{'step': i, 'expert': 0 if i < 2 else 1} for i in range(4)] or len(noise) != 1:
        raise RuntimeError('expert dispatch/noise coverage ' + src)
    n_sites = len(list(cap_dir.glob('*.pt')))
    if n_sites != len(TARGET_SITES):
        raise RuntimeError('capture site count %d != 12 for %s' % (n_sites, src))
    check_resources()
    return {'record_id': src, 'wall_seconds': elapsed, 'initial_noise': noise[0],
            'expert_dispatch': dispatch_rows, 'capture_sites': n_sites}


def same_state_check(cond, chip_root, bank_root):
    """Bit-compare captured PANEL512 cross-check + gate vs landed bank at each
    matched (source,step,layer). Mismatch blocks only this source's Chip cells."""
    import torch
    src = cond['record_id']
    cap_dir = chip_root / src
    bank_dir = bank_root / src
    per_site = {}
    for (step, layer), site_key in sorted(SITE_OF.items(), key=lambda kv: (kv[1], kv[0])):
        role = 'anchor' if step in (0, 2) else 'hit'
        fname = '%s__seed42__s%02d__b%02d.pt' % (src, step, layer)
        cap = torch.load(cap_dir / fname, map_location='cpu', weights_only=True)
        bank = torch.load(bank_dir / fname, map_location='cpu', weights_only=True)
        u_match = bool(torch.equal(cap['U_panel_check'], bank['U']))
        gate_match = bool(torch.equal(cap['gate'], bank['gate']))
        key = '%s|%s' % (site_key, role)
        per_site[key] = {'step': step, 'layer': layer, 'site_key': site_key, 'endpoint_role': role,
                         'panel_u_bit_match': u_match, 'gate_bit_match': gate_match,
                         'bank_file_sha256': sha(bank_dir / fname),
                         'capture_file_sha256': sha(cap_dir / fname)}
        if not u_match:
            per_site[key]['u_absmax_diff'] = float((cap['U_panel_check'].float() - bank['U'].float()).abs().max())
    all_match = all(v['panel_u_bit_match'] and v['gate_bit_match'] for v in per_site.values())
    return {'all_match': all_match, 'per_endpoint': per_site}


def build_source_record(cond, arm, ss, chip_root, fixed64_ids, fixed64_sha, scope, scope_sha):
    """Per-source fixed64 input manifest (anchor/hit U_fixed64 paths + SHA + same-state)."""
    src = cond['record_id']
    cap_dir = chip_root / src
    sites = {}
    for e, L in SITES:
        sk = '%s%02d' % (e, L)
        a_step, h_step = {'high': (0, 1), 'low': (2, 3)}[e]
        entry = {}
        for role, step in (('anchor', a_step), ('hit', h_step)):
            fname = '%s__seed42__s%02d__b%02d.pt' % (src, step, L)
            side = json.loads((cap_dir / (fname + '.json')).read_text())
            entry[role] = {'step': step, 'layer': L, 'path': str(cap_dir / fname),
                           'file_sha256': side['file_sha256'], 'bytes': side['bytes'],
                           'U_fixed64_shape': side['U_fixed64_shape'], 'U_fixed64_dtype': side['U_fixed64_dtype']}
        entry['same_state'] = {
            'anchor_panel_u_bit_match': ss['per_endpoint']['%s|anchor' % sk]['panel_u_bit_match'],
            'hit_panel_u_bit_match': ss['per_endpoint']['%s|hit' % sk]['panel_u_bit_match'],
            'anchor_gate_bit_match': ss['per_endpoint']['%s|anchor' % sk]['gate_bit_match'],
            'hit_gate_bit_match': ss['per_endpoint']['%s|hit' % sk]['gate_bit_match']}
        entry['chip_cell_unblocked'] = all(entry['same_state'].values())
        sites[sk] = entry
    panel_ids, panel_info = panel512.make_panel(src, cond['seed'], EXPECTED_GRID)
    return {'record_id': src, 'cycle_role': 'DEV', 'source_cluster_id': cond['source_cluster_id'],
            'fixed64_token_ids': fixed64_ids, 'fixed64_token_id_sha256_le_int64': fixed64_sha,
            'panel_token_id_sha256_le_int64': panel_info['token_id_sha256_le_int64'],
            'initial_noise_sha256': arm['initial_noise']['sha256'],
            'image_sha256': cond['image_sha256'], 'prompt_sha256': cond['prompt_sha256'],
            'parent_recipe_sha256': scope['recipe_binding']['recipe_sha256'],
            'capture_scope_sha256': scope_sha, 'wall_seconds': arm['wall_seconds'],
            'sites': sites, 'same_state_all_match': ss['all_match'],
            'chip_cells_unblocked_count': sum(1 for v in sites.values() if v['chip_cell_unblocked'])}


def aggregate_inputs(results, fixed64_ids, fixed64_sha, p64, scope, scope_sha, out):
    """Aggregate the per-source records into B_CHIP_P64_INPUTS.json (formal path)."""
    manifest = {
        'schema': 'G20_B_CHIP_P64_INPUTS_V1', 'method_id': 'CHIPMUNK_GLOBAL_P64',
        'numeric_adapter': 'OWNW_LOCAL512_P64ANCHOR64_V1',
        'fixed64_token_ids': fixed64_ids, 'fixed64_token_id_sha256_le_int64': fixed64_sha,
        'selected_k': p64['selected_k'], 'rows_are_panel_first64': False,
        'grid': list(EXPECTED_GRID), 'flattened_tokens': FULL_M,
        'parent_recipe_sha256': scope['recipe_binding']['recipe_sha256'],
        'capture_scope_sha256': scope_sha, 'sources': results}
    exclusive_json(out / 'B_CHIP_P64_INPUTS.json', manifest)
    return sha(out / 'B_CHIP_P64_INPUTS.json')


def write_summary(conds, results, scope, out, started, torch):
    unblocked_sources = [s for s, r in results.items() if r['same_state_all_match']]
    blocked_sources = [s for s, r in results.items() if not r['same_state_all_match']]
    total_unblocked_cells = sum(r['chip_cells_unblocked_count'] for r in results.values())
    summary = {'scope_id': scope['scope_id'], 'status': 'B_CHIP_STATS_COMPLETE',
               'new_D4_full_executions': len(conds), 'new_O40_full_executions': 0,
               'new_video_decode': 0, 'new_video_scoring': 0,
               'trajectory_byproduct_videos_not_decoded': len(conds),
               'sources': len(conds), 'sites': len(SITES), 'target_captures_per_source': len(TARGET_SITES),
               'fixed64_count': FIXED64_COUNT,
               'total_fixed64_U_rows_captured': len(conds) * len(TARGET_SITES) * FIXED64_COUNT,
               'same_state_pass_sources': len(unblocked_sources),
               'same_state_fail_sources': blocked_sources,
               'bw_chip_cells_unblocked': total_unblocked_cells,
               'bw_chip_cells_expected': len(conds) * len(SITES),
               'inputs_manifest_sha256': sha(out / 'B_CHIP_P64_INPUTS.json'),
               'bank_manifest_sha256': scope['bank_manifest_sha256'],
               'elapsed_seconds': time.monotonic() - started,
               'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
               'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
               'quality_metrics': 'NOT_MEASURED', 'performance_comparison': 'NOT_BENCHMARKED',
               'timing_note': 'instrumented tap timing is NEVER production timing'}
    exclusive_json(out / 'SUMMARY.json', summary)
    return summary


def run(stage, scope, scope_sha):
    stage = Path(stage)
    budget = scope['budget']
    out = stage / 'output'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    conds = json.loads((stage / 'chip_conditions.json').read_text())
    bindings = json.loads((stage / 'B_BINDINGS.json').read_text())
    dev11 = bindings['bank']['dev11']
    verify_chip_conditions(conds, dev11)
    verify_pipeline_binding(scope)

    bank_root = Path(scope['bank_root'])
    bank_manifest_path = bank_root.parent / 'BANK_MANIFEST.json'
    if sha(bank_manifest_path) != scope['bank_manifest_sha256']:
        raise RuntimeError('bank manifest SHA mismatch')

    p64 = bindings['p64']
    fixed64_ids = p64['fixed64_token_ids']
    if len(fixed64_ids) != FIXED64_COUNT:
        raise RuntimeError('binding fixed64 count')

    def check_budget():
        if time.monotonic() - started > budget['soft_seconds']:
            raise TimeoutError('soft time budget')
        import resource
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > budget['host_memory_gib'] * 1024 ** 3:
            raise MemoryError('host peak RSS budget')
        total = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        if total > budget['output_bytes']:
            raise RuntimeError('output limit')

    for path in reversed(scope['python_import_paths']):
        sys.path.insert(0, path)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 \
            or torch.cuda.get_device_name() != 'NVIDIA A100-SXM4-80GB':
        raise RuntimeError('requires one A100 80GB')
    if torch.__version__ != scope['runtime_versions']['torch'] \
            or np.__version__ != scope['runtime_versions']['numpy']:
        raise RuntimeError('runtime version mismatch')
    if regenerate_fixed64(torch) != fixed64_ids:
        raise RuntimeError('regenerated fixed64 differs from frozen binding')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    gpu_limit = budget['gpu_memory_gib'] * 1024 ** 3

    def check_resources():
        check_budget()
        if torch.cuda.max_memory_reserved() > gpu_limit:
            raise MemoryError('GPU_MEMORY_BUDGET_EXCEEDED')

    pipe = build_checked_pipeline(scope, out)
    check_resources()
    scheduler = pipe.runner.scheduler
    chip_root = out / 'chip_inputs'
    chip_root.mkdir(exist_ok=False)
    byproduct = out / 'byproduct_video'
    byproduct.mkdir(exist_ok=False)
    fixed64_sha = hashlib.sha256(np.ascontiguousarray(fixed64_ids, dtype='<i8').tobytes()).hexdigest()
    results = {}

    for cond in conds:
        src = cond['record_id']
        arm = capture_source(cond, scope, scope_sha, chip_root, byproduct, fixed64_ids, budget,
                             check_resources, pipe, scheduler)
        ss = same_state_check(cond, chip_root, bank_root)
        rec = build_source_record(cond, arm, ss, chip_root, fixed64_ids, fixed64_sha, scope, scope_sha)
        results[src] = rec
        exclusive_json(out / ('CHIP_CAPTURE_%s.json' % src), {**arm, 'same_state': ss})
        print('B_CHIP_CAPTURED', src, 'same_state_all_match=%s' % ss['all_match'],
              'unblocked=%d/6' % rec['chip_cells_unblocked_count'], flush=True)

    aggregate_inputs(results, fixed64_ids, fixed64_sha, p64, scope, scope_sha, out)
    check_budget()
    summary = write_summary(conds, results, scope, out, started, torch)
    return {'status': 'B_CHIP_STATS_COMPLETE', 'new_D4_full_executions': len(conds),
            'sources': len(conds), 'same_state_pass_sources': summary['same_state_pass_sources'],
            'bw_chip_cells_unblocked': summary['bw_chip_cells_unblocked'],
            'elapsed_seconds': time.monotonic() - started,
            'peak_rss_bytes': int(__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss) * 1024,
            'summary_sha256': sha(out / 'SUMMARY.json')}