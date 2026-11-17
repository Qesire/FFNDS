"""Two-source, six-site true-M Chipmunk replay. No full-model or video calls."""

from contextlib import contextmanager
import json
import hashlib
import importlib.util
import math
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from chip_provider import load_author_provider
from icc_stage3_assets import checked_path, load_weights, verify_live_weights, sha
from icc_local_replay import delta_alignment, gated_risk


SITES = tuple((e, b) for e in ('high', 'low') for b in (4, 19, 34))
SOURCES = ('vbi_1013', 'vbi_0800')
FULL_ROWS = 31668
PREFIX_ROWS = 247 * 128
WATCH_BLOCKS = (0, 123, 246)


class DenseLinear:
    def __init__(self, weight, bias):
        self.weight, self.bias = weight, bias

    def __call__(self, value):
        return F.linear(value, self.weight, self.bias)


def load_input(spec, device, *, source, step, layer):
    if (spec['kind'] != 'stage1_full_u' or spec['source'] != source or
        spec['step'] != step or spec['layer'] != layer or spec['seed'] != 42):
        raise RuntimeError('capture site identity mismatch')
    panel = torch.load(checked_path(spec['panel']), map_location='cpu', weights_only=True)
    u = torch.load(checked_path(spec['full_u']), map_location='cpu', mmap=True, weights_only=True)
    if u.shape != (FULL_ROWS, 5120) or u.dtype != torch.bfloat16:
        raise RuntimeError('real full-M BF16 operand required')
    idx = panel['token_indices'].long()
    if (idx.shape != (512,) or idx.min().item() < 0 or idx.max().item() >= FULL_ROWS
            or torch.unique(idx).numel() != 512): raise RuntimeError('invalid panel rows')
    identity = hashlib.sha256(idx.numpy().astype('<i8', copy=False).tobytes()).hexdigest()
    if identity != spec['token_id_sha256_le_int64']: raise RuntimeError('panel row identity mismatch')
    y, gate = panel['Y_fullM_panel'], panel['gate'].reshape(-1)
    if y.shape != (512,5120) or y.dtype != torch.bfloat16 or gate.numel() != 5120 or not torch.isfinite(gate).all():
        raise RuntimeError('invalid panel output or gate')
    return u.to(device), {'indices': idx.to(device), 'gate': gate.to(device), 'target': y.to(device)}


def terms(u, w1, b1, w2, h_cache, y_cache, indices, *, author):
    aw, ab, bw = w1[indices], b1[indices], w2[:, indices]
    if author:
        preactivation = torch.addmm(ab.float(), u.float(), aw.float().T)
    else:
        preactivation = F.linear(u, aw, ab)
    new_h = F.gelu(preactivation, approximate='tanh').to(torch.bfloat16)
    delta = (new_h.float() - h_cache.float()).to(torch.bfloat16)
    w2_delta = delta.float() @ bw.float().T
    if author:
        w2_delta = w2_delta.to(torch.bfloat16).float()
    return new_h, delta, w2_delta, (y_cache.float() + w2_delta).to(torch.bfloat16)


def score_parts(reference, actual):
    if reference.shape != actual.shape:
        raise ValueError('same-mask arithmetic shapes differ')
    if reference.numel() == 0:
        return {'max_abs': None, 'rms': None, 'compared_values': 0}
    diff = actual.float() - reference.float()
    return {'max_abs': float(diff.abs().max()), 'rms': float(diff.square().mean().sqrt()),
            'compared_values': diff.numel()}


@contextmanager
def trace_author_intermediates(watches):
    import chipmunk.triton as triton

    old1 = triton.csp_mlp_mm1_bf16
    trace = {}

    def mm1(x, w, packed, b, h, indices, counts):
        value = old1(x, w, packed, b, h, indices, counts)
        for block in watches:
            n = int(counts[block])
            sl = slice(block * 128, (block + 1) * 128)
            trace[block] = {'indices': indices[block, :n].long().clone(),
                            'delta': packed[sl, :n].clone()}
        return value

    triton.csp_mlp_mm1_bf16 = mm1
    try:
        yield trace
    finally:
        triton.csp_mlp_mm1_bf16 = old1


@contextmanager
def trace_selector_seeds():
    import chipmunk.ops as ops
    original = ops.topk_indices
    records = []
    def traced(activation, indices, counts, *args):
        if activation.dtype != torch.bfloat16 or not activation.is_contiguous():
            raise RuntimeError('unexpected selector proxy layout')
        # The CUDA kernel first advances activation to each row, then adds the
        # signed int32 made from that row's first two BF16 values to its thread id.
        words = activation.view(torch.int32)[..., 0].to('cpu').reshape(-1).tolist()
        records.append({'rows': activation.shape[1], 'batch': activation.shape[0],
                        'threads_per_block': 1024, 'first_bf16_pair_as_int32': words})
        return original(activation, indices, counts, *args)
    ops.topk_indices = traced
    try:
        yield records
    finally:
        ops.topk_indices = original


def replay_author(u0, u1, weights, perm, inv, provider, layer, *, compare, tail_class):
    w1, b1, w2, b2 = weights
    sparse, counter = provider
    clock = counter(40, 1)
    clock.cur_layer = layer
    first, second = DenseLinear(w1, b1), DenseLinear(w2, b2)
    module = sparse(layer, clock, first, lambda x: F.gelu(x, approximate='tanh'), second)
    x0, x1 = u0.index_select(0, perm), u1.index_select(0, perm)
    module(x0[:PREFIX_ROWS].unsqueeze(0).contiguous())
    old_h, old_y = module.storage.get_sparse_act_T()[0], module.storage.get_out_cache()[0]
    samples = {}
    if compare:
        for block in WATCH_BLOCKS:
            sl = slice(block * 128, (block + 1) * 128)
            samples[block] = old_h[:, sl].T.clone(), old_y[sl].clone()
    clock.cur_inference_step, clock.cur_layer = 1, layer
    with trace_author_intermediates(samples) as trace:
        prefix = module(x1[:PREFIX_ROWS].unsqueeze(0).contiguous())[0]
    tail_current = F.gelu(first(x1[PREFIX_ROWS:]), approximate='tanh')
    tail = second(tail_current)
    records = []
    valid_tail = None
    if compare:
        h = module.storage.get_sparse_act_T()[0]
        for block, (h_cache, y_cache) in samples.items():
            sl = slice(block * 128, (block + 1) * 128)
            indices = trace[block]['indices']
            a, delta, w2_delta, pred = terms(x1[sl], w1, b1, w2, h_cache[:, indices], y_cache, indices, author=True)
            n, nd, nw, ny = terms(x1[sl], w1, b1, w2, h_cache[:, indices], y_cache, indices, author=False)
            records.append({'block': block, 'selected': int(indices.numel()),
                            'author_h_check': score_parts(a, h[indices, sl].T), 'author_delta_check': score_parts(delta, trace[block]['delta']),
                            'author_y_check': score_parts(pred, prefix[sl]), 'native_order_h': score_parts(a, n),
                            'native_order_delta': score_parts(delta, nd), 'native_order_w2': score_parts(w2_delta, nw),
                            'native_order_y': score_parts(pred, ny)})
        # Execute actual author kernels with a selector formed solely from 52 rows.
        import chipmunk.ops as ops
        from chipmunk.util.config import GLOBAL_CONFIG
        candidate = tail_class(first, second, ops, GLOBAL_CONFIG['mlp'])
        _, anchor_info = candidate.forward(x0[PREFIX_ROWS:], 0)
        ah = candidate.h[:, :52].T.clone()
        ay = candidate.y[:52].clone()
        valid_tail, hit_info = candidate.forward(x1[PREFIX_ROWS:], 1, padding_probe=True)
        indices = candidate.indices[0, 0, :hit_info['count']].long()
        ref_h, _, _, ref_y = terms(x1[PREFIX_ROWS:], w1, b1, w2,
                                   ah[:, indices], ay, indices, author=True)
        records.append({'block': 'tail52', 'selected': hit_info['count'],
                        'execution': 'AUTHOR_TRITON_VALID52_PAD128',
                        'anchor': anchor_info, 'hit': hit_info,
                        'author_h_check': score_parts(ref_h, candidate.h[indices, :52].T),
                        'author_y_check': score_parts(ref_y, valid_tail),
                        'dense_vs_valid': score_parts(tail, valid_tail),
                        'reference_note': 'FP32 reference GEMM reduction order may differ from Triton'})
    y = torch.cat((prefix, tail)).index_select(0, inv)
    counts = module.storage.get_counts()[0].tolist()
    if len(counts) != 247:
        raise RuntimeError('full-M author path did not select 247 real blocks')
    return y, {'block_counts': counts, 'same_mask_diagnostics': records}, valid_tail


def load_bound_module(spec, name):
    path = checked_path(spec)
    modspec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(modspec)
    modspec.loader.exec_module(module)
    return module


def load_canonical(scope, pair, source, expert, layer):
    manifest = json.loads(checked_path(scope['canonical_cache_manifest']).read_text(encoding='utf-8'))
    grid_scope_path = checked_path(scope['canonical_grid_scope'])
    if manifest['scope_sha256'] != sha(grid_scope_path):
        raise RuntimeError('canonical cache grid scope mismatch')
    grid_scope = json.loads(grid_scope_path.read_text(encoding='utf-8'))
    sites = [v for v in grid_scope['input_pairs'] if (v['source'],v['expert'],v['layer']) == (source,expert,layer)]
    if len(sites) != 1 or sites[0] != pair:
        raise RuntimeError('canonical target has different input identity')
    if grid_scope['checkpoints'][expert] != scope['checkpoints'][expert]:
        raise RuntimeError('canonical checkpoint mismatch')
    name = f'native__{source}__{expert}__{layer:02d}.pt'
    files = [v for v in manifest['files'] if v['file'] == name]
    if len(files) != 1 or manifest['status'] != 'PRESERVED_PENDING_SELECTED_FAMILY_ERROR_DECOMPOSITION':
        raise RuntimeError('canonical cache receipt missing or duplicate')
    path = Path(scope['canonical_cache_manifest']['path']).resolve().parent / 'cache' / name
    if not path.is_file() or path.stat().st_size != files[0]['bytes'] or sha(path) != files[0]['sha256']:
        raise RuntimeError('canonical endpoint bytes mismatch')
    native = torch.load(path, map_location='cpu', weights_only=True)
    for key in ('h0','h1','y0','y1'):
        shape = (512, 13824 if key.startswith('h') else 5120)
        if key not in native or native[key].shape != shape or native[key].dtype != torch.bfloat16:
            raise RuntimeError('canonical endpoint shape/dtype mismatch')
    return native['y0'].cuda(), native['y1'].cuda(), files[0]['sha256']


def execute(scope, output, *, authorize, check_resources):
    """Reviewed external guard must authorize scope and resources before GPU work."""
    if not callable(authorize) or not callable(check_resources): raise RuntimeError('guard required')
    authorize(scope); check_resources()
    for path, identity in scope['implementation_sha256'].items():
        if sha(path) != identity: raise RuntimeError('implementation identity mismatch')
    if len(scope['implementation_sha256']) < 5: raise RuntimeError('unbound implementation')
    proxy = load_bound_module(scope['gp64_proxy_source'], 'ffnds_true_m_proxy')
    action = load_bound_module(scope['gp64_action_source'], 'ffnds_true_m_action')
    rms_score, selection, packed_pregate = proxy.rms_score, proxy.selection, action.packed_pregate
    tail_class = load_bound_module(scope['tail_candidate_source'], 'ffnds_valid_tail52').ValidCountTail52
    if not os.environ.get('SLURM_JOB_ID') or torch.cuda.device_count() != 1: raise RuntimeError('one Slurm GPU required')
    prop = torch.cuda.get_device_properties(0)
    if (prop.major,prop.minor) != (8,0) or 'A100' not in prop.name: raise RuntimeError('A100 required')
    if scope['num_videos'] != 0 or scope['num_model_nfe'] != 0: raise RuntimeError('unexpected model or video budget')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(int(scope['selector_seed']))

    if scope['max_predictions'] != 24 or scope['max_arithmetic_pairs'] != 4:
        raise RuntimeError('unregistered true-M comparison count')
    floor = scope['decomposition_norm_floor']
    if not math.isfinite(floor) or floor != 1e-12:
        raise ValueError('need frozen 1e-12 norm floor')
    pairs = json.loads(checked_path(scope['dev_pairs']).read_text(encoding='utf-8'))['input_pairs']
    selected = [row for row in pairs if row['source'] in SOURCES]
    index = {(row['source'], row['expert'], row['layer']): row for row in selected}
    if len(selected) != 12: raise RuntimeError('duplicate or missing true-M pairs')
    if set(index) != {(source, expert, layer) for source in SOURCES for expert, layer in SITES}:
        raise RuntimeError('exactly two neutral sources and six sites per source required')
    coords = torch.load(checked_path(scope['voxel']), map_location='cpu', weights_only=True)
    perm, inv = coords['voxel_to_raster'].cuda(), coords['raster_to_voxel'].cuda()
    if not torch.equal(perm[inv], torch.arange(FULL_ROWS, device='cuda')):
        raise RuntimeError('author voxel map does not round-trip')
    rows = torch.tensor(scope['fixed64_token_ids'], dtype=torch.long, device='cuda')
    capture = json.loads(checked_path(scope['capture_manifest']).read_text(encoding='utf-8'))
    if scope['fixed64_token_ids'] != capture['fixed64_token_ids']:
        raise RuntimeError('fixed64 does not match capture manifest')
    if rows.shape != (64,) or torch.unique(rows).numel() != 64 or rows.min().item() < 0 or rows.max().item() >= FULL_ROWS:
        raise RuntimeError('invalid fixed64 raster rows')
    provider = load_author_provider(scope['provider'])
    stats = []
    output = Path(output)
    output.mkdir()
    for (source, expert, layer), pair in index.items():
        check_resources()
        anchor_local, target_local, canonical_sha = load_canonical(scope, pair, source, expert, layer)
        if pair['anchor']['input_identity'] != pair['hit']['input_identity']:
            raise RuntimeError('anchor/hit condition identity mismatch')
        if pair['anchor']['input_identity']['checkpoint_sha256'] != scope['checkpoints'][expert]['sha256']:
            raise RuntimeError('input/checkpoint identity mismatch')
        step0, step1 = (0, 1) if expert == 'high' else (2, 3)
        inp0, panel0 = load_input(pair['anchor'], 'cuda', source=source, step=step0, layer=layer)
        inp1, panel1 = load_input(pair['hit'], 'cuda', source=source, step=step1, layer=layer)
        score_rows, gate, target = panel1['indices'], panel1['gate'], panel1['target']
        if not torch.equal(score_rows, panel0['indices']): raise RuntimeError('anchor/hit panel rows differ')
        mats = load_weights(scope['checkpoints'][expert], layer, 'cuda')
        verify_live_weights(scope['capture_manifest'], expert, layer, *mats)
        z = F.linear(inp0, mats[0], mats[1])
        cached = F.gelu(z, approximate='tanh')
        anchor = F.linear(cached, mats[2], mats[3])
        proxy = rms_score(F.linear(inp1[rows], mats[0], mats[1]), cached[rows])
        mask = selection(proxy, 5120)
        gp64 = packed_pregate(inp1, mats[0], mats[1], mats[2], cached, anchor, mask)
        record = {'source': source, 'expert': expert, 'layer': layer, 'gp64_risk': gated_risk(gp64[score_rows], target_local, gate), 'gp64_risk_vs_Y_fullM_panel': gated_risk(gp64[score_rows], target, gate), 'dense_local512_vs_fullM_panel': gated_risk(target_local,target,gate), 'canonical_sha256': canonical_sha}
        record['gp64_delta_alignment'] = delta_alignment(anchor_local, target_local, gp64[score_rows], gate, floor)
        del z, cached, anchor, gp64
        with trace_selector_seeds() as seed_trace:
            author, trace, valid_tail = replay_author(inp0, inp1, mats, perm, inv, provider, layer, compare=layer == 19, tail_class=tail_class)
        record['compiled_curand_seed_inputs'] = seed_trace
        record['author_risk'] = gated_risk(author[score_rows], target_local, gate)
        record['author_risk_vs_Y_fullM_panel'] = gated_risk(author[score_rows], target, gate)
        record['author_delta_alignment'] = delta_alignment(anchor_local, target_local, author[score_rows], gate, floor)
        record.update(trace)
        if valid_tail is not None:
            voxel_rows = inv[score_rows]
            is_tail = voxel_rows >= PREFIX_ROWS
            sample_alt = author[score_rows].clone()
            sample_alt[is_tail] = valid_tail[voxel_rows[is_tail] - PREFIX_ROWS]
            record['tail_e6_candidate'] = {
                'status': 'AUTHOR_TRITON_CANDIDATE_PENDING_REVIEW',
                'panel_tail_rows': int(is_tail.sum().item()),
                'risk_vs_local512': gated_risk(sample_alt, target_local, gate),
                'risk_vs_fullM_panel': gated_risk(sample_alt, target, gate),
                'gated_change_from_dense_tail': gated_risk(sample_alt, author[score_rows], gate),
                'gated_change_on_all_52_tail_rows': gated_risk(valid_tail, author[perm[PREFIX_ROWS:]], gate),
            }

        with (output / f'{source}_{expert}_{layer:02d}.json').open('x', encoding='utf-8') as file:
            json.dump(record, file, indent=2, allow_nan=False)
        stats.append(record)
        del inp0, inp1, panel0, panel1, anchor_local, target_local, mats, author, valid_tail
        check_resources()
    if len(stats) != 12 or sum(bool(row['same_mask_diagnostics']) for row in stats) != 4: raise RuntimeError('incomplete prescribed comparisons')
    with (output / 'SUMMARY.json').open('x', encoding='utf-8') as file:
        json.dump({'status': 'COMPLETE', 'prediction_count': 2 * len(stats), 'arithmetic_pairs': 4, 'risk_formula': 'mean_float64(((prediction-target)*gate)**2); denominator=512*5120', 'torch_seed': 42, 'curand_seed_rule': 'uint64(block_x*1024 + block_y*grid_x*1024 + thread_x + signed_int32(first_two_row_bf16))', 'rows': stats}, file, indent=2, allow_nan=False)
