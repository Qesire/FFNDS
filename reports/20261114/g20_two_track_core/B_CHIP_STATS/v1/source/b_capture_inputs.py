"""B_CHIP_STATS fixed64 U-input capture producer (conditional input production).

Adapted from the proven V2 capture_producer.NativeTap512 (sha a9748f38), preserving
the Wan hook structure and every identity guard: infer_block exposes the runtime
patch grid; infer_ffn wraps phase.ffn_0.apply (W1); the pre-W1 input z is the full-M
U; c_gate_msa is the gate G; expert dispatch is HHLL (high steps 0,1; low 2,3).

DIFFERENCE vs NativeTap512 (the B_CHIP_STATS estimand):
  * NativeTap512 saves U at the 512 PANEL coordinates at all 160 sites. The landed
    V2 bank therefore contains ONLY the PANEL512 slice, NOT full-M U. The Chip
    GLOBAL_P64 selector needs the real fixed64 spatial-token rows (CPU seed42
    randperm(31668)[:64], NOT the panel first64). 696 (DEV source x token-ID) gaps
    are absent from the bank, so this producer re-runs the SAME W_D D4 native
    trajectory and captures U at the fixed64 coordinates.
  * Captures ONLY at the 12 target endpoint sites (6 sites x 2 endpoints):
    high{04,19,34} steps 0->1, low{04,19,34} steps 2->3. The model still runs the
    full 4x40 trajectory; non-target FFN calls pass through untouched.
  * Per target site saves U_fixed64 = z[fixed64_ids] (64,5120) AND a PANEL512
    cross-check U_panel_check = z[panel_ids] (512,5120). The runner bit-compares
    U_panel_check against the landed bank U at the exact matched (source,step,layer)
    to confirm the captured fixed64 rows come from the SAME native state. A mismatch
    blocks only that source's Chip cells (never silently spliced into the BW table).
  * U-only: the Chip own-W anchor/hit are recomputed locally from U0_64/U1_64 in BW;
    no Y_pre_gate at fixed64 is required, so W2 is not wrapped. The eval512 anchor/
    target and G_ref continue to come from the landed bank PANEL512.

The observer is trajectory-neutral: D2H copies complete before W1, no detach alias
is held across apply, and weights/gate/scheduler/latents are never modified.
Instrumented (tap) timing is NEVER production timing. No full-M block statistics,
no full-M F1, no tubelet, no stratified mask, no new source/image/prompt.
"""
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

import panel512

PANEL_VERSION = panel512.VERSION
PANEL_COUNT = panel512.PANEL_COUNT
EXPECTED_GRID = (21, 29, 52)
DIM = 5120
FFN_DIM = 13824
STEPS = 4
LAYERS = 40
FULL_M = int(np.prod(EXPECTED_GRID))  # 31668
FIXED64_COUNT = 64
SEED = 42

SITES = [('high', 4), ('high', 19), ('high', 34), ('low', 4), ('low', 19), ('low', 34)]
EXPERT_STEP = {'high': (0, 1), 'low': (2, 3)}
# (step, layer) -> site_key ; the 12 target endpoint captures per source.
SITE_OF = {}
for _e, _L in SITES:
    for _s in EXPERT_STEP[_e]:
        SITE_OF[(_s, _L)] = '%s%02d' % (_e, _L)
TARGET_SITES = set(SITE_OF)


def _sha_i64(ids):
    return hashlib.sha256(np.ascontiguousarray(ids, dtype='<i8').tobytes()).hexdigest()


def regenerate_fixed64(torch):
    """The frozen CHIPMUNK_GLOBAL_P64 fixed64 spatial token IDs (CPU seed42)."""
    g = torch.Generator(device='cpu').manual_seed(SEED)
    return torch.randperm(FULL_M, generator=g)[:FIXED64_COUNT].tolist()


class NativeTapFixed64:
    """Full-trajectory observer saving fixed64 U + PANEL512 cross-check at 12 sites.

    torch is injected by the runner (this module does not import torch at load).
    sink is a Fixed64Sink. expected_w2_class is the dense MMWeight class.
    """

    def __init__(self, torch, sink, fixed64_ids, expected_w2_class=None, expected_grid=EXPECTED_GRID):
        self.torch, self.sink = torch, sink
        if len(fixed64_ids) != FIXED64_COUNT or len(set(fixed64_ids)) != FIXED64_COUNT:
            raise ValueError('fixed64 must be 64 distinct token IDs')
        if any(not (0 <= int(t) < FULL_M) for t in fixed64_ids):
            raise ValueError('fixed64 token ID outside full-M domain')
        self.fixed64_ids = [int(t) for t in fixed64_ids]
        self.fixed64_sha = _sha_i64(self.fixed64_ids)
        self.expected_w2_class = expected_w2_class
        self.expected_grid = tuple(expected_grid) if expected_grid is not None else None
        self.context = None
        self.grid = None
        self.seen = set()
        self.owner = None
        self.dispatch = None
        self.source_grid = None
        self.panel_ids = None
        self.panel_info = None
        self.panel_sha = None
        self.originals = None

    def begin_source(self, source, seed):
        if self.context is not None or not isinstance(source, str) or not source \
                or not isinstance(seed, int) or isinstance(seed, bool) or seed != SEED:
            raise ValueError('invalid or overlapping source/seed')
        self.context = (source, seed)
        self.seen.clear()
        self.source_grid = None
        self.panel_ids = None
        self.panel_info = None
        self.panel_sha = None

    def finish_source(self):
        if self.context is None or self.seen != TARGET_SITES:
            raise RuntimeError('incomplete/duplicate trajectory: %d/12 target sites' % len(self.seen))
        self.context = None

    def _ensure_panel(self, grid):
        if self.panel_ids is None:
            ids, info = panel512.make_panel(self.context[0], self.context[1], grid, PANEL_COUNT)
            self.panel_ids = ids
            self.panel_info = info
            self.panel_sha = info['token_id_sha256_le_int64']
        else:
            ids, info = panel512.make_panel(self.context[0], self.context[1], grid, PANEL_COUNT)
            if info['token_id_sha256_le_int64'] != self.panel_sha:
                raise RuntimeError('cross-site panel identity mismatch (key must have no step/block)')
            if not np.array_equal(ids, self.panel_ids):
                raise RuntimeError('cross-site panel ids mismatch')
        return self.panel_ids, self.panel_info

    def install(self, owner):
        if self.owner is not None:
            raise RuntimeError('already installed')
        tap = self
        oldblock = owner.infer_block
        oldffn = owner.infer_ffn

        def block(self, block, x, pre):
            if tap.grid is not None:
                raise RuntimeError('nested block unsupported')
            g = getattr(getattr(pre, 'grid_sizes', None), 'tuple', None)
            if g is None:
                raise RuntimeError('runtime grid not exposed')
            tap.grid = tuple(int(v) for v in g)
            if tap.expected_grid is not None and tap.grid != tap.expected_grid:
                tap.grid = None
                raise RuntimeError('grid differs from reviewed geometry')
            try:
                return oldblock(self, block, x, pre)
            finally:
                tap.grid = None

        def ffn(self, phase, x, attn_out, c_shift_msa, c_scale_msa, c_gate_msa=None):
            if tap.context is None or tap.grid is None:
                raise RuntimeError('missing context/grid')
            if hasattr(phase, 'smooth_norm2_weight') or hasattr(phase, 'smooth_norm2_bias'):
                raise RuntimeError('folded smooth norm is forbidden')
            if self._use_mxfp8_quant_fuse():
                raise RuntimeError('fused quantization forbidden')
            step = int(self.scheduler.step_index)
            layer = int(self.block_idx)
            if step not in range(STEPS) or layer not in range(LAYERS):
                raise RuntimeError('bad step/layer')
            if c_gate_msa is None:
                raise RuntimeError('missing gate')
            if tap.dispatch not in (0, 1) or tap.dispatch != (0 if step < 2 else 1):
                raise RuntimeError('actual expert dispatch missing or inconsistent (HHLL required)')
            # Non-target sites pass through untouched (no capture, no D2H).
            if (step, layer) not in TARGET_SITES:
                return oldffn(self, phase, x, attn_out, c_shift_msa, c_scale_msa, c_gate_msa)
            if (step, layer) in tap.seen:
                raise RuntimeError('duplicate target step/layer')
            times = getattr(self.scheduler, 'timesteps', None)
            if times is None:
                raise RuntimeError('actual scheduler timesteps unavailable')
            value = times[step]
            timestep = float(value.item() if hasattr(value, 'item') else value)
            if tap.expected_w2_class is not None:
                if any(type(f) is not tap.expected_w2_class or getattr(f, 'bias', None) is None or getattr(f, 'has_lora_branch', True) for f in (phase.ffn_0, phase.ffn_2)):
                    raise RuntimeError('W1/W2 must be dense MMWeight with bias and no LoRA')
                for f, shape in ((phase.ffn_0, (DIM, FFN_DIM)), (phase.ffn_2, (FFN_DIM, DIM))):
                    if hasattr(f, 'weight_diff') or hasattr(f, 'bias_diff'):
                        raise RuntimeError('weight modifications forbidden')
                    w = f._get_actual_weight()
                    buf = getattr(f, 'weight_cuda_buffer', None)
                    if buf is None or w.device.type != 'cuda' or tuple(w.shape) != shape or w.data_ptr() != buf.data_ptr():
                        raise RuntimeError('actual prefetched weight ownership unverified')
            oldapply = phase.ffn_0.apply
            called = 0

            def apply(z):
                nonlocal called
                called += 1
                if called != 1:
                    raise RuntimeError('multiple W1 calls')
                if z.dtype != tap.torch.bfloat16 or z.ndim != 2 or z.shape[1] != DIM:
                    raise RuntimeError('unexpected W1 input dtype/shape')
                if len(tap.grid) != 3 or int(np.prod(tap.grid)) != z.shape[0] or z.shape[0] != FULL_M:
                    raise RuntimeError('grid/token geometry mismatch')
                if tap.source_grid is None:
                    tap.source_grid = tap.grid
                if tap.source_grid != tap.grid:
                    raise RuntimeError('grid changed across source steps')
                panel_ids, info = tap._ensure_panel(tap.grid)
                t, h, w = tap.grid
                ix64 = tap.torch.as_tensor(tap.fixed64_ids, device=z.device, dtype=tap.torch.long)
                ixp = tap.torch.as_tensor(panel_ids, device=z.device, dtype=tap.torch.long)
                coords64 = np.stack((np.array(tap.fixed64_ids) // (h * w),
                                     (np.array(tap.fixed64_ids) // w) % h,
                                     np.array(tap.fixed64_ids) % w), axis=1)
                # D2H completes here; never hold a detach alias until after apply.
                payload = {
                    'U_fixed64': z.index_select(0, ix64).detach().to('cpu').clone(),
                    'U_panel_check': z.index_select(0, ixp).detach().to('cpu').clone(),
                    'gate': c_gate_msa.detach().to('cpu').clone(),
                    'token_indices_fixed64': tap.torch.as_tensor(np.array(tap.fixed64_ids, dtype='<i8'), dtype=tap.torch.long),
                    'token_indices_panel': tap.torch.as_tensor(panel_ids.copy(), dtype=tap.torch.long),
                    'token_thw_fixed64': tap.torch.as_tensor(coords64.copy(), dtype=tap.torch.long),
                }
                if any(not tap.torch.isfinite(v).all().item() for k, v in payload.items()
                       if k not in ('token_indices_fixed64', 'token_indices_panel', 'token_thw_fixed64')):
                    raise RuntimeError('nonfinite capture')
                event = {
                    'source': tap.context[0], 'seed': tap.context[1], 'step': step, 'layer': layer,
                    'expert': 'high' if tap.dispatch == 0 else 'low', 'timestep': timestep,
                    'site_key': SITE_OF[(step, layer)],
                    'endpoint_role': 'anchor' if step in (0, 2) else 'hit',
                    'grid': list(tap.grid), 'flattened_tokens': int(z.shape[0]),
                    'fixed64_count': FIXED64_COUNT, 'panel_count': PANEL_COUNT,
                    'panel_version': PANEL_VERSION,
                    'fixed64_token_id_sha256_le_int64': tap.fixed64_sha,
                    'token_id_sha256_le_int64': info['token_id_sha256_le_int64'],
                    'sorted_token_id_sha256_le_int64': info['sorted_token_id_sha256_le_int64'],
                    'coordinate_status': 'FIXED64_FULLM_TOKEN_IDS_PLUS_PANEL512_CROSSCHECK_SHARED_CROSS_SITE',
                    'U_fixed64_shape': [FIXED64_COUNT, DIM], 'U_fixed64_dtype': 'bfloat16',
                    'U_panel_check_shape': [PANEL_COUNT, DIM], 'U_panel_check_dtype': 'bfloat16',
                    'gate_shape': list(c_gate_msa.shape), 'gate_dtype': str(c_gate_msa.dtype),
                    'rows_are_panel_first64': False,
                }
                tap.sink(event, payload)
                return oldapply(z)

            phase.ffn_0.apply = apply
            try:
                out = oldffn(self, phase, x, attn_out, c_shift_msa, c_scale_msa, c_gate_msa)
            finally:
                phase.ffn_0.apply = oldapply
            if called != 1:
                raise RuntimeError('W1 was not called')
            tap.seen.add((step, layer))
            return out

        owner.infer_block = block
        owner.infer_ffn = ffn
        self.owner = owner
        self.originals = (oldblock, oldffn)

    def uninstall(self):
        if self.owner is None:
            return
        self.owner.infer_block, self.owner.infer_ffn = self.originals
        self.owner = None


class Fixed64Sink:
    """Exclusive per-site sink: write-once .pt + typed .pt.json sidecar, byte budget."""

    def __init__(self, torch, root, byte_limit):
        self.torch = torch
        self.root = Path(root)
        self.root.mkdir(parents=False, exist_ok=False)
        if not isinstance(byte_limit, int) or byte_limit <= 0:
            raise ValueError('invalid byte budget')
        self.limit = byte_limit
        self.used = 0

    def __call__(self, event, payload):
        source = event['source']
        step = event['step']
        layer = event['layer']
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,96}', source) or (step, layer) not in TARGET_SITES:
            raise ValueError('bad site identity')
        if payload['U_fixed64'].shape != (FIXED64_COUNT, DIM) or payload['U_panel_check'].shape != (PANEL_COUNT, DIM):
            raise ValueError('U_fixed64 must be (64,5120) and U_panel_check (512,5120)')
        if payload['token_indices_fixed64'].shape != (FIXED64_COUNT,) or payload['token_indices_panel'].shape != (PANEL_COUNT,):
            raise ValueError('coordinate shape mismatch')
        if event['fixed64_token_id_sha256_le_int64'] != _sha_i64(payload['token_indices_fixed64'].numpy()):
            raise ValueError('fixed64 token id sha mismatch')
        if event['token_id_sha256_le_int64'] != _sha_i64(payload['token_indices_panel'].numpy()):
            raise ValueError('panel token id sha mismatch')
        name = f'{source}__seed{event["seed"]}__s{step:02}__b{layer:02}.pt'
        path = self.root / name
        raw = sum(v.numel() * v.element_size() for v in payload.values())
        allowance = raw + 1024 * 1024
        if self.used + allowance > self.limit:
            raise RuntimeError('capture byte budget exceeded')
        self.used += allowance
        with path.open('xb') as f:
            self.torch.save({'metadata': event, **payload}, f)
            f.flush()
            os.fsync(f.fileno())
        n = path.stat().st_size
        if n > allowance:
            raise RuntimeError('serializer exceeded reserved bytes')
        self.used -= allowance - n
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
                h.update(chunk)
        receipt = {**event, 'file': name, 'file_sha256': h.hexdigest(), 'bytes': n,
                   'consumer_contract': 'CHIP_FIXED64_U_TYPED_V1'}
        with (self.root / (name + '.json')).open('x', encoding='utf-8') as f:
            json.dump(receipt, f, ensure_ascii=False, sort_keys=True, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        self.used += (self.root / (name + '.json')).stat().st_size
        if self.used > self.limit:
            raise RuntimeError('capture byte budget including sidecar exceeded')


CONSUMER_CONTRACT = {
    'contract_id': 'CHIP_FIXED64_U_TYPED_V1',
    'panel_version': PANEL_VERSION,
    'target_sites_per_source': len(TARGET_SITES),
    'sites': ['%s%02d' % (e, L) for e, L in SITES],
    'expert_schedule': 'HHLL', 'anchor_steps': [0, 2], 'hit_steps': [1, 3],
    'grid': list(EXPECTED_GRID), 'flattened_tokens': FULL_M,
    'fixed64_count': FIXED64_COUNT, 'panel_count': PANEL_COUNT, 'dim': DIM, 'ffn_dim': FFN_DIM,
    'rows_are_panel_first64': False,
    'payload_tensors': {
        'U_fixed64': {'dtype': 'bfloat16', 'shape': [FIXED64_COUNT, DIM], 'semantics': 'pre-W1 input rows at the frozen fixed64 full-M token IDs'},
        'U_panel_check': {'dtype': 'bfloat16', 'shape': [PANEL_COUNT, DIM], 'semantics': 'pre-W1 input rows at PANEL512 coords; bit-compared to landed bank U for same-state identity'},
        'gate': {'dtype': 'bfloat16', 'shape': [1, 1, DIM], 'semantics': 'c_gate_msa G'},
        'token_indices_fixed64': {'dtype': 'int64_le', 'shape': [FIXED64_COUNT], 'semantics': 'frozen fixed64 flat THW ids, original order'},
        'token_indices_panel': {'dtype': 'int64_le', 'shape': [PANEL_COUNT], 'semantics': 'PANEL512 flat THW ids, draw order'},
        'token_thw_fixed64': {'dtype': 'int64', 'shape': [FIXED64_COUNT, 3], 'semantics': '(t,h,w) coords of fixed64 ids'},
    },
    'same_state_rule': 'U_panel_check must torch.equal the landed bank U at exact matched (source,step,layer); mismatch blocks only that source Chip cells',
    'no_full_M_block_stats': True, 'no_full_M_F1': True, 'no_tubelet': True, 'no_stratified_mask': True,
    'observer_neutral_policy': 'D2H before W1; no detach alias across apply; no weight/gate/scheduler/latent modification; instrumented timing is never production timing',
}