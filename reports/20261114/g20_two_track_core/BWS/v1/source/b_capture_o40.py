"""BWS mu_O original40 endpoint U-input capture producer (composite native state).

Adapted from the proven B_CHIP_STATS NativeTapFixed64 (sha 4dd8ab95) and the V2
NativeTap512 (sha a9748f38), preserving the Wan hook structure and identity guards:
infer_block exposes the runtime patch grid; infer_ffn wraps phase.ffn_0.apply (W1);
the pre-W1 input z is the full-M U; c_gate_msa is the gate G.

DIFFERENCE vs the D4 fixed64 producer (the BWS mu_O estimand):
  * ORIGINAL model W_O (Wan2.2-I2V-A14B-bf16), 40-step scheduler/CFG trajectory
    (infer_steps=40, boundary=0.9, guide_scale=[3.5,3.5], enable_cfg). Expert
    dispatch is the original-model scheduler split (high then low), NOT D4 HHLL.
  * Endpoint policy (handoff B7): per expert, the FIRST adjacent same-expert
    scheduler-step pair. The first two consecutive steps of each expert are that
    pair (anchor = first step, hit = second step). The actual timestep/noise/gap
    are recorded; no nearest-time matching, no interval grid. Conditional branch
    only is scored (the unconditional CFG call still runs and is counted in cost).
  * Geometry: aspect_ratio 16:9 + width 832 forces actual height 464 (the config
    target_height 480 is a max-area hint). The hook ASSERTS the runtime grid is
    (21,29,52), M=31668 — identical to the V2 D4 bank. A 480->30 grid is rejected.
  * Per target site saves U_panel = z[panel_ids] (512,5120) for ALL six sources
    (the mu_O Q1/ICC state). For the three DEV sources it ALSO saves
    U_fixed64 = z[fixed64_ids] (64,5120) for the Chip mu_O selector. CAL3 needs no
    P64 stats. G_ref is NOT taken from O40: scoring uses the same-source D4 hit
    gate; the O40 gate is stored only for identity/provenance.

The observer is trajectory-neutral: D2H copies complete before W1, no detach alias
is held across apply, and weights/gate/scheduler/latents are never modified.
Instrumented (tap) timing is NEVER production timing. One dense original40 pass per
missing condition (<=6); no seventh trajectory, no full-M block stats, no full-M F1,
no new source/image/prompt, no video decode/scoring.
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
STEPS = 40
LAYERS = 40
FULL_M = int(np.prod(EXPECTED_GRID))  # 31668
FIXED64_COUNT = 64
SEED = 42
SITE_LAYERS = (4, 19, 34)
SITES = [('high', 4), ('high', 19), ('high', 34), ('low', 4), ('low', 19), ('low', 34)]


def _sha_i64(ids):
    return hashlib.sha256(np.ascontiguousarray(ids, dtype='<i8').tobytes()).hexdigest()


def regenerate_fixed64(torch):
    g = torch.Generator(device='cpu').manual_seed(SEED)
    return torch.randperm(FULL_M, generator=g)[:FIXED64_COUNT].tolist()


class NativeTapO40:
    """Original40 observer saving PANEL512 U (+ DEV3 fixed64) at the first adjacent
    same-expert endpoint pair, six sites. torch injected by the runner; sink is an
    O40Sink. expected_w2_class is the dense MMWeight class.

    Endpoint selection is online: the dispatch tap (once per scheduler step) records
    step/expert/timestep and marks the first two consecutive steps of each expert as
    anchor/hit. The FFN hook captures only at (endpoint step, site layer).
    """

    def __init__(self, torch, sink, fixed64_ids, capture_fixed64,
                 expected_w2_class=None, expected_grid=EXPECTED_GRID):
        self.torch, self.sink = torch, sink
        if len(fixed64_ids) != FIXED64_COUNT or len(set(fixed64_ids)) != FIXED64_COUNT:
            raise ValueError('fixed64 must be 64 distinct token IDs')
        if any(not (0 <= int(t) < FULL_M) for t in fixed64_ids):
            raise ValueError('fixed64 token ID outside full-M domain')
        self.fixed64_ids = [int(t) for t in fixed64_ids]
        self.fixed64_sha = _sha_i64(self.fixed64_ids)
        self.capture_fixed64 = bool(capture_fixed64)
        self.expected_w2_class = expected_w2_class
        self.expected_grid = tuple(expected_grid) if expected_grid is not None else None
        self.context = None
        self.grid = None
        self.source_grid = None
        self.owner = None
        self.originals = None
        self.panel_ids = None
        self.panel_info = None
        self.panel_sha = None
        # online endpoint selection (per expert 0=high,1=low)
        self.expert_steps = {0: [], 1: []}
        self.endpoint_of = {}          # (step,layer) -> (site_key, role)
        self.step_expert = {}          # step -> expert
        self.step_timestep = {}        # step -> float timestep
        self.seen = set()
        self.dispatch = None

    def begin_source(self, source, seed):
        if self.context is not None or not isinstance(source, str) or not source \
                or not isinstance(seed, int) or isinstance(seed, bool) or seed != SEED:
            raise ValueError('invalid or overlapping source/seed')
        self.context = (source, seed)
        self.source_grid = None
        self.panel_ids = None
        self.panel_info = None
        self.panel_sha = None
        self.expert_steps = {0: [], 1: []}
        self.endpoint_of = {}
        self.step_expert = {}
        self.step_timestep = {}
        self.seen = set()

    def record_step(self, step, expert, timestep):
        """Called once per scheduler step by the dispatch tap (before that step's
        FFN calls). Builds the first-adjacent-pair endpoint set online."""
        if step in self.step_expert:
            raise RuntimeError('duplicate scheduler step %d' % step)
        if expert not in (0, 1):
            raise RuntimeError('bad expert index')
        self.step_expert[step] = expert
        self.step_timestep[step] = float(timestep)
        steps = self.expert_steps[expert]
        steps.append(step)
        if len(steps) <= 2:
            role = 'anchor' if len(steps) == 1 else 'hit'
            for L in SITE_LAYERS:
                self.endpoint_of[(step, L)] = ('%s%02d' % ('high' if expert == 0 else 'low', L), role)

    def finish_source(self):
        if self.context is None:
            raise RuntimeError('no source')
        # both experts must have an adjacent anchor/hit pair; 12 captures total
        for e in (0, 1):
            st = self.expert_steps[e]
            if len(st) < 2 or st[1] != st[0] + 1:
                raise RuntimeError('expert %d lacks adjacent first pair: %s' % (e, st[:2]))
        if len(self.seen) != 12:
            raise RuntimeError('incomplete trajectory: %d/12 endpoint sites' % len(self.seen))
        # expert schedule must be a single high->low switch (contiguous blocks)
        seq = [self.step_expert[s] for s in sorted(self.step_expert)]
        switches = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
        if switches != 1 or seq[0] != 0 or seq[-1] != 1:
            raise RuntimeError('expert schedule not a single high->low switch')
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
                raise RuntimeError('cross-site panel identity mismatch')
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
                raise RuntimeError('grid differs from reviewed geometry (need 21x29x52)')
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
            if tap.dispatch not in (0, 1):
                raise RuntimeError('actual expert dispatch missing')
            # CFG runs conditional + unconditional passes per step; capture ONLY the
            # conditional branch (handoff B7). scheduler.infer_condition is the bool
            # CFG marker (proven AnchoredHook). Unconditional passes through untouched.
            condition = self.scheduler.infer_condition
            if type(condition) is not bool:
                raise RuntimeError('invalid CFG marker')
            if not condition:
                return oldffn(self, phase, x, attn_out, c_shift_msa, c_scale_msa, c_gate_msa)
            if (step, layer) not in tap.endpoint_of:
                return oldffn(self, phase, x, attn_out, c_shift_msa, c_scale_msa, c_gate_msa)
            if (step, layer) in tap.seen:
                raise RuntimeError('duplicate conditional endpoint step/layer')
            site_key, role = tap.endpoint_of[(step, layer)]
            expert = 'high' if tap.dispatch == 0 else 'low'
            if not site_key.startswith(expert):
                raise RuntimeError('endpoint expert/site mismatch')
            timestep = tap.step_timestep[step]
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
                ixp = tap.torch.as_tensor(panel_ids, device=z.device, dtype=tap.torch.long)
                payload = {
                    'U_panel': z.index_select(0, ixp).detach().to('cpu').clone(),
                    'gate': c_gate_msa.detach().to('cpu').clone(),
                    'token_indices_panel': tap.torch.as_tensor(panel_ids.copy(), dtype=tap.torch.long),
                }
                if tap.capture_fixed64:
                    ix64 = tap.torch.as_tensor(tap.fixed64_ids, device=z.device, dtype=tap.torch.long)
                    coords64 = np.stack((np.array(tap.fixed64_ids) // (h * w),
                                         (np.array(tap.fixed64_ids) // w) % h,
                                         np.array(tap.fixed64_ids) % w), axis=1)
                    payload['U_fixed64'] = z.index_select(0, ix64).detach().to('cpu').clone()
                    payload['token_indices_fixed64'] = tap.torch.as_tensor(np.array(tap.fixed64_ids, dtype='<i8'), dtype=tap.torch.long)
                    payload['token_thw_fixed64'] = tap.torch.as_tensor(coords64.copy(), dtype=tap.torch.long)
                if any(not tap.torch.isfinite(v).all().item() for k, v in payload.items()
                       if k not in ('token_indices_panel', 'token_indices_fixed64', 'token_thw_fixed64')):
                    raise RuntimeError('nonfinite capture')
                event = {
                    'source': tap.context[0], 'seed': tap.context[1], 'step': step, 'layer': layer,
                    'expert': expert, 'timestep': timestep, 'site_key': site_key, 'endpoint_role': role,
                    'grid': list(tap.grid), 'flattened_tokens': int(z.shape[0]),
                    'panel_count': PANEL_COUNT, 'panel_version': PANEL_VERSION,
                    'token_id_sha256_le_int64': info['token_id_sha256_le_int64'],
                    'sorted_token_id_sha256_le_int64': info['sorted_token_id_sha256_le_int64'],
                    'U_panel_shape': [PANEL_COUNT, DIM], 'U_panel_dtype': 'bfloat16',
                    'gate_shape': list(c_gate_msa.shape), 'gate_dtype': str(c_gate_msa.dtype),
                    'state_mu': 'mu_O', 'capture_fixed64': tap.capture_fixed64,
                }
                if tap.capture_fixed64:
                    event['fixed64_count'] = FIXED64_COUNT
                    event['fixed64_token_id_sha256_le_int64'] = tap.fixed64_sha
                    event['U_fixed64_shape'] = [FIXED64_COUNT, DIM]
                    event['U_fixed64_dtype'] = 'bfloat16'
                    event['rows_are_panel_first64'] = False
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


class O40Sink:
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
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,96}', source) or layer not in SITE_LAYERS:
            raise ValueError('bad site identity')
        if payload['U_panel'].shape != (PANEL_COUNT, DIM):
            raise ValueError('U_panel must be (512,5120)')
        if payload['token_indices_panel'].shape != (PANEL_COUNT,):
            raise ValueError('panel coordinate shape mismatch')
        if event['token_id_sha256_le_int64'] != _sha_i64(payload['token_indices_panel'].numpy()):
            raise ValueError('panel token id sha mismatch')
        if event['capture_fixed64']:
            if payload['U_fixed64'].shape != (FIXED64_COUNT, DIM):
                raise ValueError('U_fixed64 must be (64,5120)')
            if event['fixed64_token_id_sha256_le_int64'] != _sha_i64(payload['token_indices_fixed64'].numpy()):
                raise ValueError('fixed64 token id sha mismatch')
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
                   'consumer_contract': 'BWS_O40_U_TYPED_V1'}
        with (self.root / (name + '.json')).open('x', encoding='utf-8') as f:
            json.dump(receipt, f, ensure_ascii=False, sort_keys=True, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        self.used += (self.root / (name + '.json')).stat().st_size
        if self.used > self.limit:
            raise RuntimeError('capture byte budget including sidecar exceeded')


CONSUMER_CONTRACT = {
    'contract_id': 'BWS_O40_U_TYPED_V1',
    'panel_version': PANEL_VERSION,
    'target_sites_per_source': 12,
    'sites': ['%s%02d' % (e, L) for e, L in SITES],
    'expert_schedule': 'original40 single high->low switch (boundary 0.9)',
    'endpoint_policy': 'first adjacent same-expert scheduler-step pair per expert (anchor=first, hit=second)',
    'grid': list(EXPECTED_GRID), 'flattened_tokens': FULL_M,
    'panel_count': PANEL_COUNT, 'fixed64_count': FIXED64_COUNT, 'dim': DIM, 'ffn_dim': FFN_DIM,
    'state_mu': 'mu_O', 'rows_are_panel_first64': False,
    'G_ref_source': 'same-source/site D4 hit gate (NOT O40); O40 gate stored for identity only',
    'no_full_M_block_stats': True, 'no_full_M_F1': True,
    'observer_neutral_policy': 'D2H before W1; no detach alias across apply; no weight/gate/scheduler/latent modification; instrumented timing is never production timing',
}