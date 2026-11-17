"""Full-U observer layered around the unchanged reviewed V2 panel tap.

No model invocation or runtime dispatch occurs in this module. V2 remains the
owner of the 160-site, dispatch, weight, geometry, and panel validations.
"""
import hashlib
import json
import os
from pathlib import Path

SOURCES = ('vbi_1013', 'vbi_0800')
LAYERS = (4, 19, 34)
GRID = (21, 29, 52)
TOKENS = 31668
DIM = 5120
STEPS = range(4)
ARCHIVE_LIMIT = 9 * 1024**3


def selected(step, layer):
    return step in STEPS and layer in LAYERS


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class FullUSink:
    """Exclusive, synchronous and one-tensor-at-a-time on-disk archive."""
    def __init__(self, torch, root, byte_limit=ARCHIVE_LIMIT):
        self.torch, self.root = torch, Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.limit, self.bytes, self.rows = byte_limit, 0, []
        self.seen_u, self.seen_panel = set(), set()
        self.seen_weights = set()
        self.staging_observed_max_bytes = 0

    def _save(self, rel, data):
        path = self.root / rel
        tensors = data.values() if isinstance(data, dict) else (data,)
        bytes_bound = sum(t.untyped_storage().nbytes() for t in tensors) + 256 * 1024
        if self.bytes + bytes_bound > self.limit:
            raise RuntimeError('insufficient full-U archive capacity for the next payload')
        with path.open('xb') as f:
            self.torch.save(data, f)
            f.flush(); os.fsync(f.fileno())
        size = path.stat().st_size
        self.bytes += size
        if self.bytes > self.limit:
            raise RuntimeError('full-U archive byte limit exceeded')
        return {'path': rel, 'bytes': size, 'sha256': digest(path)}

    def record_weights(self, step, layer, phase):
        expert = 'high' if step < 2 else 'low'
        key = (expert, layer)
        if key in self.seen_weights:
            return
        row = {'kind': 'live_w1_w2_identity', 'expert': expert, 'layer': layer, 'matrices': []}
        for label, f in (('W1', phase.ffn_0), ('W2', phase.ffn_2)):
            value = f._get_actual_weight()
            buf = getattr(f, 'weight_cuda_buffer', None)
            if buf is None or value.device.type != 'cuda' or value.data_ptr() != buf.data_ptr():
                raise RuntimeError('point-of-use weight storage identity unverified')
            h = hashlib.sha256()
            for i in range(0, value.shape[0], 128):
                chunk = value[i:i + 128].detach().to('cpu').contiguous()
                h.update(chunk.view(self.torch.uint8).numpy().tobytes())
            bias = f._get_actual_bias()
            bias_buf = getattr(f, 'bias_cuda_buffer', None)
            if bias is None or bias_buf is None or bias.device.type != 'cuda' or bias.data_ptr() != bias_buf.data_ptr():
                raise RuntimeError('point-of-use bias storage identity unverified')
            bias_bytes = bias.detach().to('cpu').contiguous().view(self.torch.uint8).numpy().tobytes()
            row['matrices'].append({'name': label, 'shape': list(value.shape),
                                    'stride': list(value.stride()), 'dtype': str(value.dtype),
                                    'runtime_weight_name': f.weight_name, 'runtime_bias_name': f.bias_name,
                                    'sha256': h.hexdigest(), 'bias_shape': list(bias.shape),
                                    'bias_dtype': str(bias.dtype), 'bias_sha256': hashlib.sha256(bias_bytes).hexdigest()})
        self.rows.append(row); self.seen_weights.add(key)
    def write_u(self, source, step, layer, value):
        key = (source, step, layer)
        if key in self.seen_u or source not in SOURCES or not selected(step, layer):
            raise RuntimeError('duplicate or unauthorized full-U site')
        if tuple(value.shape) != (TOKENS, DIM) or value.dtype != self.torch.bfloat16:
            raise RuntimeError('full-U geometry/dtype mismatch')
        # Record RSS before/after D2H and readback; transient peaks require independent measurement.
        before_rss = next(int(line.split()[1]) * 1024 for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:'))
        # Synchronous D2H; copy completes before native W1 can touch its input.
        cpu = value.detach().to(device='cpu', copy=True).contiguous()
        for start in range(0, TOKENS, 256):
            if not self.torch.isfinite(cpu[start:start + 256]).all().item():
                raise RuntimeError('nonfinite full-U')
        info = self._save(f'{source}__s{step:02d}__b{layer:02d}__U.pt', cpu)
        restored = self.torch.load(str(self.root / info['path']), map_location='cpu', mmap=True, weights_only=True)
        if restored.dtype != cpu.dtype or restored.shape != cpu.shape:
            raise RuntimeError('full-U readback shape/dtype mismatch')
        for start in range(0, TOKENS, 256):
            if not self.torch.equal(restored[start:start + 256], cpu[start:start + 256]):
                raise RuntimeError('full-U readback mismatch')
        after_rss = next(int(line.split()[1]) * 1024 for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:'))
        staging_observed = max(0, after_rss - before_rss)
        self.staging_observed_max_bytes = max(self.staging_observed_max_bytes, staging_observed)
        if staging_observed > 1024**3:
            raise MemoryError('observed capture staging exceeds 1 GiB')
        del restored
        info.update(source=source, step=step, layer=layer,
                    expert='high' if step < 2 else 'low', shape=[TOKENS, DIM], dtype='bfloat16')
        self.rows.append({'kind': 'full_u', **info}); self.seen_u.add(key)

    def write_panel(self, event, payload):
        step, layer, source = event['step'], event['layer'], event['source']
        if not selected(step, layer):
            return
        key = (source, step, layer)
        if key in self.seen_panel or key not in self.seen_u:
            raise RuntimeError('missing or duplicate selected panel')
        if tuple(payload['U'].shape) != (512, DIM) or tuple(payload['Y_pre_gate'].shape) != (512, DIM):
            raise ValueError('invalid P512 U/Y shapes')
        if tuple(payload['token_indices'].shape) != (512,) or tuple(payload['token_thw'].shape) != (512, 3):
            raise ValueError('invalid P512 coordinates')
        ids_bytes = payload['token_indices'].numpy().astype('<i8', copy=False).tobytes()
        if hashlib.sha256(ids_bytes).hexdigest() != event['token_id_sha256_le_int64']:
            raise ValueError('P512 token identity mismatch')
        full_path = self.root / f'{source}__s{step:02d}__b{layer:02d}__U.pt'
        full = self.torch.load(str(full_path), map_location='cpu', mmap=True, weights_only=True)
        recovered = full.index_select(0, payload['token_indices'].long())
        if not self.torch.equal(recovered, payload['U']):
            raise RuntimeError('saved full-U rows differ from independently copied live W1 panel')
        del full, recovered
        # Full-U is already retained; avoid duplicating its panel U in payload.
        selected_payload = {k: v for k, v in payload.items() if k != 'U'}
        selected_payload['Y_fullM_panel'] = selected_payload.pop('Y_pre_gate')
        info = self._save(f'{source}__s{step:02d}__b{layer:02d}__panel.pt', selected_payload)
        self.rows.append({'kind': 'panel', 'u_panel_readback_equal': True, 'event': event, **info}); self.seen_panel.add(key)

    def write_voxel(self, permutation, inverse, source_sha256):
        identity = self.torch.arange(TOKENS, dtype=self.torch.long)
        if permutation.shape != identity.shape or inverse.shape != identity.shape:
            raise RuntimeError('voxel permutation shape mismatch')
        if not self.torch.equal(permutation.index_select(0, inverse), identity):
            raise RuntimeError('voxel inverse does not round-trip')
        info = self._save('voxel_mapping.pt', {'voxel_to_raster': permutation, 'raster_to_voxel': inverse})
        self.rows.append({'kind': 'author_voxel_mapping', 'voxel_shape': [4, 6, 8],
                          'source_sha256': source_sha256, **info})
    def finish_source(self, source):
        expected = {(source, s, l) for s in STEPS for l in LAYERS}
        if not expected.issubset(self.seen_u) or not expected.issubset(self.seen_panel):
            raise RuntimeError('incomplete source capture')

    def finish(self, fixed64):
        expected = {(source, s, l) for source in SOURCES for s in STEPS for l in LAYERS}
        if self.seen_weights != {(expert, layer) for expert in ('high', 'low') for layer in LAYERS}:
            raise RuntimeError('point-of-use weight identities incomplete')
        if self.seen_u != expected or self.seen_panel != expected:
            raise RuntimeError('capture must contain exactly 24 full-U and 24 panel files')
        ids = [int(x) for x in fixed64]
        if len(ids) != 64 or len(set(ids)) != 64 or any(x < 0 or x >= TOKENS for x in ids):
            raise RuntimeError('invalid fixed64')
        manifest = {'schema': 'FFNDS_FULL_U_STAGE1_V1', 'sources': SOURCES, 'seed': 42,
                    'grid_thw': GRID, 'raster_mapping': 'index = (t * 29 + h) * 52 + w',
                    'fixed64_token_ids': ids, 'archive_bytes': self.bytes,
                    'target_semantics': 'Y_fullM_panel samples the native W2 pre-gate return; local512 own-W target is separate',
                    'weight_digest_semantics': 'six first-observed subject site active CUDA weights and biases; hashed in logical row order',
                    'capture_staging_observed_delta_rss_bytes': self.staging_observed_max_bytes,
                    'capture_staging_measurement': 'RSS snapshot with U and memory-mapped readback resident; transient peaks not resolved', 'files': self.rows}
        with (self.root / 'MANIFEST.json').open('x', encoding='utf8') as f:
            json.dump(manifest, f, indent=2, allow_nan=False); f.write('\n')
            f.flush(); os.fsync(f.fileno())
        return manifest


class FullUTap:
    """Compose V2 tap with a selective full-U wrapper without changing its guards."""
    def __init__(self, torch, v2_tap, sink):
        self.torch, self.v2, self.sink = torch, v2_tap, sink
        self.owner = self.inner_ffn = None

    def install(self, owner):
        if self.owner is not None:
            raise RuntimeError('already installed')
        try:
            self.v2.install(owner)
        except BaseException:
            self.v2.uninstall()
            raise
        self.owner, self.inner_ffn = owner, owner.infer_ffn
        tap, inner = self, self.inner_ffn

        def wrapper(model, phase, x, attn_out, shift, scale, gate=None):
            step, layer = int(model.scheduler.step_index), int(model.block_idx)
            if not selected(step, layer):
                return inner(model, phase, x, attn_out, shift, scale, gate)
            oldapply = phase.ffn_0.apply
            calls = 0
            def observed(u):
                nonlocal calls
                calls += 1
                if calls != 1 or tap.v2.context is None:
                    raise RuntimeError('invalid full-U W1 invocation')
                tap.sink.record_weights(step, layer, phase)
                tap.sink.write_u(tap.v2.context[0], step, layer, u)
                return oldapply(u)
            phase.ffn_0.apply = observed
            try:
                out = inner(model, phase, x, attn_out, shift, scale, gate)
                if calls != 1: raise RuntimeError('selected W1 was not called')
                return out
            finally:
                phase.ffn_0.apply = oldapply
        owner.infer_ffn = wrapper

    def uninstall(self):
        if self.owner is None: return
        self.owner.infer_ffn = self.inner_ffn
        self.v2.uninstall()
        self.owner = self.inner_ffn = None









