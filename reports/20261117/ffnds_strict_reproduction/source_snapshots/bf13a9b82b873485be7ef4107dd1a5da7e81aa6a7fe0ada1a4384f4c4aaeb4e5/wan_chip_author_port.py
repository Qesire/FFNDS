"""Pre-gate author Chipmunk with safe selector, true raster voxel order and dense tail."""

import hashlib
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from wan_ffn_port import dense


class NativeLinear:
    """Expose the currently active native MMWeight with the author Linear layout."""

    def __init__(self, operation):
        self.op = operation
        actual = operation._get_actual_weight()
        self.weight, self.bias = actual.T.contiguous(), operation._get_actual_bias()
        self._weight_ptr, self._bias_ptr = actual.data_ptr(), self.bias.data_ptr() if self.bias is not None else None
        if self.bias is None or actual.data_ptr() != operation.weight_cuda_buffer.data_ptr():
            raise RuntimeError('native Chip encountered invalid live weights')

    def __call__(self, inputs):
        actual, bias = self.op._get_actual_weight(), self.op._get_actual_bias()
        if actual.data_ptr() != self._weight_ptr or bias is None or bias.data_ptr() != self._bias_ptr:
            raise RuntimeError('native point-of-use weight changed')
        rows = inputs.reshape(-1, inputs.shape[-1])
        result = self.op.apply(rows)
        return result.reshape(*inputs.shape[:-1], result.shape[-1])


class NativeAuthorChipPort:
    """The author MLP owns active proxy, masks and outputs; tail stays native Dense."""

    def __init__(self, *, force_dense, provider=None, voxel=None, scratch=None, check_resources=None):
        if not force_dense and (provider is None or voxel is None):
            raise ValueError('missing approved author provider or voxel map')
        self.force_dense, self.provider = force_dense, provider
        if not force_dense and (scratch is None or not callable(check_resources)):
            raise ValueError('disk cache and resource guard required')
        self.scratch = None if scratch is None else Path(scratch)
        if not force_dense:
            self.scratch.mkdir(parents=True, exist_ok=False)
        self.check_resources = check_resources
        self.seen, self.storages, self.io_receipts = [], {}, []
        self.voxel = voxel
        if voxel is not None:
            perm, inv = self.voxel
            index = torch.arange(31668, device=perm.device)
            if perm.shape != index.shape or inv.shape != index.shape or not torch.equal(perm.index_select(0, inv), index):
                raise RuntimeError('invalid native author voxel mapping')

    def __call__(self, model, phase, u):
        step, layer = int(model.scheduler.step_index), int(model.block_idx)
        if (step, layer) != divmod(len(self.seen), 40) or u.shape != (31668, 5120) or u.dtype != torch.bfloat16:
            raise RuntimeError('invalid native Chip site')
        expert = 'high' if step < 2 else 'low'
        if self.force_dense or layer < 2:
            y = dense(u, phase.ffn_0.apply, phase.ffn_2.apply)[2]
            kind = 'forced_dense' if self.force_dense else 'fixed_dense'
            counts = None
        else:
            y, counts = self._sparse(step, layer, expert, phase, u)
            kind = 'anchor' if step in (0, 2) else 'hit'
        if y.dtype != u.dtype or y.shape != u.shape:
            raise RuntimeError('invalid Chipmunk pre-gate result')
        self.seen.append({'step': step, 'layer': layer, 'expert': expert, 'kind': kind,
                          'block_counts': counts, 'dense_tail_rows': 52 if not self.force_dense and layer >= 2 else 0})
        return y

    def _sparse(self, step, layer, expert, phase, u):
        sparse_type, counter_type = self.provider
        key = expert, layer
        counter = counter_type(40, 1)
        counter.cur_inference_step, counter.cur_layer = step % 2, layer
        first, second = NativeLinear(phase.ffn_0), NativeLinear(phase.ffn_2)
        if (first.weight.shape != (13824, 5120) or second.weight.shape != (5120, 13824)
                or first.bias.shape != (13824,) or second.bias.shape != (5120,)
                or any(v.dtype != torch.bfloat16 for v in (first.weight, first.bias, second.weight, second.bias))):
            raise RuntimeError('invalid native Wan BF16 weight geometry')
        self.check_resources()
        weight_identity = self._weight_identity(first, second)
        if step % 2 and self.storages.get(key, {}).get('weights') != weight_identity:
            raise RuntimeError('anchor/hit weight values differ')
        activation = lambda x: F.gelu(x, approximate='tanh')
        module = sparse_type(layer, counter, first, activation, second)
        perm, inv = (x.to(u.device) for x in self.voxel)
        arranged = u.index_select(0, perm).unsqueeze(0)
        head = arranged[:, :31616].contiguous()
        tail = arranged[:, 31616:]
        if step % 2 == 1:
            self._restore(module.storage, key)
        result = module(head)
        counts = module.storage.get_counts()
        if step % 2 == 0:
            if key in self.storages or counts is not None:
                raise RuntimeError('Chip anchor unexpectedly inherits another request/step state')
            self._save(module.storage, key, weight_identity)
        else:
            if counts is None or counts.shape != (1, 247):
                raise RuntimeError('missing 247 author support counts')
            counts = counts[0].tolist()
        tail_result = second(activation(first(tail)))
        if step % 2 == 1:
            for rec in self.storages.pop(key)['records']: Path(rec['path']).unlink()
        return torch.cat((result, tail_result), dim=1)[0].index_select(0, inv), counts

    @staticmethod
    def _hash(path):
        with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

    def _weight_identity(self, first, second):
        start = time.monotonic()
        hashes = []
        for value in (first.weight, first.bias, second.weight, second.bias):
            h = hashlib.sha256()
            rows = value.reshape(1, -1) if value.ndim == 1 else value
            for offset in range(0, rows.shape[0], 128):
                block = rows[offset:offset+128].detach().to('cpu').contiguous()
                h.update(block.view(torch.uint8).numpy().tobytes())
            hashes.append(h.hexdigest())
        self.io_receipts.append({'action':'weight_identity', 'seconds':time.monotonic()-start})
        return hashes

    def _save(self, storage, key, weights):
        if key in self.storages: raise RuntimeError('duplicate cache key')
        records = []
        for name in ('sparse_act_T', 'out_cache', 'blockmean_mid_cache'):
            self.check_resources()
            value = getattr(storage, 'get_' + name)()
            if value is None or value.dtype != torch.bfloat16: raise RuntimeError('invalid author cache')
            path = self.scratch / f'{key[0]}_{key[1]:02d}_{name}.pt'
            torch.cuda.synchronize(); start = time.monotonic()
            cpu = value.detach().to('cpu', copy=True).contiguous()
            with path.open('xb') as f:
                torch.save(cpu, f); f.flush(); os.fsync(f.fileno())
            del cpu
            elapsed = time.monotonic() - start
            rec = {'path': str(path.resolve()), 'sha256': self._hash(path), 'shape': list(value.shape),
                   'dtype': str(value.dtype), 'bytes': path.stat().st_size, 'write_seconds': elapsed}
            records.append(rec); self.io_receipts.append({'action':'write', **rec})
        self.storages[key] = {'records': records, 'weights': weights}

    def _restore(self, storage, key):
        if key not in self.storages: raise RuntimeError('missing anchor cache')
        for name, rec in zip(('sparse_act_T', 'out_cache', 'blockmean_mid_cache'), self.storages[key]['records']):
            self.check_resources()
            path = Path(rec['path'])
            if path.stat().st_size != rec['bytes'] or self._hash(path) != rec['sha256']:
                raise RuntimeError('cache asset changed')
            torch.cuda.synchronize(); start = time.monotonic()
            value = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
            if list(value.shape) != rec['shape'] or str(value.dtype) != rec['dtype']:
                raise RuntimeError('cache metadata changed')
            getattr(storage, 'set_' + name)(value.to('cuda', copy=True))
            torch.cuda.synchronize()
            self.io_receipts.append({'action':'read', 'path':str(path), 'bytes':rec['bytes'],
                                     'read_seconds':time.monotonic()-start})
            del value

    def finish(self):
        if len(self.seen) != 160 or self.storages:
            raise RuntimeError('incomplete author Chip request or retained state')
        return {'sites': self.seen, 'cache_io': self.io_receipts}
