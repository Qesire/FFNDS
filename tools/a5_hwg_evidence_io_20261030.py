"""Bounded immutable evidence reads and charged scientific call records.

This library grants no execution authority. Its caller supplies the actual
reviewed budgets, confinement root, durable journal and whole-job guard.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time


def bound_bytes(root, ref, budget, *, maximum_file=512 << 20):
    root = Path(root).resolve()
    path = (root / ref['path']).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('evidence confinement')
    size = path.stat().st_size
    if size != ref['bytes'] or size > maximum_file:
        raise ValueError('evidence length/file budget')
    budget.before(size, ref['path'])
    digest = hashlib.sha256()
    pieces = []
    with path.open('rb') as stream:
        while True:
            piece = stream.read(min(8 << 20, size + 1 - sum(map(len, pieces))))
            if not piece:
                break
            pieces.append(piece)
            digest.update(piece)
            budget.after(len(piece))
    if digest.hexdigest() != ref['sha256']:
        raise ValueError('evidence SHA mismatch: ' + ref['path'])
    budget.finish()
    return b''.join(pieces)


def bound_json(root, ref, budget):
    return json.loads(bound_bytes(root, ref, budget, maximum_file=16 << 20))


def bound_tensors(root, ref, budget, schema):
    from safetensors.torch import load
    import torch
    data = bound_bytes(root, ref, budget)
    values = load(data)
    if set(values) != set(schema):
        raise ValueError('evidence tensor field set')
    for name, tensor in values.items():
        row = schema[name]
        if list(tensor.shape) != row['shape'] or str(tensor.dtype).removeprefix('torch.') != row['dtype']:
            raise ValueError('evidence tensor shape/dtype: ' + name)
        actual = hashlib.sha256(memoryview(tensor.contiguous().view(torch.uint8).numpy())).hexdigest()
        budget.explicit_memory_raw_SHA_bytes = getattr(budget, 'explicit_memory_raw_SHA_bytes', 0) + tensor.numel() * tensor.element_size()
        if actual != row['raw_tensor_sha256']:
            raise ValueError('evidence raw tensor SHA: ' + name)
    return values


def checkpoint_biases(path, expected, budget, full_file_sha):
    """Charge actual checkpoint SHA, header and exact original BF16 biases.

    The caller supplies the already charged whole-file SHA function. No
    safetensors mmap/deserializer performs hidden checkpoint reads here.
    """
    import torch
    path = Path(path)
    before = path.stat()
    identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    if before.st_size != expected['bytes'] or full_file_sha(path) != expected['sha256']:
        raise ValueError('original bias checkpoint SHA/length')
    def exact_read(stream, count, label):
        budget.before(count, label)
        data = bytearray()
        while len(data) < count:
            piece = stream.read(min(8 << 20, count - len(data)))
            if not piece:
                raise ValueError('short checkpoint slice')
            data.extend(piece)
            budget.after(len(piece))
        budget.finish()
        return data
    result = {}
    with path.open('rb') as stream:
        prefix = exact_read(stream, 8, 'bias_checkpoint_header_length')
        size = struct.unpack('<Q', prefix)[0]
        if not 0 < size <= 16 << 20 or 8 + size > before.st_size:
            raise ValueError('bounded checkpoint header')
        header = json.loads(exact_read(stream, size, 'bias_checkpoint_header'))
        spans = []
        for block in range(40):
            for linear, width in [(0, 13824), (2, 5120)]:
                key = f'blocks.{block}.ffn.{linear}.bias'
                row = header[key]
                start, end = row['data_offsets']
                if (row['dtype'] != 'BF16' or row['shape'] != [width]
                        or end - start != 2 * math.prod(row['shape'])
                        or not 0 <= start <= end <= before.st_size - size - 8):
                    raise ValueError('original bias tensor geometry')
                spans.append((start, end))
                stream.seek(8 + size + start)
                raw = exact_read(stream, end - start, key)
                result[key] = torch.frombuffer(raw, dtype=torch.bfloat16).clone()
        ordered = sorted(spans)
        if any(a[1] > b[0] for a, b in zip(ordered[:-1], ordered[1:])):
            raise ValueError('aliased checkpoint biases')
        if identity(path.stat()) != identity(before) or identity(os.fstat(stream.fileno())) != identity(before):
            raise ValueError('checkpoint changed during bias reads')
    return result


class CallLedger:
    def __init__(self, limits, note, check, torch_module):
        self.limits = dict(limits)
        self.counts = {}
        self.note, self.check, self.torch = note, check, torch_module
        self.records = []
        self.context = {}

    def __call__(self, name, operation, **identity):
        self.check()
        count = self.counts.get(name, 0) + 1
        if name not in self.limits or count > self.limits[name]:
            raise ValueError('call budget: ' + name)
        self.counts[name] = count
        row = dict(name=name, ordinal=count, identity={**self.context, **identity},
                   status='INFLIGHT_UNKNOWN', actual_CUDA_seconds=None, actual_host_seconds=None,
                   timing_scope='SYNCHRONIZED_CUDA_EVENT_INTERVAL_AND_HOST_WALL_NOT_SUM_OF_KERNEL_ACTIVE_TIME')
        self.note('SCIENTIFIC_CALL_BEGIN', **row)
        torch = self.torch
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        began = time.monotonic()
        start.record()
        try:
            value = operation()
            end.record()
            torch.cuda.synchronize()
            row.update(status='COMPLETE', actual_CUDA_seconds=start.elapsed_time(end) / 1000)
            return value
        finally:
            row['actual_host_seconds'] = time.monotonic() - began
            self.note('SCIENTIFIC_CALL_RETURN', **row)
            self.records.append(row)
            self.check()
