"""Validate an immutable OP2 tensor payload without reconstructing its values."""
import json
import math
from pathlib import Path

from ar2_checkpoint import digest

FIELDS = {'V1', 'BaseY', 'TeacherY', 'StudentY', 'b2', 'gate'}


def verify_payload(scope, expected_parent):
    path = Path(scope['path'])
    if path.resolve().parent != Path(expected_parent).resolve():
        raise RuntimeError('Native payload escaped its accepted input directory')
    size = path.stat().st_size
    if size != scope['bytes'] or digest(path) != scope['sha256']:
        raise RuntimeError('Native payload size or full checksum differs')
    with path.open('rb') as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise RuntimeError('Truncated safetensors length prefix')
        length = int.from_bytes(prefix, 'little')
        if not 0 < length <= min(1 << 20, size - 8):
            raise RuntimeError('Invalid native safetensors header length')
        header = json.loads(handle.read(length))
    if set(header) - {'__metadata__'} != FIELDS:
        raise RuntimeError('Native payload fields differ from the recorded producer')
    segments = []
    for name in sorted(FIELDS):
        record = header[name]
        shape, offsets = record['shape'], record['data_offsets']
        if record['dtype'] != 'BF16' or not shape or any(type(v) != int or v <= 0 for v in shape):
            raise RuntimeError('Native payload is not positive-shape BF16')
        if name in {'V1', 'BaseY', 'TeacherY', 'StudentY'}:
            if shape != [256, 5120]:
                raise RuntimeError('Native main tensor sample shape differs')
        elif math.prod(shape) != 5120:
            raise RuntimeError('Native original bias or gate shape differs')
        if (len(offsets) != 2 or any(type(v) != int for v in offsets)
                or offsets[0] < 0 or offsets[1] - offsets[0] != 2 * math.prod(shape)):
            raise RuntimeError('Native tensor offsets do not match BF16 storage')
        segments.append(tuple(offsets))
    segments.sort()
    if (segments[0][0] != 0 or segments[-1][1] != size - 8 - length
            or any(a[1] != b[0] for a, b in zip(segments, segments[1:]))):
        raise RuntimeError('Native payload body has a gap, overlap or trailing bytes')
    return size
