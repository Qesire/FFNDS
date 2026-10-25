"""Observation preparation utilities. No model, torch or GPU imports."""
import hashlib
import json
import math
import random
import struct
from pathlib import Path

FIELDS = {
    "x_before_cross_add": 5120, "cross_attention": 5120,
    "residual_pre_ffn": 5120, "modulated_input": 5120,
    "q1_pre": 5120, "q1_post": 5120, "z": 13824, "gelu": 13824,
    "q2_pre": 13824, "q2_post": 13824, "ffn_output": 5120,
    "residual_post_ffn": 5120,
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(8 << 20), b""):
            h.update(data)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def row_plan(grid, source_id, uniform_count=32):
    """Fixed coordinates across steps/blocks; neighbor rows are NOT IID samples."""
    if len(grid) != 3 or any(type(v) is not int or v <= 0 for v in grid):
        raise ValueError("grid must be positive integer [T,H,W]")
    if type(uniform_count) is not int or not 1 <= uniform_count <= 256:
        raise ValueError("uniform_count must be in [1,256]")
    total = math.prod(grid)
    seed = int.from_bytes(hashlib.sha256(
        (str(source_id) + "|" + json.dumps(grid)).encode()).digest()[:8], "little")
    rng = random.Random(seed)  # no global or model RNG change
    uniform = sorted(rng.sample(range(total), min(total, uniform_count)))
    tmax, hmax, wmax = grid

    def flat(t, y, x):
        return (t * hmax + y) * wmax + x

    def coord(i):
        t, rem = divmod(i, hmax * wmax)
        y, x = divmod(rem, wmax)
        return [t, y, x]

    pairs = []
    for k in range(4):
        t = min(tmax - 1, (k * tmax) // 4)
        y, x = (hmax - 1) // 2, (wmax - 1) // 2
        a = flat(t, y, x)
        for axis, pos in [("space_x", (t, y, x + 1)),
                          ("space_y", (t, y + 1, x)),
                          ("latent_time", (t + 1, y, x))]:
            if pos[0] < tmax and pos[1] < hmax and pos[2] < wmax:
                pair = {"axis": axis, "a": a, "b": flat(*pos)}
                if pair not in pairs:
                    pairs.append(pair)
    rows = sorted(set(uniform) | {p[k] for p in pairs for k in ("a", "b")})
    return {"grid": grid, "rows": rows, "coordinates": [coord(i) for i in rows],
            "uniform_rows": uniform, "uniform_inclusion_probability": len(uniform) / total,
            "neighbor_pairs": pairs, "sampling_seed": seed,
            "independence_unit": "source, not rows/blocks/steps",
            "alignment": "same latent coordinate, not tracked physical object"}


def read_header(path):
    """Read only a safetensors header; no pickle/torch/model loading."""
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("truncated safetensors length")
        length = struct.unpack("<Q", raw)[0]
        if not 2 <= length <= min(size - 8, 64 << 20):
            raise ValueError("invalid safetensors header length")
        def unique(pairs):
            d = {}
            for k, v in pairs:
                if k in d:
                    raise ValueError("duplicate JSON key")
                d[k] = v
            return d
        header = json.loads(f.read(length), object_pairs_hook=unique)
        if not isinstance(header, dict):
            raise ValueError("header must be an object")
    widths = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8,
              "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
    spans, fields = [], {}
    for name, item in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(item, dict):
            raise ValueError("tensor entry must be an object")
        shape, dtype, offsets = item["shape"], item["dtype"], item["data_offsets"]
        if not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError("invalid shape or offsets")
        if dtype not in widths or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("unsupported dtype or invalid shape")
        a, b = offsets
        if type(a) is not int or type(b) is not int or not 0 <= a <= b <= size - 8 - length:
            raise ValueError("tensor offset out of bounds")
        if b - a != math.prod(shape) * widths[dtype]:
            raise ValueError("tensor byte count mismatch")
        spans.append((a, b))
        fields[name] = item
    end = 0
    for a, b in sorted(spans):
        if a != end:
            raise ValueError("overlap or gap in tensor data")
        end = b
    if end != size - 8 - length:
        raise ValueError("unclaimed tensor bytes")
    return {"header_bytes": length, "fields": fields, "bytes": size,
            "metadata": header.get("__metadata__", {})}


def audit_headers(paths):
    result = []
    for p in paths:
        try:
            h = read_header(p)
            names = set(h["fields"])
            result.append({"path": str(p), "status": "HEADER_VALID",
                           "bytes": h["bytes"], "fields": h["fields"],
                           "paired_q1": {"q1_pre", "q1_post"} <= names,
                           "paired_q2": {"q2_pre", "q2_post"} <= names,
                           "legacy_post_q2_only": "V2" in names,
                           "missing_new_fields": sorted(set(FIELDS) - names)})
        except (OSError, ValueError, KeyError, TypeError, struct.error) as e:
            result.append({"path": str(p), "status": "HEADER_FAILED", "error": str(e)})
    return {"purpose": "header availability, not payload hash verification or model evidence",
            "files": result}

