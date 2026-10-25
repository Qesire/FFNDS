"""CPU-only descriptive analysis of accepted observation bytes, not candidate evaluation."""
import argparse
import json
from pathlib import Path

import numpy as np

from ffnds_observation_common import read_header, sha256, write_json


def load_arrays(path):
    h = read_header(path)
    arrays = {}
    with Path(path).open("rb") as f:
        for name, d in h["fields"].items():
            a, b = d["data_offsets"]
            f.seek(8 + h["header_bytes"] + a)
            raw = f.read(b - a)
            if d["dtype"] == "BF16":
                bits = np.frombuffer(raw, dtype="<u2").astype("<u4") << 16
                value = bits.view("<f4")
            else:
                dt = {"F16": "<f2", "F32": "<f4", "F64": "<f8", "I64": "<i8",
                      "I32": "<i4", "I16": "<i2", "I8": "i1", "U8": "u1", "BOOL": "?"}[d["dtype"]]
                value = np.frombuffer(raw, dtype=dt)
            arrays[name] = value.reshape(d["shape"]).copy()
    return arrays


def ratio(n, d):
    return float(n / d) if d > 0 else None


def quantiles(x):
    x = np.asarray(x).reshape(-1)
    return [float(v) for v in np.quantile(x, [0, .25, .5, .75, 1])] if x.size else []


def paired(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("paired arrays must have identical [rows,width] shape")
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    valid = denom > 0
    cos = np.sum(a[valid] * b[valid], axis=1) / denom[valid]
    return {"relative_SSE": ratio(np.sum((b - a) ** 2), np.sum(a ** 2)),
            "changed_fraction": float(np.mean(a != b)),
            "cosine_quantiles": quantiles(cos), "cosine_valid_rows": int(valid.sum())}


def sample_rank(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    gram = x @ x.T
    eig = np.maximum(np.linalg.eigvalsh(gram), 0)
    return {"centered_sample_participation_rank": ratio(eig.sum() ** 2, np.sum(eig ** 2)),
            "rank_upper_bound": min(max(len(x) - 1, 0), x.shape[1]),
            "scope": "sample row spectrum, not global model rank or a chosen low-rank method"}


def describe(arrays, scope=None):
    result = {"fields": {}, "quantization_pairs": {}, "neighbor_pairs": [],
              "native_dynamic_metadata": "not exported; no invented MX/NV codes or scales"}
    scope = scope or {}
    plan = scope.get("row_plan")
    row_lookup = {r: i for i, r in enumerate(plan["rows"])} if plan else {}
    uniform = [row_lookup[r] for r in plan["uniform_rows"]] if plan else None
    names = ["q1_pre", "q1_post", "z", "gelu", "q2_pre", "q2_post",
             "ffn_output", "residual_pre_ffn", "residual_post_ffn", "V1", "V2", "Y"]
    for name in names:
        if name not in arrays:
            continue
        x = arrays[name]
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("non-finite/non-matrix field: " + name)
        x = x[uniform] if uniform is not None else x
        x = x.astype(np.float64)
        stats = {"rows": len(x), "width": x.shape[1], "zero_fraction": float(np.mean(x == 0)),
                 "negative_fraction": float(np.mean(x < 0)),
                 "row_RMS_quantiles": quantiles(np.sqrt(np.mean(x * x, axis=1))),
                 "population": "uniform panel" if uniform is not None else "provided rows; sampling context required"}
        if x.shape[1] % 64 == 0:
            g = x.reshape(len(x), -1, 64)
            stats["group64_amax_quantiles"] = quantiles(np.max(np.abs(g), axis=2))
            stats["group64_RMS_quantiles"] = quantiles(np.sqrt(np.mean(g * g, axis=2)))
        if name in ("q1_post", "q2_post", "ffn_output", "V1", "V2", "Y") and len(x) > 1:
            stats.update(sample_rank(x))
        result["fields"][name] = stats
    for stage in ("q1", "q2"):
        p, q = stage + "_pre", stage + "_post"
        if p in arrays and q in arrays:
            a = arrays[p][uniform] if uniform is not None else arrays[p]
            b = arrays[q][uniform] if uniform is not None else arrays[q]
            result["quantization_pairs"][stage] = paired(a, b)
    if "z" in arrays and "q2_post" in arrays:
        z = arrays["z"][uniform] if uniform is not None else arrays["z"]
        a = arrays["q2_post"][uniform] if uniform is not None else arrays["q2_post"]
        mask = z < 0
        result["negative_z_to_q2_zero_fraction"] = ratio(np.count_nonzero((a == 0) & mask), np.count_nonzero(mask))
    if plan:
        for p in plan["neighbor_pairs"]:
            item = dict(p)
            item["measurements"] = {}
            for name in ("q1_post", "q2_post", "ffn_output"):
                if name in arrays:
                    item["measurements"][name] = paired(arrays[name][[row_lookup[p["a"]]]],
                                                       arrays[name][[row_lookup[p["b"]]]])
            result["neighbor_pairs"].append(item)
    return result


def panel_geometry(arrays, weights, groups, selected=None):
    """Real decoded FP64 post-Q2 panel contributions. NOT a native action replay."""
    a = np.asarray(arrays["q2_post"], dtype=np.float64)
    if selected is not None:
        a = a[selected]
    b = np.asarray(weights["W2_panel"], dtype=np.float64)
    gate = np.asarray(arrays["gate"], dtype=np.float64)
    if b.shape != (5120, 64 * len(groups)) or gate.shape != (1, 5120):
        raise ValueError("panel/gate layout mismatch")
    parts = [(a[:, 64*g:64*(g+1)] @ b[:, 64*k:64*(k+1)].T) * gate for k, g in enumerate(groups)]
    independent = sum(np.sum(p*p, axis=1) for p in parts)
    total = sum(parts)
    return {"groups": groups,
            "signed_sum_energy_over_sum_individual_energies": ratio(np.sum(total*total), np.sum(independent)),
            "group_gated_energy": [float(np.sum(p*p)) for p in parts],
            "scope": "fixed panel geometry; ratio may exceed one; no skip policy, downstream or native quality claim"}


def analyze_run(run_path):
    record = json.loads(run_path.read_text())
    if record.get("status") != "PASS_PASSIVE_CAPTURE_NOOP":
        raise ValueError("Only accepted plain/observed captures may be analyzed as native observations")
    root = run_path.parent / "observed"
    out = {"source_id": record["source_id"], "source_clusters": 1, "scopes": [], "cross_step": [],
           "claim": "descriptive observation; no new method, causal intervention, quality or speed acceptance"}
    previous = {}
    for s in sorted(record["scopes"], key=lambda s: (s["step"], s["block"])):
        path = (root / s["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != s["sha256"]:
            raise ValueError("payload path/hash mismatch")
        a = load_arrays(path)
        item = {"step": s["step"], "block": s["block"], "expert": s["expert"], **describe(a, s)}
        bind = s["weight_binding"]
        wp = (root / bind["path"]).resolve()
        if not wp.is_relative_to(root.resolve()) or sha256(wp) != bind["sha256"]:
            raise ValueError("weight path/hash mismatch")
        lookup = {r: i for i, r in enumerate(s["row_plan"]["rows"])}
        ids = [lookup[r] for r in s["row_plan"]["uniform_rows"]]
        item["panel_geometry"] = panel_geometry(a, load_arrays(wp), bind["groups"], ids)
        out["scopes"].append(item)
        prev = previous.get(s["block"])
        fields = {k: a[k] for k in ("q1_post", "q2_post", "ffn_output")}
        if prev:
            ps, pa = prev
            if ps["row_plan"]["rows"] != s["row_plan"]["rows"] or ps["row_plan"]["grid"] != s["row_plan"]["grid"]:
                raise ValueError("cross-step coordinates are not paired")
            out["cross_step"].append({"block": s["block"], "steps": [ps["step"], s["step"]],
                                     "experts": [ps["expert"], s["expert"]],
                                     "scope": "same latent coordinates, not tracked objects or independent samples",
                                     "measurements": {k: paired(pa[k][ids], fields[k][ids]) for k in fields}})
        previous[s["block"]] = (s, fields)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", type=Path)
    group.add_argument("--legacy-payload", type=Path)
    p.add_argument("--expected-sha")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.run:
        report = analyze_run(args.run)
    else:
        if not args.expected_sha or sha256(args.legacy_payload) != args.expected_sha:
            raise ValueError("Legacy payload requires its archived expected SHA256")
        report = {"status": "LEGACY_PAYLOAD_DESCRIPTIVE_ONLY",
                  "payload_sha256": args.expected_sha,
                  "missing_context": "attach original source/step/block/row sampling receipt; no cross-position pairing inferred",
                  **describe(load_arrays(args.legacy_payload))}
    write_json(args.out, report)


if __name__ == "__main__":
    main()

