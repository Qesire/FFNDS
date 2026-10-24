#!/usr/bin/env python3
"""CPU-only FFNDS archived evidence reaggregation + exhaustive FP4 orbit check.

Uses existing Git-tracked JSON archives; never imports torch, accesses GPU, fetches
HF tensors, or runs model inference. Quantizer code occupancy of native Wan
cannot be inferred from the available scalar summaries.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path


ALPHABET_X2 = (0, 1, -1, 2, -2, 3, -3, 4, -4, 6, -6, 8, -8, 12, -12)


def avg(a):
    return sum(a) / len(a)


def read(root, path):
    return json.loads((root / path).read_text(encoding="utf-8"))


def orbit_signature(code):
    g = 0
    for x in code:
        g = math.gcd(g, abs(x))
    if g == 0:
        return (0,)
    first = next(x for x in code if x)
    sgn = 1 if first > 0 else -1
    return tuple(sgn * x // g for x in code)


def exact_orbit_counts():
    out = []
    for k in range(1, 5):
        signatures = {orbit_signature(x) for x in itertools.product(ALPHABET_X2, repeat=k)}
        expected = 1 + (15 ** k - 11 ** k - 7 ** k + 5 ** k) // 2
        assert len(signatures) == expected, (k, len(signatures), expected)
        out.append({
            "k": k,
            "enumerated_tuples": 15 ** k,
            "unique_orbits": expected,
            "min_guaranteed_repeat_in_128_aligned_tokens": max(0, 128 - expected),
            "min_guaranteed_repeat_in_31668_aligned_tokens": max(0, 31668 - expected),
        })
    return out


def recompute_op2d(op):
    assert len(op["prefix_controls"]) == 24
    assert all(all(row[k] for k in ("V1_bitwise", "TeacherY_bitwise", "gate_and_raw_bias_bitwise"))
               for row in op["prefix_controls"])
    positions = sorted(op["positions"], key=lambda x: x["step"])
    assert [p["step"] for p in positions] == [0, 1, 2, 3]
    sources = {x["index"] for x in positions[0]["cases"]}
    assert len(sources) == 24
    by_step = []
    for p in positions:
        d = {x["index"]: x for x in p["cases"]}
        assert set(d) == sources and p["block"] == 20
        a = list(d.values())
        by_step.append({
            "step": p["step"],
            "n_sources": len(a),
            "mean_V1_cosine": avg([x["V1"]["cosine"] for x in a]),
            "mean_V1_relative_RMSE": avg([x["V1"]["relative_RMSE"] for x in a]),
            "mean_TeacherY_cosine": avg([x["TeacherY"]["cosine"] for x in a]),
            "UB1_over_C24_own_state_gated_SSE": (
                sum(x["UB1_own_state_gated_native_SSE"] for x in a) /
                sum(x["C24_own_state_gated_native_SSE"] for x in a)
            ),
        })
    byid = [{x["index"]: x for x in p["cases"]} for p in positions]
    delta01 = [byid[1][i]["V1"]["cosine"] - byid[0][i]["V1"]["cosine"] for i in sorted(sources)]
    delta31 = [byid[1][i]["V1"]["cosine"] - byid[3][i]["V1"]["cosine"] for i in sorted(sources)]
    return {
        "positions": by_step,
        "sources_step1_V1_cosine_gt_step0": sum(x > 0 for x in delta01),
        "sources_step1_V1_cosine_gt_step3": sum(x > 0 for x in delta31),
        "mean_step1_minus_step0_V1_cosine": avg(delta01),
        "step1_minus_step0_leave_one_source_out_mean_range": [
            min(avg(delta01[:i] + delta01[i + 1:]) for i in range(24)),
            max(avg(delta01[:i] + delta01[i + 1:]) for i in range(24)),
        ],
    }


def recompute_c1(record):
    rows = record["rows"]
    assert len(rows) == 72 and len({r["source"] for r in rows}) == 6
    assert set(r["rows"] for r in rows) == {31668}
    n = sum(r["nextnorm_total_SSE"] for r in rows)
    post_capture = sum(r["post_selected_nextnorm_SSE"] for r in rows)
    next_capture = sum(r["nextnorm_selected_nextnorm_SSE"] for r in rows)
    return {
        "source_count": len({r["source"] for r in rows}),
        "site_count": len(rows),
        "rows_per_site": 31668,
        "mean_post_nextnorm_Spearman": avg([r["Spearman"] for r in rows]),
        "mean_top_quarter_overlap": avg([r["selected_overlap_fraction"] for r in rows]),
        "post_selected_capture_of_nextnorm": post_capture / n,
        "nextnorm_selected_capture_of_nextnorm": next_capture / n,
        "oracle_boundary_gain_fraction": (next_capture - post_capture) / n,
        "all_sites_nextnorm_oracle_dominates_post_target_selection": all(
            r["nextnorm_selected_nextnorm_SSE"] >= r["post_selected_nextnorm_SSE"] - 1e-7
            for r in rows
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    root = args.repo_root.resolve()
    op = read(root, "reports/worker_20261020/OP2D_native_state_description_RUN.json")
    c1 = read(root, "research/20261024/C1_boundary_target_audit.json")
    receipt = read(root, "reports/worker_20261020/OP2A_complete_dataset_integrity_RUN.json")
    r12 = read(root, "research/20261022/R12_HiF4_group_separability.json")
    assert receipt["verified_payloads"] == 3840
    assert r12["saved_post_Q1_reference_cases"] and all(
        all(x["checks"].values()) for x in r12["saved_post_Q1_reference_cases"])
    output = {
        "id": "OFFLINE_SAVED_ASSET_QSTRUCTURE_REPRODUCE_R1",
        "scope": "CPU saved-scalar reaggregation and exact E2M1 combinatorial enumeration",
        "assets_not_read": "Original OP2/HF native tensor payloads and checkpoints",
        "op2d": recompute_op2d(op),
        "c1": recompute_c1(c1),
        "code_orbits": exact_orbit_counts(),
        "native_Q1_Q2_code_collision_rate": None,
        "native_Q2_scale_certification_hit_rate": None,
        "native_FFN_or_end_to_end_speedup": None,
    }
    serialized = json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
