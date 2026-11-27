"""Independently recompute school job206224 timing and typed cache budgets.

Uses raw timing receipts, not the archived summary's computed ratios. No GPU.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


ARMS = ("dense", "wide8192", "centered")
PAIRS = (("centered", "dense"), ("wide8192", "dense"), ("centered", "wide8192"))
REPS, SEED = 10000, 20261127


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_indices(labels, rng):
    labels = np.asarray(labels)
    return np.concatenate([idx[rng.integers(0, len(idx), (REPS, len(idx)))]
                           for label in sorted(set(labels))
                           for idx in [np.flatnonzero(labels == label)]], axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    raw_path = args.source / "TIMING_RAW_206224.json"
    raw, cohort, final = (read(args.source / name) for name in
                          ("TIMING_RAW_206224.json", "COHORT_50_PUBLIC.json", "QUALITY_50_SUMMARY.json"))
    early = read(args.source / "TIMING_SUMMARY_206224.json")["timing"]
    ids = cohort["timing_source_ids"]
    people = {r["source_id"]: r for r in cohort["sources"]}
    assert len(ids) == len(set(ids)) == 12 and len(raw["timing"]) == 36
    assert len({people[s]["source_cluster_id"] for s in ids}) == 12
    assert Counter(people[s]["stratum"] for s in ids) == {"structured": 4, "natural": 4, "subject": 4}
    grouped = {sid: {} for sid in ids}
    orders = Counter()
    ff_cache_bytes = set()
    requests = []
    for row in raw["timing"]:
        sid, repeat = row["source_id"], row["repeat"]
        assert repeat not in grouped[sid] and repeat in (0, 1, 2)
        grouped[sid][repeat] = row
        orders[tuple(row["order"])] += 1
        assert set(row["arms"]) == set(ARMS)
        traces = []
        for arm, wrapper in row["arms"].items():
            record = wrapper["record"]
            assert not wrapper["final_latent_saved"] and not wrapper["quality_capture_in_timing_window"]
            assert not wrapper["trajectory_saved"] and not wrapper["diagnostic_replay"]
            assert wrapper["binding_sha256"] == raw["method_binding_sha256"]
            assert wrapper["source_manifest_sha256"] == raw["cohort_sha256"]
            assert record["experts"] == [0, 0, 1, 1]
            assert len(record["step_ms"]) == len(record["flows"]) == len(record["latents"]) == 4
            traces.append((record["initial"], record["flows"][0], record["latents"][0]))
            if arm != "dense":
                counts = record["counters"]
                assert (counts["anchors"], counts["hits"], counts["new_packed_sites"]) == (80, 80, 0)
                assert counts["anchor_d2h"] == counts["anchor_h2d"] == 0
                ff_cache_bytes.add(counts["anchor_gpu_capacity_bytes"])
            if arm == "centered":
                counts = wrapper["attention"]
                assert tuple(counts[k] for k in ("attention_sites", "anchors", "hits", "native_probe_queries")) == (160, 80, 80, 5120)
            requests.append(dict(source_id=sid, repeat=repeat, arm=arm,
                                 ordinal_position=row["order"].index(arm), **record))
        assert len(set(traces)) == 1
    assert len(orders) == 6 and set(orders.values()) == {6}
    for sid in ids:
        assert set(grouped[sid]) == {0, 1, 2}
        for arm in ARMS:
            rows = [grouped[sid][j] for j in range(3)]
            assert sorted(r["order"].index(arm) for r in rows) == [0, 1, 2]
            records = [r["arms"][arm]["record"] for r in rows]
            signatures = {json.dumps([r["initial"], r["flows"], r["latents"],
                                      r["video"]["decoded_rgb24_sha256"]]) for r in records}
            assert len(signatures) == 1
    early_idx = bootstrap_indices([people[s]["stratum"] for s in ids], np.random.default_rng(SEED))
    rng = np.random.default_rng(SEED)
    bootstrap_indices([r["stratum"] for r in cohort["sources"]], rng)
    final_idx = bootstrap_indices([people[s]["stratum"] for s in ids], rng)
    report = dict(schema="FFNDS_JOB206224_INDEPENDENT_SPEED_MEMORY_V1", raw_sha256=sha(raw_path),
                  source_commit="a1eb64d63910b4d90b94ea4c641ef461c83d0d92", gpu=raw["gpu"],
                  checks=dict(requests=108, source_clusters=12, repeats=3, all_initial_and_first_anchors_paired=True,
                              all_36_source_arm_repeats_bitwise_equal=True, extra_tensor_capture_disabled=True,
                              six_orders_each_six_times=True, probes_and_full_trajectory_counts=True),
                  aggregation="arithmetic mean within source/arm, geometric mean of reference/candidate ratios across sources",
                  intervals={}, means={}, order_diagnostics={}, outliers=[])
    max_point_error = max_final_ci_error = max_early_ci_error = 0.0
    for interval, getter in (("wall_s", lambda r: r["wall"]),
                             ("gpu_s", lambda r: sum(r["step_ms"]) / 1000)):
        values = {arm: np.asarray([[getter(grouped[s][j]["arms"][arm]["record"])
                                    for j in range(3)] for s in ids]) for arm in ARMS}
        means = {a: v.mean(axis=1) for a, v in values.items()}
        report["intervals"][interval] = {}
        for candidate, reference in PAIRS:
            label = candidate + "/" + reference
            logs = np.log(means[reference] / means[candidate])
            point = float(np.exp(logs.mean()))
            ci = np.quantile(np.exp(logs[final_idx].mean(1)), [.025, .975])
            ci_early = np.quantile(np.exp(logs[early_idx].mean(1)), [.025, .975])
            result = dict(speedup=point, time_saving_percent=100 * (1 - 1 / point),
                          ci95_final_rng=ci.tolist(), ci95_timing_only_rng=ci_early.tolist(),
                          reference_mean_seconds=float(means[reference].mean()),
                          candidate_mean_seconds=float(means[candidate].mean()),
                          candidate_faster_source_means=int((means[candidate] < means[reference]).sum()),
                          candidate_faster_paired_repeats=int((values[candidate] < values[reference]).sum()),
                          min_source_speedup=float(np.exp(logs).min()), max_source_speedup=float(np.exp(logs).max()),
                          sensitivity_within_source_median_speedup=float(np.exp(np.log(np.median(values[reference], axis=1) /
                                                                                      np.median(values[candidate], axis=1)).mean())),
                          by_source=[dict(source_id=s, candidate_seconds=float(means[candidate][i]),
                                          reference_seconds=float(means[reference][i]), speedup=float(np.exp(logs[i])))
                                     for i, s in enumerate(ids)])
            report["intervals"][interval][label] = result
            archived = final["timing"]["by_interval"][interval][label]
            max_point_error = max(max_point_error, abs(point - archived["speedup"]))
            max_final_ci_error = max(max_final_ci_error, float(np.max(np.abs(ci - archived["speedup_ci95"]))))
            max_early_ci_error = max(max_early_ci_error, float(np.max(np.abs(ci_early - early["by_interval"][interval][label]["speedup_ci95"]))))
        # Ordinal means use each source once at every position, thus remain source balanced.
        report["order_diagnostics"][interval] = {
            arm: {str(pos + 1): float(np.mean([getter(r) for r in requests
                                            if r["arm"] == arm and r["ordinal_position"] == pos]))
                  for pos in range(3)} for arm in ARMS}
        estimated_attention_only = means["centered"] + means["dense"] - means["wide8192"]
        assert np.all(estimated_attention_only > 0)
        report.setdefault("attention_only_conditional_estimate", {})[interval] = dict(
            formula="T_A = T_M + T_D - T_W within each source after averaging three rounds",
            condition="Assumes the FFN time saving transfers from native-attention to centered-attention background; omits timing interaction and is not measured",
            mean_seconds=float(estimated_attention_only.mean()),
            estimated_speedup=float(np.exp(np.log(means["dense"] / estimated_attention_only).mean())),
            by_source=[dict(source_id=s, estimated_seconds=float(estimated_attention_only[i]),
                            estimated_speedup=float(means["dense"][i] / estimated_attention_only[i]))
                       for i, s in enumerate(ids)])
    for arm in ARMS:
        rows = [r for r in requests if r["arm"] == arm]
        steps = np.asarray([r["step_ms"] for r in rows]) / 1000
        wall = np.asarray([r["wall"] for r in rows])
        gpu = steps.sum(1)
        peaks = np.asarray([r["peak_allocated"] for r in rows]) / 2**30
        reserved = np.asarray([r["peak_reserved"] for r in rows]) / 2**30
        report["means"][arm] = dict(wall_s=float(wall.mean()), gpu_s=float(gpu.mean()), step_s=steps.mean(0).tolist(),
                                   anchors_s=float(steps[:, [0, 2]].sum(1).mean()),
                                   hits_s=float(steps[:, [1, 3]].sum(1).mean()),
                                   outside_step_events_s=float((wall - gpu).mean()),
                                   wall_p50_s=float(np.median(wall)), wall_p95_s=float(np.quantile(wall, .95)),
                                   allocated_peak_gib=[float(peaks.min()), float(peaks.max())],
                                   reserved_peak_gib=[float(reserved.min()), float(reserved.max())])
    for r in sorted(requests, key=lambda x: x["wall"] - sum(x["step_ms"]) / 1000, reverse=True)[:8]:
        report["outliers"].append({k: r[k] for k in ("source_id", "arm", "repeat", "wall", "ordinal_position")})
        report["outliers"][-1].update(gpu_s=sum(r["step_ms"]) / 1000,
                                      outside_step_events_s=r["wall"] - sum(r["step_ms"]) / 1000)
    a, d, w = (report["means"][key] for key in ("centered", "dense", "wide8192"))
    report["cost_decomposition"] = dict(
        dense_minus_centered_anchor_s=d["anchors_s"] - a["anchors_s"],
        dense_minus_centered_hit_s=d["hits_s"] - a["hits_s"],
        dense_minus_centered_gpu_s=d["gpu_s"] - a["gpu_s"],
        dense_minus_centered_wall_s=d["wall_s"] - a["wall_s"],
        wide_minus_centered_gpu_s=w["gpu_s"] - a["gpu_s"],
        wide_minus_centered_wall_s=w["wall_s"] - a["wall_s"],
        share_of_mean_gpu_saving_added_by_attention=(w["gpu_s"] - a["gpu_s"]) / (d["gpu_s"] - a["gpu_s"]),
        share_of_mean_wall_saving_added_by_attention=(w["wall_s"] - a["wall_s"]) / (d["wall_s"] - a["wall_s"]))
    assert max_point_error < 1e-12 and max_final_ci_error < 1e-12 and max_early_ci_error < 1e-12
    report["archive_reproduction"] = dict(max_point_error=max_point_error, max_final_ci_error=max_final_ci_error,
                                           max_timing_only_ci_error=max_early_ci_error)
    N, D, K, H, L, R, P = 31668, 5120, 2560, 40, 40, 31, 64
    fields = dict(ffn_u0=2 * L * N * D, ffn_y0=2 * L * N * D, ffn_z0k=2 * L * N * K,
                  low_attention_centered_history=2 * L * N * D,
                  low_attention_projected_anchor=2 * L * H * N * R,
                  low_attention_rho=4 * L * H * N)
    assert sum(fields[k] for k in ("ffn_u0", "ffn_y0", "ffn_z0k")) == next(iter(ff_cache_bytes))
    assert len(ff_cache_bytes) == 1
    T = (N + 127) // 128
    # Logical active block IDs are not a measurement of actual BlockMask storage.
    route_active = 4 * L * H * T * 31
    route_one_padded = 4 * L * H * T * T
    ff8_payload = L * N * (2 * D + K)
    ff8_scale = 2 * L * N * ((2 * D + K) // 128)
    ff8_exact = 2 * L * P * (2 * D + K)
    ff8 = ff8_payload + ff8_scale + ff8_exact
    ff16 = sum(fields[k] for k in ("ffn_u0", "ffn_y0", "ffn_z0k"))
    att = sum(v for k, v in fields.items() if k.startswith("low_attention"))
    report["memory"] = dict(
        field_bytes=fields, field_gib={k: v / 2**30 for k, v in fields.items()},
        ffn_counter_bytes=ff16, low_attention_principal_fields_gib=att / 2**30,
        combined_principal_fields_gib=(ff16 + att) / 2**30,
        active_route_ID_payload_gib=route_active / 2**30,
        one_padded_BlockMask_index_array_gib=route_one_padded / 2**30,
        route_note="Actual BlockMask may retain padded and reverse-order index arrays/counts; active-ID payload is logical support, not measured allocation.",
        lifetime="FFN has 40 persistent slots shared across experts. high attention keeps routes only then pops on step1. low keeps route/rho/history/features after step2 and pops each layer on step3; no two-expert doubling.",
        added_measured_peak_gib=a["allocated_peak_gib"][1] - w["allocated_peak_gib"][1],
        added_measured_peak_fraction=a["allocated_peak_gib"][1] / w["allocated_peak_gib"][1] - 1,
        baseline_peak_scope="Dense and W share a resident FFN port including30.20096GiB history;47.57GiB is not a clean standalone Dense peak.",
        int8_ffn=dict(payload_bytes=ff8_payload, scale_bytes=ff8_scale, exact_P64_bytes=ff8_exact,
                      principal_gib=ff8 / 2**30, saved_gib=(ff16 - ff8) / 2**30,
                      combined_principal_with_unchanged_attention_gib=(ff8 + att) / 2**30,
                      conditional_same_transient_peak_estimate_gib=a["allocated_peak_gib"][1] - (ff16 - ff8) / 2**30,
                      estimate_status="Static capacity combination only; full centered+INT8 quality/speed/peak not measured in job206224"))
    report["setup"] = dict(cold_init_seconds=raw["cold_init_seconds"], wide_port_setup_seconds=raw["setup"][0]["seconds"],
                           setup_only_payback_requests=raw["setup"][0]["seconds"] / (d["wall_s"] - a["wall_s"]),
                           payback_scope="Conditional accounting of port preparation alone; excludes initial compilation and shared model loading")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(archive_reproduction=report["archive_reproduction"], intervals=report["intervals"],
                         means=report["means"], cost_decomposition=report["cost_decomposition"], memory=report["memory"]),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
