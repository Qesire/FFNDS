"""Verify and summarize the sealed current-S control and four-arm Wan replay."""
import argparse
import json
import math
from pathlib import Path
from statistics import geometric_mean

import numpy as np


SOURCES = ("vbi_0392", "vbi_0800", "vbi_1013")
LAYERS = (0, 13, 26, 39)
ARMS = ("A_PFRR", "B_diagonal", "C_paired", "D_shuffled")
LOSSES = ("head", "point", "spatial", "temporal")
TARGETS = dict(zip(LOSSES, (0.3464, 0.3835, 0.4537, 0.3929)))


def checked_gram(raw, size):
    g = np.asarray(raw, dtype=np.float64)
    assert g.shape == (size, size) and np.isfinite(g).all()
    scale = max(g.diagonal().max(), np.finfo(g.dtype).tiny)
    scaled = g / scale
    assert np.allclose(scaled, scaled.T, atol=1e-12, rtol=0)
    assert np.linalg.eigvalsh((scaled + scaled.T) / 2).min() > -1e-10
    return g


def dense_energy(g):
    return g.diagonal()[:-1] + g[-1, -1] + 2 * g[:-1, -1]


def ratios(sites, name, numerator, denominator):
    site_ratios = {(s, layer): y[name][numerator] / y[name][denominator] for (s, layer), y in sites.items()}
    source = {s: geometric_mean(site_ratios[s, layer] for layer in LAYERS) for s in SOURCES}
    layer = {str(l): geometric_mean(site_ratios[s, l] for s in SOURCES) for l in LAYERS}
    return dict(ratio=geometric_mean(source.values()), by_source=source, by_layer=layer,
                winning_sites=sum(1 for v in site_ratios.values() if v < 1), sites=len(site_ratios))


def panel_report(panel):
    assert panel["status"] == "complete"
    sites = {(r["source"], r["layer"]): {loss: {k: a[loss] for k, a in r["scores"].items()} for loss in LOSSES}
             for r in panel["sites"]}
    assert len(panel["sites"]) == 12 and set(sites) == {(s, layer) for s in SOURCES for layer in LAYERS}
    output = {arm: {loss: ratios(sites, loss, "PFRR-S128", arm) for loss in LOSSES} for arm in ("OUT-ABS", "OUT-D")}
    for arm, losses in output.items():
        for loss, row in losses.items():
            assert math.isclose(row["ratio"], panel["ratios"][arm][loss]["ratio"], rel_tol=1e-12)
    return dict(energy_ratios=output, historical_OUT_D_over_OUT_ABS_targets=TARGETS, coverage=panel["coverage"],
                max_legacy_fit_metric_relative_error=max(v for r in panel["sites"] for m in r["legacy_relative_deviation"].values() for v in m.values()))


def fourarm_report(replay):
    assert replay["status"] == "complete" and tuple(replay["arms"]) == ARMS
    assert replay["additional_native_attention_queries"] == 0 and replay["ridge"] == 0.001
    fit, held = replay["value_fit_rows"], replay["value_held_rows"]
    assert len(fit) == 256 and len(held) == 64 and len(set(fit + held)) == 320
    shuffled = replay["shuffled_current_order"]
    assert sorted(shuffled) == list(range(256)) and all(i != j for i, j in enumerate(shuffled))
    assert set(replay["records"]) == set(SOURCES)
    sites, audit = {}, []
    for source, record in replay["records"].items():
        assert record["trace_exact"] and set(map(int, record["sites"])) == set(LAYERS)
        for layer, row in record["sites"].items():
            tail = row["tail"]
            assert row["fixed_attention_probes_exact"] and tail["shared_probe_ffn_input_exact"]
            assert tail["reused_live_ffn_probe_teacher"] and tail["dense_native_ffn_extra_held_rows"] == 448
            v = checked_gram(row["value_held_gram"], 3)
            h = checked_gram(row["held_attention_gram"], 4)
            base = checked_gram(row["held_base_gram"], 4)
            stage = {k: checked_gram(x, 4) for k, x in tail["stage_gram_raw_transport"].items()}
            dense = {k: checked_gram(x, 5) for k, x in tail["stage_gram_raw_transport_background"].items()}
            edges = {k: checked_gram(x, 5) for k, x in tail["block_edges"].items()}
            for k, g in dense.items():
                assert np.allclose(stage[k], g[:4, :4], rtol=1e-10, atol=1e-14)
            assert np.allclose(h, stage["attention_head"], rtol=1e-10, atol=1e-14)
            sites[source, int(layer)] = dict(
                held_value=dict(zip(ARMS[1:], v.diagonal() / row["held_value_target_energy"])),
                pre_recovery_attention=dict(zip(ARMS, base.diagonal())),
                held_attention=dict(zip(ARMS, h.diagonal())),
                **{k + "_to_wide": dict(zip(ARMS, g.diagonal())) for k, g in stage.items()},
                **{k + "_to_dense": dict(zip(ARMS, dense_energy(g))) for k, g in dense.items()},
                **{k + "_edge_to_dense": dict(zip(ARMS, dense_energy(g))) for k, g in edges.items()},
            )
            block = dense["block_exit"]
            delta = block[2, 2] - block[1, 1]
            bg = 2 * (block[2, -1] - block[1, -1])
            audit.append(dict(source=source, layer=int(layer),
                block_paired_minus_diagonal_to_wide=float(delta), block_paired_minus_diagonal_background_cross=float(bg),
                block_paired_minus_diagonal_to_dense=float(delta + bg), block_diagonal_to_dense=float(dense_energy(block)[1]),
                live_tail_relative_l2=tail["live_relative_l2"], probe_relative_l2=tail["shared_probe_vs_live_relative_l2"],
                ridge_max_condition=row["ridge_system_max_condition"]))
    comparisons = {}
    for metric in next(iter(sites.values())):
        comparisons[metric] = {arm: ratios(sites, metric, arm, ARMS[1]) for arm in ARMS if arm != ARMS[1]
                               and (arm != ARMS[0] or metric not in ("held_value", "pre_recovery_attention"))}
        comparisons[metric]["C_over_D"] = ratios(sites, metric, ARMS[2], ARMS[3])
    primary = ("held_value", "held_attention", "block_exit_to_dense")
    continuation = all(max(comparisons[k][comparison]["by_source"].values()) < 1
                       for k in primary for comparison in (ARMS[2], "C_over_D"))
    return dict(ratios=comparisons, same_block_dense_background=audit, continuation_gate_passed=continuation,
                condition="C/B and C/D each below one in all three sources on value, post-recovery attention and same-input Dense block exit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--fourarm", type=Path)
    args = parser.parse_args()
    panel = json.loads(args.panel.read_bytes())
    summary = dict(plan_commit=panel["plan_commit"], panel=panel_report(panel))
    if args.fourarm:
        replay = json.loads(args.fourarm.read_bytes())
        assert replay["plan_commit"] == panel["plan_commit"]
        summary["fourarm"] = fourarm_report(replay)
    args.out.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
