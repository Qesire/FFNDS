#!/usr/bin/env python3
"""CPU witnesses for causal error transport; all numerical operands are synthetic.

This does not load Wan weights, run a model, or estimate real video quality.
The identities apply to attention derivatives and an augmented-state DAG.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def attention(q, k, v):
    logits = q @ k.T / np.sqrt(q.shape[1])
    a = np.exp(logits - logits.max(axis=1, keepdims=True))
    a /= a.sum(axis=1, keepdims=True)
    return a @ v, a


def main(out):
    rng = np.random.default_rng(20261128)
    n, d = 19, 7
    p = np.array([1, 5, 11])
    q, k, v = (rng.normal(size=(n, d)) for _ in range(3))
    dq = np.zeros_like(q)
    dk, dv = (rng.normal(size=(n, d)) for _ in range(2))
    dk[p] = 0
    dv[p] = 0
    o, a = attention(q, k, v)
    ds = (dq @ k.T + q @ dk.T) / np.sqrt(d)
    analytical = a @ dv + (a * (ds - (a * ds).sum(1, keepdims=True))) @ v
    eps = 1e-6
    plus = attention(q + eps*dq, k + eps*dk, v + eps*dv)[0]
    minus = attention(q - eps*dq, k - eps*dk, v - eps*dv)[0]
    derivative_error = float(np.max(np.abs((plus-minus)/(2*eps)-analytical)))
    assert derivative_error < 2e-8
    assert float(np.linalg.norm(analytical[p])) > 1e-4

    # A first block can have zero local error on P; the next all-key attention
    # still reads non-P states through K/V. Compute its P rows natively.
    x = rng.normal(size=(n, d))
    error = rng.normal(size=x.shape)*0.08
    error[p] = 0
    wq, wk, wv = (rng.normal(size=(d, d))/np.sqrt(d) for _ in range(3))
    baseline = attention(x @ wq, x @ wk, x @ wv)[0]
    changed = attention((x+error) @ wq, (x+error) @ wk, (x+error) @ wv)[0]
    p_change = float(np.linalg.norm((changed-baseline)[p]))
    assert np.array_equal((x+error)[p], x[p])
    assert p_change > 1e-5

    # A small augmented DAG carries both state and cache edges. Its linear
    # propagation identity is exact. It is a mathematical construction, not
    # the Wan execution graph or a fit to the experimental timing data.
    dim = 5
    ident = np.eye(dim)
    nodes = [
        ("state0", {}, np.zeros(dim)),
        ("anchorH", {"state0": 1.02*ident}, np.zeros(dim)),
        ("state1", {"anchorH": .91*ident}, np.zeros(dim)),
        ("cacheH", {"anchorH": .7*ident}, np.zeros(dim)),
        ("hitH", {"state1": 1.08*ident, "cacheH": .12*ident}, rng.normal(size=dim)*.03),
        ("state2", {"state1": ident, "hitH": -.2*ident}, np.zeros(dim)),
        ("anchorL", {"state2": 1.1*ident}, np.zeros(dim)),
        ("cacheL", {"anchorL": .9*ident}, np.zeros(dim)),
        ("state3", {"state2": ident, "anchorL": -.3*ident}, np.zeros(dim)),
        ("hitL", {"state3": .8*ident, "cacheL": .2*ident}, rng.normal(size=dim)*.01),
        ("state4", {"state3": ident, "hitL": -.6*ident}, np.zeros(dim)),
    ]
    values, transport, local = {}, {}, {}
    for name, parents, innovation in nodes:
        assert all(parent in values for parent in parents)
        values[name] = innovation + sum((w @ values[parent] for parent, w in parents.items()), start=np.zeros(dim))
        transport[name] = {name: ident.copy()}
        local[name] = innovation
        for parent, w in parents.items():
            for origin, previous in transport[parent].items():
                transport[name][origin] = transport[name].get(origin, np.zeros((dim, dim))) + w @ previous
    summed = sum((matrix @ local[origin] for origin, matrix in transport["state4"].items()), start=np.zeros(dim))
    path_error = float(np.max(np.abs(summed-values["state4"])))
    assert path_error < 1e-14
    assert np.linalg.norm(local["anchorL"]) == 0
    assert np.linalg.norm(values["anchorL"]) > 0
    contributions = [transport["state4"][origin] @ local[origin] for origin in ("hitH", "hitL")]
    cross = 2*float(contributions[0] @ contributions[1])
    energy_sum = sum(float(c @ c) for c in contributions)
    total_energy = float(values["state4"] @ values["state4"])
    energy_error = abs(total_energy-energy_sum-cross)
    assert energy_error < 1e-15

    result = {
        "status": "PASS_SYNTHETIC_MATHEMATICAL_WITNESSES",
        "scope": "All numerical operands synthetic; no Wan model measurements or performance prediction",
        "new_model_forwards": 0,
        "new_gpu_jobs": 0,
        "attention_differential": {
            "max_abs_central_difference_error": derivative_error,
            "delta_q_at_P_norm": 0.0,
            "delta_kv_at_P_norm": 0.0,
            "nonP_perturbations_induce_P_output_derivative_norm": float(np.linalg.norm(analytical[p])),
            "meaning": "Native P queries at the next layer depend on non-P keys and values",
        },
        "conditional_P_exactness": {
            "first_block_P_local_error": 0.0,
            "next_attention_P_output_difference_norm": p_change,
            "meaning": "Local same-input P exactness does not imply equality to a separate Dense trajectory",
        },
        "augmented_state_DAG": {
            "nodes": len(nodes),
            "path_sum_identity_max_abs": path_error,
            "low_anchor_local_innovation_norm": 0.0,
            "low_anchor_inherited_difference_norm": float(np.linalg.norm(values["anchorL"])),
            "final_difference_energy": total_energy,
            "sum_individual_transport_energies": energy_sum,
            "twice_transport_inner_product": cross,
            "energy_decomposition_abs_error": energy_error,
            "meaning": "Refreshing a native anchor retains inherited state differences; joint final error includes cross terms",
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"status": result["status"], "output": str(out), "max_derivative_error": derivative_error}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("CAUSAL_TRANSPORT_VERIFIED.json"))
    main(parser.parse_args().out)
