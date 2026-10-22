#!/usr/bin/env python3
"""Read-only finite-action oracle arithmetic. Does not establish Wan evidence."""
import argparse
import json
import math
from pathlib import Path


def analyze(data):
    if data.get("schema") != "ffnds.counterfactual_actions.v1":
        raise ValueError("schema mismatch")
    actions = data.get("actions")
    if not isinstance(actions, list) or len(actions) < 2 or len(set(actions)) != len(actions):
        raise ValueError("invalid action set")
    ref = data.get("reference_action")
    budget = data.get("action_budget_groups", {})
    if ref not in actions or data.get("metric") != "native_flow_SSE":
        raise ValueError("unmatched metric/reference")
    if set(budget) != set(actions) or any(type(x) is not int or x <= 0 for x in budget.values()) or len(set(budget.values())) != 1:
        raise ValueError("unequal or invalid group budget")
    rows = data.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("empty rows")
    ids = [row.get("source_id") for row in rows]
    if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("source IDs must be unique strings")
    has_policy = ["predicted_action" in row for row in rows]
    if any(has_policy) and not all(has_policy):
        raise ValueError("partially specified predictions")
    sums = {a: 0.0 for a in actions}
    oracle_sum = policy_sum = 0.0
    detailed = []
    for row in rows:
        losses = row.get("losses")
        if not isinstance(losses, dict) or set(losses) != set(actions):
            raise ValueError("all action losses required for every source")
        if any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in losses.values()):
            raise ValueError("loss must be finite nonnegative numeric")
        for action in actions:
            sums[action] += losses[action]
        selected = min(actions, key=lambda a: losses[a])
        oracle_sum += losses[selected]
        d = {"source_id": row["source_id"], "oracle_action": selected, "oracle_loss": losses[selected]}
        if all(has_policy):
            pa = row["predicted_action"]
            if pa not in actions:
                raise ValueError("predicted action missing from actions")
            policy_sum += losses[pa]
            d.update(predicted_action=pa, per_source_regret=losses[pa] - losses[selected])
        detailed.append(d)
    n = len(rows)
    fixed = min(actions, key=lambda a: sums[a])
    oracle = oracle_sum / n
    best_fixed = sums[fixed] / n
    if oracle > best_fixed + 1e-8:
        raise AssertionError("inconsistent oracle ordering")
    result = {
        "status": "FINITE_ACTION_MATRIX_DIAGNOSTIC_ONLY",
        "evidence_level": data.get("evidence_level", "UNVERIFIED_USER_INPUT"),
        "metric": data["metric"], "source_count": n, "action_budget_groups": budget[ref],
        "mean_loss_by_action": {a: sums[a]/n for a in actions},
        "best_fixed_action": fixed, "best_fixed_mean_loss": best_fixed,
        "reference_mean_loss": sums[ref]/n,
        "oracle_mean_loss": oracle,
        "oracle_gain_over_best_fixed": best_fixed - oracle,
        "oracle_gain_over_reference": sums[ref]/n - oracle,
        "per_source": detailed,
        "interpretation": "Post-action oracle is nondeployable; no timing, original Wan run, independent quality, or matched native execution is certified here."
    }
    if all(has_policy):
        policy = policy_sum/n
        result.update(policy_mean_loss=policy, policy_gain_over_best_fixed=best_fixed-policy,
                      policy_regret_over_oracle=policy-oracle)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = analyze(json.loads(Path(args.input).read_text(encoding="utf-8")))
    s = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        Path(args.output).write_text(s, encoding="utf-8")
    else:
        print(s, end="")


if __name__ == "__main__":
    main()
