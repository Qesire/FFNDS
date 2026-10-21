#!/usr/bin/env python3
"""Check recorded research decisions; never launch models or infer novelty.

Exit 0: consistent registry (and requested method eligible, if supplied).
Exit 1: inconsistent/malformed records. Exit 2: method not eligible or unknown.
The R5 evidence audit is deliberately version-scoped. Future decisions must
preserve this closed recipe and explicitly migrate their evidence checks.
"""
import argparse
import json
import math
from pathlib import Path


READY = "ELIGIBLE_REGISTERED_SCREEN"
CURRENT_STATE = "REGISTERED_R4_SEQUENCE_CLOSED_NO_ELIGIBLE_FOLLOWUP"
RECIPE_ID = "R4E1_FIXED_H_16GROUP_Q2"
RATIO_FIELDS = {
    "K1_matched_over_K0_matched": ("K0_matched", .98),
    "K1_matched_over_frozen_UB1": ("K0_frozen", .98),
    "K1_matched_over_C_matched": ("C_matched", .99),
    "K1_matched_over_random_matched": ("random_matched", .99),
}


def load_records(root):
    def read(path):
        return json.loads((root / path).read_text(encoding="utf-8"))
    decision = read("research/CURRENT_DECISION.json")
    return (decision, read(decision["triage"]),
            read(decision["evidence_summary"]), read(decision["execution_record"]))


def build_index(triage):
    index, errors = {}, []
    for method in triage["methods"]:
        for key in [method["id"], *method.get("aliases", [])]:
            if key in index:
                errors.append("Duplicate method ID/alias: " + key)
            index[key] = method
    return index, errors


def validate_records(decision, triage, summary, execution):
    index, errors = build_index(triage)
    methods = {m["id"]: m for m in triage["methods"]}
    if decision["execution_state"] != CURRENT_STATE:
        errors.append("Current evidence audit requires an explicit migration for a new decision state")
    if triage["policy"]["execution_state"] != decision["execution_state"]:
        errors.append("Policy and current decision disagree")
    if triage["policy"].get("current_decision") != "research/CURRENT_DECISION.json":
        errors.append("Missing canonical decision pointer")
    for field in ("closed_method_statuses", "guarded_current_statuses"):
        for method_id, status in decision[field].items():
            if method_id not in methods or methods[method_id]["status"] != status:
                errors.append("Changed or removed guarded disposition: " + method_id)
    if decision["closed_method_statuses"].get(RECIPE_ID) != "CLOSED_REJECTED_REGISTERED_NEIGHBORHOOD":
        errors.append("R4E1 finite rejection must remain in the closed ledger")

    eligible = {m["id"] for m in methods.values()
                if m.get("model_execution_eligibility") == READY}
    canonical = decision["eligible_model_method_ids"]
    if len(canonical) != len(set(canonical)) or eligible != set(canonical):
        errors.append("Eligible methods disagree with canonical decision")
    if canonical:
        errors.append("R5 closure records no eligible model follow-up; update evidence before changing the decision")
    for method_id in eligible:
        if (method_id in decision["closed_method_statuses"]
                or method_id in decision["guarded_current_statuses"]
                or methods[method_id]["status"].startswith(("CLOSED", "DEFERRED", "NOT_TRIGGERED"))):
            errors.append("Closed/deferred method made eligible: " + method_id)

    # Recompute the published conjunction, without implying a native re-execution.
    rows = summary["CV"]
    if len(rows) != 4 or {(r["step"], r["block"], r["fold"]) for r in rows} != {
            (2, 20, 0), (2, 20, 1), (3, 20, 0), (3, 20, 1)}:
        errors.append("Registered four position/fold records changed")
    gates = []
    for row in rows:
        label = "step%s/fold%s" % (row["step"], row["fold"])
        if row["finite_candidates"] != 849:
            errors.append("Finite proxy pool changed: " + label)
        risks = row["native_risks"]
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0
               for v in [risks["K1_matched"], *(risks[key] for key, _ in RATIO_FIELDS.values())]):
            errors.append("Invalid native risk denominator/value: " + label)
            continue
        candidate = row["candidate"]
        proxy = candidate["proxy_advantage"]
        if not math.isfinite(proxy):
            errors.append("Invalid proxy: " + label)
        passed = candidate["swap_count"] > 0 and proxy > 0
        for field, (denominator, threshold) in RATIO_FIELDS.items():
            ratio = risks["K1_matched"] / risks[denominator]
            if not math.isclose(ratio, row[field], rel_tol=1e-12, abs_tol=1e-12):
                errors.append("Saved ratio mismatch: " + label + "/" + field)
            passed = passed and risks["K1_matched"] <= threshold * risks[denominator]
        if row["gate_pass"] is not bool(passed):
            errors.append("Saved native gate mismatch: " + label)
        gates.append(bool(passed))
    if len(gates) != 4 or any(gates) or summary["native_gate_pass"] is not False:
        errors.append("R5 finite-rejection evidence changed; reconcile before selecting a route")
    if summary["status"] != "COMPLETED_R4E1_FINITE_NATIVE_SUPPORT_GATE_REJECTED":
        errors.append("Saved R4E1 status changed")
    if execution["R4E1_status"] != summary["status"]:
        errors.append("Execution record and evidence summary disagree")
    for stage, method_id in decision["blocked_stages"].items():
        if (execution[stage + "_status"] != "NOT_RUN_R4E1_NATIVE_GATE_REJECTED"
                or method_id not in methods
                or methods[method_id]["status"] != "NOT_TRIGGERED_PREREQUISITE_REJECTED"
                or methods[method_id].get("model_execution_eligibility") == READY):
            errors.append("Failed prerequisite bypassed: " + stage)
        if stage not in index or index[stage]["id"] != method_id:
            errors.append("Missing blocked-stage alias: " + stage)
    if set(decision["blocked_stages"]) != {"R4E2", "R4E3"}:
        errors.append("R4 dependent stages removed from guard")
    return errors


def method_eligibility(decision, triage, method_id):
    index, errors = build_index(triage)
    if errors:
        return False, "INVALID_REGISTRY"
    method = index.get(method_id)
    if method is None:
        return False, "UNKNOWN_UNREVIEWED_METHOD"
    allowed = (method["id"] in decision["eligible_model_method_ids"]
               and method.get("model_execution_eligibility") == READY)
    return allowed, method["status"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--method-id", help="Canonical ID or alias to check before selecting a model method")
    args = parser.parse_args(argv)
    try:
        decision, triage, summary, execution = load_records(args.root)
        errors = validate_records(decision, triage, summary, execution)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "INVALID_RECORDS", "error": str(exc)}, ensure_ascii=True))
        return 1
    if errors:
        print(json.dumps({"status": "INCONSISTENT", "errors": errors}, ensure_ascii=True))
        return 1
    result = {"status": "CONSISTENT", "eligible_model_method_ids": decision["eligible_model_method_ids"],
              "scope": "Saved-record consistency only; no model execution or semantic novelty certification"}
    if args.method_id:
        allowed, reason = method_eligibility(decision, triage, args.method_id)
        result.update(method_id=args.method_id, eligible=allowed, disposition=reason)
        print(json.dumps(result, ensure_ascii=True))
        return 0 if allowed else 2
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
