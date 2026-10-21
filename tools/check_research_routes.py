#!/usr/bin/env python3
"""Check recorded research decisions; never launch models or infer novelty.

Exit 0: consistent registry (and requested method eligible, if supplied).
Exit 1: inconsistent/malformed records. Exit 2: method not eligible or unknown.
The R5 evidence audit is deliberately version-scoped. Future decisions must
preserve this closed recipe and explicitly migrate their evidence checks.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


READY = "ELIGIBLE_REGISTERED_SCREEN"
CURRENT_STATE = "REGISTERED_R4_SEQUENCE_CLOSED_NO_ELIGIBLE_FOLLOWUP"
DIAGNOSTIC_STATE = "R4_CLOSED_FIXED_TRANSPORT_DIAGNOSTIC_REGISTERED"
DIAGNOSTIC_ID = "R7D1_CROSS_LAYER_TRANSPORT"
GAIN_STATE = "R4_CLOSED_FIXED_GAIN_OBJECTIVE_REGISTERED"
GAIN_ID = "R7D2_RESIDUAL_GAIN_OBJECTIVE"
FOLD_STATE = "R4_CLOSED_FULL_SUPPORT_FOLD_CONTROL_REGISTERED"
FOLD_ID = "R8F1_FULL_SUPPORT_NATIVE_FOLD"
ENCODING_STATE = "R4_CLOSED_GROUP64_HESSIAN_FOLD_REGISTERED"
ENCODING_ID = "R8F2_GROUP64_HESSIAN_AFFINE"
POPULATION_STATE = "R4_CLOSED_UNCHANGED_POPULATION_COLLECTION_REGISTERED"
POPULATION_ID = "R8D1_FULL_POPULATION_Q1_COVARIANCE"
LEARNED_STATE = "R4_CLOSED_ORIGINAL_LEARNED_AFFINE_OPERATOR_REGISTERED"
LEARNED_ID = "R9A1_LEARNED_AFFINE_NATIVE_REMAINDER"
HEAD_STATE = "R4_CLOSED_NATIVE_HEAD_GRADIENT_READINESS_REGISTERED"
HEAD_ID = "R11G1_NATIVE_HEAD_GRADIENT_READINESS"
MATCHED_STATE = "R4_CLOSED_MATCHED_NATIVE_HEAD_OBJECTIVE_REGISTERED"
MATCHED_ID = "R11T1_MATCHED_NATIVE_HEAD_OBJECTIVE"
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


def validate_records(decision, triage, summary, execution, root=None):
    index, errors = build_index(triage)
    methods = {m["id"]: m for m in triage["methods"]}
    state = decision["execution_state"]
    if state not in (CURRENT_STATE, DIAGNOSTIC_STATE, GAIN_STATE, FOLD_STATE, ENCODING_STATE, POPULATION_STATE, LEARNED_STATE, HEAD_STATE, MATCHED_STATE):
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
    if canonical and state == CURRENT_STATE:
        errors.append("R5 closure records no eligible model follow-up; update evidence before changing the decision")
    if state in (DIAGNOSTIC_STATE, GAIN_STATE, FOLD_STATE, ENCODING_STATE, POPULATION_STATE, LEARNED_STATE, HEAD_STATE, MATCHED_STATE):
        selected = {DIAGNOSTIC_STATE: DIAGNOSTIC_ID, GAIN_STATE: GAIN_ID, FOLD_STATE: FOLD_ID,
                    ENCODING_STATE: ENCODING_ID, POPULATION_STATE: POPULATION_ID, LEARNED_STATE: LEARNED_ID,
                    HEAD_STATE: HEAD_ID, MATCHED_STATE: MATCHED_ID}[state]
        if canonical != [selected]:
            errors.append("Only the fixed causal diagnostic is registered by this migration")
        registration = decision.get('registered_diagnostic', {})
        if (registration.get('method_id') != selected
                or registration.get('scope') != 'FIXED_CAUSAL_DIAGNOSTIC_NOT_CANDIDATE'
                or registration.get('preserved_model_closure') != CURRENT_STATE):
            errors.append('Missing scoped diagnostic migration; R4 closure must remain preserved')
        try:
            base = root or Path(__file__).resolve().parents[1]
            path = base / registration['protocol']
            if hashlib.sha256(path.read_bytes()).hexdigest() != registration['protocol_sha256']:
                errors.append('Registered diagnostic protocol hash changed')
            protocol = json.loads(path.read_text())
            expected_positions = ({'step': 3, 'A_block': 30, 'B_block': 31})
            expected_indices = [0, 5, 10, 15]
            if selected == HEAD_ID:
                expected_positions, expected_indices = [[3, 39]], [0]
            elif selected == MATCHED_ID:
                expected_positions = [[3, 39]]
            elif selected == LEARNED_ID:
                expected_positions, expected_indices = [[s, 20] for s in range(4)], [0, 6, 12, 18]
            elif selected in (FOLD_ID, ENCODING_ID, POPULATION_ID):
                expected_positions = [[2, 20], [3, 20]]
                expected_indices = sorted(set(range(1, 24, 2)) | {0, 6, 12, 18}) if selected == POPULATION_ID else [0, 6, 12, 18]
            common = (protocol['id'] == selected
                    and protocol['positions'] == expected_positions
                    and protocol['inputs']['indices'] == expected_indices
                    and protocol['execution_plan']['full_native_steps_per_run'] == (0 if selected in (FOLD_ID, ENCODING_ID, LEARNED_ID) else 4)
                    and protocol['execution_plan']['interventions_require_all_controls'] is True)
            if selected == DIAGNOSTIC_ID:
                specific = (protocol['status'] == 'REGISTERED_CONTROLS_THEN_FIXED_ARMS'
                    and set(protocol['arms']) == {'none', 'A', 'B', 'AB'}
                    and protocol['execution_plan']['required_control_runs'] == 12
                    and protocol['execution_plan']['intervention_runs'] == 12)
                required_sources = {'tools/run_cross_layer_transport.py', 'tools/cross_layer_transport.py'}
            elif selected == GAIN_ID:
                specific = (protocol['status'] == 'REGISTERED_IDENTITY_DIRECTIONS_FITS_NATIVE_COMPARISON'
                    and protocol['inputs']['training_indices'] == [0, 5, 10]
                    and protocol['inputs']['development_holdout_index'] == 15
                    and protocol['execution_plan']['required_control_runs'] == 4
                    and protocol['execution_plan']['direction_runs'] == 6
                    and protocol['execution_plan']['native_comparison_runs'] == 8
                    and protocol['parameters']['gain_bounds'] == [.75, 1.25]
                    and protocol['parameters']['finite_direction_epsilon'] == .125)
                required_sources = {'tools/run_residual_gain_transport.py', 'tools/residual_gain_transport.py'}
                prior = registration['prior_diagnostic']
                evidence_path = base / prior['result']
                evidence = json.loads(evidence_path.read_text())
                if (hashlib.sha256(evidence_path.read_bytes()).hexdigest() != prior['sha256']
                        or evidence['status'] != 'PASS_COMPLETE_NATIVE_FIXED_PAIR_DIAGNOSTIC'
                        or evidence['completed_generations'] != 24
                        or evidence['rank_disagreement_source_count'] < 1
                        or evidence['repeated_none_bitwise_equal'] is not True):
                    errors.append('Gain-objective diagnostic lacks its native prerequisite evidence')
            elif selected == FOLD_ID:
                specific = (protocol['status'] == 'REGISTERED_FULL_SUPPORT_PRECISION_CONTROL'
                    and protocol['execution_plan']['original_UB1_native_controls'] == 48
                    and protocol['execution_plan']['full_teacher_identity_controls'] == 8
                    and protocol['execution_plan']['compression_or_training_runs'] == 0
                    and protocol['gates'] == {'BF16_error_over_UB1_max': .01, 'HiF4_error_over_UB1_max': .1}
                    and len(protocol['arms']) == 3)
                required_sources = {'tools/probe_native_affine_fold.py', 'tools/native_affine_fold.py'}
                prior = registration['prior_diagnostic']
                previous_path = base / prior['result']
                previous = json.loads(previous_path.read_text())
                if (hashlib.sha256(previous_path.read_bytes()).hexdigest() != prior['sha256']
                        or previous['status'] != 'PASS_COMPLETE_FIXED_OBJECTIVE_COMPARISON'
                        or previous['completed_generations'] != 18
                        or previous['fits_reproduced_exactly'] is not True):
                    errors.append('Folding review lacks its completed native prerequisite evidence')
            elif selected == ENCODING_ID:
                specific = (protocol['status'] == 'REGISTERED_FIXED_GROUP64_HESSIAN_ENCODING'
                    and protocol['inputs']['training_indices'] == list(range(1, 24, 2))
                    and protocol['encoding']['group_size'] == 64
                    and protocol['encoding']['damping_fraction'] == .01
                    and protocol['encoding']['activation_order'] is False
                    and protocol['execution_plan']['encoded_matrices'] == 2
                    and protocol['execution_plan']['compression_runs'] == 0
                    and protocol['gates']['HiF4_error_over_UB1_max'] == .1)
                required_sources = {'tools/probe_hessian_affine_fold.py', 'tools/group_hessian_quant.py'}
                prior = registration['prior_diagnostic']
                previous_path = base / prior['result']
                previous = json.loads(previous_path.read_text())
                if (hashlib.sha256(previous_path.read_bytes()).hexdigest() != prior['sha256']
                        or previous['status'] != 'COMPLETED_FULL_SUPPORT_NATIVE_FOLD_CHECK'
                        or previous['BF16_numerical_gate_all8'] is not True
                        or previous['HiF4_precision_gate_all8'] is not False
                        or len(previous['rows']) != 8):
                    errors.append('Encoding repair lacks its measured native prerequisite evidence')
            elif selected == POPULATION_ID:
                specific = (protocol['status'] == 'REGISTERED_UNCHANGED_NATIVE_POPULATION_COLLECTION'
                    and protocol['inputs']['training_indices'] == list(range(1, 24, 2))
                    and protocol['inputs']['validation_indices'] == [0, 6, 12, 18]
                    and protocol['execution_plan']['pipeline_generations'] == 16
                    and protocol['execution_plan']['new_encodings_or_candidate_fits'] == 0
                    and protocol['execution_plan']['max_GPUs'] == 2)
                required_sources = {'tools/collect_q1_population.py'}
                prior = registration['prior_diagnostic']
                previous_path = base / prior['result']; previous = json.loads(previous_path.read_text())
                data = registration['data_audit']
                audit_path = base / data['result']; audit = json.loads(audit_path.read_text())
                if (hashlib.sha256(previous_path.read_bytes()).hexdigest() != prior['sha256']
                        or previous['status'] != 'COMPLETED_FIXED_GROUP64_HESSIAN_NATIVE_CHECK'
                        or previous['native_precision_gate_all8'] is not False
                        or len(previous['rows']) != 8):
                    errors.append('Population diagnosis lacks its completed native prerequisite evidence')
                if (hashlib.sha256(audit_path.read_bytes()).hexdigest() != data['sha256']
                        or audit['status'] != 'PASS_CPU_SAVED_HESSIAN_TRANSFER_AUDIT'
                        or len(audit['positions']) != 2
                        or any(p['Hessian_over_RTN_affine_SSE']['train'] >= .3
                               or p['Hessian_over_RTN_affine_SSE']['validation'] <= .9 for p in audit['positions'])):
                    errors.append('Population diagnosis lacks its frozen transfer data audit')
            elif selected == LEARNED_ID:
                specific = (protocol['status'] == 'REGISTERED_LEARNED_AFFINE_FULL_SUPPORT_THEN_FIXED_COMPRESSION'
                    and protocol['inputs']['training_indices'] == list(range(1, 24, 2))
                    and protocol['inputs']['validation_indices'] == [0, 6, 12, 18]
                    and protocol['groups_by_step'] == [14, 68, 68, 122]
                    and protocol['R256_active'] is False
                    and protocol['precision']['derived_matrix'] == 'BF16'
                    and protocol['precision']['extra_remainder_QDQ'] is False
                    and protocol['gates'] == {'full_support_error_over_UB1_max': .01, 'compressed_error_over_UB1_max': .98})
                required_sources = {'tools/screen_learned_affine_remainder.py', 'tools/affine_remainder.py'}
                prior = registration['prior_diagnostic']
                previous_path = base / prior['result']; previous = json.loads(previous_path.read_text())
                if (hashlib.sha256(previous_path.read_bytes()).hexdigest() != prior['sha256']
                        or previous['status'] != 'COMPLETED_FULL_SUPPORT_NATIVE_FOLD_CHECK'
                        or previous['BF16_numerical_gate_all8'] is not True):
                    errors.append('Learned affine operator lacks its native prerequisite evidence')
                reconciliation = registration['scope_reconciliation']
                design_path = base / reconciliation['original_R2_document']
                if hashlib.sha256(design_path.read_bytes()).hexdigest() != reconciliation['original_R2_sha256']:
                    errors.append('Original R2 design/scope reconciliation changed')
            elif selected == HEAD_ID:
                specific = (protocol['status'] == 'REGISTERED_ONE_NATIVE_CAPTURE_THEN_ONE_GRADIENT_STEP'
                    and protocol['execution_plan']['pipeline_generations'] == 1
                    and protocol['execution_plan']['gradient_updates'] == 1
                    and protocol['execution_plan']['max_GPUs'] == 2
                    and protocol['update']['relative_global_L2_step'] == .001
                    and protocol['trainable_parameters'] == 'Existing step3/block39 R256 first/bias_first/second/bias_second only'
                    and protocol['additional_inference_parameters'] == 0
                    and protocol['candidate_training_or_quality_acceptance'] is False)
                required_sources = {'tools/probe_native_head_gradient.py', 'tools/native_head_replay.py'}
                prior = registration['prior_diagnostic']
                previous_path = base / prior['result']; previous = json.loads(previous_path.read_text())
                if (hashlib.sha256(previous_path.read_bytes()).hexdigest() != prior['sha256']
                        or previous['status'] != 'PASS_COMPLETE_FIXED_OBJECTIVE_COMPARISON'
                        or previous['completed_generations'] != 18):
                    errors.append('Native-gradient readiness lacks its completed fixed-secant prerequisite')
            else:
                specific = (protocol['status'] == 'REGISTERED_FIXED20_MATCHED_LOCAL_NATIVE_OBJECTIVES'
                    and protocol['inputs']['training_indices'] == [0, 5, 10]
                    and protocol['inputs']['development_index'] == 15
                    and protocol['execution_plan']['capture_generations'] == 4
                    and protocol['execution_plan']['deployment_validation_generations'] == 2
                    and protocol['optimization']['steps_per_arm'] == 20
                    and protocol['optimization']['relative_global_L2_step'] == .001
                    and protocol['gates'] == {'development_flow_over_UB1_max': .98, 'development_flow_over_local_max': .99}
                    and protocol['additional_inference_parameters'] == 0
                    and protocol['fresh_quality_acceptance'] is False)
                required_sources = {'tools/run_native_window_objective.py', 'tools/native_head_replay.py'}
                prior = registration['prior_diagnostic']
                previous_path = base / prior['result']; previous = json.loads(previous_path.read_text())
                if (hashlib.sha256(previous_path.read_bytes()).hexdigest() != prior['sha256']
                        or previous['status'] != 'COMPLETED_FIXED_NATIVE_HEAD_GRADIENT_READINESS'
                        or previous['strict_byte_audit']['status'] != 'PASS_COMPLETE_NATIVE_INTEGER_BYTES'
                        or previous['gradient_readiness']['additional_inference_parameters'] != 0
                        or previous['gradient_readiness']['peak_allocated_bytes'] > 8 << 30
                        or not all(previous['gradient_readiness']['controls'].values())):
                    errors.append('Matched objectives lack exact native gradient/memory readiness')
            if not common or not specific:
                errors.append('Fixed diagnostic controls/positions/budget changed')
            bindings = registration['source_sha256']
            if not required_sources.issubset(bindings):
                errors.append('Missing native intervention implementation bindings')
            for relative, expected in bindings.items():
                file = base / relative
                if (not file.resolve().is_relative_to(base.resolve())
                        or hashlib.sha256(file.read_bytes()).hexdigest() != expected):
                    errors.append('Registered diagnostic source changed: ' + relative)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append('Invalid diagnostic registration: ' + str(exc))
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
        errors = validate_records(decision, triage, summary, execution, args.root)
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
