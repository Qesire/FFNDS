"""Check saved R12 E0 consistency; does not certify unobserved native actions."""
import hashlib
import json
import math
from pathlib import Path

from audit_r12_saved_operands import historical_headroom

ROOT = Path(__file__).resolve().parents[1]
BASE = "research/20261022/"


def read(path):
    return json.loads((ROOT / path).read_text())


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def validate(audit, geometry, draft, runtime, current, triage):
    errors = []

    def need(value, message):
        if not value:
            errors.append(message)

    need(audit["status"] == "PASS_CPU_SAVED_OPERAND_AUDIT_NO_MODEL_RUN", "wrong operand audit status")
    expected = {"TF2_full_step_payloads": 32, "OP2_sampled_pair_payloads": 48, "R7_full_flow_payloads": 24,
                "R11_wrong_position_full_payloads": 4, "DQ_effective_pair_weight_shards": 2,
                "R12_counterfactual_action_flows": 0}
    need(audit["counts"] == expected, "scoped counts or unobserved R12 actions changed")
    need(audit["new_native_generations"] == 0 and audit["CUDA_initialized"] is False, "E0 must stay CPU/saved-only")
    groups = {}
    for o in audit["observations"]:
        groups.setdefault(o["family"], []).append(o)
        need(not o["locator"].startswith("/"), "public locator must be relative/neutral")
    need({k: len(v) for k, v in groups.items()} == {"TF2": 32, "OP2": 48, "R7D1": 24, "R11T1": 4, "ORIGINAL_DQ_WEIGHTS": 2},
         "observation coverage differs")
    for b in (30, 31):
        sampled = [o for o in groups.get("OP2", []) if o["block"] == b]
        need({o["index"] for o in sampled} == set(range(24)), "incomplete OP2 source coverage")
        need(all(o["saved_rows"] == 256 and o["full_rows_seen_by_producer"] == 31668 for o in sampled), "sampled rows relabeled fullM")
        need(all(o["tensors"]["V1"]["shape"] == [256, 5120] for o in sampled), "sampled V1 geometry changed")
        need(len({o["tensors"]["gate"]["sha256"] for o in sampled}) == 1, "saved gate-constancy claim changed")
        need(len({o["tensors"]["V1"]["sha256"] for o in sampled}) == 24, "saved V1 distinctness claim changed")
    for o in groups.get("ORIGINAL_DQ_WEIGHTS", []):
        old_rows = read("research/20261021/R7D1_results.json")["rows"]
        expected_weight_shas = {s["teacher_shard_sha256"] for r in old_rows for s in r["scopes"]["none"]
                                if s["block"] == o["block"]}
        need(expected_weight_shas == {o["sha256"]}, "audited weight shard differs from the original native teacher")
        variants = o["CPU_pack_variants"]
        for count in (108, 162, 216):
            v = variants[str(count)]
            need(len(v["groups"]) == count and len(set(v["groups"])) == count, "invalid support cardinality")
            need(v["groups"] == sorted(v["groups"]) and set(v["groups"]) <= set(range(216)), "invalid support order/range")
            need(v["packed_effective_records"]["0"]["weight_shape"] == [64 * count, 5120], "wrong W1 slice")
            need(v["packed_effective_records"]["2"]["weight_shape"] == [5120, 64 * count], "wrong W2 slice")
        need(set(variants["108"]["groups"]) <= set(variants["162"]["groups"]), "support nesting claim changed")
        need(o["pack_status"] == "CPU_EFFECTIVE_WEIGHT_SLICES_ONLY_NO_ACTIVATION_QDQ_OR_NATIVE_FORWARD", "CPU pack relabeled native")
    historical = historical_headroom(read("research/20261021/R7D1_results.json")["rows"])
    need(audit["historical_restore_analysis"]["restore_oracle_gain_over_best_fixed_fraction"] == historical["restore_oracle_gain_over_best_fixed_fraction"],
         "historical restore headroom does not reproduce")
    need(geometry["evidence_level"] == "HISTORICAL_FLOW_GEOMETRY_NOT_EXECUTED_DYNAMIC_ACTIONS", "output stitching relabeled dynamic execution")
    need(geometry["new_model_jobs"] == 0, "geometry analysis cannot add model jobs")
    for key in ("source", "frame", "native_token"):
        loss = math.fsum(r["post_action_mosaic_total_losses"][key] for r in geometry["per_source"])
        need(math.isclose(loss, geometry["post_action_mosaic_total_losses"][key], rel_tol=1e-12), "partition pooled loss inconsistent")
        need(math.isclose(1 - loss / geometry["best_fixed_total_loss"], geometry["mosaic_reduction_over_best_fixed_fraction"][key], rel_tol=1e-11),
             "partition reduction denominator changed")
    need(geometry["post_action_mosaic_total_losses"]["source"] == historical["restore_oracle_total_loss"], "whole-source oracle changed")
    for row in runtime["observed_TF2"]:
        for coordinate, timestep in zip(runtime["configured_denoising_step_list"], row["model_timesteps"]):
            predicted = 1000 * 5 * (coordinate / 1000) / (1 + 4 * coordinate / 1000)
            need(math.isclose(predicted, timestep, rel_tol=1e-7), "configured/actual native time distinction inconsistent")
    r7_rows = read("research/20261021/R7D1_results.json")["rows"]
    full_input_counts = {str(b): len({s["norm_input_sha256"] for r in r7_rows for s in r["scopes"]["none"] if s["block"] == b})
                         for b in (30, 31)}
    need(runtime["pre_W1_full_input_metadata"]["distinct_full_norm_input_hashes_by_block"] == full_input_counts,
         "full input diversity must use full input hashes, not unpaired sampled rows")
    need(runtime["native_change"] is False and runtime["new_model_jobs"] == 0, "reconstruction changed native execution")
    need(draft["status"] == "DRAFT_ONLY_NOT_REGISTERED_NOT_ELIGIBLE", "draft status changed")
    need(draft["execution_cost_cap_proposed"]["full_four_step_generations"] == 24, "draft run budget changed")
    for a in ("N0_162_162_raw", "NA_216_108_raw", "NB_108_216_raw"):
        need(sum(draft["action_family_D0"][a]) == 324, "draft unequal pair mainMAC")
    costs = draft["MAC_ledger"]
    need(costs["pair_main_MAC_per_video"] == 2 * 31668 * 5120 * 64 * 324, "pair MAC arithmetic changed")
    need(costs["all_dense_main_MAC_per_video"] == 2 * 31668 * 5120 * 13824 * 160, "dense MAC arithmetic changed")
    need(costs["all_UB1_main_MAC_per_video"] * 2 == costs["all_dense_main_MAC_per_video"], "global mainMAC changed")
    # E0 is immutable historical evidence. A separately authorized later native
    # registration cannot be inferred from E0 or its original draft.
    if current['eligible_model_method_ids']:
        allowed = False
        try:
            registration = current['registered_diagnostic']
            protocol = read(registration['protocol'])
            authorization = read(protocol['authorization']['path'])
            allowed = (draft['status'] == 'DRAFT_ONLY_NOT_REGISTERED_NOT_ELIGIBLE'
                and current['eligible_model_method_ids'] == ['R12E1_D0_EQUAL_BUDGET_NATIVE_VALUE']
                and current['execution_state'] == 'R4_CLOSED_R12_D0_NATIVE_VALUE_REGISTERED'
                and protocol['status'] == 'REGISTERED_FIXED_R12_D0_CONTROLS_THEN_ACTIONS'
                and digest(registration['protocol']) == registration['protocol_sha256']
                and protocol['operand_audit']['sha256'] == digest(BASE + 'R12_E0_operand_audit.json')
                and digest(protocol['authorization']['path']) == protocol['authorization']['sha256']
                and authorization['status'] == 'EXPLICIT_USER_AUTHORIZATION')
            if current['eligible_model_method_ids'] == ['R14N0_COMPLETE_STRIPE_NATIVE_VALUE']:
                allowed = (draft['status'] == 'DRAFT_ONLY_NOT_REGISTERED_NOT_ELIGIBLE'
                    and current['execution_state'] == 'R4_CLOSED_FIXED_COMPLETE_STRIPE_VALUE_REGISTERED'
                    and protocol['status'] == 'REGISTERED_FIXED_COMPLETE_STRIPE_NATIVE_VALUE'
                    and digest(registration['protocol']) == registration['protocol_sha256']
                    and digest(protocol['authorization']['path']) == protocol['authorization']['sha256']
                    and authorization['status'] == 'EXPLICIT_USER_AUTHORIZATION'
                    and authorization['authorized_method_id'] == 'R14N0_COMPLETE_STRIPE_NATIVE_VALUE')
        except (KeyError, OSError, ValueError, TypeError):
            pass
        if current['eligible_model_method_ids'] == ['R15V0_MEAN_CONTRAST_PAIRED_ROLLOUT']:
            # A later, independently authorized full-video scope does not turn
            # E0 or its unregistered draft into an execution authorization.
            from rollout_contract import STATE, validate_registration
            allowed = (draft['status'] == 'DRAFT_ONLY_NOT_REGISTERED_NOT_ELIGIBLE'
                and current['execution_state'] == STATE
                and not validate_registration(current.get('registered_rollout', {}), ROOT))
        if current['eligible_model_method_ids'] == ['R16V0_FULL160_UB1_DENSE_ROW_DISPATCH']:
            from row_dispatch_contract import STATE, validate_registration
            allowed = (draft['status'] == 'DRAFT_ONLY_NOT_REGISTERED_NOT_ELIGIBLE'
                and current['execution_state'] == STATE
                and not validate_registration(current.get('registered_row_dispatch', {}), ROOT))
        if current['eligible_model_method_ids'] == ['R17E0_FRESH23_NINE_METRICS_AND_SYSTEM_TIMING']:
            from fresh_quality_contract import STATE, validate_registration
            allowed = (draft['status'] == 'DRAFT_ONLY_NOT_REGISTERED_NOT_ELIGIBLE'
                and current['execution_state'] == STATE
                and not validate_registration(current.get('registered_fresh_quality', {}), ROOT))
        need(allowed, 'E0 cannot grant model eligibility')
    need(current["latest_static_review"] == triage["latest_static_review"], "current/triage E0 mismatch")
    need(current['handoff'] == BASE + 'HANDOFF_R12_E0.md'
         or (BASE + 'HANDOFF_R12_E0.md' in current.get('historical_handoffs', [])
             and (ROOT / current['handoff']).is_file()), 'current handoff not updated')
    return errors


def main():
    audit = read(BASE + "R12_E0_operand_audit.json")
    geometry = read(BASE + "R12_E0_flow_partitions.json")
    paths = [BASE + "R12_E0_operand_audit.json", BASE + "R12_E0_flow_partitions.json", BASE + "R12_E1_native_value_DRAFT.json",
             BASE + "R12_E0_runtime_reconstruction.json", "research/CURRENT_DECISION.json", "research/20261021/method_triage.json"]
    errors = validate(*(read(p) for p in paths))
    for path, expected in [("tools/audit_r12_saved_operands.py", audit["audit_source_sha256"]),
                           ("tools/analyze_saved_flow_partitions.py", geometry["source_sha256"]),
                           (BASE + "R12_E0_saved_flow_geometry_protocol.json", geometry["protocol_sha256"]),
                           (BASE + "R12_E0_operand_audit.json", geometry["operand_audit_sha256"])]:
        if digest(path) != expected:
            errors.append("Executed source/protocol/result SHA changed: " + path)
    if errors:
        raise SystemExit("\n".join(errors))
    print("R12 E0 saved records, CPU packs, output geometry and unregistered draft: CONSISTENT")
    print("No new native action value, dynamic policy, quality, speed or execution eligibility certified.")


if __name__ == "__main__":
    main()
