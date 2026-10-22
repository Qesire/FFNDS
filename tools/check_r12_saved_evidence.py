#!/usr/bin/env python3
"""Reproduce metadata-level R12 E0 arithmetic from immutable-looking source files."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT/path).read_text())


def near(a, b):
    assert math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-11), (a, b)


def main():
    saved = read("research/20261022/R12_E0_SAVED_EVIDENCE_AUDIT.json")
    r4 = read("reports/worker_20261020/R4E1_summary.json")
    r11 = read("research/20261021/R11T1_matched_objective_results.json")
    assert saved["status"] == "COMPLETED_RECORD_ONLY"
    assert saved["r4"]["all_registered_gates_failed"] is True
    assert len(saved["r4"]["four_folds"]) == len(r4["CV"]) == 4
    for old, new in zip(r4["CV"], saved["r4"]["four_folds"]):
        assert (new["step"], new["block"], new["fold"]) == (old["step"], old["block"], old["fold"])
        nr = old["native_risks"]
        near(new["k1_matched_over_k0_matched"], nr["K1_matched"]/nr["K0_matched"])
        near(new["k1_matched_over_ub1"], nr["K1_matched"]/nr["K0_frozen"])
        near(new["limited_saved_source_oracle_gain_fraction"], old["descriptive_saved_row_source_oracle"]["relative_improvement_over_best_static"])
        assert new["gate_pass"] == old["gate_pass"] is False
    out = saved["r11"]
    near(out["development_flow_over_UB1"], r11["development_flow_over_UB1"])
    near(out["development_flow_over_local"], r11["development_flow_over_local"])
    near(out["local_arm_own_loss_improvement"], r11["training_descent_audit"]["local"]["own_objective_relative_improvement"])
    near(out["flow_arm_own_loss_improvement"], r11["training_descent_audit"]["flow"]["own_objective_relative_improvement"])
    near(out["flow_arm_local_MSE_multiplier"], r11["flow_fit_local_train_MSE_over_initial_local_MSE"])
    assert out["original_gate_pass"] == r11["development_gate"] is False
    print("R12 E0 saved-evidence arithmetic: OK (not new model data)")


if __name__ == "__main__":
    main()
