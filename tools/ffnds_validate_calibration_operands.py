"""Validate exact native BF16 operands against a frozen 160-scope calibration row plan.

The index must come from a real observer/capture; this adapter never substitutes
old 256-row operands for fresh draws or treats old filenames as new data.
Import safetensors only for the optional actual tensor-header verification.
"""
import argparse
from collections import Counter
from pathlib import Path
import json

from ffnds_cohort_preflight import audit
from ffnds_experiment_plans import load, save_new


def check(plan, index, *, verify_payload=False):
    if plan.get("phase") != "REGISTERED_CALIBRATION_ROW_PLAN":
        raise ValueError("Expected a real, frozen 160-scope calibration row plan")
    original = audit(plan)
    if original["errors"]:
        raise ValueError("Calibration plan has invalid source/row definitions")
    expected = {(x["source_id"], x["step"], x["block"]): x for x in plan["calibration"]}
    observed, payloads = set(), []
    if len(index) != len(expected):
        raise ValueError("Actual calibrated operand count differs from selected 160-site plan")
    for row in index:
        k = (row["source_id"], row["step"], row["block"])
        if k not in expected or k in observed:
            raise ValueError("Unexpected or duplicate native operand scope: " + str(k))
        observed.add(k)
        ref = expected[k]
        for field in ("selected_row_ids", "n_rows", "population_rows", "source_state", "expert"):
            if row.get(field) != ref.get(field):
                raise ValueError(f"Native {field} disagrees with registered sampled rows at {k}")
        if not row.get("tensor_boundary") == "original_native_Q1_V1_Q2_V2_W2_Y_b2_gate":
            raise ValueError("Operand boundary is incompatible with original HiF4 model")
        path = Path(row.get("operand_path", ""))
        if not path.is_file():
            raise FileNotFoundError("Native operand payload absent: " + str(path))
        if verify_payload:
            from safetensors import safe_open
            shape = {"V1": [ref["n_rows"], 5120],
                     "V2": [ref["n_rows"], 13824],
                     "Y": [ref["n_rows"], 5120],
                     "gate": [1, 5120], "b2": [5120]}
            with safe_open(path, framework="numpy") as tensor:
                if set(tensor.keys()) != set(shape):
                    raise ValueError("Actual native operand fields differ from frozen QDQ/W2 contract")
                for name, dims in shape.items():
                    s = tensor.get_slice(name)
                    if s.get_shape() != dims or s.get_dtype() != "BF16":
                        raise ValueError(f"Actual native operand geometry/dtype differs: {k}/{name}")
        payloads.append({"source_id": row["source_id"], "step": row["step"],
                         "block": row["block"], "operand_path": str(path),
                         "n_rows": ref["n_rows"], "source_state": ref["source_state"]})
    if observed != set(expected):
        raise ValueError("Missing actual native operand positions")
    counts = Counter(p["source_id"] for p in payloads)
    if any(n != 160 for n in counts.values()):
        raise ValueError("Actual source operand coverage must include all160 positions")
    return {"schema": "ffnds.native.operands.checked.v1",
            "status": "PASS_REAL_OPERAND_HEADER_AND_COORDINATE_CHECK" if verify_payload
                      else "PASS_REAL_OPERAND_EXISTENCE_AND_COORDINATE_CHECK",
            "source_count": len(counts), "native_scope_count": len(payloads),
            "rows_per_position": plan["training_rows_per_position"],
            "actual_operands": payloads, "native_tensor_header_verified": verify_payload,
            "fitting_completed": False, "quality_accepted": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibration-plan", type=Path, required=True)
    p.add_argument("--operands-index", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--verify-payload", action="store_true")
    args = p.parse_args()
    result = check(load(args.calibration_plan), load(args.operands_index),
                   verify_payload=args.verify_payload)
    save_new(args.out, result)
    print(json.dumps({"status": result["status"], "scopes": result["native_scope_count"],
                      "sources": result["source_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
