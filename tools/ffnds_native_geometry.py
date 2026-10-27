"""Extract the *observed* 4-step × 40-block token populations from native calibration runs.

Consumes original completed calibration collection receipts and per-source case reports.
This is a read-only adapter for historical native capture; it does not infer geometry
from resolution or a future experimental plan.
"""
import argparse
import json
import math
from pathlib import Path
from ffnds_experiment_plans import load, save_new, source_bank


def extract(collections, bank):
    registry = source_bank(bank)
    by_group = {}
    for s in registry:
        if s["role"] != "CALIBRATION":
            continue
        key = s["source_group"]
        if key in by_group:
            raise ValueError("Ambiguous old source group; split archives by dataset/revision")
        by_group[key] = s
    positions, found = [], set()
    for collection in collections:
        root = load(collection)
        if root.get("status") not in ("PASS_CALIBRATION24_SHARD_FULL160_OPERANDS_AND_NOOP",
                                       "PASS_SIX_INPUT_FULL160_OPERANDS_AND_NOOP"):
            raise ValueError("Collection must be a completed native calibration capture")
        for entry in root["completed"]:
            path = Path(entry["input_report"])
            if not path.is_file():
                raise FileNotFoundError("Original per-source native capture unavailable: " + str(path))
            case = load(path)
            if case.get("status") != "PASS_NATIVE_OPERANDS_AND_SAME_PIPELINE_NOOP":
                raise ValueError("Calibration native capture has incomplete status")
            group = case["input"]["source_group"]
            if group not in by_group:
                raise ValueError("Native source lacks a registered CALIBRATION role: " + group)
            source = by_group[group]
            if source["source_id"] in found:
                raise ValueError("Repeated original native source capture")
            found.add(source["source_id"])
            keys = set()
            for row in case["scopes"]:
                step, block = int(row["step"]), int(row["block"])
                if step not in range(4) or block not in range(40) or (step, block) in keys:
                    raise ValueError("Missing/duplicated Wan scope")
                keys.add((step, block))
                grid = row["grid"]
                if not isinstance(grid, list) or not grid or any(not isinstance(n, int) or n < 1 for n in grid):
                    raise ValueError("Actual native grid absent")
                N = math.prod(grid)
                if row["rows"] > N or row["rows"] != len(row["selected_rows"]):
                    raise ValueError("Stored rows differ from actual scope geometry")
                positions.append({"source_id": source["source_id"], "step": step, "block": block,
                                  "population_rows": N})
            if len(keys) != 160:
                raise ValueError("Native capture must contain all160 positions")
    if not found:
        raise ValueError("No native captured source matched cohort")
    return sorted(positions, key=lambda x: (x["source_id"], x["step"], x["block"]))


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--bank", required=True)
    a.add_argument("--collections", nargs="+", required=True)
    a.add_argument("--out", required=True)
    args = a.parse_args()
    rows = extract(args.collections, args.bank)
    save_new(args.out, rows)
    print(json.dumps({"status": "ACTUAL_NATIVE_GEOMETRY_EXTRACTED",
                      "observed_sources": len(rows) // 160, "native_scopes": len(rows)}))


if __name__ == "__main__":
    main()
