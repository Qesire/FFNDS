"""Describe hindsight output mosaics; they are not executed FFN routing actions."""
import argparse
import json
import math
import os
from pathlib import Path
import time

from audit_r12_saved_operands import ROOT, FLOW_SHAPE, read, sha, verified_safetensors


def summarize_partitions(rows):
    totals = {a: math.fsum(r["whole_loss"][a] for r in rows) for a in ("A", "B")}
    fixed = min(totals, key=totals.get)
    source = math.fsum(min(r["whole_loss"].values()) for r in rows)
    mosaics = {"source": source}
    for key in ("frame", "native_token"):
        values = []
        for row in rows:
            losses = row[key + "_losses"]
            if len(losses["A"]) != len(losses["B"]):
                raise ValueError("Unmatched output partitions")
            for action in ("A", "B"):
                if any(not math.isfinite(x) or x < 0 for x in losses[action]):
                    raise ValueError("Invalid component loss")
                if not math.isclose(math.fsum(losses[action]), row["whole_loss"][action], rel_tol=1e-11, abs_tol=1e-8):
                    raise ValueError("Partition loss does not sum to original whole loss")
            values.append(math.fsum(min(a, b) for a, b in zip(losses["A"], losses["B"])))
        mosaics[key] = math.fsum(values)
    if not (totals[fixed] + 1e-8 >= source and source + 1e-8 >= mosaics["frame"]
            and mosaics["frame"] + 1e-8 >= mosaics["native_token"]):
        raise ValueError("Inconsistent nested output partitions")
    return {"best_fixed_historical_restore": fixed, "best_fixed_total_loss": totals[fixed],
            "historical_restore_total_loss": totals, "post_action_mosaic_total_losses": mosaics,
            "mosaic_reduction_over_best_fixed_fraction": {k: 1 - v / totals[fixed] for k, v in mosaics.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from worker_identity import require_worker
    require_worker()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.out.exists():
        raise ValueError("Requires CPU wrapper and a new output file")
    import numpy as np

    started = time.monotonic()
    audit = read(args.audit)
    if audit["status"] != "PASS_CPU_SAVED_OPERAND_AUDIT_NO_MODEL_RUN":
        raise ValueError("Complete original operand audit required")
    protocol = ROOT / "research/20261022/R12_E0_saved_flow_geometry_protocol.json"
    tf2 = read(ROOT / "reports/worker_20261020/TF2_DQ_input00.json")
    p = Path(tf2["steps"][3]["path"])
    hf = Path(*p.parts[:p.parts.index("ffnds-derived") + 1])
    original = read(ROOT / "research/20261021/R7D1_results.json")
    raw_rows, details = [], []
    for row in original["rows"]:
        flows = {}
        for action in ("DQ", "A", "B"):
            binding = row["payloads"][action]
            path = hf / binding["HF_derived_relative"]
            expected = {k: {"shape": FLOW_SHAPE, "dtype": "F32", "sha256": binding[k + "_sha256"]} for k in ("flow", "post")}
            _, header, start = verified_safetensors(path, binding["sha256"], expected)
            if not any(o.get("family") == "R7D1" and o.get("sha256") == binding["sha256"] for o in audit["observations"]):
                raise ValueError("Flow not present in original successful audit")
            flows[action] = np.memmap(path, dtype="<f4", mode="r", offset=start + header["flow"]["data_offsets"][0],
                                      shape=tuple(FLOW_SHAPE)).astype(np.float64)
        losses = {a: np.square(flows[a] - flows["DQ"]) for a in ("A", "B")}
        if any(not np.isfinite(v).all() for v in losses.values()):
            raise ValueError("Nonfinite full-flow error")
        whole = {a: float(v.sum()) for a, v in losses.items()}
        if any(not math.isclose(v, row["flow_SSE"][a], rel_tol=1e-12, abs_tol=1e-8) for a, v in whole.items()):
            raise ValueError("Whole-flow loss differs from original R7 record")
        frame = {a: v.sum(axis=(0, 2, 3)).tolist() for a, v in losses.items()}
        # Original Wan output patch is (1,2,2); each token has 16*1*2*2=64 outputs.
        patch = {a: v.reshape(16, 21, 29, 2, 52, 2).sum(axis=(0, 3, 5)).reshape(-1).tolist() for a, v in losses.items()}
        item = {"index": row["index"], "whole_loss": whole, "frame_losses": frame, "native_token_losses": patch}
        per_source = summarize_partitions([item])
        per_source.update(index=row["index"], frame_loss_A_minus_B=[a - b for a, b in zip(frame["A"], frame["B"])],
                          frames_preferring_A=sum(a < b for a, b in zip(frame["A"], frame["B"])),
                          native_tokens_preferring_A=sum(a < b for a, b in zip(patch["A"], patch["B"])),
                          native_tokens_tied=sum(a == b for a, b in zip(patch["A"], patch["B"])))
        raw_rows.append(item)
        details.append(per_source)
    result = {"id": "R12_E0_SAVED_FLOW_PARTITIONS", "status": "PASS_CPU_POST_ACTION_OUTPUT_MOSAIC_DIAGNOSTIC",
              "protocol_sha256": sha(protocol), "source_sha256": sha(Path(__file__)), "operand_audit_sha256": sha(args.audit),
              "evidence_level": "HISTORICAL_FLOW_GEOMETRY_NOT_EXECUTED_DYNAMIC_ACTIONS",
              "new_model_jobs": 0, "new_raw_tensor_storage_bytes": 0,
              "native_output_grid": [21, 29, 52], "native_output_values_per_token": 64,
              **summarize_partitions(raw_rows), "per_source": details,
              "interpretation_boundary": read(protocol)["interpretation_boundary"],
              "elapsed_seconds": time.monotonic() - started}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "mosaic_reduction_over_best_fixed_fraction", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
