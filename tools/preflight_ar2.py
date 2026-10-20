"""Check the frozen AR2 scopes and external asset paths without importing torch.

This is a metadata/existence check. Presence is never reported as numerical
replay success. Path relocation is explicit; historical source paths remain in
the report. No missing D-Q asset is substituted with a BF16/new-cache operand.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / ".runtime/archive/server/work/v5/analysis_r1"


def audit_sampling(scope):
    rows, weights, coords, grid = (scope[name] for name in ["selected_rows", "design_weights", "coordinates", "grid"])
    if isinstance(weights, str):
        weights = json.loads(weights)
    if len(grid) != 3 or any(int(size) != size or size <= 0 for size in grid):
        raise ValueError("Invalid saved latent grid")
    if rows != sorted(set(rows)) or not rows or rows[-1] >= math.prod(grid) or rows[0] < 0:
        raise ValueError("Invalid sampled flat rows")
    expected = [[row // (grid[1] * grid[2]), (row % (grid[1] * grid[2])) // grid[2], row % grid[2]] for row in rows]
    if len(weights) != len(rows) or any(not math.isfinite(weight) or weight <= 0 for weight in weights):
        raise ValueError("Actual rows and explicit design weights disagree")
    if coords is not None and coords != expected:
        raise ValueError("Saved coordinates disagree with the saved flat rows/grid")
    return {"rows": len(rows), "grid": grid,
            "coordinate_provenance": scope["coordinate_provenance"],
            "coordinate_index_view": "COPIED_IN_SCOPE_INDEX" if coords is not None else
                "DERIVED_FROM_SAVED_FLAT_ROWS_AND_GRID_NO_RESAMPLING_NOT_A_NEW_RUNTIME_OBSERVATION"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scope-index", type=Path, default=REFERENCE / "scope_index.jsonl")
    ap.add_argument("--dq-index", type=Path, default=REFERENCE / "asset_recovery_index_R1.json")
    ap.add_argument("--map-path", action="append", default=[], metavar="OLD=NEW")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Output exists; choose a new report")
    maps = []
    for value in args.map_path:
        old, sep, new = value.partition("=")
        if not sep or not Path(old).is_absolute() or not Path(new).is_absolute():
            ap.error("Path mappings must be absolute OLD=NEW pairs")
        maps.append((old.rstrip("/"), new.rstrip("/")))
    maps.sort(key=lambda item: len(item[0]), reverse=True)

    def relocate(value):
        for old, new in maps:
            if value == old or value.startswith(old + "/"):
                return new + value[len(old):]
        return value

    data = args.scope_index.read_bytes()
    index = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    recovery_data = args.dq_index.read_bytes()
    recovery = json.loads(recovery_data)
    errors, scopes, required = [], [], {}

    def require(role, original):
        if not original:
            errors.append(f"Missing asset locator for {role}")
            return
        target = Path(relocate(original))
        entry = required.setdefault(original, {"roles": [], "historical_path": original, "path": str(target),
                                                "is_file": target.is_file()})
        entry["roles"].append(role)
        if entry["is_file"]:
            entry["bytes"] = target.stat().st_size

    for task, source, split in [(0, "validation_000_multihuman_5", "fit"),
                                (6, "validation_003_singleobj_29", "validation")]:
        for seed in [20261001, 20261002]:
            for step, expert in [(0, "high_noise_model"), (2, "low_noise_model")]:
                hits = [s for s in index if s.get("arm") == "D-Q" and int(s["task_id"]) == task
                        and int(s["seed"]) == seed and int(s["step"]) == step and int(s["block"]) == 20
                        and s["expert"] == expert and s["source"] == source and s["split"] == split]
                if len(hits) != 1:
                    errors.append(f"Expected one frozen scope task{task}/seed{seed}/step{step}; got {len(hits)}")
                    continue
                scope = hits[0]
                try:
                    sampling = audit_sampling(scope)
                except ValueError as error:
                    errors.append(f"{scope['scope_key']}: {error}")
                    sampling = {"rows": len(scope["selected_rows"]), "grid": scope["grid"], "sampling_status": "INVALID"}
                scopes.append({"scope_key": scope["scope_key"], "task": task, "seed": seed, "step": step,
                               "expert": expert, "split": split, **sampling})
                require(scope["scope_key"] + ":tensor", scope["tensor_file"])
    weights = recovery["dependencies"]["dq_quantized"]["weight_assets"]
    for expert in ["high_noise_model", "low_noise_model"]:
        hits = [w for w in weights if w["expert"] == expert and int(w["block_index"]) == 20
                and w["module_name"].endswith("ffn.2")]
        if len(hits) != 1:
            errors.append(f"Expected one effective W2 metadata record: {expert}/block20")
            continue
        row = hits[0]
        require(expert + ":W2_cache", row["cache_path"])
        require(expert + ":cache_metadata", row["cache_path"] + ".json")
        require(expert + ":cache_complete_marker", row["cache_path"] + ".COMPLETE")
        require(expert + ":b2", row.get("bias_locator"))
    missing = [item for item in required.values() if not item["is_file"]]
    status = "INVALID_SCOPE_METADATA" if errors else ("MISSING_EXTERNAL_ASSETS" if missing else "ASSETS_PRESENT_UNVERIFIED_CONTENT")
    result = {"status": status, "scope": "frozen AR2 metadata and path existence only; no numerical replay",
              "scope_count": len(scopes), "independent_sources": len({s["task"] for s in scopes}),
              "saved_rows": sum(s["rows"] for s in scopes), "scopes": scopes,
              "scope_index_sha256": hashlib.sha256(data).hexdigest(),
              "dq_index_sha256": hashlib.sha256(recovery_data).hexdigest(),
              "path_maps": maps, "required_assets": list(required.values()),
              "missing_asset_count": len(missing), "errors": errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["status", "scope_count", "saved_rows", "missing_asset_count", "errors"]}, indent=2))
    return 1 if errors else (2 if missing else 0)


if __name__ == "__main__":
    raise SystemExit(main())
