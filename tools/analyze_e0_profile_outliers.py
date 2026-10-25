"""Post-hoc leave-one-row audit of the two extreme Q2 coarse-profile changes."""
import argparse
import json
from pathlib import Path

from analyze_ffnds_observations import load_arrays
from analyze_legacy_ffn_structure import profile
from ffnds_observation_common import sha256, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve first outlier audit")
    summary = json.loads(args.summary.read_text())
    def metric(x):
        return profile(x)["row_amplitude_times_static_profile_relative_SSE"]
    def change(s):
        return s["profiles"]["q2_post"]["row_amplitude_times_static_profile_relative_SSE"] - s["profiles"]["q2_pre"]["row_amplitude_times_static_profile_relative_SSE"]
    ordered = sorted(summary["scopes"], key=change)
    result = {"status": "COMPLETE_POSTHOC_EXTREME_PROFILE_LEAVE_ONE_ROW_DESCRIPTION",
              "summary_sha256": sha256(args.summary),
              "selection": "Post-hoc min/max signed Q2 profile-metric changes among all120 scopes; no acceptance gate or method selection",
              "source_sha256": {p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name("analyze_legacy_ffn_structure.py")]},
              "GPU_calls": 0, "model_calls": 0, "cases": []}
    for selected in [ordered[0], ordered[-1]]:
        run_path = args.raw_root / f"source{selected['source_index']:02d}_attempt1/RUN.json"
        run = json.loads(run_path.read_text())
        if run["status"] != "PASS_PASSIVE_CAPTURE_NOOP":
            raise ValueError("Unaccepted source")
        scope = next(s for s in run["scopes"] if (s["step"], s["block"]) == (selected["step"], selected["block"]))
        path = run_path.parent / "observed" / scope["path"]
        if sha256(path) != scope["sha256"]:
            raise ValueError("Native operand changed")
        arrays = load_arrays(path)
        lookup = {r: i for i, r in enumerate(scope["row_plan"]["rows"])}
        rows = scope["row_plan"]["uniform_rows"]
        ids = [lookup[r] for r in rows]
        pre, post = [arrays[k][ids] for k in ["q2_pre", "q2_post"]]
        case = {"source_index": selected["source_index"], "step": selected["step"],
                "block": selected["block"], "full_metric_change": metric(post) - metric(pre), "leave_one_out": []}
        for k, row in enumerate(rows):
            keep = [i for i in range(len(rows)) if i != k]
            value = metric(post[keep]) - metric(pre[keep])
            case["leave_one_out"].append({"excluded_row": row,
                "latent_coordinate": scope["row_plan"]["coordinates"][lookup[row]],
                "metric_change": value})
        result["cases"].append(case)
    write_json(args.out, result)


if __name__ == "__main__":
    main()
