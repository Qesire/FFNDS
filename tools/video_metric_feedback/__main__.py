"""CLI for matrix export, metric import, scoring and bounded feedback."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .workflow import load_config, plan, read_json, run_search, write_json


def _rows(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("records", "videos", "cohort"):
            if isinstance(value.get(key), list):
                return value[key]
    raise ValueError("expected a list or an object with records/videos/cohort")


def assemble_result(request: dict, videos: list[dict], official: list[dict],
                    local_reports: list[dict]) -> dict:
    """Seal imported metrics with the exact request and generation identities."""
    from .metrics import _index, import_local_diagnostics, merge_record_sets, score_records
    cohort = _index(request["cohort"], "request cohort")
    generated = _index(videos, "generated videos")
    if set(cohort) != set(generated):
        raise ValueError("generated videos must exactly match the requested cohort")
    for key, video in generated.items():
        if video["image_id"] != cohort[key]["image_id"]:
            raise ValueError("generated video uses another image cluster")
    index = {(r["source_id"], r["seed"], r.get("sample_index", 0)): r for r in videos}
    synthetic = any(part.get("synthetic", False) for part in [*official, *local_reports])
    if synthetic and request.get("mode", "video") != "synthetic_unit_fixture":
        raise ValueError("synthetic metric results cannot be assembled as real video evidence")
    for bundle in official:
        for field in ("candidate_id", "evaluation_key", "execution_fingerprint"):
            if field in bundle and bundle[field] != request[field]:
                raise ValueError(f"metric bundle {field} does not match request")
    parts = list(official)
    local_records = []
    for report in local_reports:
        identity = report.get("identity", {})
        key = (identity.get("source_id"), identity.get("seed"), identity.get("sample_index", 0))
        if key not in index:
            raise ValueError("local diagnostic is not from the requested generation cohort")
        reference_path = cohort[key].get("reference_video_path")
        if reference_path is not None:
            from .matrices import sha256_file
            reference = report.get("reference")
            if (not isinstance(reference, dict) or not reference.get("path")
                    or Path(reference["path"]).resolve() != Path(reference_path).resolve()):
                raise ValueError("local diagnostic used a different Dense reference video")
            if reference.get("sha256") != sha256_file(reference_path):
                raise ValueError("local diagnostic Dense reference SHA256 does not match the cohort asset")
        local_records.append(import_local_diagnostics(report, index[key]))
    if local_records:
        parts.append({"schema_version": 1, "candidate_id": request["candidate_id"],
                      "evaluator": {"name": "ffnds_local_diagnostics"},
                      "records": local_records})
    if not parts:
        raise ValueError("no measured metric records supplied")
    result = merge_record_sets(*parts)
    if result.get("candidate_id") not in (None, request["candidate_id"]):
        raise ValueError("metric bundle belongs to a different candidate")
    for row in result["records"]:
        key = (row["source_id"], row["seed"], row.get("sample_index", 0))
        if key not in index:
            raise ValueError("metric record does not match generated video manifest")
        video = index[key]
        if Path(row["video_path"]).resolve() != Path(video["video_path"]).resolve():
            raise ValueError("metric record uses another video path")
        if row.get("image_id") != video["image_id"]:
            raise ValueError("metric record uses another image cluster")
        noise = video.get("initial_noise_sha256")
        if row.get("initial_noise_sha256", noise) != noise:
            raise ValueError("metric and generation initial-noise identities disagree")
        if noise is not None:
            row["initial_noise_sha256"] = noise
    result.update(schema_version=1, candidate_id=request["candidate_id"],
                  evaluation_key=request["evaluation_key"],
                  execution_fingerprint=request["execution_fingerprint"],
                  synthetic=bool(synthetic or request.get("mode") == "synthetic_unit_fixture"))
    score_records(result, result, request["cohort"], request["metrics"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for command in ("check", "run", "export"):
        p = sub.add_parser(command)
        p.add_argument("--config", required=True)
        if command != "check":
            p.add_argument("--output", required=True)
        if command == "export":
            p.add_argument("--theta", help='JSON object; defaults to configured initial values')
    p = sub.add_parser("import-vbench")
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--videos-manifest", required=True)
    p.add_argument("--candidate-id", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("assemble")
    p.add_argument("--request", required=True)
    p.add_argument("--videos-manifest", required=True)
    p.add_argument("--official-bundle", nargs="*", default=[])
    p.add_argument("--local-report", nargs="*", default=[])
    p.add_argument("--output", required=True)
    p = sub.add_parser("score")
    p.add_argument("--candidate", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--cohort", required=True)
    p.add_argument("--specification", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.action in ("check", "run", "export"):
        config = load_config(args.config)
        if args.action == "check":
            result = plan(config)
        elif args.action == "run":
            result = run_search(config, args.output)
            result = {"parameters": result["parameters"],
                      "status": result["optimization"]["status"],
                      "result_path": str(Path(args.output).resolve() / "optimization_result.json"),
                      "external_calls": result["external_calls"]}
        else:
            from .matrices import export_candidate
            theta = json.loads(args.theta) if args.theta else {p["name"]: p["initial"] for p in config["parameters"]}
            result = export_candidate(theta=theta, output_dir=args.output, **config["matrices"])
    elif args.action == "import-vbench":
        from .metrics import import_vbench_results, merge_record_sets
        manifest = read_json(args.videos_manifest)
        videos = _rows(manifest)
        result = merge_record_sets(*[import_vbench_results(read_json(path), videos, args.candidate_id, args.revision)
                                     for path in args.results])
        if isinstance(manifest, dict):
            if manifest.get("candidate_id", args.candidate_id) != args.candidate_id:
                raise ValueError("generation manifest belongs to another candidate")
            result["synthetic"] = bool(manifest.get("synthetic", False))
            for key in ("evaluation_key", "execution_fingerprint"):
                if key in manifest:
                    result[key] = manifest[key]
        write_json(args.output, result)
    elif args.action == "assemble":
        request = read_json(args.request)
        manifest = read_json(args.videos_manifest)
        if isinstance(manifest, dict):
            if manifest.get("synthetic", False) and request.get("mode", "video") != "synthetic_unit_fixture":
                raise ValueError("synthetic generation cannot be assembled as real video evidence")
            for key in ("candidate_id", "execution_fingerprint", "evaluation_key"):
                if manifest.get(key) != request[key]:
                    raise ValueError(f"generation manifest {key} does not match the request")
        else:
            raise ValueError("assemble requires a generation manifest object with request identities")
        from .vbench_worker import validate_videos_manifest
        videos = validate_videos_manifest(request, manifest)
        result = assemble_result(request, videos,
                                 [read_json(p) for p in args.official_bundle],
                                 [read_json(p) for p in args.local_report])
        write_json(args.output, result)
    else:
        from .metrics import score_records
        result = score_records(read_json(args.candidate), read_json(args.reference),
                               _rows(read_json(args.cohort)), read_json(args.specification))
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
