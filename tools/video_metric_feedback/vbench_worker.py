"""Run the actual VBench compute functions in the evaluator's own environment.

The compute call, submodule initializer, and metadata contract follow the
official pinned checkout and FFNDS tools/run_fresh_official_metrics.py. Heavy
dependencies are imported only after validating the requested video panel and
the checkout. ``--local`` is VBench's local-cache option, not an offline promise:
the official initializer can download missing model assets.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Callable

from .metrics import (
    LOCAL_FIELDS, _index, _path, import_vbench_results,
    validate_metric_specification,
)


PINNED_REVISION = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
I2V_DIMENSIONS = frozenset(("i2v_subject", "i2v_background", "camera_motion"))
OFFICIAL_DIMENSIONS = frozenset((
    "subject_consistency", "background_consistency", "temporal_flickering",
    "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality",
)) | I2V_DIMENSIONS
CAMERA_INSTRUCTIONS = frozenset((
    "camera pans left", "camera pans right", "camera tilts up", "camera tilts down",
    "camera zooms in", "camera zooms out", "camera static",
))


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        return _json_value(value.item())
    return value


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_value(value), ensure_ascii=False,
                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha(path: Path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def validate_videos_manifest(request: dict, manifest: dict) -> list[dict]:
    """Bind a native generation manifest to this request before model loading."""
    if not isinstance(request, dict) or request.get("schema_version") != 1:
        raise ValueError("request must have schema_version=1")
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or manifest.get("schema") != "ffnds.video_metric_feedback.videos.v1"):
        raise ValueError("videos manifest must be a complete native generation manifest object")
    for field in ("candidate_id", "evaluation_key", "execution_fingerprint"):
        expected = request.get(field)
        if not isinstance(expected, str) or not expected or manifest.get(field) != expected:
            raise ValueError(f"generation manifest {field} does not match the request")
    if not isinstance(manifest.get("synthetic"), bool):
        raise ValueError("generation manifest synthetic must be a boolean")
    if manifest["synthetic"] and request.get("mode") != "synthetic_unit_fixture":
        raise ValueError("synthetic generation manifest requires mode='synthetic_unit_fixture'; video requests must use real generation")
    exported = request.get("matrix_export")
    if not isinstance(exported, dict) or exported.get("fingerprint") != request["execution_fingerprint"]:
        raise ValueError("request matrix_export fingerprint does not match execution_fingerprint")
    expected_sha = exported.get("matrix_manifest_sha256")
    if (not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
            or manifest.get("matrix_manifest_sha256") != expected_sha):
        raise ValueError("generation matrix_manifest_sha256 does not match the request")
    paths = []
    for value in (request.get("matrix_manifest_path"), manifest.get("matrix_manifest_path")):
        if not isinstance(value, str) or not value:
            raise ValueError("request and generation require an explicit matrix_manifest_path")
        paths.append(Path(value).expanduser().resolve())
    if paths[0] != paths[1]:
        raise ValueError("generation matrix_manifest_path does not match the request")
    if not paths[0].is_file() or _sha(paths[0]) != expected_sha:
        raise ValueError("matrix manifest file does not match its recorded SHA256")
    rows = manifest.get("records")
    _index(rows, "generation manifest")
    return rows


def _verify_checkout(repo: str | Path, revision: str, dimensions: list[str]) -> dict:
    root = Path(repo).resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("VBench revision must be a complete lowercase commit SHA")
    actual = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    if actual != revision:
        raise ValueError(f"VBench checkout is {actual}, expected {revision}")
    modified = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if modified:
        raise ValueError("VBench checkout has tracked modifications; fixed-revision scoring requires original source")
    paths = {"vbench/__init__.py", "vbench/utils.py",
             "vbench2_beta_i2v/__init__.py", "vbench2_beta_i2v/utils.py"}
    for dimension in dimensions:
        package = "vbench2_beta_i2v" if dimension in I2V_DIMENSIONS else "vbench"
        paths.add(f"{package}/{dimension}.py")
    return {
        "repository": str(root), "revision": actual, "tracked_source_clean": True,
        "source_sha256": {path: _sha(root / path) for path in sorted(paths)},
    }


def _official_compute(repo, revision, dimensions, device, resolution, local):
    if repo is None:
        raise ValueError("vbench_repo is required for real official scoring")
    provenance = _verify_checkout(repo, revision, dimensions)
    root = Path(provenance["repository"])
    # A separate worker process prevents mixing an installed wheel with the
    # explicitly selected source checkout. Refuse an already imported mismatch.
    for name in ("vbench", "vbench2_beta_i2v"):
        if name in sys.modules:
            origin = getattr(sys.modules[name], "__file__", None)
            if origin is None or not Path(origin).resolve().is_relative_to(root):
                raise ValueError(f"{name} was already imported from another source")
    sys.path.insert(0, str(root))
    import torch
    from vbench2_beta_i2v.utils import init_submodules

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    selected_device = torch.device(device)
    submodules = init_submodules(dimensions, local=local, read_frame=False,
                                 resolution=resolution)
    if set(submodules) != set(dimensions):
        raise ValueError("official initializer did not provide every requested dimension")
    provenance.update(device=str(selected_device), torch_version=torch.__version__,
                      local_cache_mode=bool(local), resolution=resolution)

    def compute(dimension, info_path):
        package = "vbench2_beta_i2v" if dimension in I2V_DIMENSIONS else "vbench"
        module = importlib.import_module(f"{package}.{dimension}")
        expected = root / package / f"{dimension}.py"
        if Path(module.__file__).resolve() != expected.resolve():
            raise ValueError(f"{dimension} was imported from a different checkout")
        function = getattr(module, f"compute_{dimension}")
        with torch.inference_mode():
            result = function(str(info_path), selected_device, submodules[dimension])
        if selected_device.type == "cuda":
            torch.cuda.synchronize(selected_device)
        gc.collect()
        if selected_device.type == "cuda":
            torch.cuda.empty_cache()
        return result

    return compute, provenance


def _prepare(request: dict, videos: list[dict], work: Path):
    if not isinstance(request, dict) or request.get("schema_version") != 1:
        raise ValueError("request must have schema_version=1")
    if not isinstance(request.get("candidate_id"), str) or not request["candidate_id"]:
        raise ValueError("request candidate_id is required")
    specification = request.get("metrics", {})
    validate_metric_specification(specification)
    cohort = _index(request.get("cohort"), "request cohort")
    manifest = _index(videos, "videos manifest")
    if not cohort or set(cohort) != set(manifest):
        raise ValueError("videos manifest must exactly match the requested sample keys")
    required = {item["name"] for group in ("objectives", "guards")
                for item in specification.get(group, [])}
    unknown = required - OFFICIAL_DIMENSIONS - set(LOCAL_FIELDS)
    if unknown:
        raise ValueError(f"unsupported official/local metric names: {sorted(unknown)}")
    dimensions = sorted(required & OFFICIAL_DIMENSIONS)
    paths = set()
    canonical = []
    for key, row in manifest.items():
        source = cohort[key]
        if row["image_id"] != source["image_id"]:
            raise ValueError(f"video image_id differs from cohort: {key!r}")
        original = Path(row.get("video_path", "")).resolve()
        if not row.get("video_path") or not original.is_file():
            raise ValueError(f"video is missing: {row.get('video_path')}")
        if str(original) in paths:
            raise ValueError("each manifest sample must identify a distinct generated video path")
        paths.add(str(original))
        item = copy.deepcopy(row)
        item["video_path"] = str(original)
        if "eligible_metrics" in source:
            item["eligible_metrics"] = list(source["eligible_metrics"])
        else:
            item.pop("eligible_metrics", None)
        canonical.append(item)
    indexed = _index(canonical, "canonical videos")
    jobs = []
    for dimension in dimensions:
        eligible = [(key, row) for key, row in indexed.items()
                    if "eligible_metrics" not in cohort[key]
                    or dimension in cohort[key]["eligible_metrics"]]
        if not eligible:
            raise ValueError(f"requested metric {dimension} has no eligible videos")
        full_info, mapping = [], {}
        folder = work / dimension
        folder.mkdir(parents=True, exist_ok=True)
        for key, row in eligible:
            source = cohort[key]
            prompt = source.get("prompt", "")
            if not isinstance(prompt, str):
                raise ValueError(f"prompt must be a string for {key!r}")
            evaluated_path = row["video_path"]
            information = {"prompt_en": prompt, "dimension": [dimension]}
            if dimension in ("i2v_subject", "i2v_background"):
                actual_image = row.get("image_path")
                requested_image = source.get("image_path")
                if not actual_image or not requested_image:
                    raise ValueError(f"I2V scoring requires explicit manifest and cohort image_path for {key!r}")
                image = Path(actual_image).resolve()
                if image != Path(requested_image).resolve() or not image.is_file():
                    raise ValueError(f"I2V condition image does not match request for {key!r}")
                information["custom_image_path"] = str(image)
            if dimension == "camera_motion":
                instruction = source.get("camera_instruction")
                if instruction not in CAMERA_INSTRUCTIONS:
                    raise ValueError(f"eligible camera_motion sample needs an explicit official camera_instruction: {key!r}")
                identity = hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest()[:20]
                link = folder / f"sample-{identity} {instruction}.mp4"
                target = Path(row["video_path"])
                if link.is_symlink() and link.resolve() != target:
                    raise ValueError(f"camera staging path points to a different video: {link}")
                if not link.is_symlink():
                    if link.exists():
                        raise ValueError(f"camera staging path is not a symlink: {link}")
                    link.symlink_to(target)
                evaluated_path = str(link)
                information["camera_instruction_source"] = "cohort.camera_instruction"
            mapping[_path(evaluated_path)] = row["video_path"]
            information["video_list"] = [evaluated_path]
            full_info.append(information)
        info_path = folder / "FULL_INFO.json"
        _write_json(info_path, full_info)
        jobs.append({"dimension": dimension, "full_info_path": str(info_path),
                     "full_info_sha256": _sha(info_path), "video_path_mapping": mapping})
    return canonical, jobs, sorted(required & set(LOCAL_FIELDS))


def evaluate_vbench(request: dict, videos: list[dict] | dict, work_dir: str | Path, *,
                    vbench_repo: str | Path | None = None,
                    revision: str = PINNED_REVISION, device: str = "cuda:0",
                    resolution: str = "7-4", local: bool = True,
                    compute: Callable | None = None) -> dict:
    """Score each requested official dimension once on its eligible videos.

    ``compute(dimension, full_info_path)`` is a unit-test injection only. Its
    result is marked synthetic and cannot enter a real video-mode search.
    Native generation and local diagnostics are separate worker steps.
    A native manifest object verifies request and matrix identities and carries
    its synthetic flag forward. The CLI requires this complete object.
    """
    if resolution not in ("1-1", "8-5", "7-4", "16-9"):
        raise ValueError("resolution must be an official VBench-I2V aspect-ratio label")
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    generation = None
    if isinstance(videos, dict):
        generation = videos
        videos = validate_videos_manifest(request, generation)
    canonical, jobs, deferred = _prepare(request, videos, work)
    injected = compute is not None
    provenance = {"revision": revision, "execution": "injected_test_double" if injected else "no_official_dimensions"}
    if jobs and not injected:
        compute, provenance = _official_compute(vbench_repo, revision,
                                                [job["dimension"] for job in jobs],
                                                device, resolution, local)
    results = {}
    for job in jobs:
        dimension = job["dimension"]
        payload = _json_value(compute(dimension, job["full_info_path"]))
        expected_count = 3 if dimension == "camera_motion" else 2
        if not isinstance(payload, list) or len(payload) != expected_count or not isinstance(payload[-1], list):
            raise ValueError(f"official compute_{dimension} returned an invalid structure")
        mapping = job["video_path_mapping"]
        seen = set()
        for record in payload[-1]:
            if not isinstance(record, dict):
                raise ValueError(f"official {dimension} returned a non-object video record")
            path = _path(record.get("video_path"))
            if path not in mapping or path in seen:
                raise ValueError(f"official {dimension} returned an unexpected/duplicate video path")
            seen.add(path)
            record["video_path"] = mapping[path]
        if seen != set(mapping):
            raise ValueError(f"official {dimension} omitted eligible videos")
        if dimension == "camera_motion":
            if not isinstance(payload[1], dict):
                raise ValueError("official camera_motion per-type result must be an object")
            # Official camera summaries are NaN for types absent from this panel.
            # Preserve their absence as JSON null; actual video/aggregate scores
            # must still be finite and are checked by the import adapter below.
            payload[1] = {name: (None if isinstance(value, (float, int)) and not math.isfinite(value)
                                else value) for name, value in payload[1].items()}
        results[dimension] = payload
    bundle = import_vbench_results(results, canonical, request["candidate_id"], revision)
    for field in ("evaluation_key", "execution_fingerprint"):
        if field in request:
            bundle[field] = request[field]
    bundle["synthetic"] = injected or (generation is not None and generation["synthetic"])
    if generation is not None:
        bundle["generation_identity"] = {
            field: generation[field] for field in (
                "candidate_id", "evaluation_key", "execution_fingerprint",
                "matrix_manifest_path", "matrix_manifest_sha256", "synthetic",
            )
        }
    bundle["evaluator"]["source_verification"] = provenance
    bundle["official_evaluator_calls"] = len(jobs)
    bundle["deferred_local_metrics"] = deferred
    bundle["official_calls"] = jobs
    _write_json(work / "official_results.json", results)
    _write_json(work / "worker_provenance.json", {"evaluator": bundle["evaluator"], "calls": jobs,
                                                "synthetic": bundle["synthetic"],
                                                "generation_identity": bundle.get("generation_identity"),
                                                "deferred_local_metrics": deferred})
    return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--videos-manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--vbench-repo", type=Path, required=True)
    parser.add_argument("--revision", default=PINNED_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution", default="7-4")
    parser.add_argument("--local", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    videos = json.loads(args.videos_manifest.read_text(encoding="utf-8"))
    if not isinstance(videos, dict):
        raise ValueError("CLI requires a complete native generation manifest object, not a records list")
    work = args.work_dir or args.result.parent / (args.result.stem + "_work")
    bundle = evaluate_vbench(request, videos, work, vbench_repo=args.vbench_repo,
                             revision=args.revision, device=args.device,
                             resolution=args.resolution, local=args.local)
    _write_json(args.result, bundle)


if __name__ == "__main__":
    main()
