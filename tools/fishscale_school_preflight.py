#!/usr/bin/env python3
"""School-only, read-only provenance and data-integrity preflight for fishscale studies.

Inventory is not a scientific/execution PASS. Strict phases hash every required
input file and require execution on a Slurm node. Nothing uses remote HF roots.
No imports of torch, CUDA, or LightX2V. Writes only a JSON receipt under the
registered school project.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
from typing import Any

DEFAULT_CONFIG = "research/diagnostics/wan_temporal_fishscale/SCHOOL_CLUSTER_CONFIG.json"
DEFAULT_REGISTRY = "research/diagnostics/wan_temporal_fishscale/ASSET_REGISTRY.json"
LEGACY_PREFIXES = ("/home/xiang/", "/var/tmp/ffnds-xiang/", "/root/")

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()

def under(root: Path, path: Path) -> bool:
    return path == root or root in path.parents

def checked_file(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Explicit school file path is required")
    path = Path(name).expanduser()
    if not path.is_absolute():
        path = root / path
    # Reject old paths even if a user accidentally mounted them inside school.
    if any(str(path).startswith(prefix) for prefix in LEGACY_PREFIXES):
        raise ValueError("External legacy worker path is forbidden")
    resolved = path.resolve()
    if not under(root.resolve(), resolved):
        raise ValueError("Path is not inside resolved school project root")
    return resolved

def file_evidence(root: Path, spec: dict[str, Any], strict: bool) -> dict[str, Any]:
    item: dict[str, Any] = {"declared_path": spec.get("path") or spec.get("relative_path")}
    try:
        path = checked_file(root, item["declared_path"])
        item["realpath"] = str(path)
        item["exists"] = path.is_file()
        if not path.is_file():
            item["ok"] = False
            item["error"] = "NOT_A_READABLE_FILE"
            return item
        st = path.stat()
        item["bytes"] = st.st_size
        expected_size = spec.get("bytes")
        expected_sha = spec.get("sha256")
        if expected_size is not None and st.st_size != int(expected_size):
            item["ok"] = False
            item["error"] = "BYTE_SIZE_MISMATCH"
            return item
        if strict:
            if not isinstance(expected_sha, str) or len(expected_sha) != 64:
                item["ok"] = False
                item["error"] = "NO_REGISTERED_SHA256"
                return item
            item["sha256"] = sha256(path)
            item["ok"] = item["sha256"] == expected_sha
            if not item["ok"]:
                item["error"] = "SHA256_MISMATCH"
        else:
            item["ok"] = False
            item["state"] = "STAT_ONLY_UNVERIFIED"
        return item
    except (OSError, ValueError) as exc:
        item["ok"] = False
        item["error"] = f"{type(exc).__name__}: {exc}"
        return item

def parse_safetensors(path: Path) -> dict[str, Any]:
    # Header-only parser to avoid importing CUDA/torch.
    with path.open("rb") as fd:
        raw = fd.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated safetensors")
        length = struct.unpack("<Q", raw)[0]
        if length < 2 or length > 16 * 1024 * 1024:
            raise ValueError("Unreasonable safetensors header length")
        header = json.loads(fd.read(length).decode("utf-8"))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if len(tensors) != 1:
        raise ValueError("Expected one latent tensor")
    name, tensor = next(iter(tensors.items()))
    return {"tensor_name": name, "dtype": tensor["dtype"], "shape": tensor["shape"]}

def video_info(path: Path) -> dict[str, Any]:
    probe = shutil.which("ffprobe")
    if probe is None:
        raise RuntimeError("ffprobe is required for strict video validation")
    call = subprocess.run([
        probe, "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_read_frames,avg_frame_rate",
        "-of", "json", str(path)
    ], text=True, capture_output=True, timeout=60, check=True)
    streams = json.loads(call.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError("Expected exactly one video stream")
    s = streams[0]
    p, q = map(int, s["avg_frame_rate"].split("/"))
    return {"width": int(s["width"]), "height": int(s["height"]),
            "frames": int(s["nb_read_frames"]), "fps": p / q}

def check_geometry(actual: dict[str, Any], reference: dict[str, Any]) -> bool:
    return (all(actual[k] == reference[k] for k in ("width", "height", "frames"))
            and abs(actual["fps"] - float(reference["fps"])) < 0.02)

def school_environment(cfg: dict[str, Any], strict: bool) -> tuple[dict[str, Any], Path | None]:
    home = Path(cfg["school_home"])
    alias = Path(cfg["school_home_documented_alias"])
    project = Path(cfg["school_project_root"])
    out = {"declared_home": str(home), "documented_alias": str(alias),
           "home_exists": home.is_dir(), "alias_exists": alias.is_dir(),
           "hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "school_project_exists": project.is_dir(), "errors": []}
    if not home.is_dir():
        out["errors"].append("DECLARED_SCHOOL_HOME_MISSING")
    if home.is_dir() and alias.is_dir() and not os.path.samefile(home, alias):
        out["errors"].append("SCHOOL_HOME_ALIAS_NOT_SAMEFILE")
    if not project.is_dir():
        out["errors"].append("SCHOOL_PROJECT_ROOT_MISSING")
    if strict and not out["slurm_job_id"]:
        out["errors"].append("STRICT_PREFLIGHT_REQUIRES_SLURM_COMPUTE_ALLOCATION")
    if project.is_dir() and home.is_dir() and not under(home.resolve(), project.resolve()):
        out["errors"].append("SCHOOL_PROJECT_NOT_UNDER_DECLARED_HOME")
    out["home_realpath"] = str(home.resolve())
    out["project_realpath"] = str(project.resolve())
    return out, project if not out["errors"] else None

def find_school_candidates(project: Path, dirs: list[str], max_files: int = 300) -> list[dict[str, Any]]:
    """Bounded path discovery for explicitly named school subdirectories only."""
    found: list[dict[str, Any]] = []
    for directory in dirs:
        try:
            root = checked_file(project, directory)
            if not root.is_dir():
                found.append({"search_directory": str(root), "status": "DIRECTORY_MISSING"})
                continue
            queue = [(root, 0)]
            while queue and len(found) < max_files:
                current, depth = queue.pop(0)
                if depth > 4:
                    continue
                for child in sorted(current.iterdir(), key=lambda p: p.name):
                    if len(found) >= max_files:
                        break
                    if child.is_symlink():
                        continue
                    if child.is_dir() and depth < 4:
                        queue.append((child, depth + 1))
                    elif child.is_file() and (
                        child.name.endswith(".mp4") or child.name == "latent_step3.safetensors"
                        or child.name == "final_latent_bf16.safetensors"
                    ):
                        found.append({"path": str(child.resolve()), "bytes": child.stat().st_size,
                                      "status": "DISCOVERED_NOT_SHA_VERIFIED"})
        except (OSError, ValueError) as exc:
            found.append({"search_directory": directory, "error": str(exc)})
    return found


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    if cfg["execution_domain"] != "USTC_SLURM_ONLY":
        raise ValueError("Not a school-only config")
    if args.source_manifest is not None:
        # Private school manifest; never copy raw input/prompt paths to public GitHub.
        candidate_project = Path(cfg["school_project_root"]).resolve()
        manifest_path = checked_file(candidate_project, str(args.source_manifest))
        cfg["new_source_capture_spec"] = json.loads(manifest_path.read_text(encoding="utf-8"))
    strict = args.phase != "inventory"
    env, project = school_environment(cfg, strict)
    rows: list[dict[str, Any]] = []
    status = "INVENTORY_ONLY_NOT_AN_EXECUTION_PASS" if not strict else "BLOCKED"
    candidates = find_school_candidates(project, args.candidate_dir) if project is not None and args.phase == "inventory" else []
    if project is not None:
        requirements = {"inventory": [], "review": [],
                        "decode": ["vae"], "generate": ["high", "low", "t5", "vae"]}
        for identity in requirements[args.phase]:
            models = [m for m in cfg["expected_model_assets"] if m["id"] == identity]
            if len(models) != 1:
                raise ValueError("Missing model definition: " + identity)
            row = file_evidence(project, models[0], strict)
            row["id"] = "model_" + identity
            rows.append(row)
        if args.phase == "inventory":
            for model in cfg["expected_model_assets"]:
                row = file_evidence(project, model, False)
                row["id"] = "model_" + model["id"]
                rows.append(row)
        if args.phase in ("review", "decode"):
            pairs = registry.get("school_source_pairs", [])
            selected = [p for p in pairs if p.get("id") == args.pair_id] if args.pair_id else pairs
            if len(selected) != 1:
                env["errors"].append("EXACTLY_ONE_REGISTERED_SCHOOL_PAIR_REQUIRED")
            else:
                pair = selected[0]
                fields = ["video"] if args.phase == "review" else ["video", "final_latent", "parent_run", "image"]
                if args.phase == "decode" and (
                    not pair.get("prompt_sha256") or pair.get("seed") is None
                ):
                    env["errors"].append("SOURCE_PROMPT_OR_SEED_NOT_BOUND")
                for field in fields:
                    if field not in pair:
                        rows.append({"id": "pair_" + field, "ok": False, "error": "MISSING_SOURCE_BINDING"})
                        continue
                    row = file_evidence(project, pair[field], strict)
                    row["id"] = "pair_" + field
                    if row.get("ok"):
                        try:
                            path = checked_file(project, pair[field]["path"])
                            if field == "video":
                                actual = video_info(path)
                                row["video_info"] = actual
                                if not check_geometry(actual, cfg["model_recipe"]["reference_geometry"]):
                                    row["ok"] = False
                                    row["error"] = "VIDEO_GEOMETRY_MISMATCH"
                            if field == "final_latent":
                                actual = parse_safetensors(path)
                                row["latent_info"] = actual
                                expect = pair[field]
                                if actual["dtype"] != "BF16" or actual["shape"] != [16,21,58,104]:
                                    row["ok"] = False
                                    row["error"] = "NATIVE_BF16_LATENT_REQUIRED"
                                if expect.get("tensor_name") and expect["tensor_name"] != actual["tensor_name"]:
                                    row["ok"] = False
                                    row["error"] = "LATENT_KEY_MISMATCH"
                        except (OSError, ValueError, RuntimeError, KeyError, ZeroDivisionError,
                                subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                            row["ok"] = False
                            row["error"] = f"CONTENT_CHECK_FAILED: {exc}"
                    rows.append(row)
                if args.phase == "decode" and all(r.get("ok") for r in rows):
                    try:
                        run_path = checked_file(project, pair["parent_run"]["path"])
                        native_record = json.loads(run_path.read_text(encoding="utf-8"))
                        native_text = json.dumps(native_record, ensure_ascii=False)
                        if (pair["video"]["sha256"] not in native_text
                                or pair["final_latent"]["sha256"] not in native_text):
                            env["errors"].append("PARENT_RUN_DOES_NOT_BIND_BOTH_VIDEO_AND_LATENT")
                        if (pair["image"]["sha256"] not in native_text
                                or pair["prompt_sha256"] not in native_text
                                or native_record.get("seed") != pair["seed"]):
                            env["errors"].append("PARENT_RUN_INPUT_SEED_NOT_BOUND")
                    except (OSError, ValueError, KeyError) as exc:
                        env["errors"].append("PARENT_RUN_INVALID: " + str(exc))
        if args.phase == "generate":
            inp = cfg.get("new_source_capture_spec", {})
            if not inp.get("image_path") or not inp.get("prompt") or not inp.get("image_sha256"):
                env["errors"].append("NEW_SCHOOL_SOURCE_IMAGE_PROMPT_NOT_BOUND")
            else:
                row = file_evidence(project, {"path": inp["image_path"], "sha256": inp["image_sha256"]}, True)
                row["id"] = "new_source_input"
                rows.append(row)
    else:
        # Inventory reports missing root without trying legacy paths.
        env["errors"].append("SCHOOL_FILE_CHECK_SKIPPED_DUE_TO_ROOT_ERRORS")
    if strict and not env["errors"] and rows and all(r.get("ok") for r in rows):
        status = "VERIFIED_SCHOOL_INPUTS_READY_FOR_" + args.phase.upper()
    receipt = {"schema": "FFNDS_FISHSCALE_SCHOOL_PREFLIGHT_V1", "phase": args.phase,
               "status": status, "school_environment": env, "checks": rows,
               "gpu_model_calls": 0, "vae_calls": 0, "school_candidate_discovery": candidates,
               "note": "Strict PASS verifies current files in a Slurm process but is not ACTIVE permission or an actual generation result."}
    return receipt, 0 if (status.startswith("VERIFIED_") or
                          (args.phase == "inventory" and not env["errors"])) else 2

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument("--registry", type=Path, default=Path(DEFAULT_REGISTRY))
    parser.add_argument("--phase", required=True, choices=("inventory", "review", "decode", "generate"))
    parser.add_argument("--pair-id", help="Required if registry has more than one pair")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--candidate-dir", action="append", default=[],
                        help="School project relative directory, inventory only; bound to four levels")
    parser.add_argument("--source-manifest", type=Path,
                        help="School-local JSON specifying image_path/image_sha256/prompt")
    args = parser.parse_args()
    receipt, code = run(args)
    # Only create a small report in the declared school project; never write to assets.
    project = Path(json.loads(args.config.read_text(encoding="utf-8"))["school_project_root"])
    if not project.is_dir():
        raise SystemExit("School project root is not present; refusing to create it during preflight")
    path = args.out.expanduser().resolve()
    if not under(project.resolve(), path):
        raise SystemExit("Report must be written within school project root")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
    print(json.dumps({"status": receipt["status"], "out": str(path)}, ensure_ascii=False))
    return code

if __name__ == "__main__":
    sys.exit(main())
