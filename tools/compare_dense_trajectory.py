"""Require bitwise equality of four scheduler latents for observer no-op.

Video hashes are also compared but are not the numerical criterion. This check
applies only to the input, environment and observer sources in these two runs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plain", type=Path, required=True)
    ap.add_argument("--observed", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    if not 1 <= args.threads <= 16:
        ap.error("Use 1..16 CPU threads")
    torch.set_num_threads(args.threads)
    plain = json.loads((args.plain / "run_meta.json").read_text())
    observed = json.loads((args.observed / "run_meta.json").read_text())
    for data in [plain, observed]:
        if data["status"] != "PASS" or data["arm"] not in ["D-BF16", "D-Q"]:
            raise RuntimeError("Both plain and observed dense distill runs must finish successfully")
    fields = ["arm", "source_id", "seed", "prompt_sha256", "image_sha256", "checkpoint_sha256",
              "versions", "contract", "resolved_contract", "scheduler_class", "device", "cuda_runtime"]
    if plain["arm"] == "D-Q":
        fields.append("quantized_cache_identity")
    identity = {name: plain[name] == observed[name] for name in fields}
    # Runner additions enable observation but cannot alter native implementation files.
    original_sources = {path: sha for path, sha in plain["source_sha256"].items()
                        if "/.runtime/code/" in path}
    identity["native_source_files"] = original_sources == {
        path: sha for path, sha in observed["source_sha256"].items() if "/.runtime/code/" in path}
    if not all(identity.values()):
        raise RuntimeError(f"Trajectory comparison identity mismatch: {identity}")
    cap_path = args.observed / "ffn/CAPTURE_MANIFEST.json"
    capture = json.loads(cap_path.read_text())
    if capture["status"] != "PASS_CAPTURE_STRUCTURE_ONLY":
        raise RuntimeError("Capture structure must pass before observer no-op can be assessed")
    checks = []
    for step in range(4):
        paths = [root / f"latent_step{step}.safetensors" for root in [args.plain, args.observed]]
        for path, meta in zip(paths, [plain, observed]):
            if hashlib.sha256(path.read_bytes()).hexdigest() != meta["trajectory"][step]["sha256"]:
                raise RuntimeError(f"Latent artifact hash changed: {path}")
        aa, bb = [load_file(str(path))["latents"] for path in paths]
        finite = bool(torch.isfinite(aa).all() and torch.isfinite(bb).all())
        equal = finite and aa.shape == bb.shape and aa.dtype == bb.dtype and torch.equal(aa, bb)
        checks.append({"step": step, "finite": finite, "bitwise_equal": equal,
                       "plain_sha256": plain["trajectory"][step]["sha256"],
                       "observed_sha256": observed["trajectory"][step]["sha256"],
                       "max_abs_error": float((aa - bb).abs().max()) if aa.shape == bb.shape else None})
    passed = all(row["bitwise_equal"] for row in checks)
    report = {"status": "PASS_OBSERVER_NOOP_THIS_INPUT" if passed else "FAILED_OBSERVER_NOOP",
              "criterion_frozen_before_run": "bitwise identical FP32 saved native scheduler latents at all four steps",
              "plain": str(args.plain.resolve()), "observed": str(args.observed.resolve()),
              "identity_checks": identity, "source_id": plain["source_id"], "seed": plain["seed"],
              "observer_source_sha256": observed["observer_source_sha256"], "checks": checks,
              "cpu_threads": args.threads,
              "arm": plain["arm"], "hif4_observer_source_sha256": observed.get("hif4_observer_source_sha256"),
              "video_hash_equal": plain["video"]["sha256"] == observed["video"]["sha256"],
              "scope": "Only this same-input, same-arm native capture and source/environment identity; no candidate intervention equivalence claimed"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
