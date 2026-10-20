"""Run fresh task0/task6 seed pairs with an independent observer no-op check.

Every input gets a plain dense run and an observed run in separate processes.
The same frozen four scheduler latents must match bitwise. This orchestrates
worker GPU jobs and local CPU comparisons; it never uses a local GPU.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_run(root, record, observed, arm, cache):
    meta = json.loads((root / "run_meta.json").read_text())
    if meta["status"] != "PASS" or meta["arm"] != arm or meta["input_record"] != record:
        raise RuntimeError(f"Run identity/status differs: {root}")
    if sha(record["image_path"]) != meta["image_sha256"]:
        raise RuntimeError("Input image bytes changed")
    if [row["step"] for row in meta["trajectory"]] != [0, 1, 2, 3]:
        raise RuntimeError("Missing native denoising steps")
    for row in meta["trajectory"]:
        if sha(root / f"latent_step{row['step']}.safetensors") != row["sha256"]:
            raise RuntimeError("Native latent artifact changed")
    if sha(root / "output.mp4") != meta["video"]["sha256"]:
        raise RuntimeError("Output video changed")
    for path, expected in meta["source_sha256"].items():
        if "/.runtime/code/" in path and sha(path) != expected:
            raise RuntimeError(f"Native source identity changed: {path}")
    if arm == "D-Q" and meta["quantized_cache_identity"]["manifest_sha256"] != sha(cache):
        raise RuntimeError("Quantized cache identity differs")
    if observed:
        if meta["observer_source_sha256"] != sha(ROOT / "tools/bf16_observer.py"):
            raise RuntimeError("Capture observer changed; select a new matrix directory")
        if arm == "D-Q" and meta["hif4_observer_source_sha256"] != sha(ROOT / "tools/hif4_observer.py"):
            raise RuntimeError("Quantized capture observer changed")
        capture = json.loads((root / "ffn/CAPTURE_MANIFEST.json").read_text())
        if capture["status"] != "PASS_CAPTURE_STRUCTURE_ONLY" or capture["completed_scopes"] != 16:
            raise RuntimeError("Incomplete FFN capture")
        for scope in capture["scopes"]:
            if sha(scope["tensor_file"]) != scope["tensor_sha256"]:
                raise RuntimeError("Captured tensor artifact changed")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["D-BF16", "D-Q"], required=True)
    ap.add_argument("--gpu", type=int, choices=[0, 1], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-manifest", type=Path)
    ap.add_argument("--tasks", type=int, nargs="+", choices=[0, 6], default=[0, 6])
    ap.add_argument("--seeds", type=int, nargs="+", choices=[20261001, 20261002], default=[20261001, 20261002])
    args = ap.parse_args()
    if args.arm == "D-Q" and args.cache_manifest is None:
        ap.error("D-Q requires --cache-manifest")
    if len(set(args.tasks)) != len(args.tasks) or len(set(args.seeds)) != len(args.seeds):
        ap.error("Duplicate tasks/seeds")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    tag = "dbf16" if args.arm == "D-BF16" else "dq"
    name = f"MATRIX_{tag}_gpu{args.gpu}_tasks{'-'.join(map(str,args.tasks))}_seeds{'-'.join(map(str,args.seeds))}"
    lock = (args.out / f"{name}.lock").open("w")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    logs = args.out / "logs"
    logs.mkdir(exist_ok=True)
    entry = ROOT / "tools" / ("run_dense_bf16.py" if args.arm == "D-BF16" else "run_dense_hif4.py")
    result = {"status": "RUNNING", "arm": args.arm, "gpu": args.gpu,
              "purpose": "fresh fixed-input boundary assets; not video quality or speed acceptance",
              "tasks": args.tasks, "seeds": args.seeds, "records": [], "controller_sha256": sha(__file__)}
    report = args.out / f"{name}.json"

    def checkpoint():
        temp = report.with_suffix(".json.tmp")
        temp.write_text(json.dumps(result, indent=2) + "\n")
        temp.replace(report)

    checkpoint()
    try:
        for task in args.tasks:
            for seed in args.seeds:
                input_path = ROOT / f".runtime/inputs/task_{task:03d}_seed_{seed}.json"
                rows = json.loads(input_path.read_text())
                if len(rows) != 1:
                    raise RuntimeError("Expected one frozen research input")
                record = rows[0]
                prefix = f"{tag}_task{task:03d}_seed{seed}"
                destinations = {}
                for kind in ["plain", "observed"]:
                    destination = args.out / f"{prefix}_{kind}"
                    destinations[kind] = destination
                    if not destination.exists():
                        command = [sys.executable, str(ROOT / "tools/run_on_worker.py"), "--gpu", str(args.gpu), "--",
                                   "python", str(entry), "--input-manifest", str(input_path), "--out", str(destination)]
                        if args.arm == "D-Q":
                            command += ["--cache-manifest", str(args.cache_manifest.resolve())]
                        if kind == "observed":
                            command += ["--capture-ffn"]
                        print(f"START {prefix} {kind} on worker GPU{args.gpu}", flush=True)
                        with (logs / f"{prefix}_{kind}.log").open("w") as log:
                            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                    validate_run(destination, record, kind == "observed", args.arm, args.cache_manifest)
                compare = args.out / f"{prefix}_observer_noop.json"
                environment = {**os.environ, "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "CUDA_VISIBLE_DEVICES": ""}
                subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "tools/compare_dense_trajectory.py"),
                                "--plain", str(destinations["plain"]), "--observed", str(destinations["observed"]),
                                "--output", str(compare)], env=environment, check=True,
                               stdout=(logs / f"{prefix}_observer_noop.log").open("w"), stderr=subprocess.STDOUT)
                comparison = json.loads(compare.read_text())
                if comparison["status"] != "PASS_OBSERVER_NOOP_THIS_INPUT":
                    raise RuntimeError("Observer no-op comparison failed")
                result["records"].append({"task": task, "seed": seed, "input_manifest": str(input_path),
                    "input_sha256": sha(input_path), "plain": str(destinations["plain"]),
                    "observed": str(destinations["observed"]),
                    "capture_manifest_sha256": sha(destinations["observed"] / "ffn/CAPTURE_MANIFEST.json"),
                    "observer_noop": str(compare), "observer_noop_sha256": sha(compare)})
                checkpoint()
                print(f"DONE {prefix}: native capture + all four latents bitwise identical", flush=True)
        result["status"] = "COMPLETE_DATA_CAPTURE_AND_NOOP"
    except BaseException as error:
        result.update(status="FAILED_RESUMABLE_COMPLETED_INPUTS_VERIFIED_ON_RESTART", error=str(error))
        raise
    finally:
        checkpoint()


if __name__ == "__main__":
    main()
