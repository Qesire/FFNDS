"""Check a reviewed, single-job FFNDS scope and retain its run/exit evidence."""

import argparse
from datetime import datetime, timezone
from hashlib import file_digest
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def checked(path, expected):
    actual = sha(path)
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch: {path}: {actual} != {expected}")


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write(path, data):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")


def run(stage, scope_sha):
    stage = stage.resolve()
    scope_path = stage / "scope.json"
    checked(scope_path, scope_sha)
    scope = json.loads(scope_path.read_bytes())
    active = json.loads((stage / "ACTIVE_RESEARCH.json").read_bytes())
    review_path = stage / "REVIEW.json"
    checked(review_path, active["review_sha256"])
    review = json.loads(review_path.read_bytes())
    if not (
        active["state"] == "AUTHORIZED_FOR_ONE_JOB"
        and active["scope_sha256"] == scope_sha
        and active["scope_id"] == scope["scope_id"] == review["scope_id"]
        and review["scope_sha256"] == scope_sha
        and active["reviewer_agent_id"] == review["reviewer_agent_id"] != review["author_agent_id"]
        and review["recheck_disposition"] == "APPROVE"
        and active["max_jobs"] == 1
        and active["gpus_per_job"] == scope["resources"]["gpus"]
        and active["session_max_gpus"] >= active["gpus_per_job"]
    ):
        raise RuntimeError("review and ACTIVE scopes differ")
    if Path(scope["stage"]).resolve() != stage:
        raise RuntimeError("stage location changed")
    if not Path(sys.executable).samefile(scope["python_executable"]):
        raise RuntimeError("Python executable changed")
    checked(Path(sys.executable), scope["python_executable_sha256"])
    for name, digest in scope["code_sha256"].items():
        checked(stage / name, digest)
    for asset in scope["inputs"]:
        checked(Path(asset["path"]), asset["sha256"])
    job_id = os.environ["SLURM_JOB_ID"]
    if os.environ["SLURM_JOB_PARTITION"] != scope["resources"]["partition"]:
        raise RuntimeError("Slurm partition changed")
    if int(os.environ["SLURM_CPUS_PER_TASK"]) != scope["resources"]["cpus"]:
        raise RuntimeError("Slurm CPU count changed")
    if scope["resources"]["gpus"]:
        import torch
        if torch.cuda.device_count() != scope["resources"]["gpus"] or "A100" not in torch.cuda.get_device_name():
            raise RuntimeError("A100 allocation changed")
        if torch.__version__ != scope["torch_version"]:
            raise RuntimeError("Torch build changed")
    write(stage / "RUN.json", {
        "scope_id": scope["scope_id"], "scope_sha256": scope_sha,
        "review_sha256": active["review_sha256"], "slurm_job_id": job_id,
        "host": socket.gethostname(), "start_utc": utc(), "argv": scope["command"],
    })
    code = 1
    outputs = {}
    try:
        completed = subprocess.run([sys.executable, "-B", *scope["command"]], cwd=stage, check=False)
        code = completed.returncode
        if code == 0:
            outputs = {name: sha(stage / name) for name in scope["expected_outputs"]}
    except Exception:
        code = 1
        raise
    finally:
        write(stage / "EXIT.json", {
            "scope_id": scope["scope_id"], "slurm_job_id": job_id, "exit_code": code,
            "end_utc": utc(), "output_sha256": outputs,
        })
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("scope_sha256")
    arguments = parser.parse_args()
    sys.exit(run(arguments.stage, arguments.scope_sha256))

