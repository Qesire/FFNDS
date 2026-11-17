"""Verify the reviewed Dense bundle and allocation before any model load."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
from datetime import datetime, timezone


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def fail_on_termination(signum, frame):
    raise TimeoutError(f"termination signal {signum}")


def main(scope_sha):
    stage = Path.cwd()
    path = stage / "scope.json"
    scope = json.loads(path.read_text(encoding="utf-8"))
    review_path = stage / "REVIEW.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    active = json.loads((stage / "ACTIVE_RESEARCH.json").read_text(encoding="utf-8"))[scope["scope_id"].lower()]
    if (digest(path) != scope_sha or active["scope_sha256"] != scope_sha
            or not active["runnable"] or active["review_sha256"] != digest(review_path)
            or review["scope_sha256"] != scope_sha or review["recheck_disposition"] != "APPROVE"):
        raise RuntimeError("review or ACTIVE identity mismatch")
    ref = Path(scope["v2_reference_dir"])
    if digest(ref / "runtime_guard.py") != scope["reference_sha256"]["runtime_guard.py"]:
        raise RuntimeError("runtime guard source mismatch")
    sys.path.insert(0, str(ref))
    from runtime_guard import require_compute_allocation

    allocation = require_compute_allocation()
    job = dict(re.findall(r"(\w+)=([^\s]+)", allocation["scheduler_job_record"]))
    wall = re.fullmatch(r"(\d+):(\d+):(\d+)", job["TimeLimit"])
    budget = scope["resources"]
    if wall is None or sum(int(field) * factor for field, factor in zip(wall.groups(), (3600, 60, 1))) > budget["wall_seconds"]:
        raise RuntimeError("runtime wall bound exceeded")
    if job["Partition"] != "P107-A100" or allocation["memory_mib"] > budget["host_memory_gib"] * 1024:
        raise RuntimeError("partition or memory allocation mismatch")
    if shutil.disk_usage(stage).free < budget["min_free_disk_gib"] * 1024 ** 3:
        raise RuntimeError("output filesystem lacks free capacity")
    executable = Path(sys.executable)
    expected_python = Path(scope["python_executable"])
    if not executable.samefile(expected_python) or digest(executable) != scope["python_executable_sha256"]:
        raise RuntimeError("Python environment mismatch")
    with (stage / "receipt" / "RUN.json").open("x", encoding="utf-8") as receipt:
        json.dump({"scope_sha256": scope_sha, "job_id": os.environ["SLURM_JOB_ID"],
                   "allocation": allocation, "start_utc": datetime.now(timezone.utc).isoformat()}, receipt, indent=2)
    signal.signal(signal.SIGTERM, fail_on_termination)
    import torch

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != "NVIDIA A100-SXM4-80GB":
        raise RuntimeError("one A100 80 GB required")
    memory = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, budget["gpu_memory_gib"] * 1024 ** 3 / memory))
    from stage1_dense_runner import run

    with torch.inference_mode():
        run(path)


if __name__ == "__main__":
    main(sys.argv[1])
