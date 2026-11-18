#!/bin/bash
set -euo pipefail
quality="$1"
entry="$2"
python="/public/home/scc/$USER/miniconda3/envs/hif4/bin/python3.11"
test -f "$entry"
test "$(squeue -u "$USER" -h -o '%j' | awk '/^ffnds-state-p2-/{n++} END{print n+0}')" -eq 0

submit() {
    local dim="$1" dependency="$2" stage="$quality/$1" scope_sha job_id
    scope_sha=$("$python" -B - "$stage" <<'PY'
from hashlib import file_digest
from pathlib import Path
import json
import sys

root = Path(sys.argv[1])

def sha(p):
    with Path(p).open("rb") as file:
        return file_digest(file, "sha256").hexdigest()

def read(p):
    return json.loads(p.read_bytes())

scope = read(root / "scope.json")
review = read(root / "REVIEW.json")
active = read(root / "ACTIVE_RESEARCH.json")[scope["active_key"]]
digest = sha(root / "scope.json")
assert review["recheck_disposition"] == "APPROVE" and active["runnable"]
assert review["scope_sha256"] == active["scope_sha256"] == digest
assert active["review_sha256"] == sha(root / "REVIEW.json")
assert review["reviewer_agent_id"] not in scope["authors"]
assert active["session_max_gpus"] == 2
assert scope["comparison"] == "STATE_OPERATOR_DEV3"
assert scope["remote_stage"] == str(root)
r = scope["resources"]
assert (r["gpus"], r["cpu_cores"], r["host_memory_gib"], r["wall_seconds"]) == (1, 4, 160, 7200)
assert (r["partition"], r["account"], r["qos"]) == ("P107-A100", "competition", "qos_p107-a100")
assert all(sha(p) == h for p, h in scope["implementation_sha256"].items())
print(digest)
PY
)
    mkdir "$stage/dispatch_once"
    local -a prereq=()
    if [[ -n "$dependency" ]]; then
        prereq+=("--dependency=afterany:$dependency")
    fi
    if job_id=$(sbatch --parsable --no-requeue --job-name="ffnds-state-quality-$dim" \
            -p P107-A100 -A competition --qos=qos_p107-a100 --gres=gpu:A100:1 \
            --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=160G --time=02:00:00 \
            --output="$stage/slurm-%j.out" --error="$stage/slurm-%j.err" \
            "${prereq[@]}" "$entry" "$stage" "$scope_sha"); then
        printf '%s\n' "$job_id" > "$stage/dispatch_once/JOB_ID"
        printf '{"job_id":"%s","scope_sha256":"%s","previous_lane_job":"%s","submitted_utc":"%s"}\n' \
            "$job_id" "$scope_sha" "$dependency" "$(date -u +%FT%TZ)" > "$stage/SUBMITTED.json"
        printf '%s\n' "$job_id"
    else
        rmdir "$stage/dispatch_once"
        return 1
    fi
}

# Build two dependency chains. An independent metric can run after either exit of its predecessor.
declare -a lane0=(dynamic_degree i2v_background background_consistency aesthetic_quality temporal_flickering)
declare -a lane1=(i2v_subject imaging_quality subject_consistency motion_smoothness)
last0=
last1=
for index in "${!lane0[@]}"; do
    dim0="${lane0[$index]}"
    dim1="${lane1[$index]:-}"
    last0=$(submit "$dim0" "$last0")
    printf '%s %s\n' "$dim0" "$last0"
    if [[ -n "$dim1" ]]; then
        last1=$(submit "$dim1" "$last1")
        printf '%s %s\n' "$dim1" "$last1"
    fi
done

