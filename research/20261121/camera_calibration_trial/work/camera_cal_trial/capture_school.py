"""Submit the sealed paired capture and download verified dense videos and receipts."""
import argparse
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"anchored_scale"))
import school

OUT = school.OUT.parent/"FFNDS_camera_cal_trial"
SUBMISSION = HERE/"capture_submission.json"


def submit():
    anchor = HERE.parent/"anchored_scale"
    expected = {n:h for h,n in (l.split("  ") for l in (anchor/"MANIFEST.sha256").read_text().splitlines())}
    with tarfile.open(anchor/"stage.tar.gz") as tar:
        files = {m.name:tar.extractfile(m).read() for m in tar if m.isfile() and m.name not in {"MANIFEST.sha256","reviewed.json","cohort.json"}}
    assert all(school.sha(b) == expected[n] for n,b in files.items())
    for name in ("capture.py", "protocol.json"):
        files[name] = (HERE/name).read_bytes()
    files["job.sh"] = files["job.sh"].replace(b" -B generate.py", b" -B capture.py")
    assert b" -B capture.py" in files["job.sh"]
    for n,b in files.items():
        if n.endswith(".py"):
            compile(b,n,"exec")
    manifest = "".join(f"{school.sha(b)}  {n}\n" for n,b in files.items()).encode()
    files["MANIFEST.sha256"] = manifest
    stage = school.BASE+"/camera_cal_capture_"+school.sha(manifest)[:16]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for n,b in files.items():
            m = tarfile.TarInfo(n)
            m.size = len(b)
            m.mode = 0o700 if n == "job.sh" else 0o600
            tar.addfile(m,io.BytesIO(b))
    command = (f"umask 077 && mkdir -p {stage} && cd {stage} && if [ -s JOB_ID ]; then cat JOB_ID; else "
               "tar xzf - && sha256sum --status -c MANIFEST.sha256 && "
               "sbatch --parsable --no-requeue -p P107-A100 -A competition --qos=qos_p107-a100 --gres=gpu:A100:1 "
               "-N1 -n1 -c4 --mem=160G -t 01:45:00 --job-name=ffnds-cam-cap -o job.log -e job.err job.sh > JOB_ID && cat JOB_ID; fi")
    job = subprocess.check_output(school.SSH+[command], input=buf.getvalue(), timeout=120).decode().strip().splitlines()[-1].split(";")[0]
    assert job.isdigit()
    entry = dict(stage=stage,job=job,manifest=school.sha(manifest),protocol_sha256=school.sha(files["protocol.json"]))
    school.atomic(SUBMISSION, json.dumps(entry,indent=2).encode())
    school.atomic(OUT/"capture_MANIFEST.sha256",manifest)
    print(entry)


def status(state):
    stage, job = state["stage"], state["job"]
    print(school.command(f"cd {stage} && squeue -j {job} -h -o '%i %T %M %R'; cd {stage} && tail -c 1200 job.log job.err 2>/dev/null").decode(errors="replace"))


def collect(state, partial=False):
    stage = state["stage"]
    index = json.loads(school.command(f"cd {stage} && cat output/CAPTURE_INDEX.json"))
    assert index["protocol_sha256"] == state["protocol_sha256"] and index["manifest_sha256"] == state["manifest"]
    if not partial:
        assert index["status"] == "complete" and len(index["sites"]) == 52 and index["nfe"] == 52
        assert len(index["trajectories"]) == 13
    assert len(index["sites"]) == 4*len(index["trajectories"])
    replay = {}
    ref = school.read(HERE.parent/"video_joint_run/stage/scope.json")["v2_reference_dir"]
    for row in index["trajectories"]:
        stem = row["trajectory_id"]
        assert re.fullmatch(r"vbi_[0-9]{4}_(camera|base)_s[0-9]{2}", stem)
        path = OUT/"dense"/(stem+".mp4")
        if not path.is_file() or school.sha(path.read_bytes()) != row["video"]["mp4_sha256"]:
            data = school.command(f"cd {stage} && cat output/dense/{stem}.mp4",160)
            assert school.sha(data) == row["video"]["mp4_sha256"]
            school.atomic(path,data)
        school.atomic(path.with_suffix(".json"), json.dumps(row,indent=2).encode())
        if row["seed"] == 42:
            source = row["source_id"]
            assert source in {"vbi_0032", "vbi_0304", "vbi_0456"}
            previous = json.loads(school.command(f"cat {ref}/output/CAPTURE_{source}.json"))
            replay[source] = dict(initial=row["initial"] == previous["initial_noise"]["sha256"],
                                  latents=[a == b["sha256"] for a,b in zip(row["latents"],previous["latent_digests"],strict=True)],
                                  decoded_video=row["video"]["decoded_rgb24_sha256"] == previous["decoded_rgb24_sha256"])
    school.atomic(OUT/"CAPTURE_INDEX.json",json.dumps(index,indent=2).encode())
    school.atomic(OUT/"capture_seed42_replay.json",json.dumps(replay,indent=2).encode())
    by_source = {}
    for row in index["trajectories"]:
        by_source.setdefault(row["source_id"], {})[(row["seed"],row["condition"])] = row
    checks = {}
    for source, group in by_source.items():
        seed = 43 if (42,"base") in group else 44
        item = {}
        if (42,"base") in group and (43,"base") in group:
            item["base42_differs_from_base43"] = group[42,"base"]["initial"] != group[43,"base"]["initial"]
        if (seed,"base") in group and (seed,"camera") in group:
            item["paired_initial_equal"] = group[seed,"base"]["initial"] == group[seed,"camera"]["initial"]
            item["paired_first_flow_differs"] = group[seed,"base"]["flows"][0] != group[seed,"camera"]["flows"][0]
        assert all(item.values()), (source,item)
        checks[source] = item
    school.atomic(OUT/"capture_seed_pairs.json",json.dumps(checks,indent=2).encode())
    print("Verified videos:", len(index["trajectories"]), "indexed teacher panels:", len(index["sites"]), "status:", index["status"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("submit","status","partial","collect"))
    action = parser.parse_args().action
    if action == "submit": submit()
    elif action == "status": status(school.read(SUBMISSION))
    else: collect(school.read(SUBMISSION), partial=action=="partial")
