"""Seal the four-site A/B generation against the actual fitted weights and Dense receipts."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "anchored_scale"))
import school

OUT = school.OUT.parent / "FFNDS_camera_cal_trial"
STATE = HERE / "render_ab_submission.json"


def submit():
    fit = school.read(HERE / "fit_submission.json")
    capture = school.read(HERE / "capture_submission.json")
    assert fit["dependency"] == capture["job"] and fit["protocol_sha256"] == capture["protocol_sha256"]
    index_sha = school.read(OUT / "fit/RESULT.json")["capture_index_sha256"]
    binding = dict(protocol_sha256=fit["protocol_sha256"], fitted=fit["stage"], capture=capture["stage"],
                   fit_result_sha256=school.sha((OUT / "fit/RESULT.json").read_bytes()), capture_index_sha256=index_sha)
    folder = HERE.parent / "anchored_scale"
    with tarfile.open(folder / "stage.tar.gz") as tar:
        files = {m.name: tar.extractfile(m).read() for m in tar if m.isfile() and m.name != "MANIFEST.sha256"}
    registered = {n:h for h,n in (s.split("  ") for s in (folder / "MANIFEST.sha256").read_text().splitlines())}
    assert all(registered[k] == school.sha(v) for k, v in files.items())
    # Earlier generation hardcoded seed 42. Use each row's sealed seed 44.
    marker = b"self.pipe.generate(seed=42,image_path=row"
    assert files["generate.py"].count(marker) == 1
    files["generate.py"] = files["generate.py"].replace(marker, b'self.pipe.generate(seed=row["seed"],image_path=row')
    files["job.sh"] = files["job.sh"].replace(b" -B generate.py", b" -B render_ab.py")
    files["render_ab.py"] = (HERE / "render_ab.py").read_bytes()
    files["protocol.json"] = (HERE / "protocol.json").read_bytes()
    files["render_bindings.json"] = json.dumps(binding, indent=2).encode()
    for name, data in files.items():
        if name.endswith(".py"):
            compile(data, name, "exec")
    manifest = "".join(f"{school.sha(data)}  {name}\n" for name, data in files.items()).encode()
    files["MANIFEST.sha256"] = manifest
    stage = school.BASE + "/camera_cal_video_" + school.sha(manifest)[:16]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in files.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            tar.addfile(entry, io.BytesIO(data))
    cmd = (f"umask 077 && mkdir -p {stage} && cd {stage} && if [ -s JOB_ID ]; then cat JOB_ID; else "
           "tar xzf - && sha256sum --status -c MANIFEST.sha256 && "
           "sbatch --parsable --no-requeue -p P107-A100 -A competition --qos=qos_p107-a100 "
           "--gres=gpu:A100:1 -N1 -n1 -c4 --mem=160G -t 01:30:00 -J ffnds-four-ab -o job.log -e job.err job.sh > JOB_ID && cat JOB_ID; fi")
    job = subprocess.check_output(school.SSH + [cmd], input=buffer.getvalue(), timeout=120).decode().strip().splitlines()[-1].split(";")[0]
    assert job.isdigit()
    entry = dict(stage=stage, job=job, manifest=school.sha(manifest), protocol_sha256=binding["protocol_sha256"])
    school.atomic(STATE, json.dumps(entry, indent=2).encode())
    school.atomic(OUT / "render_ab_MANIFEST.sha256", manifest)
    print(entry, flush=True)


def collect():
    state = school.read(STATE)
    stage = state["stage"]
    result = json.loads(school.command(f"cd {stage} && sha256sum --status -c MANIFEST.sha256 && cat output/RESULT.json"))
    assert result["status"] == "complete" and result["job"] == state["job"] and result["protocol_sha"] == state["manifest"]
    assert result["nfe_this_job"] == 32 and len(result["attempts"]) == 8
    valid = {i["trajectory_id"] for i in school.read(HERE / "protocol.json")["trajectories"] if i["split"]=="validation"}
    expected = {(arm,t) for arm in ("A","B") for t in valid}
    assert {(r["arm"],r["trajectory"]) for r in result["attempts"]} == expected
    out = OUT / "render_ab"
    for arm, t in sorted(expected):
        name = f"{arm}/{t}"
        data = school.command(f"cd {stage} && cat output/{name}.json")
        receipt = json.loads(data)
        assert receipt["source"] == t and receipt["group"] == arm and receipt["job"] == state["job"]
        assert receipt["initial_and_native_prefix_matched"] and receipt["seed"] == 44
        video = out / (name + ".mp4")
        sha = next(r["sha256"] for r in result["attempts"] if (r["arm"],r["trajectory"])==(arm,t))
        assert sha == receipt["video"]["mp4_sha256"]
        if not video.exists() or school.sha(video.read_bytes()) != sha:
            media = school.command(f"cd {stage} && cat output/{name}.mp4", 180)
            assert school.sha(media) == sha
            school.atomic(video, media)
        school.atomic(out / (name + ".json"), data)
    school.atomic(out / "RESULT.json", json.dumps(result, indent=2).encode())
    print("Verified all eight closed-loop MP4 files and native-prefix receipts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("submit", "status", "collect"))
    action = parser.parse_args().action
    if action == "submit":
        submit()
    elif action == "collect":
        collect()
    else:
        print(school.status(school.read(STATE)))
