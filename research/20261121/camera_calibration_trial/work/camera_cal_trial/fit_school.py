"""Submit or collect the paired, high-step fold experiment after native capture."""
import argparse
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "anchored_scale"))
import school

OUT = school.OUT.parent / "FFNDS_camera_cal_trial"
STATE = HERE / "fit_submission.json"


def submit():
    capture = school.read(HERE / "capture_submission.json")
    dependency = capture["job"]
    parent = capture["stage"]
    assert dependency.isdigit()
    assert re.fullmatch(re.escape(school.BASE) + "/camera_cal_capture_[0-9a-f]{16}", parent)
    assert re.fullmatch("[0-9a-f]{64}", capture["manifest"])
    with tarfile.open(HERE.parent / "anchored_scale/stage.tar.gz") as archive:
        script = archive.extractfile("job.sh").read().decode().split("sha256sum --status", 1)[0]
    script += (
        "sha256sum --status -c MANIFEST.sha256\n"
        f"(cd {parent} && echo '{capture['manifest']}  MANIFEST.sha256' | sha256sum --status -c -"
        " && sha256sum --status -c MANIFEST.sha256)\n"
        f"{school.PY} -B fit_camera.py --capture-stage {parent} --output output\n"
        "sha256sum output/RESULT.json output/high*.json > RESULTS.sha256\n"
    )
    files = {"job.sh": script.encode(), "fit_camera.py": (HERE / "fit_camera.py").read_bytes(),
             "folded_runtime.py": (HERE.parent / "diagnose_mechanism_0951/folded_runtime.py").read_bytes()}
    for name, data in files.items():
        if name.endswith(".py"):
            compile(data, name, "exec")
    manifest = "".join(f"{school.sha(data)}  {name}\n" for name, data in files.items()).encode()
    files["MANIFEST.sha256"] = manifest
    stage = school.BASE + "/camera_cal_fit_" + school.sha(manifest)[:16]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        for name, data in files.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
    cmd = (f"umask 077 && mkdir -p {stage} && cd {stage} && if [ -s JOB_ID ]; then cat JOB_ID; else "
           "tar xzf - && sha256sum --status -c MANIFEST.sha256 && sbatch --parsable --no-requeue "
           f"--dependency=afterok:{dependency} -p P107-A100 -A competition --qos=qos_p107-a100 "
           "--gres=gpu:A100:1 -N1 -n1 -c4 --mem=48G -t 00:30:00 -J ffnds-cam-fit "
           "-o job.log -e job.err job.sh > JOB_ID && cat JOB_ID; fi")
    job = subprocess.check_output(school.SSH + [cmd], input=buf.getvalue(), timeout=120).decode().strip().splitlines()[-1].split(";")[0]
    assert job.isdigit()
    entry = dict(job=job, stage=stage, dependency=dependency, capture_stage=parent,
                 protocol_sha256=capture["protocol_sha256"], manifest=school.sha(manifest))
    school.atomic(STATE, json.dumps(entry, indent=2).encode())
    school.atomic(OUT / "fit_MANIFEST.sha256", manifest)
    print(entry)


def collect():
    entry = school.read(STATE)
    stage = entry["stage"]
    receipt = school.command(f"cd {stage} && sha256sum --status -c MANIFEST.sha256 && sha256sum --status -c RESULTS.sha256 && cat RESULTS.sha256")
    for line in receipt.decode().splitlines():
        sha, name = line.split()
        assert re.fullmatch(r"output/(RESULT|high(00|13|26|39))\.json", name)
        data = school.command(f"cd {stage} && cat {name}")
        assert school.sha(data) == sha
        school.atomic(OUT / "fit" / Path(name).name, data)
    result = school.read(OUT / "fit/RESULT.json")
    assert result["status"] == "PASS" and result["job"] == entry["job"]
    assert result["protocol_sha256"] == entry["protocol_sha256"] and len(result["layers"]) == 4
    school.atomic(OUT / "fit/RESULTS.sha256", receipt)
    print("Verified four-layer held-out gated replay and both arms.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("submit", "status", "collect"))
    action = parser.parse_args().action
    if action == "submit":
        submit()
    elif action == "status":
        print(school.status(school.read(STATE)))
    else:
        collect()
