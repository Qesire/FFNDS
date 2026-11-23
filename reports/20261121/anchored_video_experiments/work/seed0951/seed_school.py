"""Seal, run and collect the matched 0951 left-pan seed study."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/"anchored_scale"
sys.path.insert(0,str(PARENT))
import school

OUT=school.OUT.parent/"FFNDS_0951_seeds"
STATE=HERE/"submission.json"


def register():
    old=school.OUT/"camera7"
    row=next(r for r in school.read(old/"COHORT.json") if r["record_id"]=="vbi_0951_pan_left")
    assert row["prompt"]=="A teddy bear is climbing over a wooden fence, camera pans left"
    assert school.sha(row["prompt"].encode()) == row["prompt_sha256"] and row["seed"] == 42
    old_runs=[school.read(old/arm/"vbi_0951_pan_left.json") for arm in ("dense","candidate")]
    for arm,record in zip(("dense","candidate"),old_runs):
        assert school.sha((old/arm/"vbi_0951_pan_left.mp4").read_bytes())==record["video"]["mp4_sha256"]
    assert old_runs[0]["initial"] == old_runs[1]["initial"]
    return dict(reference=dict(seed=42,initial=old_runs[0]["initial"],mp4_sha256=[r["video"]["mp4_sha256"] for r in old_runs]),
                reference_config=school.sha((old/"EFFECTIVE_CONFIG.json").read_bytes()),
                rows=[dict(row, seed=s, record_id=f"vbi_0951_pan_left_s{s}") for s in (43,44,45,46)])


def submit():
    with tarfile.open(PARENT/"stage.tar.gz") as archive:
        files={m.name:archive.extractfile(m).read() for m in archive if m.isfile() and m.name!="MANIFEST.sha256"}
    known={n:h for h,n in (l.split("  ") for l in (PARENT/"MANIFEST.sha256").read_text().splitlines())}
    assert all(school.sha(v)==known[n] for n,v in files.items())
    old_camera={n:h for h,n in (l.split("  ") for l in (school.OUT/"camera7/MANIFEST.sha256").read_text().splitlines())}
    for n in ("generate.py","exact_port.py","fast_hit.py","native_joint.py","base.json","fused_ops.py"):
        assert known[n] == old_camera[n]
    marker=b"self.pipe.generate(seed=42,image_path=row"
    assert files["generate.py"].count(marker)==1
    files["generate.py"]=files["generate.py"].replace(marker,b'self.pipe.generate(seed=row["seed"],image_path=row')
    files["job.sh"]=files["job.sh"].replace(b" -B generate.py", b" -B seed_run.py")
    files["seed_run.py"]=(HERE/"seed_run.py").read_bytes()
    files["seed_protocol.json"]=json.dumps(register(),indent=2).encode()
    for n,v in files.items():
        if n.endswith(".py"):
            compile(v,n,"exec")
    manifest="".join(f"{school.sha(v)}  {n}\n" for n,v in files.items()).encode()
    files["MANIFEST.sha256"]=manifest
    tag=school.sha(manifest)
    dst=school.BASE+"/seed0951_"+tag[:16]
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode="w:gz") as archive:
        for name,data in files.items():
            entry=tarfile.TarInfo(name)
            entry.size=len(data)
            archive.addfile(entry,io.BytesIO(data))
    cmd=(f"umask 077 && mkdir -p {dst} && cd {dst} && if [ -s JOB_ID ]; then cat JOB_ID; else "
         "mkdir .dispatch && tar xzf - && sha256sum --status -c MANIFEST.sha256 && "
         "sbatch --parsable --no-requeue -p P107-A100 -A competition --qos=qos_p107-a100 "
         "--gres=gpu:A100:1 -N1 -n1 -c4 --mem=160G -t 01:00:00 -J ffnds-seed0951 -o job.log -e job.err job.sh > JOB_ID && cat JOB_ID; fi")
    job=subprocess.check_output(school.SSH+[cmd],input=buf.getvalue(),timeout=120).decode().strip().splitlines()[-1].split(";")[0]
    assert job.isdigit()
    state=dict(job=job,manifest=tag,stage=dst)
    school.atomic(STATE,json.dumps(state,indent=2).encode())
    school.atomic(OUT/"seed_protocol.json",files["seed_protocol.json"])
    school.atomic(OUT/"MANIFEST.sha256",manifest)
    print(state,flush=True)


def collect(state,complete):
    job,stage=state["job"],state["stage"]
    name="RESULT" if complete else "progress_"+job
    rr=json.loads(school.command(f"cd {stage} && sha256sum --status -c MANIFEST.sha256 && cat output/{name}.json"))
    assert rr["job"]==job and rr["protocol_sha"]==state["manifest"]
    for pair in rr["attempts"]:
        key=pair["source"]
        assert key==f"vbi_0951_pan_left_s{pair['seed']}" and pair["seed"] in (43,44,45,46)
        for arm in ("dense","candidate"):
            dest=OUT/arm/(key+".mp4")
            local_record=dest.with_suffix(".json")
            sha=pair["sha256"][arm]
            if local_record.exists() and dest.exists() and school.sha(dest.read_bytes())==sha:
                continue
            data=school.command(f"cd {stage} && cat output/{arm}/{key}.json")
            record=json.loads(data)
            assert record["video"]["mp4_sha256"]==sha and record["seed"]==pair["seed"] and record["initial"]==pair["initial"]
            video=school.command(f"cd {stage} && cat output/{arm}/{key}.mp4",180)
            assert school.sha(video)==sha
            school.atomic(dest,video)
            school.atomic(local_record,data)
    if complete:
        assert rr["status"]=="complete" and rr["nfe_this_job"]==32 and len(rr["attempts"])==4
        config=school.command(f"cd {stage} && cat output/EFFECTIVE_CONFIG.json")
        assert school.sha(config)==register()["reference_config"]
        school.atomic(OUT/"EFFECTIVE_CONFIG.json",config)
    school.atomic(OUT/(name+".json"),json.dumps(rr,indent=2).encode())
    return len(rr["attempts"])


def watch():
    st=school.read(STATE)
    stage,job=st["stage"],st["job"]
    previous=None
    while True:
        slurm=dict(t.split("=",1) for t in school.command(f"scontrol show job -o {job}").decode().split() if "=" in t)
        state=slurm["JobState"]
        pairs=int(school.command(f"cd {stage} && if [ -f output/progress_{job}.json ]; then python3 -c \"import json;print(len(json.load(open('output/progress_{job}.json'))['attempts']))\"; else echo 0; fi"))
        if previous!=(state,pairs):
            if pairs:
                collect(st,False)
            print(job,state,slurm["RunTime"],"pairs retrieved",pairs,flush=True)
            previous=(state,pairs)
        if state=="COMPLETED":
            assert slurm["ExitCode"]=="0:0"
            collect(st,True)
            school.atomic(OUT/"slurm.json",json.dumps({k:slurm[k] for k in ("JobId","JobState","RunTime","ExitCode","StartTime","EndTime","AllocTRES")},indent=2).encode())
            break
        if state not in {"RUNNING","COMPLETING","CONFIGURING","PENDING"}:
            raise RuntimeError(school.status(st))
        time.sleep(20)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["submit","watch","status"])
    action=parser.parse_args().action
    if action=="submit":
        submit()
    elif action=="watch":
        watch()
    else:
        print(school.status(school.read(STATE)))
