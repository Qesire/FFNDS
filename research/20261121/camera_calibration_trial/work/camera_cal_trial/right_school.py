"""Seal, submit and retrieve the two Dense counter-direction controls."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"anchored_scale"))
import school

ROOT=school.OUT.parent/"FFNDS_camera_cal_trial"
OUT=ROOT/"right_control"
STATE=HERE/"right_submission.json"


def protocol():
    rows=[]
    for src in school.read(HERE/"protocol.json")["sources"]:
        if src["split"]!="validation":
            continue
        ident=src["record_id"]
        refs=[school.read(ROOT/"dense"/f"{ident}_{m}_s44.json") for m in ("base","camera")]
        a,b=refs
        assert a["initial"]==b["initial"] and a["image_sha256"]==b["image_sha256"]==src["image_sha256"]
        assert a["prompt_sha256"]==school.sha(src["prompt"].encode()) and b["prompt_sha256"]==school.sha((src["prompt"]+", camera pans left").encode())
        prompt=src["prompt"]+", camera pans right"
        rows.append(dict(trajectory_id=ident+"_right_s44",source_id=ident,base_prompt=src["prompt"],prompt=prompt,
                         prompt_sha256=school.sha(prompt.encode()),seed=44,image_sha256=src["image_sha256"],image_path=src["image_path"],
                         initial=a["initial"],prior_first_flows=[r["flows"][0] for r in refs]))
    assert {r["source_id"] for r in rows}=={"vbi_0905","vbi_1014"}
    return json.dumps(rows,indent=2).encode()


def submit():
    parent=HERE.parent/"anchored_scale"
    with tarfile.open(parent/"stage.tar.gz") as src:
        files={m.name:src.extractfile(m).read() for m in src if m.isfile() and m.name!="MANIFEST.sha256"}
    known={n:h for h,n in (l.split("  ") for l in (parent/"MANIFEST.sha256").read_text().splitlines())}
    assert all(school.sha(data)==known[n] for n,data in files.items())
    mark=b"self.pipe.generate(seed=42,image_path=row"
    assert files["generate.py"].count(mark)==1
    files["generate.py"]=files["generate.py"].replace(mark,b'self.pipe.generate(seed=row["seed"],image_path=row')
    files["job.sh"]=files["job.sh"].replace(b" -B generate.py",b" -B right_dense.py")
    files["right_dense.py"]=(HERE/"right_dense.py").read_bytes()
    files["right_protocol.json"]=protocol()
    for n,d in files.items():
        if n.endswith(".py"):
            compile(d,n,"exec")
    manifest="".join(f"{school.sha(data)}  {n}\n" for n,data in files.items()).encode()
    files["MANIFEST.sha256"]=manifest
    tag=school.sha(manifest)
    stage=school.BASE+"/camera_right_dense_"+tag[:16]
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode="w:gz") as bundle:
        for n,d in files.items():
            m=tarfile.TarInfo(n)
            m.size=len(d)
            bundle.addfile(m,io.BytesIO(d))
    command=(f"umask 077 && mkdir -p {stage} && cd {stage} && if [ -s JOB_ID ]; then cat JOB_ID; else "
             "mkdir .dispatch && tar xzf - && sha256sum --status -c MANIFEST.sha256 && "
             "sbatch --parsable --no-requeue -p P107-A100 -A competition --qos=qos_p107-a100 "
             "--gres=gpu:A100:1 -N1 -n1 -c4 --mem=160G -t 00:30:00 -J ffnds-cam-right -o job.log -e job.err job.sh > JOB_ID && cat JOB_ID; fi")
    job=subprocess.check_output(school.SSH+[command],input=buf.getvalue(),timeout=120).decode().strip().splitlines()[-1].split(";")[0]
    assert job.isdigit()
    state=dict(stage=stage,job=job,manifest=tag)
    school.atomic(STATE,json.dumps(state,indent=2).encode())
    school.atomic(OUT/"right_protocol.json",files["right_protocol.json"])
    school.atomic(OUT/"MANIFEST.sha256",manifest)
    print(state)


def collect(state,partial):
    stage=state["stage"]
    job=state["job"]
    name=f"progress_{job}" if partial else "RESULT"
    result=json.loads(school.command(f"cd {stage} && sha256sum --status -c MANIFEST.sha256 && cat output/{name}.json"))
    assert result["protocol_sha"]==state["manifest"] and result["job"]==job
    rows={r["trajectory_id"]:r for r in school.read(OUT/"right_protocol.json")}
    for rec in result["attempts"]:
        key=rec["trajectory_id"]
        ref=rows[key]
        data=school.command(f"cd {stage} && cat output/dense/{key}.json")
        detail=json.loads(data)
        assert detail["prompt_sha256"]==ref["prompt_sha256"] and detail["initial"]==ref["initial"]
        assert detail["seed"]==44 and rec["sha256"]==detail["video"]["mp4_sha256"]
        video=OUT/"dense"/(key+".mp4")
        if not video.exists() or school.sha(video.read_bytes())!=rec["sha256"]:
            content=school.command(f"cd {stage} && cat output/dense/{key}.mp4",180)
            assert school.sha(content)==rec["sha256"]
            school.atomic(video,content)
        school.atomic(video.with_suffix(".json"),data)
    if not partial:
        assert result["status"]=="complete" and result["nfe_this_job"]==8 and len(result["attempts"])==2
    school.atomic(OUT/(name+".json"),json.dumps(result,indent=2).encode())
    print("verified",len(result["attempts"]),result["status"])


def watch(state):
    stage=state["stage"]
    job=state["job"]
    last=("",0)
    while True:
        values=school.command(f"scontrol show job -o {job}").decode().strip().split()
        slurm=dict(token.split("=",1) for token in values if "=" in token)
        job_state=slurm["JobState"]
        files=school.command(f"cd {stage} && if [ -d output/dense ]; then ls -1 output/dense/*.json 2>/dev/null || true; fi").splitlines()
        mark=(job_state,len(files))
        if mark!=last:
            print(job,job_state,slurm["RunTime"],len(files),"receipts",flush=True)
            last=mark
            if len(files)>0:
                collect(state,True)
        if job_state=="COMPLETED":
            assert slurm["ExitCode"]=="0:0"
            collect(state,False)
            record={k:slurm[k] for k in ("JobId","JobState","ExitCode","RunTime","StartTime","EndTime","Partition","AllocTRES")}
            school.atomic(OUT/"slurm.json",json.dumps(record,indent=2).encode())
            return
        if job_state not in {"PENDING","RUNNING","COMPLETING","CONFIGURING"}:
            raise RuntimeError(job_state)
        time.sleep(25)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("submit","status","partial","collect","watch"))
    action=p.parse_args().action
    if action=="submit":
        submit()
    elif action=="status":
        print(school.status(school.read(STATE)))
    elif action=="watch":
        watch(school.read(STATE))
    else:
        collect(school.read(STATE),action=="partial")
