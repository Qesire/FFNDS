"""Build a per-seed paired-trajectory audit from archived MP4s and receipts."""
from hashlib import sha256
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]/"outputs"
OUT=ROOT/"FFNDS_0951_seeds"


def read(path):
    return json.loads(path.read_bytes())


def main():
    batch=read(OUT/"RESULT.json")
    protocol=read(OUT/"seed_protocol.json")
    trials=[]
    for seed in (42,43,44,45,46):
        path=ROOT/"FFNDS_anchor_scale24/camera7" if seed==42 else OUT
        stem="vbi_0951_pan_left"+("" if seed==42 else f"_s{seed}")
        dense,old=[read(path/g/(stem+".json")) for g in ("dense","candidate")]
        videos={}
        for arm,r in (("dense",dense),("candidate",old)):
            mp4=path/arm/(stem+".mp4")
            h=sha256(mp4.read_bytes()).hexdigest()
            assert h==r["video"]["mp4_sha256"] and r["video"]["frames"]==81
            videos[arm]=dict(mp4_sha256=h, decoded_rgb24_sha256=r["video"]["decoded_rgb24_sha256"],wall_seconds=r["wall"])
            if seed!=42:
                assert r["job"]==batch["job"] and r["seed"]==seed
                assert all(r[k]==protocol["rows"][0][k] for k in ("prompt_sha256","image_sha256"))
        same_latents=[a==b for a,b in zip(dense["latents"],old["latents"],strict=True)]
        same_predictions=[a==b for a,b in zip(dense["flows"],old["flows"],strict=True)]
        assert dense["initial"]==old["initial"] and same_predictions==same_latents==[True,False,False,False]
        assert old["counters"]["hits"]==old["counters"]["anchors"]==80
        row=dict(seed=seed,job=dense["job"],pair_initial_sha256=dense["initial"],same_step_predictions=same_predictions,
                 same_step_post_update_latents=same_latents,old_anchor_hits=old["counters"]["hits"],videos=videos)
        if seed in (43,44):
            prior=("vbi_0032_base_s43" if seed==43 else "vbi_0905_base_s44")+".json"
            ref=read(ROOT/"FFNDS_camera_cal_trial/dense"/prior)
            assert ref["seed"]==seed and ref["initial"]==dense["initial"] and ref["image_sha256"]!=dense["image_sha256"]
            row["independent_same_seed_reference"]=prior
        trials.append(row)
    assert len({r["pair_initial_sha256"] for r in trials})==5
    assert batch["nfe_this_job"]==32 and batch["status"]=="complete"
    result=dict(job=batch["job"], new_seeds=[43,44,45,46], historical_seed=42, input_sha256=protocol["rows"][0]["image_sha256"],
                prompt=protocol["rows"][0]["prompt"], trials=trials)
    (OUT/"paired_trajectory_checks.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print("Verified 5 paired initial hashes and traces, 10 videos; 4 new seeds, 1 archive.")


if __name__=="__main__":
    main()
