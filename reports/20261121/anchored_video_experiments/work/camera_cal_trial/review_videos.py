"""Make frame-checked reference and blinded review videos for the two validation images."""
import argparse
import json
from pathlib import Path
import re
import secrets
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1]/"outputs/FFNDS_camera_cal_trial"
REVIEW = OUT/"review"
sys.path.insert(0,str(HERE.parent/"zero_video/visual"))
from review import ffmpeg, source_info


def compose(path, panes):
    command, filters, sources = [], [], []
    for i, (label, file, known_sha) in enumerate(panes):
        info = source_info(file)
        assert info["sha256"] == known_sha
        command.extend(["-i", file])
        filters.append(r"[{}:v]pad=832:496:0:32:color=black,drawtext=fontfile='C\:/Windows/Fonts/arial.ttf':text='{}':x=14:y=4:fontsize=20:fontcolor=white[p{}]".format(i,label,i))
        sources.append(dict(label=label, sha256=info["sha256"]))
    filters.append("".join(f"[p{i}]" for i in range(len(panes)))+f"hstack=inputs={len(panes)}[out]")
    REVIEW.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=REVIEW) as tmp:
        target = Path(tmp)/"composite.mp4"
        ffmpeg([*command, "-filter_complex_threads", "2", "-filter_complex", ";".join(filters),
                "-map","[out]","-an","-fps_mode","passthrough","-c:v","libx264","-profile:v","high",
                "-qp","1","-preset","veryfast","-pix_fmt","yuv420p","-threads","4","-movflags","+faststart",target])
        result = source_info(target, 832*len(panes), 496)
        for i, (_, file, _) in enumerate(panes):
            expr = f"[0:v]crop=832:464:{832*i}:32[c];[c][1:v]psnr=stats_file=stats.txt[v]"
            ffmpeg(["-i",target,"-i",file,"-filter_complex",expr,"-map","[v]","-f","null","-"], cwd=tmp)
            rows = (Path(tmp)/"stats.txt").read_text().splitlines()
            values = [float(re.search(r"\bpsnr_avg:(\S+)", line)[1]) for line in rows]
            assert len(values) == 81 and min(values) >= 44
            sources[i]["min_frame_psnr_db"] = round(min(values),2) if min(values)<float("inf") else "identical"
        target.replace(path)
    result["panes"] = sources
    return result


def media(file):
    row = json.loads(file.with_suffix(".json").read_bytes())
    return (file,row["video"]["mp4_sha256"])


def run(arms, right=False, available=False):
    ledger, identities = {}, {}
    for source in ("0905","1014"):
        stem = f"vbi_{source}"
        dense = [OUT/"dense"/f"{stem}_{condition}_s44.mp4" for condition in ("base","camera")]
        if arms is None:
            name = source+"_dense_base_vs_pan_left.mp4"
            labels = ["Dense - original prompt","Dense - camera pans left"]
            if right:
                name = source+"_dense_base_left_right.mp4"
                labels.append("Dense - camera pans right")
                dense.append(OUT/"right_control"/"dense"/f"{stem}_right_s44.mp4")
                if available and (not dense[-1].is_file() or not dense[-1].with_suffix(".json").is_file()):
                    continue
            ledger[name] = compose(REVIEW/name, [(label,*media(file)) for label,file in zip(labels,dense)])
            continue
        receipt = REVIEW/"blind_key.json"
        saved = json.loads(receipt.read_bytes()) if receipt.exists() else {}
        for condition, file in zip(("base","camera"),dense):
            key = stem+"_"+condition+"_s44"
            if available and any(not p.with_suffix(".json").is_file() or not p.is_file() for p in arms[key].values()):
                continue
            order = saved.get(key)
            if order is None:
                order = ["B","A"] if secrets.randbelow(2) else ["A","B"]
            assert sorted(order) == ["A","B"]
            identities[key] = order
            panes = [("Dense",*media(file))]
            for i, arm in enumerate(order,1):
                panes.append((f"Variant {i}",*media(Path(arms[key][arm]))))
            name = source+"_"+condition+"_blind.mp4"
            ledger[name] = compose(REVIEW/name, panes)
    suffix = "blind" if arms is not None else "dense_directions" if right else "dense"
    if arms is not None:
        (REVIEW/"blind_key.json").write_text(json.dumps(identities,indent=2)+"\n")
    (REVIEW/(suffix+"_media_checks.json")).write_text(json.dumps(ledger,indent=2)+"\n")
    print("Verified",len(ledger),"review clips")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--closed-loop",action="store_true",help="Use locally verified A/B videos in render_ab.")
    group.add_argument("--right",action="store_true",help="Include the right-pan Dense control.")
    parser.add_argument("--available",action="store_true")
    args = parser.parse_args()
    arms = None
    if args.closed_loop:
        arms = {}
        for source in ("0905","1014"):
            for condition in ("base","camera"):
                stem = f"vbi_{source}_{condition}_s44"
                arms[stem] = {k:OUT/"render_ab"/k/(stem+".mp4") for k in ("A","B")}
    run(arms,args.right,args.available)
