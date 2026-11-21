"""Make independently blinded, frame-synchronous panels for the 0951 seed sweep."""

import argparse
import hashlib
import json
from pathlib import Path
import secrets
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "outputs/FFNDS_0951_seeds"
REVIEW = OUT / "review"
sys.path.insert(0, str(HERE.parent / "camera_cal_trial"))
import review_videos

SEEDS = (42, 43, 44, 45, 46)
ARMS = ("dense", "candidate")


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".pending")
    tmp.write_bytes(content.encode("utf8"))
    tmp.replace(path)


def media(seed, arm):
    if seed == 42:
        folder = ROOT / "outputs/FFNDS_anchor_scale24/camera7" / arm
        return folder / "vbi_0951_pan_left.mp4"
    return OUT / arm / f"vbi_0951_pan_left_s{seed}.mp4"


def page(ledger):
    opts = []
    # New seeds first. Seed 42 is the historical reference case.
    for seed in (*SEEDS[1:], SEEDS[0]):
        if str(seed) in ledger:
            label = f"{seed} · 历史参照" if seed == 42 else str(seed)
            opts.append(f'<option value="s{seed}_blind.mp4">{label}</option>')
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>0951 左摇 · 跨种子审片</title><style>
body{max-width:1420px;margin:0 auto;padding:clamp(16px,4vw,52px);font:17px/1.6 system-ui,sans-serif;background:#14181a;color:#edf0eb}
h1{font-size:clamp(24px,3vw,40px);margin:0}p{max-width:55em;color:#c8cfca;margin:12px 0 24px}select{font:inherit;margin-left:.5em}
video{width:100%;display:block;margin:14px 0;background:black}small{color:#c8cfca}</style>
<h1>0951 左摇 · 跨种子审片</h1>
<p>每次选择一个种子，比较同图、同提示、同初始噪声的两种生成。先看熊翻过栅栏时动作是否连贯，再看轮廓有无跳动、毛绒纹理能否延续。可分别记录 Variant 1、Variant 2、两者相近或无法判断。</p>
<label>种子<select id="choice">''' + "".join(opts) + '''</select></label>
<video id="player" controls playsinline loop preload="metadata"></video>
<small>左侧为 Variant 1，右侧为 Variant 2；各个种子的顺序独立抽取。每栏原画幅 832 × 464，81 帧，16 fps。</small>
<script>const sel=document.querySelector('#choice'),v=document.querySelector('#player');function choose(){v.src=sel.value;v.load()}sel.addEventListener('change',choose);choose();</script></html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--available", action="store_true", help="Prepare verified pairs available so far.")
    args = parser.parse_args()
    key_file = OUT / "unblind/seed_panel_order.json"
    if not key_file.exists():
        key = {str(seed): list(ARMS if secrets.randbelow(2) else reversed(ARMS)) for seed in SEEDS}
        write(key_file, json.dumps(key, indent=2) + "\n")
    key = json.loads(key_file.read_bytes())
    review_videos.REVIEW = REVIEW
    check_file = REVIEW / "media_checks.json"
    ledger = json.loads(check_file.read_bytes()) if check_file.exists() else {}
    for seed in SEEDS:
        order = key[str(seed)]
        assert sorted(order) == sorted(ARMS)
        paths = [media(seed, arm) for arm in order]
        if args.available and not all(p.is_file() and p.with_suffix(".json").is_file() for p in paths):
            continue
        dest = REVIEW / f"s{seed}_blind.mp4"
        receipts = [json.loads(path.with_suffix(".json").read_bytes()) for path in paths]
        assert receipts[0]["initial"] == receipts[1]["initial"]
        assert receipts[0]["flows"][0] == receipts[1]["flows"][0]
        hashes = [x["video"]["mp4_sha256"] for x in receipts]
        assert paths[0] != paths[1]
        previous = ledger.get(str(seed))
        if previous and dest.is_file():
            if previous["sha256"] == hashlib.sha256(dest.read_bytes()).hexdigest() and [p["sha256"] for p in previous["panes"]] == hashes:
                continue
        panes = [(f"Variant {i}", path, digest) for i, (path, digest) in enumerate(zip(paths, hashes), 1)]
        check = review_videos.compose(dest, panes)
        check["initial_sha256"] = receipts[0]["initial"]
        ledger[str(seed)] = check
        write(check_file, json.dumps(ledger, indent=2, allow_nan=False) + "\n")
        print(f"Verified seed {seed}: {dest.name}, {check['sha256']}", flush=True)
    if ledger:
        write(REVIEW / "审片.html", page(ledger))
    if not args.available:
        assert set(ledger) == {str(s) for s in SEEDS}
        assert len({v["initial_sha256"] for v in ledger.values()}) == len(SEEDS)


if __name__ == "__main__":
    main()
