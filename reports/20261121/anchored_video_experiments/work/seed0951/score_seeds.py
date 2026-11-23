"""Reapply the frozen 0951 foreground diagnostic and signed background fit per seed."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2 as cv
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"vbench_scale24"))
sys.path.insert(0,str(HERE.parent/"camera_cal_trial"))
import diagnose0951 as fg
import signed_background as bg
import measure_camera

ROOT=HERE.parents[1]
OUT=ROOT/"outputs/FFNDS_0951_seeds"
OLD=ROOT/"outputs/FFNDS_anchor_scale24/camera7"
SCORES=OUT/"scores"
KEYS=("body_band_correlation","body_compensated_flicker","edge_normal_accel_p75",
      "body_area_pct","body_valid_pct","edge_valid_pct")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,obj):
    tmp=path.with_suffix(".pending")
    tmp.write_text(json.dumps(obj,indent=1,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
    tmp.replace(path)


def score(seed,arm):
    files=OLD if seed==42 else OUT
    suffix="" if seed==42 else f"_s{seed}"
    path=files/arm/("vbi_0951_pan_left"+suffix+".mp4")
    note=json.loads(path.with_suffix(".json").read_bytes())
    if seed!=42:
        assert note["seed"]==seed
    cv.setNumThreads(2)
    cv.setRNGSeed(42)
    result,masks=fg.run(path)
    cv.setRNGSeed(42)
    background=bg.score(path)
    assert result["sha256"]==background["sha256"]==note["video"]["mp4_sha256"]
    result.update(seed=seed,arm=arm,background=background,scoring_code_sha256=
                  {p.name:sha(p) for p in map(Path,(fg.__file__,bg.__file__,measure_camera.__file__))})
    if seed==42:
        historic=json.loads((OLD/"diagnosis"/f"pan_left_{arm}.json").read_bytes())
        assert historic["sha256"]==result["sha256"]
        dr=max(abs(result["all"][k]-historic["all"][k]) for k in historic["all"])
        assert dr<1e-5,(arm,dr)
        result["historical_foreground_median_max_abs_difference"]=dr
    SCORES.mkdir(exist_ok=True,parents=True)
    ok,encoded=cv.imencode(".jpg",masks)
    assert ok
    (SCORES/f"s{seed}_{arm}_mask.jpg").write_bytes(encoded.tobytes())
    save(SCORES/f"s{seed}_{arm}.json",result)
    print(seed,arm,{k:round(result["all"][k],4) for k in KEYS},background["full"],flush=True)


def summarize():
    pairs=[]
    for seed in range(42,47):
        pair={}
        for arm in ("dense","candidate"):
            path=SCORES/f"s{seed}_{arm}.json"
            if not path.exists():
                break
            v=json.loads(path.read_bytes())
            assert v["arm"]==arm and v["seed"]==seed
            pair[arm]={**{k:v["all"][k] for k in KEYS},"background_center_x_px":v["background"]["net_center_xy_all_fits"][0],
                       "background_support":v["background"]["full"]["supported"],
                       "background_right_frames":v["background"]["full"]["screen_right_over_quarter_px"],
                       "texture_early":v["early"]["body_band_correlation"],"texture_late":v["late"]["body_band_correlation"],
                       "sha256":v["sha256"]}
        if len(pair)==2:
            pair["delta_anchor_minus_dense"]={k:pair["candidate"][k]-pair["dense"][k] for k in KEYS}
            pairs.append({"seed":seed,**pair})
    if not pairs:
        return
    extra=[x for x in pairs if x["seed"]!=42]
    save(SCORES/"paired.json",dict(ordered_arms=["dense","candidate"],seeds=pairs,
         diagnostic="Each clip datum is the median of 79 centre frames; horizontal background translation composes all 80 outer-frame similarity fits. The direction-support count requires >=60% inliers."))
    lines=["# 0951 左摇的跨种子主体与背景测量","",
           "同一行的两条视频使用相同的图、提示和种子。旧锚点版在高、低噪声命中步均使用近似。每格的配对数值按 Dense / 旧锚点排列。种子 42 是先前发现问题的案例；四个新增种子分别计数。","",
           "| 种子 | 熊身纹理跨帧相关 | 补偿后亮度闪烁 | 轮廓法向加速度 | 熊身暖色面积 % | 熊身有效对应 % |",
           "|---|---:|---:|---:|---:|---:|"]
    def cell(row,key,places=3):
        return " / ".join(f"{row[a][key]:.{places}f}" for a in ("dense","candidate"))
    for r in pairs:
        lines.append(f"| {r['seed']} | {cell(r,KEYS[0])} | {cell(r,KEYS[1],2)} | {cell(r,KEYS[2],2)} | {cell(r,KEYS[3],2)} | {cell(r,KEYS[4],1)} |")
    lines.extend(["","纹理相关衡量当前熊身高频与双向光流补偿后的前后帧高频均值的相关。闪烁使用 8 位灰度，轮廓法向加速度取局部边界的第 75 百分位（px/帧²），然后跨视频时刻取中位数。轮廓量也会响应真正发生的非匀速运动。暖色区域的覆盖、面积和形状会随姿态、遮挡或颜色变化而改变。每条视频单独使用原有的区域规则。抽查图显示，有的时刻主要圈住头部，有的时刻还包括口鼻、手足或躯干，暖色面积不能直接换算成整只熊的投影面积。","","| 种子 | 背景累计水平像移 px | 支持帧间 / 80（Dense / 旧锚点） | 背景向右的支持帧间 | 早段熊身纹理相关 | 末段熊身纹理相关 |","|---|---:|---:|---:|---:|---:|"])
    for r in pairs:
        lines.append(f"| {r['seed']} | {cell(r,'background_center_x_px',1)} | {cell(r,'background_support',0)} | {cell(r,'background_right_frames',0)} | {cell(r,'texture_early')} | {cell(r,'texture_late')} |")
    lines.extend(["","正像移表示画面背景向右，符合相机左摇。累计量由外圈特征的相似变换逐帧复合得出。方向计数仅含内点率至少 60%、位移大于 0.25 像素的帧间。早段与末段分别为中心帧 1–26 和 53–79。暖色区域及其边带在第 1、20、40、60、79 帧的抽查图与逐帧结果存于本目录。"])
    if extra:
        d=[r["delta_anchor_minus_dense"] for r in extra]
        lines.extend(["",f"目前已取得 {len(extra)} 个新种子的完整配对。旧锚点的熊身纹理相关低于 Dense 的有 {sum(x[KEYS[0]]<0 for x in d)}/{len(d)} 对，补偿后亮度闪烁较大的是 {sum(x[KEYS[1]]>0 for x in d)}/{len(d)} 对，轮廓法向加速度较大的是 {sum(x[KEYS[2]]>0 for x in d)}/{len(d)} 对。以上是连续统计量的方向计数，持续抖动的可见程度由配对人审判读。"])
    lines.extend(["","[配对全精度值及文件指纹](paired.json) · [复算入口](../../../work/seed0951/score_seeds.py)"])
    (SCORES/"跨种子主体与背景.md").write_text("\n".join(lines)+"\n",encoding="utf8")


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--seed",type=int,choices=range(42,47))
    cli.add_argument("--arm",choices=("dense","candidate"))
    args=cli.parse_args()
    if args.seed is not None:
        for a in (args.arm,) if args.arm else ("dense","candidate"):
            score(args.seed,a)
    else:
        summarize()
