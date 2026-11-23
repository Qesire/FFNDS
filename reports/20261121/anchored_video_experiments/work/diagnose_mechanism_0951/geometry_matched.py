"""Compare late-step effects at overlapping decoded object sizes and valid-flow fractions."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[2]/"outputs"
OUT=ROOT/"FFNDS_0951_mechanism/factorial"
ORIG=ROOT/"FFNDS_anchor_scale24/camera7/diagnosis"
ARMS=("dense","low_only","high_only","candidate")

def read(path):
    return json.loads(path.read_bytes())

def values(arm):
    path=(ORIG if arm in ("dense","candidate") else OUT)/f"pan_left_{arm}.json"
    data=read(path)
    assert read(OUT/"析因结果.json")["conditions"]["pan_left"]["rows"][arm]["sha256"]==data["sha256"]
    rows=data["by_frame"]
    assert [r["frame"] for r in rows]==list(range(1,80))
    return {k:np.array([r[k] for r in rows]) for k in rows[0]},hashlib.sha256(path.read_bytes()).hexdigest()

def assign(a,b,ratio,cover,limit,subset):
    f=a["frame"]
    area=np.abs(np.log(a["body_area_pct"][:,None]/b["body_area_pct"][None,:]))/np.log(ratio)
    coverage=np.abs(a["body_valid_pct"][:,None]-b["body_valid_pct"][None,:])/cover
    time=np.abs(f[:,None]-f[None,:])
    eligible=(area<=1)&(coverage<=1)&(time<=limit)&subset[:,None]&subset[None,:]
    cost=area+coverage+time/79
    # Every feasible edge costs <=3, each dummy pair costs 1000. Thus the
    # assignment maximises cardinality before it minimises matching distance.
    x,y=linear_sum_assignment(np.hstack((np.where(eligible,cost,1e6),np.full((len(f),len(f)),1000.))))
    keep=y<len(f)
    return x[keep],y[keep]

def summarize(a,b,x,y):
    if not len(x):return {"n":0,"pairs":[]}
    d=b["body_band_correlation"][y]-a["body_band_correlation"][x]
    flicker=b["body_compensated_flicker"][y]-a["body_compensated_flicker"][x]
    return dict(n=len(x),median_delta=float(np.median(d)),mean_delta=float(np.mean(d)),
                frac_lower=float(np.mean(d<0)),flicker_delta=float(np.median(flicker)),
                med_area_pp=float(np.median(np.abs(a["body_area_pct"][x]-b["body_area_pct"][y]))),
                med_coverage_pp=float(np.median(np.abs(a["body_valid_pct"][x]-b["body_valid_pct"][y]))),
                med_frame_gap=float(np.median(np.abs(x-y))),frame_range_a=[int(x.min()+1),int(x.max()+1)],
                frame_range_b=[int(y.min()+1),int(y.max()+1)],pairs=np.stack((x+1,y+1),axis=1).tolist())

def histogram(a,b,aw,cw):
    def freq(f):
        bins=np.stack((np.floor(f["body_area_pct"]/aw),np.floor(f["body_valid_pct"]/cw))).T
        rows,count=np.unique(bins,axis=0,return_counts=True)
        return dict(zip(map(tuple,rows),count))
    left,right=freq(a),freq(b)
    return float(sum(min(v,right.get(k,0)) for k,v in left.items())/79)

def main():
    arrays={}
    hashes={}
    for arm in ARMS:
        arrays[arm],hashes[arm]=values(arm)
    result={"input_sha256":hashes,"distributions":{},"pairs":{}}
    for a,f in arrays.items():
        result["distributions"][a]={k:np.quantile(f[k],[0,.1,.5,.9,1]).tolist() for k in ("body_area_pct","body_valid_pct","body_band_correlation")}
    for al,bl in (("dense","low_only"),("high_only","candidate")):
        a,b=arrays[al],arrays[bl]
        frame=a["frame"]
        record={"hist_overlap":{f"{u}_{v}":histogram(a,b,u,v) for u,v in ((1,5),(2.5,10),(5,20))},
                "raw":summarize(a,b,frame-1,frame-1),"matches":{}}
        for ratio,coverage in ((1.15,5),(1.30,10)):
            for window in (0,4,12,78):
                for lo,hi in ((1,79),(1,26),(27,53),(54,79)):
                    ix,iy=assign(a,b,ratio,coverage,window,(frame>=lo)&(frame<=hi))
                    record["matches"][f"{ratio}_{coverage}_{window}_{lo}-{hi}"]=summarize(a,b,ix,iy)
        result["pairs"][al]=record
    (OUT/"左摇匹配统计.json").write_text(json.dumps(result,ensure_ascii=False,indent=1),encoding="utf8")
    lines=["# 左摇四臂的主体尺度与匹配支持", "",
           "同一张 0951 首帧和种子。原生前缀配对比较 Dense 与只在晚步近似的 low_only；早步近似前缀配对比较 high_only 与两步均近似的 candidate。每对进入晚步的潜变量相同。每条视频 79 个中心帧，统计单位为帧时刻；同片相邻帧存在相关性。纹理差值均为「晚步近似减晚步原生」。", "",
           "| 臂 | 暖色熊身面积（占画面 %），P10 / 中位 / P90 | 熊身有效跟踪率 %，P10 / 中位 / P90 |",
           "|---|---:|---:|"]
    for arm,d in result["distributions"].items():
        fmt=lambda a:" / ".join(f"{v:.1f}" for v in a[1:4])
        lines.append(f'| {arm} | {fmt(d["body_area_pct"])} | {fmt(d["body_valid_pct"])} |')
    lines += ["", "二维面积、覆盖率以 2.5、10 个百分点分箱；各箱取两臂样本数较小值，相加后除以 79。原生前缀和早步近似前缀的分布重叠比例分别为 "
              + "、".join(f'{p["hist_overlap"]["2.5_10"]:.1%}' for p in result["pairs"].values())+
              "。更细和更粗的分箱值均保留在 JSON。", "",
              "匹配完全根据面积、有效跟踪率和时刻构建。每帧最多使用一次，先最大化满足卡钳的匹配数，再最小化标准化面积比、覆盖率差和时间错位之和。卡钳取较大与较小面积之比 ≤1.15 或 ≤1.30，对应覆盖率绝对差 ≤5 或 ≤10 个百分点。表列配对纹理相关差的中位数、差值为负的比例，以及匹配的实际差异。","",
              "| 前缀 | 面积比 / 覆盖率卡钳 | 时间可错位（帧） | 配对数 | 纹理相关差 | 差为负 | 面积差 / 覆盖率差中位（百分点） | 匹配时刻中位绝对差（帧） |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for al,p in result["pairs"].items():
        for spec in ((1.15,5,0),(1.30,10,0),(1.15,5,4),(1.30,10,4),(1.30,10,12),(1.30,10,78)):
            v=p["matches"]["_".join(map(str,spec))+"_1-79"]
            tail=f'{v["median_delta"]:+.3f} | {v["frac_lower"]:.0%} | {v["med_area_pp"]:.2f} / {v["med_coverage_pp"]:.2f} | {v["med_frame_gap"]:.1f}' if v["n"] else "— | — | — | —"
            lines.append(f'| {al} | {spec[0]} / {spec[1]} | {spec[2]} | {v["n"]} | {tail} |')
    lines += ["","宽卡钳（1.30、10 个百分点）、允许错位 ≤4 帧时，再将两个匹配帧同时限制在各自的早、中、晚时间窗。","",
              "| 前缀 | 1–26 帧 | 27–53 帧 | 54–79 帧 |","|---|---:|---:|---:|"]
    for al,p in result["pairs"].items():
        cells=[]
        for window in ("1-26","27-53","54-79"):
            v=p["matches"]["1.3_10_4_"+window]
            cells.append(f'{v["n"]} 对，差 {v["median_delta"]:+.3f}' if v["n"] else "0 对")
        lines.append("| "+" | ".join([al,*cells])+" |")
    lines += ["","近似晚步时，纹理相关在两组有共同支持的画面中都较低。严格同帧、面积比 ≤1.15、覆盖率差 ≤5 个百分点时，原生前缀的 49 对里有 48 对下降；早步近似前缀的 23 对里有 22 对下降。放宽卡钳或允许有限的时间错位，这个方向依然保持。两组匹配支持不同，第二组尤为有限。", "",
              "面积与覆盖率都是生成后的量；面积来自有遗漏的暖色色区，有效跟踪率来自 DIS 闭合阈值。匹配所保留的时刻可能集中于同一段姿态，不能从这里识别控制这两项之后的独立因果效应，也不能外推到没有共同支持的时段。","",
              "[逐个匹配时刻、所有阈值与面积分布](左摇匹配统计.json) · [复算代码](../../../work/diagnose_mechanism_0951/geometry_matched.py)"]
    (OUT/"左摇尺度与跟踪率匹配.md").write_text("\n".join(lines)+"\n",encoding="utf8")

if __name__=="__main__":
    main()
