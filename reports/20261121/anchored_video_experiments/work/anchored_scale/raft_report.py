"""Summarize frozen RAFT and the matched-raster comparison with DIS."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

OUT=Path(__file__).resolve().parents[2]/"outputs/FFNDS_anchor_scale24"
FOCUS=("vbi_0024","vbi_0864","vbi_0392","vbi_0800","vbi_1013")


def read(path):
    return json.loads(path.read_bytes())


def f(x,n=2):
    return "—" if x is None else f"{x:.{n}f}"


def main():
    cohort=read(OUT/"COHORT.json")
    with (OUT/"指标逐源逐窗.csv").open(encoding="utf-8-sig",newline="") as file:
        dis={(r["source"],r["window"]):r for r in csv.DictReader(file)}
    rows=[]
    legacy=[]
    seconds=0.
    identity=None
    for s in cohort:
        src=s["record_id"]
        report=read(OUT/"raft"/(src+".json"))
        if identity is None: identity=report["runtime"]
        assert report["source"]==src and report["runtime"]==identity
        seconds+=report["seconds"]
        if "legacy_deltas" in report:
            legacy.extend(abs(v) for v in report["legacy_deltas"].values())
        assert read(OUT/"metrics"/(src+".json"))["hashes"]==report["sha"]
        for win,w in report["windows"].items():
            dr=dis[src,win]
            rows.append(dict(source=src,cluster=s["source_cluster_id"],group=s["cohort_group"],
                             category=s["category"],window=win,**w,grid_reference_quiet=w["grid_dense_rms"]<.5,
                             dis_reference_quiet=dr["reference_quiet"]=="True",
                             dis_active_angle_deg=float(dr["angle_deg"]) if dr["angle_deg"] else None,
                             dis_gain=float(dr["gain"]) if dr["gain"] else None,dis_epe=float(dr["epe"]),
                             dis_active_support=float(dr["both_moving"])))
    assert len(rows)==24*6 and len(legacy)==75
    with (OUT/"RAFT逐源逐窗.csv").open("w",newline="",encoding="utf-8-sig") as file:
        wr=csv.DictWriter(file,fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    full={r["source"]:r for r in rows if r["window"]=="0-80"}
    cross={}
    for a,b in (("grid_gain","dis_gain"),("grid_active_angle_deg","dis_active_angle_deg"),("grid_epe","dis_epe")):
        pair=[(r[a],r[b]) for r in full.values() if r[a] is not None and r[b] is not None
              and (a!="grid_gain" or not r["grid_reference_quiet"])]
        x,y=np.array(pair).T
        cross[a]=(len(pair),np.corrcoef(x,y)[0,1],spearmanr(x,y).statistic,np.median(abs(x-y)))
    diff=max(legacy)
    lines=["# RAFT 对 24 个来源的运动复核","",
           f"对 24 条保留锚点候选及同种子 Dense 共 48 个原片运行归档 VBench 0.1.5 的 RAFT。每臂每源评估 40 对相隔两帧的画面，共 1,920 次两帧光流推断。每个来源、整片和五个时间窗各占 CSV 一行。每对视频核对生成回执、初始噪声与视频 SHA-256。计算设备为 {identity['device']}，逐源测算耗时合计 {seconds/60:.1f} 分钟。","",
           "## 同源两种光流估计","",
           "下表各对数值按 RAFT / DIS 排列。RAFT 先在原分辨率计算，再投到与 DIS 相同的网格。增益是候选在参考流方向上的投影系数；角差使用两臂流幅均达到 0.5 原像素／两帧的格点。覆盖率是对应估计器中满足此条件的比例。EPE 为同一网格上候选与 Dense 的端点距离均值。","",
           "| 来源 | 运动投影增益 | 活跃区角差 ° | 有效格点 % | 同网格 EPE 像素 |","|---|---:|---:|---:|---:|"]
    for src in FOCUS:
        r=full[src]
        ga=r["grid_gain"] if not r["grid_reference_quiet"] else None
        lines.append(f"| {src[-4:]} | {f(ga)} / {f(r['dis_gain'])} | {f(r['grid_active_angle_deg'],1)} / {f(r['dis_active_angle_deg'],1)} | {f(100*r['grid_support'],1)} / {f(100*r['dis_active_support'],1)} | {f(r['grid_epe'])} / {f(r['dis_epe'])} |")
    lines+=["","0024 在 RAFT 的五段投影增益为 0.08–0.14，活跃区角差为 56.2°–66.3°，各段均有明显方向分歧。0864 五段增益为 0.93–1.03，角差为 4.1°–5.0°。1013 的比较需结合表中的覆盖率，两种方法都只在不到一成的格点同时检测到足够运动。0392 的两种估计器给出不同投影幅度，且观看者曾指出 Dense 有重影。"]
    lines+=["","| 同网格指标 | 来源数 | 来源间 Spearman ρ | 来源间 Pearson r | 两估计器逐源绝对差的中位数 |","|---|---:|---:|---:|---:|"]
    for k,label in (("grid_gain","投影增益"),("grid_active_angle_deg","活跃区角差 °"),("grid_epe","端点距离 像素")):
        n,p,s,md=cross[k]
        lines.append(f"| {label} | {n} | {f(s,3)} | {f(p,3)} | {f(md,3)} |")
    lines+=["","汇总相关性时，任一估计器的参考流 RMS < 0.5 像素就省略增益；双方均有有效运动格点时才比较角差。两种估计器可选中不同的活跃格点。相关性描述这批来源之间的一致性。","",
           "## 原分辨率指标与归档复现","",
           "使用已核对源码与权重的 DynamicDegree 图像读取、输入填充、RAFT 20 轮迭代、原生动态分数函数。464×832 原图输出采用每个像素。角差沿用历史门槛，两臂流幅均 >0.001 原像素／两帧。angle_support 保留其分母覆盖率。另给 EPE、幅值平均绝对差、幅值过量／不足和聚合有符号投影。两臂的原生动态分数先取每个帧对最高 5% 流幅的平均值，再对帧对求平均。只保存逐源逐窗统计。","",
           "| 来源 | 全分辨率角差 ° | EPE 像素 | 幅值过量 / 不足 像素 | 投影增益 |","|---|---:|---:|---:|---:|"]
    for src in FOCUS:
        r=full[src]
        lines.append(f"| {src[-4:]} | {f(r['angle_deg'])} | {f(r['epe'])} | {f(r['excess'])} / {f(r['deficit'])} | {f(r['gain'])} |")
    lines+=["","下表每格为按来源等权的中位数［第 25、75 百分位］。每源只有一个种子。","",
            "| 分组 | 来源数 | 全像素角差 ° | EPE 像素 | 幅值过量 像素 | 幅值不足 像素 |",
            "|---|---:|---:|---:|---:|---:|"]
    for label,pool in (("全部",list(full.values())),("扩展",[r for r in full.values() if r["group"]=="expansion21"])):
        cells=[]
        for k in ("angle_deg","epe","excess","deficit"):
            q1,median,q3=np.quantile([r[k] for r in pool],[.25,.5,.75])
            cells.append(f"{f(median)} [{f(q1)}, {f(q3)}]")
        lines.append("| "+" | ".join([label,str(len(pool)),*cells])+" |")
    lines+=["",f"既有 0392、0800、1013 的 Dense 与候选 MP4 均匹配原记录哈希。三个来源共核对 75 项归档指标（每源 5 项整片统计、5 个时间窗各 4 项）。最大绝对差为 {diff:.8g}。权重 SHA-256 为 {identity['weight_sha256']}。",
           "","两种光流的参照均为相同输入下的 Dense。真实场景中的合理运动、细节保存与 Dense 自身的缺陷由人眼核查。","",
           "[逐源逐窗数据](RAFT逐源逐窗.csv) · [双栏播放原片](同步核查.html) · [计算脚本](../../work/anchored_scale/raft_scale.py) · [统计与交叉核验](../../work/anchored_scale/raft_report.py)"]
    (OUT/"RAFT复核.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("sources",len(full),"max legacy error",diff,"sum_minutes",seconds/60,"correlations",cross)


if __name__=="__main__":
    main()
