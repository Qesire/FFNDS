"""Recompute low OUT-D replay, exact-prefix evidence and matched once-hit cost."""
import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
from statistics import geometric_mean, mean

import numpy as np

PLAN = "ff8fd9dcbed3812b1da7f589f3d2e817f48c26c9"
LAYERS = (0,13,26,39)
ARMS = ("raw","centered")


def read_bytes(path):
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == ".gz" else data


def check_gram(values, width):
    g = np.asarray(values, dtype=np.float64)
    assert g.shape == (width,width) and np.isfinite(g).all() and np.allclose(g,g.T)
    assert np.linalg.eigvalsh(g)[0] >= -1e-10 * max(float(g.diagonal().max()),1e-10)
    return g


def ratio(num, den):
    return float(num/den) if den > 0 else None


def format_ratio(x):
    return "—" if x is None else f"{x:.4f}×"


def main(raw, media, folder):
    if media is not None:
        from review_attention_c_schedule import measure

    payload = read_bytes(raw)
    doc = json.loads(payload)
    assert doc["status"] == "complete" and doc["plan_commit"] == PLAN and len(doc["pairs"]) == 4
    job = doc["job"]
    summary = dict(job=job, plan_commit=PLAN, receipt_sha256=sha256(payload).hexdigest(), fixed={}, timing={}, motion={})
    report = ["# low OUT-D 的均值与尺度运输", "",
              f"学校 A100 作业 `{job}`。只替换冻结 C 的 low OUT-D 历史 BF16 槽。"
              "沿用 high、selected 支持、原 P64、特征、CAL 和 λ。锚点使用原生输出直接重编码，hit 使用当前 value 的通道统计。", "",
              "两条固定轨迹按 C 运行并逐位复现归档。四个 low 层在逐 query、全键 H448 教师上比较 raw 与 centered。"
              "分支共用现场 P64 和 wide8192 拟合。Dense 参照保持当前块输入，使用原生 H448 attention 与 Dense FFN。", "",
              "| 来源 | low 层 | 锚点重建相对 L2 raw / centered | H448 底座能量比 | 拟合后注意力 | wide 块出口 | 到 Dense 块出口 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    assert len(doc["fixed"]) == 2
    for source, item in doc["fixed"].items():
        assert tuple(map(int,item["sites"])) == LAYERS
        summary["fixed"][source] = {}
        for layer, row in item["sites"].items():
            tail = row["tail"]
            assert row["identical_final_attention_probes"] and tail["shared_probe_ffn_input_exact"]
            assert tail["reused_live_ffn_probe_teacher"] and tail["dense_native_ffn_extra_held_rows"] == 448
            entry = dict(anchor_raw_relative_l2=row["anchor_raw_relative_l2"], anchor_relative_l2=row["anchor_relative_l2"],
                         inactive_anchor_channels=row["inactive_anchor_channels"], inactive_current_channels=row["inactive_current_channels"],
                         negative_rho_rows=row["negative_rho_rows"], rho_min=row["rho_min"],
                         probe_to_live_relative_l2=tail["shared_probe_vs_live_relative_l2"], replay_to_live_relative_l2=tail["live_relative_l2"],
                         difference_identity_relative_l2=row["difference_identity_relative_l2"])
            for key, g in (("base",row["held_base_error_gram"]),("attention",tail["stage_gram_raw_transport"]["attention_head"]),
                           ("wide_exit",tail["stage_gram_raw_transport"]["block_exit"])):
                a = check_gram(g,2)
                entry[key] = ratio(a[1,1],a[0,0])
                entry[key+"_raw_energy"] = float(a[0,0])
            g = check_gram(tail["stage_gram_raw_transport_background"]["block_exit"],3)
            assert np.allclose(g[:2,:2],tail["stage_gram_raw_transport"]["block_exit"])
            old = g[0,0]+g[2,2]+2*g[0,2]
            new = g[1,1]+g[2,2]+2*g[1,2]
            entry["dense_exit"] = ratio(new,old)
            entry["dense_delta"] = float(new-old)
            entry["dense_cross_delta"] = float(2*(g[1,2]-g[0,2]))
            summary["fixed"][source][layer] = entry
            error = f"{entry['anchor_raw_relative_l2']:.4%} / {entry['anchor_relative_l2']:.4%}"
            report.append(f"| {source[-4:]} | {layer} | {error} | " +
                          " | ".join(format_ratio(entry[key]) for key in ("base","attention","wide_exit","dense_exit"))+" |")
    report.extend(["", "表内能量比为 centered / raw。归档保留逐阶段 Gram、BF16 编码见证、带符号的原 rho 以及线性更新恒等式的数值误差。", "",
                   "## 预热后的同卡配对", "",
                   "每源两轮并颠倒运行顺序。四组配对的前 3 步状态逐位相同。先对同源重复取均值，再对两个来源的耗时比取几何平均。", "",
                   "| 区间 | C 秒 | centered 秒 | centered / C |", "|---|---:|---:|---:|"])
    sources = list(doc["fixed"])
    per_source = {}
    for source in sources:
        pairs = [pair for pair in doc["pairs"] if pair["source"] == source]
        assert [r["repeat"] for r in pairs] == [0,1] and pairs[0]["order"] == pairs[1]["order"][::-1]
        assert all(pair["prefix_exact"] for pair in pairs)
        arms = {}
        repeat_trace_matches = {}
        for arm in ARMS:
            records = [p["arms"][arm]["record"] for p in pairs]
            repeat_trace_matches[arm] = (all(records[0][k] == records[1][k] for k in ("initial","latents","flows")) and
                records[0]["video"]["decoded_rgb24_sha256"] == records[1]["video"]["decoded_rgb24_sha256"])
            step = [mean(float(row["step_ms"][i])/1000 for row in records) for i in range(4)]
            arms[arm] = dict(anchor=step[2], hit=step[3], cycle=sum(step[2:]), gpu=sum(step),
                             wall=mean(r["wall"] for r in records), peak_gib=max(r["peak_allocated"]/2**30 for r in records))
        per_source[source] = arms
        summary["timing"].setdefault("repeat_trace_matches",{})[source] = repeat_trace_matches
        if media is not None:
            summary["motion"][source] = {}
            for arm in ARMS:
                r = pairs[0]["arms"][arm]["record"]["video"]
                m = measure(media/f"{source}_{arm}.mp4",r["mp4_sha256"])
                assert m["frames"] == 81 and m["geometry"] == [832,464]
                summary["motion"][source][arm] = {k:m[k] for k in ("mean_adjacent_luma_mae","median_flow_spatial_p90_px")}
    summary["timing"]["sources"] = per_source
    for k,label in (("anchor","low anchor 步"),("hit","low hit 步"),("cycle","low 两步复用周期"),
                    ("gpu","四步 GPU"),("wall","完整请求含保存")):
        row = {arm:mean(d[arm][k] for d in per_source.values()) for arm in ARMS}
        row["ratio"] = geometric_mean(d["centered"][k]/d["raw"][k] for d in per_source.values())
        summary["timing"][k] = row
        report.append(f"| {label} | {row['raw']:.4f} | {row['centered']:.4f} | {row['ratio']:.4f}× |")
    if summary["motion"]:
        report.extend(["", "| 来源 | 冻结 C 光流代理 | centered 光流代理 | 冻结 C / centered 灰度变化 |",
                       "|---|---:|---:|---:|"])
        for source, m in summary["motion"].items():
            r,c = (m[k] for k in ARMS)
            report.append(f"| {source[-4:]} | {r['median_flow_spatial_p90_px']:.4f} | {c['median_flow_spatial_p90_px']:.4f} | "
                          f"{r['mean_adjacent_luma_mae']:.3f} / {c['mean_adjacent_luma_mae']:.3f} |")
        report.extend(["", "运动代理取事先固定的第 1 轮，完整视频待人审。"])
    folder.mkdir(parents=True,exist_ok=True)
    (folder/"FFNDS_attention_mass_centered_low.md").write_text("\n".join(report)+"\n",encoding="utf8")
    (folder/"attention_mass_centered_low_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print("\n".join(report))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--raw",type=Path)
    cli.add_argument("--media",type=Path)
    cli.add_argument("--out",type=Path)
    args = cli.parse_args()
    if args.raw is None and args.media is None and args.out is None:
        from school_attention_online import OUT, read, receipt_path

        job = read(receipt_path("masslow"))["job"]
        args.raw = OUT/f"attention_online_masslow_{job}.json"
        args.media = OUT/f"attention_masslow_videos_{job}"
        args.out = OUT
    if args.raw is None or args.out is None:
        cli.error("--raw and --out are required together")
    main(args.raw,args.media,args.out)
