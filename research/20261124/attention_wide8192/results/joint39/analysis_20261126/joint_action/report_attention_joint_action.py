"""Verify staged Gram identities and summarize the joint action on both trajectories."""
import argparse
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

PLAN = "51b6e235964df2f1fb1370604387cb9c998f6591"


def read_bytes(path):
    return gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()


def matrix(values, size):
    g = np.asarray(values, dtype=np.float64)
    assert g.shape == (size, size) and np.isfinite(g).all()
    assert np.allclose(g, g.T, rtol=1e-10, atol=1e-12)
    assert np.linalg.eigvalsh(g)[0] >= -1e-10 * max(g.diagonal().max(), 1e-10)
    return g


def cosine(a, b, ab):
    return float(ab / math.sqrt(a*b)) if a > 0 and b > 0 else None


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def number(value):
    return "—" if value is None else f"{value:.4f}"


def two(values):
    g = matrix(values, 2)
    raw, trans = g.diagonal()
    change = raw + trans - 2*g[0, 1]
    return dict(raw=float(raw), transport=float(trans), ratio=ratio(trans, raw),
                cosine=cosine(raw, trans, g[0, 1]), update_energy=float(change),
                raw_dot_update=float(g[0, 1]-raw))


def dense(values, native):
    g = matrix(values, 3)
    assert np.allclose(g[:2, :2], native, rtol=1e-10, atol=1e-12)
    rr, tt, b = g.diagonal()
    cross = 2*(g[1, 2] - g[0, 2])
    fixed = tt - rr
    raw, transport = rr+b+2*g[0, 2], tt+b+2*g[1, 2]
    assert math.isclose(transport-raw, fixed+cross, rel_tol=1e-9, abs_tol=1e-10)
    return dict(raw=float(raw), transport=float(transport), ratio=ratio(transport, raw),
                background_energy=float(b), twice_background_raw_dot=float(2*g[0, 2]),
                twice_background_transport_dot=float(2*g[1, 2]), fixed_wide_delta=float(fixed),
                background_cross_delta=float(cross), total_delta=float(transport-raw))


def innovation(info):
    norm = matrix(info["route_shape_gram_standardized"], 2)
    before, after = (matrix(info[k], 3) for k in ("route_shape_scale_gram_physical", "route_shape_scale_post_fit_gram_physical"))
    def stats(g):
        routed, shaped, scaled = map(float, g.diagonal())
        transported = float(g[:2, :2].sum())
        raw = float(g.sum())
        return dict(route_energy=routed, shape_energy=shaped, scale_energy=scaled, route_shape_cross=float(2*g[0, 1]),
                    route_shape_cosine=cosine(routed, shaped, g[0, 1]), scale_cross=float(2*(g[0, 2]+g[1, 2])),
                    raw_energy=raw, transport_energy=transported, ratio=transported/raw)
    return dict(standardized_route_energy=float(norm[0, 0]), standardized_shape_energy=float(norm[1, 1]),
                standardized_route_shape_cosine=cosine(norm[0, 0], norm[1, 1], norm[0, 1]),
                before=stats(before), after=stats(after),
                transported_remaining_after_fit=float(after[:2, :2].sum()/before[:2, :2].sum()),
                scale_remaining_after_fit=float(after[2, 2]/before[2, 2]),
                value_shape_relative_l2=info["normalized_value_shape_relative_l2"],
                current_to_anchor_scale_quantiles=info["current_to_anchor_scale_quantiles"], witness=info["witness"])


def main(raw, reference_path, folder):
    data = read_bytes(raw)
    source = json.loads(data)
    job = source["job"]
    assert source["status"] == "complete" and source["plan_commit"] == PLAN
    reference = json.loads(read_bytes(reference_path))["replay"]["observations"]
    summary = dict(job=job, plan_commit=PLAN, receipt_sha256=sha256(data).hexdigest(), sources={})
    lines = ["# 标准化缺陷运输的逐阶段作用", "", f"学校 A100 作业 `{job}`。0392、1013 的原生注意力加 wide8192 固定轨迹均逐位复现归档。"
             "每条轨迹读取两专家的第 0、13、26、39 层，H448 留出。raw 与 transport 使用相同当前特征、"
             "全行对应的注意力 P64 教师，以及从实际 wide8192 调用捕获的 FFN P64 教师。", "",
             "逐阶段 Gram 的误差为各路径减去原生 attention 经相同 wide8192 尾部的参考。在 FFN、参数门控响应与块出口另以原生 attention 经 H448 Dense FFN 定义背景 $b$。Dense 参照固定同一块输入。", "",
             "| 来源 | 步／层 | 注意力 | 输出投影 | self gate | cross 残差 | FFN 输入 | FFN 响应 | 块出口 | 到 Dense 块出口 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    listed = ("attention_head", "output_projection", "attention_gate_response", "cross_residual", "ffn_input", "ffn_response", "block_exit")
    effects, early = [], []
    for sid, track in source["sources"].items():
        assert track["trace_exact"] and len(track["sites"]) == 8
        summary["sources"][sid] = {}
        for where, row in sorted(track["sites"].items(), key=lambda kv: tuple(map(int, kv[0].split(":")))):
            step, layer = map(int, where.split(":"))
            tail = row["wan_wide8192_tail"]
            assert tail["shared_probe_ffn_input_exact"] and tail["reused_live_ffn_probe_teacher"] and tail["dense_native_ffn_extra_held_rows"] == 448
            grams = tail["stage_gram_raw_transport"]
            stages = {k:two(g) for k,g in grams.items()}
            for kind in ("raw", "transport"):
                assert math.isclose(stages["attention_head"][kind], row["fit_energy"][kind], rel_tol=1e-10)
                if sid == "vbi_0392":
                    assert math.isclose(stages["attention_head"][kind], reference[where]["fit_energy"][kind], rel_tol=1e-5)
            backgrounds = {k:dense(g, grams[k]) for k,g in tail["stage_gram_raw_transport_background"].items()}
            witness = innovation(row["innovation_decomposition"])
            site = dict(stages=stages, backgrounds=backgrounds, innovation=witness, live_relative_l2=tail["live_relative_l2"],
                        shared_probe_vs_live_relative_l2=tail["shared_probe_vs_live_relative_l2"])
            if sid == "vbi_0392" and layer == 39:
                old = reference[where]["wan_wide8192_tail"]["held_block_energy"]
                site["historical_tail_ratios"] = {k: stages["block_exit"][k] / old[k] for k in ("raw", "transport")}
            summary["sources"][sid][where] = site
            end = backgrounds["block_exit"]
            cells = [number(stages[k]["ratio"]) for k in listed]
            lines.append(f"| {sid[-4:]} | {step}/{layer} | " + " | ".join(cells) + f" | {number(end['ratio'])} |")
            effects.append((sid, step, layer, site))
            if layer == 0:
                early.append((sid, step, site))
    lines.extend(["", "表中为 transport/raw 的误差能量比。Gram 原矩阵保留在收据中。", "", "## 第 39 层的方向与 Dense 背景", "",
                  "| 来源 | 步 | 注意力误差余弦 | 块出口误差余弦 | 固定 wide 能量差 | $2\\langle b,z_t-z_r\\rangle$ | 到 Dense 能量差 |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for sid, step, layer, site in effects:
        if layer != 39:
            continue
        st, d = site["stages"], site["backgrounds"]["block_exit"]
        lines.append(f"| {sid[-4:]} | {step} | {st['attention_head']['cosine']:.4f} | {st['block_exit']['cosine']:.4f} | "
                     f"{d['fixed_wide_delta']:+.4g} | {d['background_cross_delta']:+.4g} | {d['total_delta']:+.4g} |")
    lines.extend(["", "## 第 0 层的路由、形状与尺度", "",
                  "用固定支持和成员标签、同一 512 行、全部 keys 独立求三项：$R_a\\widetilde V_a$、$R_a\\widetilde V_s$、$R_s\\widetilde V_s$。"
                  "路由与形状先在标准化坐标做差，再用当前尺度解码。raw 创新增添尺度项 $R_a\\widetilde V_a(D_s-D_a)$。"
                  "分项均通过同一 P64 的线性恢复并评分 H448。", "",
                  "| 来源 | 步 | 恢复后路由 / 形状能量 | 路由与形状交叉 / 两项能量之和 | 尺度净惩罚 / raw | 运输创新恢复后保留能量 | 独立复算拟合能量比 | 实际注意力拟合比 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for sid, step, site in early:
        w = site["innovation"]
        z = w["after"]
        penalty = (z["scale_energy"]+z["scale_cross"])/z["raw_energy"]
        x = z["route_shape_cross"]/(z["route_energy"]+z["shape_energy"])
        lines.append(f"| {sid[-4:]} | {step} | {z['route_energy']/z['shape_energy']:.3f}× | {x:+.2%} | {penalty:+.2%} | "
                     f"{w['transported_remaining_after_fit']:.2%} | {z['ratio']:.4f}× | {site['stages']['attention_head']['ratio']:.4f}× |")
    worst = max(max(site["innovation"]["witness"][key] for key in ("anchor_relative_l2_physical", "hit_relative_l2_physical")) for _, _, _, site in effects)
    lines.extend(["", f"独立 FP32 路由作用对运行时缺陷在物理单位的最大相对 L2 差为 {worst:.4%}。"
                  "同特征拟合的线性分解及其与实际 BF16 输出的比较均保留在数据中。"
                  "本次为带诊断的固定轨迹回放。增加的 H448 Dense FFN 和路由交叉收缩属于测量开销。"])
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "FFNDS_attention_joint_action.md").write_text("\n".join(lines)+"\n", encoding="utf8")
    (folder / "attention_joint_action_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf8")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.raw is None and args.reference is None and args.out is None:
        from school_attention_online import OUT, read, receipt_path

        job = read(receipt_path("jointaction"))["job"]
        args.raw = OUT / f"attention_online_jointaction_{job}.json"
        args.reference = OUT / "attention_online_transport_198062.json"
        args.out = OUT
    if args.raw is None or args.reference is None or args.out is None:
        parser.error("--raw, --reference and --out are required together")
    main(args.raw, args.reference, args.out)
