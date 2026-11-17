# Wan2.2 四步 Dense 鱼鳞纹 · USTC 学校集群

**2026-11-17 · SCHOOL_ONLY / DESIGN_ONLY / NOT_RUN。** 旧 `ffnds-worker` 和 `/home/xiang` 只属于历史研究；**没有任何远端 S2 视频/latent 是本诊断的运行依赖**。本轮只用学校 Slurm 下经严格验证的学校资产，或在学校新捕获最少一条 Dense。没有通过实际计算节点文件可读性检查，也没有生成新视频。

| 文件 | 内容 |
|---|---|
| [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) | 完整学校 P0→P1/P2→A0/A1/A2/A3→按需B/C 路线、物理边界、人工关口、资产树 |
| [SCHOOL_CLUSTER_CONFIG.json](SCHOOL_CLUSTER_CONFIG.json) | 学校根目录、模型路径候选、预期 checkpoint SHA、四步执行配方（所有实物待计算节点核验） |
| [ASSET_REGISTRY.json](ASSET_REGISTRY.json) | **学校专用**登记：`school_source_pairs=[]`，没有假定 S2 14 条文件存在 |
| [EXPERIMENT_MATRIX.json](EXPERIMENT_MATRIX.json) | 学校资产复用和缺失时单条 Dense 新捕获的条件预算 |
| [SCHOOL_CLUSTER_RUNBOOK.md](SCHOOL_CLUSTER_RUNBOOK.md) | 学校 compute-node 严格 preflight 与完整视频交付命令 |
| [HUMAN_REVIEW_PROTOCOL.md](HUMAN_REVIEW_PROTOCOL.md) | 完整视频由用户人眼主导评审，机器指标只辅助 |

**学校统一执行根**：`/public/home/scc/pb23061276/projects/wan-quant`；较早学校文档 `/home/scc/pb23061276` 只有在当前计算节点确认同一真实目录时才可作为别名。所有新输出写学校 `experiments/ffnds_fishscale_20261117/`。并不意味着目录目前已建立或输出已有视频。

- [原学校模型配置](../../../docs/01_模型与服务器配置.md) 给出 D-BF16、原版 O 的学校路径；本诊断需当次验证实际 high/low/VAE 字节、原生 LightX2V 代码及项目目录。
- [历史 S2 记录](../../20261029/S2_NEW12_DENSE_BASELINE_RESULTS_20261029.md) 可供回顾研究，但其 MP4/latents 保存于旧独立 worker 的 HF 逻辑根，**不能被当成学校可读输入**。
- **注意**：既有 `tools/run_distill.py` 在保存 latent 时会调用 `.float()`。新 capture 必须保留真正 native BF16 latent，不能只把该 FP32 观察文件更名。

此文档和脚本不是模型执行许可。新的学校 CPU/GPU 模型范围必须先完成 FFNDS 预执行独立复审与 ACTIVE 注册；只有真实学校 RUN/EXIT、可观看完整视频和用户反馈才能产生实验结论。

学校端准备工具：[`fishscale_school_preflight.py`](../../../tools/fishscale_school_preflight.py)（Slurm 严格路径/全文件 SHA/视频规格/原生 BF16 latent）；[`fishscale_school_register_pair.py`](../../../tools/fishscale_school_register_pair.py)（本地学校视频与同次运行登记）；[`fishscale_school_native_capture.py`](../../../tools/fishscale_school_native_capture.py)（学校原生 Dense BF16 状态观测组件）。纯合成回归：`tests/test_fishscale_school_preflight.py`，当前尚未实跑。
