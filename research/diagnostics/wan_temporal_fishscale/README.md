# Wan2.2 I2V 四步 Dense 鱼鳞纹排查

**2026-11-17 · DESIGN_ONLY / REVIEW_REQUIRED / NOT_RUN。** 尚未新增 CPU/GPU/VAE 运行、完整诊断视频、人工评审或根因结论。

此目录是 FFNDS 的独立视频质量诊断，不修改 FFN 稀疏方法、原有权重或历史实验。

> **2026-11-17 路径纠正**：S2 的 `root=hf` 是旧执行环境的逻辑存储根。其 GPU runner 的 `--hf-root` 默认 `/home/xiang/.cache/huggingface`，原模型 view 在 `/var/tmp/ffnds-xiang/huggingface`；没有证据证明这些文件存在于学校 `/public/home/scc/pb23061276/`。旧 worker 通过 SSH `ffnds-worker` 运行，跨 worker 主机共享 home 不等于跨学校集群共享。A0/A1 等依赖旧二进制的阶段必须先在实际执行节点核真实路径与 SHA，否则标为资产不可访问并转为新集群最小基线。详见 [实验第 0 节](EXPERIMENT_PLAN.md#0-执行环境与资产位置2026-11-17-补充纠正强制先行)。

- [完整分层实验和资产树](EXPERIMENT_PLAN.md)：A0 人眼确认历史视频 → A1 RGB/编码 → A2 BF16/FP32 → A3 干净 VAE 重建 → 必要时 B0/B1/B2/B3 → 跨来源确认。
- [14 条 Dense 真实资产身份](ASSET_REGISTRY.json)：历史 S2 原始 MP4、末步 latent、全部四步状态与 SHA；这是只读索引，worker 尚未重新核验文件。
- [机器矩阵与预算](EXPERIMENT_MATRIX.json)：A 层最多 0 DiT、2 次 VAE encode、6 次 VAE decode；每轮后用户审片决定是否继续。
- [人工审阅规范](HUMAN_REVIEW_PROTOCOL.md)：完整无损 FFV1 + MP4 双份视频必须保留，用户观察优先于所有自动指标。

另有 [G19 原版40步 Dense](../../20261105/G19_METHOD_AND_RESULTS_20261105.md) 和 [四步 G19 Dense](../../../reports/20261105/g19_evidence/browse/outputs/G19_v7_真实生成与质量证据_20261104.md) 同为 chip10_00/seed43，可先恢复并核对真实视频，作为 B3 的零新增生成描述性对照（注意 CFG/权重/步数差异）。

新模型运行必须经过 FFNDS 现行独立预执行复审和 ACTIVE 注册；此文档不是任何作业许可。
