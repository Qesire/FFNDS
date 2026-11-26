# 复算范围与执行

本目录记录 2026-11-26 的方法比较，当前方法证据固定于 FFNDS `154fe8620724bf1fbe6f0f9b6c5950e154468af4`，历史严格 FFN 复现证据固定于 `13ffb127e71a5d4d1519ecb0df9c7b4d03efe601`。作者源码固定提交及文件身份见三个子目录中的 `SOURCE_IDENTITIES.json`。

从本目录运行，需要 Python 3.11+ 和 NumPy，无 GPU 或模型权重依赖：

```sh
python chipmunk/verify_chipmunk_relation.py
python neighbors/recompute_structural_comparison.py
python evidence/recompute_strict_comparison.py
python recompute_lifecycle_tradeoffs.py
```

四个脚本按次序写回各自的 JSON 结果。前两个分别核验 attention 差式/条件运输和 ICC 误差恒等式，并计算真实 Wan 维度下的指定状态和算量；第三个重算 495 条历史逐来源指标、真实选列数与完整 FFN 周期；第四个计算指定短生命周期与 attention-only 作用范围的主算量。

`STRICT_NUMERIC_INPUT.json` 从已固定 Git blob 的原始记录提取数值，并保留原记录身份。完整模型输出和学校运行收据继续位于原证据目录。本次未新增 GPU 推理，不将合成代数或静态算量解释成实测质量/时延。完整当前方法与 Chipmunk attention 的同批成片及稳态计时仍是明确待补的比较。

总报告见 [METHOD_COMPARISON_AND_TRADEOFFS.md](METHOD_COMPARISON_AND_TRADEOFFS.md)。
