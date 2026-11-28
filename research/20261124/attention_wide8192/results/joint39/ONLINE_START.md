# 四步 40 层结构化 attention 首批执行

实验依据为 FFNDS@d9cb9952d0da。A 是 BF16 wide8192 与原生 attention，B/C 在高噪声均用 OUT-ABS，低噪声分别用 FIN-C0 与 OUT-D。0392、1013 的 seed 42 原生 attention 和 Dense 对照视频已验明指纹。

> 后续已完成四条候选。[完整质量与状态收据](ONLINE_QUALITY_REPORT.md)记录最新进展。

## 已验证

- 学校 A100 作业 194054 执行实际全长后端。融合 Flash SDPA 的原生输出逐位一致，LSE 最大绝对误差为 2.86e−6。V128 与增广 V144 的相对 L2 误差约为 0.226% 与 0.227%。
- 学校 A100 作业 194153 用 0392 封存 Q/K/V 在第 0、39 层回放低噪声 FIN-C0 和 OUT-D，H448 读出最大相对差为 0.283%。非面板查询仅形成全长 kernel 形状，评分使用原面板查询。
- 作业 194145 的 B/C 同输入核验均复现了 A 的四步状态和解码视频。在第一条 B 自身轨迹上，少量饱和 query/head 即使用共同 FP32 logits，两次 logsumexp 相减仍有舍入问题。

## 当前运行

异常 query/head 已改为直接累加未选中键的正 softmax 权重，除以全键和。正常行继续使用 fused LSE，代码记录复算的数量。完全饱和与末尾不完整 key block 已在确定性数值输入上检查。

学校质量作业 `194188` 已开始执行。此次提交只登记已验收的接口、局部证据和启动状态。候选四片与配对计时待完成。输入封存 SHA256 为 `0843591bee873b251fe9ae4e3f0133b6b04b2d1aca2bcc071ef6c8eda61505a6`。

[在线执行器](attention_online_structured.py) · [后端收据](ONLINE_BACKEND_194054.json) · [真实面板收据](ONLINE_PANEL_194153.json)
