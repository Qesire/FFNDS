# TEAL Llama3 作者全32层校准 V2 实际结果

2026-10-30，有限开发校准 **实际 EXIT0，inspect-closed PASS**。作者 `grab_acts.py` 使用固定公开 Alpaca 数据前300条文本，预检与作者读取各一遍，实际前10段均为2048 token，无 padding；两次 token IDs 与 attention mask 的 SHA 一致。相同输入在 Nous Llama3 8B 的 FP16 路径完成32层，保存64个 attention/MLP histogram：实际128次 GPU sort、128次 CPU histogram、32个层输入，以及32次 layer forward pre/post。统计对象是固定开发数据，不是独立确认样本。

| 观测 | 实际值 |
|---|---:|
| 子进程墙钟 / CPU时间 | 358.003 s / 407.010 s |
| supervisor 外层墙钟 | 359.099 s |
| 进程最大 RSS（getrusage） | 6,221,262,848 B |
| 采样全卡显存峰值 | 20,000,538,624 B |
| finite observer | 64次，5,368,709,120元素，累计同步墙钟 1.184 s |
| 64个 histogram 文件 | 10,365,440 B |
| 32个层输入文件 | 5,368,744,622 B |
| histogram 验证逻辑读取 | 20,730,880 B，0.446 s |
| 输入模型资产 SHA | 16,069,766,580 B |
| Alpaca parquet SHA | 24,246,638 B |
| worker 最终表观占用 | 5,379,127,299 B |

显存及 worker 峰值为5秒采样；瞬时峰值、模型内部读取字节和物理 GPU kernel launch 数未知。histogram 的 SHA 由实际 producer 记录，closed 检查比对当前文件 stat；本报告未重新读取或 hash 5.37 GB 层输入。模型内部实际加载和 observer 成本包含在上述墙钟中，不能将其解释为推理系统速度。

V1 尝试保留实际 **EXIT1**：GPU 条件下 Transformers 的 FA2 导入缺少 `einops`，模型加载、forward、sort 和 histogram 均为0。V2 是独立环境与独立审阅的新范围；它的成功不抹去 V1 失败。

V2 RUN SHA `941cfb7d94b27f63f9c958f37b94bd06fa395eb0f9037e5b2636151eafbafc25`；真实 EXIT SHA `f074c3ed73b960ffb9e2935d67638fa2aaeeb0773574ea568505b5109e213c28`。完整机器可读标量、来源及先前失败身份见 [结果 JSON](TEAL_LLAMA3_AUTHOR_CALIBRATION_V2_RESULTS_20261029.json)。该校准支持后续真实 histogram 消费，不是生成质量、PPL、FFN-only 或论文系统性能复现。
