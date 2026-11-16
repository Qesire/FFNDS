# 软件验证记录 · 2026-11-17

本包在 FFNDS 基础提交 `336b7e11849875e02d25205497a594d229211758` 的新增工作文件上验证。

## 实际执行

环境：Python 3.12.14，NumPy 2.3.5。当前环境没有 PyTorch、Wan checkpoint、LightX2V CUDA 运行或 VBench 模型权重。

```text
$ python -m unittest discover -s tools/video_metric_feedback/tests -q
----------------------------------------------------------------------
Ran 87 tests in 1.432s

OK
```

另实际执行 `python -m tools.video_metric_feedback check --config tools/video_metric_feedback/examples/config.example.json`：配置结构通过，输出两个参数、首轮最多 6 组、模板中 1 个 image-cluster / 1 个样本，因此最多 6 段候选视频，`model_calls_made=0`。模板标识与资产路径是占位值，该检查不宣称资产可用或范围已绑定。

## 证据范围

| 检查 | 支持的结论 | 不支持的结论 |
| --- | --- | --- |
| 小矩阵 NumPy fixtures | BF16 舍入、端点/朝向、完整 manifest 和内容指纹的对应软件语义 | 生产 5120 维参数效果、GPU 重建逐位等价 |
| 解析目标及约束 fixtures | 差分、投影、验收、预算和固定参照 guard 的算法语义 | Wan 视频分数会按解析函数变化 |
| 临时 argv 子进程 | 请求/结果绑定、缓存、噪声错配拒绝、失败重启不越预算 | 已调用真实视频 producer |
| Fake native runtime | 每样本缓存生命周期、HHLL 覆盖检查、动态模块身份、异常恢复和输出不覆盖 | 实际 CUDA/LightX2V 张量运算已通过 |
| 官方格式与 compute 替身 | MUSIQ 尺度、camera 结构、eligibility、完整生成 manifest 和评分映射 | 已运行官方评分网络或完成官方 benchmark |
| CLI 组装 fixtures | 本地诊断可独立组装、精确 Dense 参照内容绑定、synthetic 标记保留 | 已取得真实视频质量或公平 SYS 结果 |

本次真实模型生成调用 0，真实模型评分调用 0。生产候选系数未由本次验证选定。参数导出/native/官方 worker 已有对应实现，但第一次真实环境运行仍需验证资产、依赖、完整解码与物理成本。

复核修正了实际内容指纹与参数来源指纹混用、逐视频 MUSIQ 尺度、合并时 synthetic/请求身份丢失、local-only 组装、失败次数重启记账，以及 native 动态模块登记和输出锁范围等问题。测试通过后未继续扩展候选或模型试验。
