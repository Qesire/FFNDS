# 官方 I2V NVFP4 双专家资产已完成原始字节校验

2026-10-29 · CPU 实际 exit 0。固定 producer 将 `lightx2v/LightWan2.2-A14B` 的 revision `f4b91eba1fd7dc2eb038191e2bca7ed33423032d` 下两个 I2V NVFP4 文件下载到标准 HF 缓存，并各执行一次完整流式 SHA256 校验：合计 **16,759,667,184 B**，与作者公布的两项 LFS SHA 一致。producer 用时 992.10 秒，无模型、Torch/NumPy 或 GPU 调用。具体 protocol、RUN、实际退出回执和文件身份见[机器结果](NVFP4_CHECKPOINT_BYTES_RESULTS_20261029.json)。

**SRC／文件头证据：**每个专家 2,295 个 tensor，其中 40 层 W1/W2 共 400 个 FFN tensor。全部 W1 packed 权重为 U8 `[13824,2560]`，W2 为 U8 `[5120,6912]`；对应 E4M3 weight scale 分别为 `[13824,320]` 和 `[5120,864]`，符合每 16 元素一个尺度的形状。每个线性层另外具有 F32 标量 `alpha`、`input_global_scale` 和原 BF16 bias。这些是实际完整文件的结构身份；在线激活的局部 scale、布局转换和 packed GEMM 由运行时另外产生和验证。

这一对资产属于作者的量化感知蒸馏／稀疏 attention 模型族；后续 Dense／候选须在同一权重、时间表及 attention recipe 内配对。它与原始 BF16 checkpoint 的差异不归因为单一量化或压缩收益。当前[真实硬件访问](PRO6000_R0_REFRESH_20261029.md)已核，[成熟后端](PRO6000_NVFP4_BACKEND_READINESS_20261029.md)进入隔离构建；本次结果完成资产 producer，原生 W4A4 数值、完整 FFN 和视频／系统性能仍由后续实际运行证明。

原作者入口：[模型说明](https://huggingface.co/lightx2v/LightWan2.2-A14B)、[高噪文件](https://huggingface.co/lightx2v/LightWan2.2-A14B/blob/f4b91eba1fd7dc2eb038191e2bca7ed33423032d/Wan2.2-I2V-A14B_NVFP4_Sparse_high.safetensors)、[低噪文件](https://huggingface.co/lightx2v/LightWan2.2-A14B/blob/f4b91eba1fd7dc2eb038191e2bca7ed33423032d/Wan2.2-I2V-A14B_NVFP4_Sparse_low.safetensors)。
