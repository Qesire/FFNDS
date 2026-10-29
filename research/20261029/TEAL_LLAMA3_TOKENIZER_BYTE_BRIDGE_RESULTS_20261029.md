# TEAL Llama3 原始分词器文件重建结果

固定公开副本 `NousResearch/Meta-Llama-3-8B@315b20096dc791d381d514deb5f8bd9c8d6d3061` 的 HF tokenizer 词表，已真实逆变换为原始 tiktoken 文件，实际 EXIT0 并通过闭合验证。产物 **2,183,982 字节**，SHA256 **`82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55`**，与官方公开 LFS 元数据完全一致。

原匿名输入 9,085,698 字节，Git blob `b197f72effb9d5ed16ee0f5663e11e4cfac2ba62`；实际解析 128,000 个普通词项及 256 个独立特殊词项，覆盖全部 256 个单字节。ByteLevel 逆映射与按 rank 写出的 base64/空格/十进制/LF 序列化通过逐项回读和原文件大小、SHA 双重检查后，才以不可覆盖方式发布。许可、notice 及公开副本身份随范围保存；此前官方入口 HTTP403 事实保留。

| 实际资源 | 结果 |
| --- | ---: |
| Worker wall | 2.061455 秒 |
| CPU 时间 | 0.803727 秒 |
| 峰值 RSS | 113,508,352 字节 |
| HTTP 响应 | 307 → 200，一次允许的重定向 |
| GPU / 模型 / Torch 调用 | 0 / 0 / 0 |

输入、逐文件逻辑读取、应用层写入、操作预扣与完成、退出回执均在[完整结果](TEAL_LLAMA3_TOKENIZER_BYTE_BRIDGE_RESULTS_20261029.json)。应用层计数不代表物理网络流量或主机断电持久性。RUN SHA256 为 `e7af086ee9b96359248ca0debc012f4881a9a1e39a4f81e778a5e7a493f18d89`；实际 EXIT 为 `0fcd65288f65b805f94d8809991a9989876e2e279ae95f348fec6c94a01153a1`。

产物位于 HF 缓存 `ffnds-derived/e0_teal_llama3_tokenizer_byte_bridge_20261029/attempt1/worker/results/tokenizer.model`。这解决作者 gpt-fast 所需的原始文件缺口；特殊词编码、chat template、完整 LLM 校准、质量和速度仍需各自真实运行。下一步为固定公开权重与配置的有限资产获取，以及原作者 checkpoint 转换和完整模型验证。
