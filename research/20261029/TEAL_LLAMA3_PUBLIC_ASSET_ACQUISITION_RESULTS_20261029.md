# TEAL 原 Llama3 公开资产获取与独立复验

固定公开副本 `NousResearch/Meta-Llama-3-8B@315b20096dc791d381d514deb5f8bd9c8d6d3061` 的13个文件已实际获取，随后另一个已登记CPU任务完整重读并验证 **16,069,766,580 字节**。两任务均真实 EXIT0，最终状态为 `PASS_CLOSED_PUBLIC_ASSETS_NOT_MODEL_EXECUTED`。四个权重分片的实际SHA256/大小与固定官方公开LFS元数据一致；配置、索引和分词器相关文件按各自真实Git blob/SHA核验。README、USE_POLICY保留公开再分发版本身份；原官方入口HTTP403事实不变。

| 实际计量 | 获取 | 独立复验 |
| --- | ---: | ---: |
| Worker wall | 763.0524秒 | 22.4380秒 |
| CPU时间 | 169.4136秒 | 19.6632秒 |
| 峰值RSS | 106,471,424字节 | 31,227,904字节 |
| 完整应用层哈希读取 | 16,071,950,562字节 | 16,069,766,580字节 |
| 模型/GPU调用 | 0/0 | 0/0 |

获取阶段的额外读取为先前已闭合的2,183,982字节原始tokenizer。实际13次Hub文件入口调用、0个已有blob复用；底层HTTP请求总数与物理网络流量未暴露，记UNKNOWN。新缓存最终apparent大小16,069,766,771字节，完整worker目录在最终RUN前16,069,781,865字节。文件大小与全目录预算是锁定后端及采样检查下的运行限制，不是文件系统硬配额。

资产保存在HF下的本范围专属**标准Hub缓存布局**，不覆盖共享缓存：

`ffnds-derived/e0_teal_llama3_public_weight_acquisition_20261029/attempt1/worker/scoped_hf_cache`

对应 `snapshots/315b20096dc791d381d514deb5f8bd9c8d6d3061` 位于 `models--NousResearch--Meta-Llama-3-8B` 下。消费者必须显式使用该缓存位置并核验原绑定。[完整结果](TEAL_LLAMA3_PUBLIC_ASSET_ACQUISITION_RESULTS_20261029.json)保存每个文件的路径、大小、SHA/Git blob、实际预扣与完成、两份RUN/EXIT及独立闭合摘要。

这些已验证文件与[字节相同的原始tokenizer](TEAL_LLAMA3_TOKENIZER_BYTE_BRIDGE_RESULTS_20261029.md)补齐作者完整LLM的输入资产。原作者checkpoint转换、运行时FP16、校准与完整模型质量/速度仍分别等待真实执行；资产成功不代表LLM复现已通过。
