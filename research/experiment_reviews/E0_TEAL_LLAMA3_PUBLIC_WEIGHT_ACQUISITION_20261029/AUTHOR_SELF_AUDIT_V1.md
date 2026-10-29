# 作者自审：TEAL Llama3 公开资产获取

状态 REVIEW_REQUESTED；这是单次资产范围，未下载大权重、转换、加载模型、运行 GPU 或修改 ACTIVE。作者 `/root/sm120_full_quality_build_terra`；真正独立 reviewer 尚未投票。固定 NousResearch/Meta-Llama-3-8B@315b20096dc791d381d514deb5f8bd9c8d6d3061；官方 meta-llama 原 endpoint 实际 403 保留。四分片只声称其 LFS size/SHA 与官方公开元数据匹配，实际字节要由本范围读取和校验后方可声明。LICENSE 与官方 Git blob 匹配；README/USE_POLICY 是再分发仓库版本，不能用它们推定官方入口可访问。访问公开对象本身不替代对许可证实际适用条款的判断。

对照预测：H1 四分片按固定公开元数据下载并完整 SHA 一致，配置与索引的 Git blob 相符；H2 镜像对象、网络、缓存并发写入或磁盘不一致导致失败，保留 predebit/RUN/EXIT，不自动重试。13 文件总计 16,069,766,580 字节；最大单文件 4,999,802,720 字节。校验各完整 SHA256，小文件同时校验 Git blob SHA1；四 LFS 分片比对完整 SHA256。已存在的相同 blob 在 hub 调用前后各完整哈希，已有 partial 一律拒绝；新下载被逐文件 RLIMIT_FSIZE 限定到 max(该文件期望字节, 16 MiB)，以避免对小文件施加的进程级限制误伤 supervisor 日志；所有文件限额之和再加 64 MiB 余量仍低于 16 GiB，库内临时文件在标准 cache blobs 目录，异常时库通常清理；SIGKILL 可能留下 partial，该范围不会自动续跑。对每个 blob 的完整读取提前持久化预算，按 64 MiB 持久化进展，异常读取数最多再加一个 1 MiB 块；数值是应用层读账，不是断电耐久保证。verify-closed 再完整读取全部对象，另计 16,069,766,580 字节和墙钟，不属于下载计时；这项闭合只读取文件字节，不加载 tensor。

一次 scope 只有顺序 13 次 `hf_hub_download` 入口（包括缓存命中），固定匿名 token=False、禁隐式 token、禁代理及 XET，完整 revision；hub 1.20.1 源码 GET 对 408/429 backoff，连接/超时/协议流错误允许最多 5 次连续无进展，收到数据便重置，因此总请求数和线缆字节不可从入口数推断，记 UNKNOWN；单次 worker 7200 s 与 supervisor 7320 s 为硬边界，失败不自动 scope retry。HEAD 元数据后端也有 backoff。监测写入上限由顺序预绑定文件大小与每文件限制确保；共享标准缓存中其他独立进程并发写、额外 backend 产物、SIGKILL 留 partial 是待独立 reviewer 检查的反例，若不能确证隔离不能登记执行。

资源：CPU1/GPU0，AS/RSS 上限 4 GiB，初始剩余磁盘至少 24 GiB，新 HF 上限 16 GiB（包括目标及小量 supervisor 日志；标准缓存中已有对象不计新写）。先前 tokenizer bridge 的真实 RUN、EXIT 与 2,183,982 B/官方 SHA256 原字节以 HF 相对路径绑定和运行前再验；不进行 tokenization。代码绑定 .venv Python 与 huggingface-hub/httpx 等分发包的文件哈希与版本，绑 worker profile 和私有 supervisor spec；现有环境预装包为工程环境，尚未用本 scope 联网验证实际后端。每条 RUN/EXIT 记录不得假设完成的网络字节或无效文件读数。

## 源码到后续作者完整 LLM 的对应

[固定提交作者源码](https://github.com/FasterDecoding/TEAL/tree/fb7373c93ac3594817c9ee64d4e08b47430a1822)和仓库 source manifest 已读：

- 作者校准为 `teal/grab_acts.py --model_name <本地或固定模型> --output_path <目录>`，非 `generate.py --calibration`；示例 Alpaca train 前300条，截断为至多10×2048，生成逐层 histogram。数据、环境、分片和加载产生的成本需要另立校准 scope。历史预计算 hist 身份及 pickle 信任也须独立验证。
- 作者转换是 `gpt-fast/scripts/convert_hf_checkpoint.py`，把 HF shard 转为 `model.pth`；须另行 review、预留额外磁盘及 host 内存并绑定转换结果。`gpt-fast/generate.py --checkpoint_path <.../model.pth> --hist_path <.../histograms> --sparsity 0.5 --compile` 使用同目录 `tokenizer.model`，FP16 与作者模型配置，CLI `--compile` 默认关闭；打开时含额外 warmup 并编译单 token decode，prefill 仅在另指定 `--compile_prefill` 时编译。作者稀疏实现多 token prefill 走 dense fallback，单 token decode 走稀疏 GEMV。dense 原模型、显式零阈值 neutral 和 candidate 应在后续范围同输入同 RNG 对照。
- 作者质量入口为 `teal/ppl_test.py --model_name <模型> --teal_path <hist目录上级> --sparsity 0.5`，默认 Alpaca train 前250条，与校准前缀重叠；HF quality 路径多 token 后半 mask，与生成路径 prefill 的未 mask dense fallback 不同。作者示例不等于论文 WikiText 128 样本，held-out 数据及评分单位须独立冻结。

资产匹配可解除 checkpoint 身份的一项缺口；权重转换、tokenizer runtime、校准、完整 LLM 质量和系统吞吐均未执行。不得以本 scope 或 prior 单 kernel benchmark 声称完整作者 LLM 已复现。


初审后修订见 [作者回应](AUTHOR_RESPONSE_V1.md)：独占 scope cache、独立 reflink 复用、scope 总预算和单独 durable closed 验证取代本初版中共享 cache 路径及无收据的重读描述；本初版其余科学对象和作者入口说明保留。
