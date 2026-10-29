# 作者自审草案：TEAL Llama3 原作者 checkpoint 转换

状态 **DRAFT_ASSET_CLOSURE_PENDING**；尚未提交独立审查、未生成带实际资产的冻结协议、未执行真实权重读取/转换、未运行完整 LLM。模型公开获取及其独立完整重读均须实际 EXIT0 后，才能根据真实两份 RUN/EXIT 调用本 producer 的 draft，登记每个输入文件实测 SHA/size 与父回执。官方 gate 403 原始身份保留，实际对象来自 Nous 的固定 revision，公开 LFS 与官方公开 metadata 相同不等于官方 endpoint 访问通过。

转换调用固定原作者 `fb7373c93ac3594817c9ee64d4e08b47430a1822` 的 `convert_hf_checkpoint`，不重写函数或改 FP16：直接加载 4 safetensors，按作者映射重命名，Q 用 32 heads、K 用 8 heads 的原版 view/transpose/reshape 后与 V 按作者顺序拼成 QKV；其余投影/归一化等原作者映射，最终 `torch.save`。QKV 的值、顺序及 dtype 由微型两个分片的独立预期例子测试，不以源码匹配单独断言大权重语义。作者 copy 原始 `original/tokenizer.model` 至 checkpoint 目录：来自真实通过 byte bridge 的文件，而不是本范围重造 tokenizer。作者的 model.py 还导入 kernel 和 `compile_wrapper.py`；后者实际入口需增加作者 source 的 kernels 目录到 sys.path，本草案绑定此前源码源码副本的固定 SHA，未修改作者源码。原作者生成程序随后在加载模型时独立转 FP16，此处输出 expected BF16，验证实际输入输出 dtype；不声称作者推理已执行。

配置的 Llama3 形状：32 层、D4096、H14336、Q heads32、KV heads8、vocab128256。验证全部 **291 个源键和 227 个目标键**的精确名称/形状，索引与各 shard header 源归属一致，全部源/目标 BF16，逐目标键抽前 32 值有限性；有限样本并非证明整个张量 finite。每个源 shard 完整读取校验、author 的 4 shard load 和一次完整输出写、输出完整 SHA 重读、`torch.load(weights_only=True,mmap=True)` 结构/有限抽样均预扣。头扫描、加载及序列化对实际物理读流量/页缓存命中不全可见，分别给原尺寸读上限、实际入口及 UNKNOWN；不是性能 benchmark。读取源时做前后 stat 稳定检查；输入资产由真实闭合固定后再逐文件核 SHA。

候选资源：转换 CPU2/GPU0、硬虚拟地址上限 48 GiB、输出文件上限20 GiB、剩余磁盘≥24 GiB、新 HF≤21 GiB、wall5400s/outer5520s；这不是已证明能在此上限完成，原作者同时累积字典与 QKV，存在内存超限及 IO 时间超限竞争预测：H1 原逻辑在上限内产出准确可读取的完整 BF16 映射；H2 mmap/拼接/序列化在真实系统超预算或键/dtype/维度不符，必须保留失败账并停止。进程 `ru_maxrss` 是进程实测峰值，硬地址上限同时约束 RSS；SIGKILL 时以 supervisor 真实 EXIT 和产物现状为事实，不编造 finally。

闭合需单独 `..._VERIFY` ACTIVE/review、独立 supervisor 与 CPU1/GPU0、wall3600s/outer3720s、输出完整 SHA 读取≤20 GiB；验证需真实转换 EXIT0，再持久预扣/完整读取，自己的 EXIT0 后才能 `inspect-closed` PASS。不会将下载/转换 ACTIVE 自动作为验证授权。原 scope 不含任何 forward、CUDA、模型质量/速度，也不包含 tokenizer runtime；下游 32 层校准、attention/FFN 推理及 PPL 是其他 scope。

环境为新 HF prefix，复用已冻结 kernel V2 环境 site 路径、且不修改旧环境；已安装作者 torch2.4.0/triton3.0.0 + transformers4.44.1、safetensors/tiktoken/sentencepiece 等，见 `ffnds-derived/teal_author_llm_env_20261029/ENVIRONMENT_METADATA.json`。目前只确认包元数据与微型测试状态；作者 eval 选项 flash-attn2.6.1 尚未构建或兼容验证，原校准完整环境不可声称已满足。实际注册前核环境的文件/源码闭包、共享旧 site 路径不可写、候选预算及单独 reviewer，禁止用本草案作为批准。

## 实际父闭合后的提交版本

公开资产获取及独立读取已各真实 EXIT0，父闭合证明状态 PASS；固定协议从两份真实 RUN/EXIT 取 4 shard、index 与 config 的实际 SHA/size 和固定私有 cache 位置，不预设输出 SHA。各输入先以 O_NOFOLLOW 打开并持共享 flock，保留描述符；脚本通过 `/proc/self/fd` 使用同一打开对象，转换前后对 6 个资产和 tokenizer 分别完整哈希并检查 fd 身份。共享锁约束合作写者，独立 fd 防替换；不合作写者若在两次全 SHA 间短暂篡改后还原，仍是不可排除的并发反例，执行前应确认原 scoped cache 已闭合且没有其他写者，不能把锁说成强制只读。每次读预扣完整字节，后验 SHA 不符即失败、不发布成功。

转换 worker 全目录（包括日志、临时文件，排除指向输入的 symlink）的 apparent bytes 在每次 RUN 持久化时核 21 GiB、每5秒后台采样/剩余磁盘并记峰值；`model.pth` 限 20 GiB，余量包括 tokenizer 和小账。验证目录以相同方法核 64 MiB。后台采样不能证明采样间绝对峰值，突然 SIGKILL 后以 supervisor 与实际目录盘点为准。环境包元数据/RECORD及核心模块字节绑定，部分 native ELF 单独内容未重读，明确 UNKNOWN；旧 frozen 环境只经 `.pth` 只读复用。writer 不自动重试。
