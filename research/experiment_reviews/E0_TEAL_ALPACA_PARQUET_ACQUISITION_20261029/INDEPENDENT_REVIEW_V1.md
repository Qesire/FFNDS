# 独立初审：Alpaca 固定 parquet 获取 — REVISE

2026-10-30，独立评审 `/root/sm120_full_quality_build_terra`；作者 `/root/teal_full_llm_calibration_build_terra`。审阅 producer SHA `fbf873420a749a3613ca08ae56a2fb0d2c32811bd3ea4ab563b77c0ef9c40b92`、draft SHA `1fe2005139b0027b996e5e6f86ba23e5ed38982b53c2afc5d8545e38cbbd5b9a`、作者自审及测试。没有下载 parquet、读取记录、tokenize、运行模型或 GPU。本条意见不登记 ACTIVE。

## 身份与范围

只读查询 [固定 Hub revision metadata](https://huggingface.co/api/datasets/tatsu-lab/alpaca/revision/dce01c9b08f87459cf36a430d809084718273017?blobs=true) 实际返回相同 revision，唯一 parquet `data/train-00000-of-00001-a09b74b3ef9c3b56.parquet` 的 LFS size 24,246,638 和 SHA256 `06391b656a06fd3fb9d213160ef2398796c3b7f3dc75ef1e3ced30d461517073` 均匹配 draft。原作者未固定该 revision；项目固定当前版本是明确适配。两个同进程全文件 SHA pass 足以作为本 24 MB byte scope 的规定验证，不能称跨进程独立验证，也不需要人为新增第三次全文读取。HF 底层重试与网络线缆字节 UNKNOWN，不可用一次 API 入口替代。

## 可推翻或阻断的具体问题

1. **NFS 可变输出扫描失败账**。finally 使用冻结 helper `common.usage(out.parent)`，该 helper 对文件先 `is_symlink()` 后 `stat()`，`.nfs` 或其它可变项恰在两次调用之间消失可触发 FileNotFoundError，覆盖原异常并跳过最终 RUN 持久化。应只在本新 producer 对可变 worker 目录采用受测扫描：`lstat`/walk 遇严格后代的 ENOENT 计数跳过，根目录缺失/替换及其它 I/O 错误硬失败；预留足够失败记录或阐明异常无法落账时由实际 supervisor EXIT 和目录盘点保留。输入 blob、SHA、源绑定的缺失不应被此例外吞掉。提供可变项真实消失与其他错误反例测试。
2. **资源失败不能 EXIT0**。`finally` 超 128 MiB 时只修改 RUN status，之后正常返回，supervisor 可实际 EXIT0。须先持久化失败再抛异常；任何闭合校验同时要求 SUCCESS RUN 与实际 child EXIT0，不能以单个字段代替。128 MiB 是退出时目录检查及后端单文件限制，不是整个文件系统硬配额；为执行中间峰值提供可核的界或监测，明确 SIGKILL/扫描间隙观测范围。4 GiB 是 RLIMIT_AS 而不是实际 CPU RSS 测值。
3. **预扣/失败时读取计量**。后端调用前明确预扣一次目标文件最大 payload 与一次 API 调用；两个 SHA 每次开始前分别预扣 24,246,638 B。`common.verify_file` 若中断不暴露已经读完的 chunk，不能声称失败实际字节为0；记录 inflight 与已完成字节 UNKNOWN 或逐块计量，并保留 wall/CPU 时间、实际 RSS、失败原因。两次全 hash 合计上限 48,493,276 B；不加无预算的第三次全文件读取。
4. **实际闭合与启动身份**。提供精确私有 durable spec、协议/worker/CPU/环境/代码闭包和 ACTIVE 绑定，以及只读取小 RUN/EXIT JSON 的 closed 验证：检查真实 child EXIT0、非 timeout/no retry、RUN SHA 和协议/spec 身份、唯一 parquet 大小/SHA、两次完成账及预算；不要在闭合验证中重读 parquet。模型/GPU/数据记录读取依然为零；应在导入下载库前后检查数值模块导入范围，或明确未观测范围。补合成成功及失败路径测试。

## 后续

作者逐条回应并修订后，提交具体源码、测试、协议指纹和 spec 哈希；本 reviewer 再对确切版本重新核定。原先 V1 `REVISE` 始终留历史，未来的实际获取需独立复核及 ACTIVE；不能将此初审写成批准。
