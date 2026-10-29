# 协调者启动核对：单个固定 Alpaca parquet

协调者 `/root` 核对作者自审、真实独立 V1 REVISE、作者 V2/V3 修订及精确 recheck APPROVE。待启动的 V3 只获取固定公开 revision 的一个 24,246,638 字节 parquet，同一进程两次完整 SHA；文件内容解析、模型与 GPU 调用均为0。下游校准的数据记录/分段有另一个尚未注册范围。

已直接阅读 producer 与独立 recheck：下载和两次 hash 分别预扣；backend 内部网络重试和网络字节为 UNKNOWN，受实际 wall 限制，不称一次 HTTP 请求。worker 128 MiB 是退出时盘点，非硬文件系统配额。NFS 可变后代消失可跳过计数，根身份和其他错误仍失败；超预算失败会抛错，真实 supervisor EXIT0 与两次已完成 hash 均为闭合必要条件。闭合额外只读小 JSON/stat，不增加第三次全文读取。

启动前将重新核验精确 protocol/source/runtime/spec、实际 worker、独占新 cache、可用 CPU0 和前一 verifier 的实际退出；仅 root 登记 ACTIVE。后续读取资产须各自固定/校验输入；同尺寸非协作恶意改写不在 stat-only 闭合保证内。当前运行结果尚未产生，本核对不声明数据已获取。
