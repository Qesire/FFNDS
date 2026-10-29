# SharQ Wan 全160视频 scope 作者自审（待独立评议）

各 part 是一个历史暴露开发 source（1040 / 979）、seed42、四步×40层 I2V。每 part 先做两次同机 BF16 生成并要求完整10状态和视频字节 neutral，成功后真实生成 Dense32 与 SharQ，各臂 raw/initial noise 配对，共 4 条、640 FFN、1280 W1/W2 实际入口。两个 part 共 8 条、非独立 seeds 的两来源；本 scope 不自动授予质量、系统性能或官方 VBench 验收。竞争预测和双缓冲源码分析见 `research/method_discussion/discoveries/E0/p0-sharq-wan-all160-same-master-video-design-20261030.md`。

本 owner `tools/e0_sharq_wan_all160_video_owner.py` 实现 `draft/freeze/inspect/preflight/run/verify-closed`。两份 frozen pending 仅完成目标 worker 的原 SM120 master proof/author ELF、本地 package 和 source bytes 检查，仍需独立初审、作者回应、reviewer exact recheck，随后 root 登记 ACTIVE，不能据此运行。运行按 exact private CPU/GPU job、profile SHA、scope fingerprint、不同作者批准、一个独占 UUID、actual CPU preflight EXIT0 和 fresh attempt 门禁。原 4 卡会话上限、单 scope 1 卡、SM120 whole-model group 同时最多2；默认 private job暂用实际检查为空闲的物理卡0/1，调度前必须重新检查，如更改 spec 必须更新 ACTIVE/recheck 的确切 spec 身份。两 job CPU亲和性分别 0..3、4..7。

权重均来自原 BF16 master，在第二条 BF16 neutral 使用原实时 GPU block buffer 时实付 160 次 W 转置/materialize、原作者 shared-W FP4 打包和缓存；每个 part Dense32/SharQ 在两步和候选臂复用同一个 per expert/block/linear payload，W1/W2 bias、tanh GELU 和 gate均留在原 native 路径。每个候选视频 320动态量化线性，Dense32 320 dense GEMM；SharQ 320 sparse GEMM **加 320 full-K dense residual GEMM**；所有 4 视频 648 次原 FFN block prefetch observation，原 W1/W2 H2D 原样执行，不能当成节省。所有静态转换 GPU读取/materialize字节与 packed cache 保守字节均预扣；packed cache上限16GiB，完整 GPU reserved 软上限80GiB。大M、CUDA图形和潜在碎片化实际可能导致资源失败，不能预先声称能跑或速度改善。

scope 每项 host可用下限128GiB、磁盘空闲8GiB、最大7200s（supervisor 7320s）、输出2GiB；CPU预检 4 CPU/900s/16MiB，独立注册的 fresh verifier 0 GPU/1 CPU/1800s（外部1920s）/46个输出 artifact SHA读最多2GiB，最终必须真实 verifier EXIT0，逐文件持久预扣和完成、失败当前文件读取进度 UNKNOWN，小型协议/RUN/EXIT源身份读取另计。模型 checkpoint此前实际全文 SHA proof 必须保持同 worker stat/header；从 GPU现有 buffer量化不新建大权重资产。通过 offline 环境、本地包和已绑定源码/ELF 禁止方法内部有意下载；必须由独立评议实际导入闭包和意外联网路径。

账本由旧已验证的原生 NativeTap + 本 scope实际 dispatch/prefetch 两层组成，所有 linear operations 在真实调用前整组预扣，返回数与调用入口分记，失败保留 UNKNOWN CUDA完成；FFN event和wall含 observer、predebit日志和 第二条 BF16 neutral 静态打包。首条 BF16 不安装本 scope 的 FFN 替换；第二条只建立缓存并调用原 BF16，两条真实状态/视频须逐字节一致。Dense32先、SharQ后使用已经建立的同一缓存，仍有 warmup/order偏差，报告原 wall、每个static pack wall、操作/预取账和资源/观察开销，不能当相同冷启动的公平 SYS 比较。旧实际 SharQ V2 单算子 `RUN`、`EXIT0`、closed 与全部代码身份受绑定；V1及V2旧原证据未改。

请评议者重点找：完整原执行者/本 scope的 repo/import闭包、外部 preflight 真实退出、四阶段状态文件持久化与验证读预算、8 buffer patch 安装/流时序与偏置、动态量化/实际矩阵和全路径两 GEMM 计数、当前 staged GPU/CPU/host/diskscope 预算及私有job、164/162 prefetch 边界、FAILED时部分状态/视频账、运行后全文件 SHA 及 exact EXIT闭合。只有来源开发结果，不得据此扩展到独立人群、完整系统或正式质量通过。

补充：CPU预检和GPU启动各自完整SHA读取上限512MiB并分别预扣；小metadata读取另列UNKNOWN。失败时持久化已完成与在途UNKNOWN。初审回应见 AUTHOR_RESPONSE_V2.md，需独立exact recheck。
