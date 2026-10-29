# 作者自审：high-expert Y-only 自由视频 owner

状态：待独立评审；本文件不是运行授权。两个暴露开发来源、单 seed，每源 NONE、HIGH_Y_CACHE_DENSE、HIGH_Y_REUSE 三条完整四步视频。step 0 保存各层原生 BF16 pre-gate Y，step 1 在同一 student 轨迹中复用该层 Y，当前 gate 仍由原 post_process 使用一次。此范围不使用 teacher 注入、选列或训练；它检验缓存全部高专家输出的质量与完整成本，而非 Chipmunk 作者系统复现。

旧的已闭合 A100 BF16 Dense 配对作为身份锚点：NONE 和 cache-dense 各十个状态及视频须逐字节相等；比较时拒绝路径或 inode 相同，并复核双方 fd 及路径前后身份；reuse 的 raw noise、initial noise、step0 flow 和 latent 也须逐字节相等。每源 24 个状态及两个视频比较均在候选结论前记录，独立 verifier 重核覆盖与绑定。输入闭合证据和源身份绑定在各 part 协议；实际 CPU preflight、GPU 和独立 VERIFY 均须真实 EXIT0，root 单独登记。失败保留账本，不自动重试。

每源计数：三 phase 共 480 FFN 位置、960 逻辑线性 dispatch，实际 Dense GEMM 320+320+240=880；缓存 Y clone 80 次共 25,942,425,600 逻辑字节，reuse clone 40 次共 12,971,212,800 逻辑字节，释放同步 80 次、边界 cleanup 2 次。单 expert 40 层缓存有效载荷 12,971,212,800 B，复用时另有单层 clone；逻辑字节不宣称实际物理总线复制量。原权重预取和 gate 保留。76 GiB reserved 峰值上限不重置；每源 1 GPU、4 CPU、7200 秒内层及 7320 秒外层、512 MiB 输出，完整模型同宿主最多两项，全局 GPU 上限四张。

每源 30 状态、3 视频、两个 ledger journal 交独立 typed/raw verifier；记录 state 写入、raw SHA、文件 SHA、视频几何检查内的文件 SHA 和比对读预算。ffprobe 内部读取及物理文件系统流量不可由这些逻辑账本精确推断。预扣失败或中断保留 UNKNOWN；独立 verifier 另有自己的注册、资源和 EXIT。phase 原生事件、whole video、预取、缓存和同步均应分别呈现，单次固定顺序不能证明公平系统加速或视频质量通过。

实现使用新的 owner/jobs 和 teal 的独立 Y-only tap/driver；旧已执行 producer、证据和 ACTIVE 未修改。目标 CPU 入口在隔离模式下对无 ACTIVE 的 preflight 和 verify 均拒绝，11 项 CPU 合成测试通过；没有实际模型、GPU、视频生成或评分。精确源、协议、job 和 fingerprint 由独立 reviewer 在实际版本上复核，完成修订及 recheck 前不得登记启动。
