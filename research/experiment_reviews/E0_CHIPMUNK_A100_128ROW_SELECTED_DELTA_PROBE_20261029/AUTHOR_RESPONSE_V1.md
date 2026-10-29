# 作者对独立初审 REVISE 的回应

初审四项均接受；下面对应修订，不将静态检查写成已测数值。数值核本体未改；producer/test/draft 均产生新指纹。

1. **同选集独立参考。** 每个 initializer × empty/full/fixed 均以 M128 Dense 当前 h 按同一选集构造 reference delta `(h_M128-h_old).to(BF16)`；用该 delta、单独 gather 的 W2 和禁 TF32 的 FP32 matmul 加已有 Y 后再 BF16 取整，比较 Y 和当前 BF16 gate 后输出。selected h 对 M128 dense h 比较、逐 BF16 delta 对参考比较；未选 h 必须 bitwise 不变。另用实际新 h 构造 delta，要求与核产生 delta bitwise 相等，并用它构造独立 FP32 Y 来隔离 W2 阶段。固定先验阈值分别 selected h relative-L2 ≤0.015、delta ≤0.05、Y 和 gate×Y ≤0.015；归约、GELU 和 BF16 取整造成的差异仍属数值实测问题，超界直接 FAIL，不随数据调整。完整更新对 M128 的原0.03上限保留为另一整体控制。FP32 是参考，不把 BF16 delta 标为 FP32 计算精度。六个参考调用预扣和同步完成，两个缓存初始化独立保留。静态标量对照覆盖非零初值、b2、正负delta、未选通道及重复加bias反例；GPU数值仍未知。

2. **数据读取账及身份。** 两次 payload 全文件 hash 与一次完整 checkpoint hash 分别先预扣；每8MiB读取记录已完成字节，期间按256MiB周期持久化，异常 finally 记录完成读取字节与最多8MiB的未计入在途读取上界，不把在途量当已完成。hash 前、后及 tensor load 前后比较同一文件的 device/inode/size/mtime/ctime；tensor load 分别记成功逻辑字节、正在读取的 key 和最终 stable status，物理磁盘读取量 UNKNOWN。请求大对象前，按完整 checkpoint、两 payload、全包所需 tensor 及四个 checkpoint tensor 的逻辑字节计算计划总上界，并检查 30GiB 范围。64GiB host 峰值 RSS 由进程高水位在关键边界与 finally 检查，明示**并非连续硬上限**；越界标失败，实际物理页读取也不等于逻辑字节。训练/推理 checkpoint 未在本阶段读取。

3. **开始、异常与资源。** 创建 fresh 输出后立即持久化 STARTED，再进行 GPU 导入、分配和读取；所有正常/异常 finally 记录 worker wall、峰值 host RSS、包含编译缓存的输出字节、资源越界及 GPU 监控结果，且 success 必须通过完整范围检查并验证 actual EXIT0、准确 source/receipt/spec。输出预算留下64KiB RUN/receipt预留；资源检查在加载后、每 case 和最终执行，尚非对瞬态峰值或目录写入的连续硬限制，这一限制明确交独立评议。监控异常也在资源账保存后抛出。单测增加中断时预扣/inflight、文件身份变动和已读取字节持久化、host/output 越界与 closed case 错误拒绝，10项非实验性测试通过。

4. **范围与性能。** RUN/design 明确标记 AUTHOR_PROXY_NOT_EXECUTED、成本未知；固定集不是 top/random 作者选择。一次同步 wall 中存在 ledger、分配和可能的编译，只作诊断，不作加速结论。scatter 只将128项 `index_copy_` 到其余位置未初始化的占位，明确不作为 full permutation、完整输出或全M系统。本文仍是一个历史暴露 teacher source、一组、两步；作者均值代理相反token抵消的反例保持开放，原raster分组、自由student和全M性能均需独立新scope。

修订待审草案 SHA `15191a6e4c0d96278c730d1d0b1eb26e55e04e07e299bf6ce955f210742dc5b3`，fingerprint `dea01f9519448d3e9cf68545c747394e0b6a2429056634c1483b82b5eb5c0c62`；producer SHA `bdc1068d37d23fff0a50aadac222969f996117d467d8deb9a403217b33955759`，kernel SHA `ec2994a7b446d31b7aa967b85a13a528791678c72eb0e93df35e356e2cbfc0d0`，test SHA `a3efacb5bdff652ab21d13f3a7ec6e591dc2b77bf09b3d0cd284eb90ede6050f`。这些是 REVIEW_REQUESTED，不是冻结/ACTIVE/启动；请独立评议员核对数值条件、I/O与限制，并针对确切版本 recheck。 
