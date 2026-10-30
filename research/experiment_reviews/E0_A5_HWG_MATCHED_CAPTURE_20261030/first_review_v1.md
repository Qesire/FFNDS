# 独立初审：A5-inspired 匹配边界采集 V1

评议者 `/root/sync_abba_actual_review_terra`；作者 `/root`。状态 **INDEPENDENT_FIRST_REVIEW / REVISION_REQUIRED**。这是对主审及已读 M0/M1、源码路径的具体反例审查，不是对尚未完成的 producer、protocol、job 的批准，不登记 ACTIVE，也不代表运行。最终版本须主 Agent 逐项响应、冻结全部身份后由我另行 recheck。

## 可辨识性与竞争反例

M0 的同专家跨步配对坐标缺口使新增原生采集有明确价值：它能为四来源、每来源四步×40块提供同坐标状态及 post-Q1 full-M 输入，并在后续独立授权的重放中比较增量、当前支持及物理tile占用。此 scope 本身只有 plain 与 observer Dense，不做 proxy、选择、拟合、候选或质量测评；不能从采集成功声称 H1/H2/H3 成立、真实W1/W2省算或视频质量通过。四个来源和seed42均为暴露开发样本，层/行/21τ并非独立来源。原A5参数producer缺失时保留 `A5_INSPIRED_MEANABS64_ROW_LS_V1` 身份；历史阈值方法、现适配器和随机/静态/Chipmunk等预算对照不应混名。

**反例一：元素稀疏但硬件全忙。** 每行的增量仅有少量大通道，但这些通道在一组 M16/M32/M64 行内分散，取并集后每个原生 K64 组仍有活动元素；则单行保留率高仍不能跳过任何对应物理 tile。后续 E0 应以完整原生group/row tile并集、实际布局与 paid selector 成本区分它与真正可跳算的局部共享。此采集的每τ一块不能单独证明全M的tile分布；full-M V1使另行授权的完整重放成为可能，但重放并非当前已付观测。

**反例二：群组看似相干但误差方向错。** 一个区域内共享小支持可保留较高逐通道能量，却漏掉经当前 gate/W2 放大的少数反向贡献；或者前后变化小但支持阈值翻转，发生相消/闪烁。均等K或真实MAC的逐token、连续组、shuffle、static/random 和 oldY-only 必须在后续范围比较 signed 输出与视频功能；仅 RMS/Jaccard 或均值无法裁决。21τ是 latent 时间而不是21个生成step，四个去噪step也不能被合并为独立样本。

## 采样和数值边界：必须修订或在最终合同落实

1. **实际 M64 对齐与坐标。** 每τ空间为29×52=1508行，1508不是64的倍数。固定并保存每τ 64个连续原始flatten索引、起点、τ/y/x以及与全局M64的余数；若用于比较实际 M64 tile，要求全局起点模64为0且64行均属于该τ。由于图像行宽52，M64通常跨二维行边界，不能称为单一空间行或直接代表二维邻域。要支持四步/全层同坐标配对，明确选取算法、全局次序与一一对应，包含每τ首尾和边界的静态反例；一种固定tile/τ对其它tile和小物体没有覆盖保证。
2. **捕获的是原生产物。** `tools/hif4_observer.py` 所示路径是 input.float/smooth.float → BF16 → 原生HiF4 activation QDQ(force_fp32) → BF16 → `F.linear`，FFN两次QDQ分别为Q1/Q2。全M V1必须在真实Q1返回后、W1消费前按实际stream和生命周期抓取；1344行原始U/Z/H/Q1输入/V1/Q2输入/V2/Y及gate、residual/postwrite均从实际全M操作的对象在对应边界切片。不能拿子集重新运行QDQ或GEMM冒充原生产物，也不能将重排、过滤/quantize(delta)当原生数值。保存原raw tensor与序列化SHA、dtype、shape、stride、buffer身份、step/expert/block、sample row与bias/b2，不应以observer派生值顶替真实值。
3. **重放身份有边界。** 后续 full-M重放必须绑定当前expert/block有效W1/W2与bias/smooth2及Q2实现和生成时的顺序/格式；在各site用已捕获的 Z/H/V2/Y 同坐标核验，失败即不称忠实重放。仅存 post-Q1 V1不提供 full-M pre-Q1 raw x或原Q1在线scale，不能据此重建全M原Q1；任何部署成本保留原Q1及selector/pack/cache开销。`Q2(h1/s2)−Q2(h0/s2)` 与 `Q2((h1−h0)/s2)` 不等，M1中的简化代数测试不证明原生QDQ相同。
4. **被动中性与完整性。** 相同source、seed、recipe、checkpoint、HiF4 cache及原始codec下，除四step latent和完整视频外，也应对raw FP32 noise、initial noise与四个flow逐字节比较，形成每视频10个状态及完整视频的配对门。各视频均检查40×4真实站点、expert high/high/low/low、320个FFN linear/QDQ与1,280个non-FFN linear/QDQ及cache miss 0（总各1,600）；全覆盖还应核每站点记录完整、位置唯一、sample rows和full V1原始文件实际存在、SHA与typed header匹配。实际 postwrite 是原始dtype/顺序的真实输出，不用融合FP32替代。observer捕获必须不改变输入、旧缓存、模型输出或active buffer；失败保留实际RUN/EXIT和inflight UNKNOWN。
5. **同stream与写入稳定。** 捕获/clone和D2H必须与生产stream保持依赖，在原始buffer复用或释放之前完成，锁定同expert相同block和此step输入；CPU异步落盘必须有界并保留对象直到写完。原始HiF4 QDQ/linear必须恰好一次，观察行为的clone/D2H/hash/同步应逐项预扣并作为额外成本报告。回放或动态选择不得偷取当前完整h/Y作为可免费在线信息。

## 预算与验收

每个来源仅 post-Q1 full-M V1 已是 `31668×5120×2×160 = 51,884,851,200 B`，约48.322 GiB；每增加一份1344×13824 BF16采样边界约5.537 GiB/160位点，每份1344×5120 BF16边界约2.051 GiB/160位点。实际字段、dtype、可能别名去重、坐标、gate/bias、headers、plain/observer状态与视频、journals、同时保留的临时文件和失败partial需列逐字段上界，加上文件hash/重读/验证的真实I/O次数后证明128 GiB/来源上限。不能将请求字节说成实测物理I/O，也不可把采集wall当FFN候选速度。不同格式QDQ、有效weight及原生kernel/adapter的源码和cache SHA需在部署前绑定。

每任务1 A100/4 CPU、GPU reserved≤76 GiB、RSS≤320 GiB、host available≥128 GiB、14400 s、输出≤128 GiB；总会话≤4 GPU、每host最多两个完整模型。**反例三：并发准入竞态。** 两个进程同时看到可用内存各≥128 GiB但随后各加载模型，阈值不足以保证联合余量。最终方案须提供可核的同host协调/保守共享预留和实时硬停，或按实际安全条件串行；128 GiB是余量门槛而非每模型总需求上界。完整模型、56,223,524,243 B cache加载/映射、CPU buffer、多个GPU、磁盘空余及所有worker的失败产物必须在对应作用域计量。无隐式重跑；每项非零退出和资源故障单独保存。

建议独立验证当前产物的完整文件/typed raw SHA、位置覆盖、原生对照与已退出producer绑定，并冻结验证读取预算；若只用producer自己的hash，要明确其证据边界。真实多source独立统计应在 source 层汇总，并对新source/seed另行确认。静态审查、旧完成结果或本首审均不替代新scope的真实 independent recheck、ACTIVE登记、启动前SHA/资源身份核验和实际RUN/EXIT。
