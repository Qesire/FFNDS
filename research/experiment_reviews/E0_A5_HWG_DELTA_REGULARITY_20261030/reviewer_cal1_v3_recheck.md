# CAL1_1040 V3 精确独立复核

评议者 `/root/sync_abba_actual_review_terra`，作者 `/root`。本范围只针对来源1040的静态校准，在任何 E0 比较数值前冻结：21τ×M64 的1344真实匹配行，双 expert ×40 block，各216个 K64 组里用 signed Gram 前向 greedy 选80组，并另付一次 selected W2 和 FP64 SSE 与已舍入 Gram 结果作闭合。该单源采样校准是明确有限的比较基线，不是全局最优或跨来源最优；898/979为历史曝光的开发来源，来源96的实际失败保持独立身份。CAL2 草案未运行不自动授权本范围。

已独立读取核验 producer RUN/EXIT、独立 VERIFY RUN/EXIT 的身份和实际 EXIT0、无 timeout、对应 protocol/job/RUN SHA，并沿 producer RUN 原始 observed manifest 逐项核对160个 site 的 metadata 引用。VERIFY 的实际 RUN 记录342文件、160匹配位置、97,813,721,804字节独立读取以及十状态/视频 neutral。该验收只涵盖已验证父资产，不是本次校准通过或新视频质量。160个 sample payload 合计43,603,261,440字节，加56,223,524,243字节完整 cache SHA 合计99,826,785,683字节，低于128GiB显式读上限；其它元数据和源码仍由其预算预扣。输入与 cache 必须在启动时继续按冻结 SHA 验证。

检查1035项源码文件大小与 SHA、协议和作业规范指纹、私有资产 locator/profile 哈希、实际完整参数及 output fresh 条件；不公开 worker或设备身份。协议为单GPU、4 CPU、32GiB CUDA reserved、96GiB RSS、128GiB host available及64GiB forecast、每host最多两模型、session最多4 GPU、HF新增2GiB、7200秒及外层7320秒，失败无自动重试。运行时还须按既有锁和 monitor 检查当前设备及host联合预留，既有采集并发纳入4卡上限。七类调用各80次，其 event 和 host wall 各自记录；读取的 opaque mmap 内部流量另标 UNKNOWN，不当作实测完整物理I/O。

source-local FP32 Gram/target 转 FP64 后的贪心目标有数值舍入；直接投影及闭合残差可揭示抵消，不称实数或全局精确最优。独立复跑 static signed math、discovery contract、evidence I/O 13项软件fixture PASS，不构成大shape GPU、实际输出或后续 E0 成功证据。后续开发源读取需要本范围实际 EXIT0、artifact SHA 冻结及对应新 scope 独立审核，不授权 E1/E2。

**APPROVE**：精确 CAL1_1040_V3 可进入登记。实际冻结提交为 `7e4dc4d3570041a30aa5f7573d4fdd5f2e18567b`，协议 blob SHA、完整私有 job SHA 和审查 fingerprint 已独立复算一致。登记/启动时仍须核实时 HEAD、ACTIVE、可用内存和设备状态。
