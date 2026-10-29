# 新12来源Dense视频基线：协调者源码审查

ROOT，2026-10-29；实现者为`/root/s2_theory_design_independent`，其[实现者自审](s2-new12-dense-baseline-implementer-self-review-20261029.md)与本审查身份分开。当前两张pilot的L/P/I只是机制定位；先取得12张按静图分层选出的新开发图的真实Dense视频，才能按实际相机、主体、背景动态和遮挡事件扩展时空采集。这批不使用候选成绩挑图、不拟合、不生成候选，也不宣称正式确认或方法质量。

固定12个ord依独立输入审阅及74已曝光历史图筛查；六类各二张，交替分part各6。每路先对旧source898或96重跑一次Dense中性控制，核全部10个真实noise/flow/post latent状态字节；全部通过才生成本路6张新图。总14次完整81帧四步×40层生成，其中仅12个新开发图，另两条明确为重复控制。原prompt、seed42、同checkpoint/BF16/W1/GELU/W2/gate/先乘后加及scheduler不变。

复用原版本private namespace中的OriginalGuard、LightTap、native_phase及注册collect；只设置新范围的私有ID/limits/run_id/resources。Dense的transfer_geometry仅访问runtime.mode，故用常量dense对象，无factor wrapper/因子读取/两GEMM或FFN行采集。完整160site、T5、VAE、原始MM、offload与每步状态仍逐调用计数。每路7视频、28flows、1120FFN、70状态、1134次prefetch观测、49组stage事件；setup计数不随视频数乘七。输出前持久化真实RUN，结束必须原始账本预扣/实测精确相等并通过逐视频注册collect。

ROOT对照源码修正两处：CPU preflight在输出/资产读取前调用ACTIVE授权；旧Dense证明直接复用已actualEXIT0的被动采集闭包，绑定其RUN/EXIT、完整ledger、资源、中性证据及观察收据，再只重读该source的10个原视频状态和10个capture状态，逐一SHA/bytes/shape/dtype/rawSHA匹配，并验证原视频及generation身份。避免每次预检顺带读取另外59条已完成视频。旧capture代码/协议/结果完全不变。

新每路1GPU/CPU4/GPU40GiB/host可用128GiB/2400秒外层2700秒/HF新增4GiB；预检CPU4/600秒外层900秒/HF8MiB。两卡总上限延续；先做独立CPU proof，当前重放结束释放卡后才提交新GPU任务。原package、原69GB master已有CPU SHA/stat证明复用，运行时重核实际执行节点/当前stat；新图实际提取和历史筛查RUN/EXIT、源码/版本与本地package均需预检。无官方评分/CORE/RAFT调用，运动分层留给随后独立有限视频观测范围。

7项合成控制实际通过。实际Dense neutral、单臂注册collect和启动环境仍须真GPU验收；早期CPU/身份验证未创建模型输出时的失败由外层真实EXIT/stderr保留，之后失败写原RUN。不自动重试，不通过则停止该范围，保留失败事实与扣账。新12的跨裁剪近重复覆盖仍是已记录的有限筛查，不把静态cohort PASS当运动分层或质量通过。
