# S2 下一轮原生实验：真实时空邻域、Dense/Student 状态与误差传播

2026-10-29 · USER_DIRECTED_P0 · **EXECUTION_REQUESTED / PRODUCER_AND_NEW_PROTOCOL_NOT_YET_REGISTERED / NO_NEW_RUN**。此任务归属 S2-FTFD 的统计/因果证据，不是第五个并行方法方向。主 Agent 必须分配独立理论子 Agent，见 [S2 专门委派合同](../agent_protocol/S2_THEORY_SUBAGENT_AND_EXECUTION_DISPATCH_20261029.md)。本文件是研究与执行规格，不能视为已启动集群作业。

## 0. 已完成的真实基础

- 十来源完整21τ×8xy×160 FFN site 原始 BF16 Dense 张量及T1–T4统计已有实际 exit0。[统计报告](S2_FULL21TAU_TEMPORAL_STATISTICS_RESULTS_20261029.md)得去τ0后 gY lag1≈0.048、时间Gram r95≈16.96/19：固定xy时间创新宽谱且弱短程相关。
- [相同 Dense h 的W2重放](RAW_BF16_FULL21TAU_NEW10_FACTOR_REPLAY_20261029.md)：旧共享r960的gY相对DC误差≈0.02932，AC误差≈0.18415，尚未涉及student prefix。
- [同预算两套r960拟合](RAW_BF16_NEW10_TEMPORAL_REPRESENTATION_FIT_RESULTS_20261029.md)：value10/dcac10的fit源AC相对SSE约0.05123/0.04727。
- [二十来源四十条完整学生视频+八轴](RAW_BF16_TEMPORAL_VALUE_DCAC_VIDEO20_QUALITY_RESULTS_20261029.md)：RAFT相对Dense仅0.45062/0.43931，dcac10对局部拟合更好却整视频更差。不得继续默认通过调AC权重修复运动。
- [两个高运动Dense ROI前置](../method_discussion/discoveries/E0/s2-motion-roi-coordinator-review-20261029.md)：ord898（subject，鸟+鱼）与ord96（camera平移）已在81帧上完成max-flow 5×5及低流负对照坐标选择，CPU actual exit0。鸟场景最大flow落在背景植物，而非中央主体。**此前没有新增这些ROI的原生FFN时空张量，也没有student状态。**

## 1. 四个互斥程度可检验的竞争解释

H1（跨来源与ROI的局部表示缺失）：在未拟合source的真实Dense相同输入上，候选W2已对重要时间频段、方向或运动ROI有较大误差；学生prefix偏移不是主要因素。

H2（student prefix/状态分布迁移）：同Dense h上的候选算子误差并不足以解释实际运动差异，free student在多step与Attention/FFN链条中的输入状态出现累积漂移。

H3（空间位移后的时序结构）：固定xy上的lag弱，但沿真实空间移动对应后相关性更强，运动跟踪后的innovation更接近低维。其匹配须优于同开销随机、固定xy及普通邻居。

H4（Gate/残差/下游消费者重定向）：真实g、r、BF16写回、norm/Attention及下游step改变W2误差作用，rowLN(u)的FP64诊断不等于原生consumer。

## 2. 全计算图与对齐对象

原始BF16 Wan2.2 I2V：4个denoise step×40 Block；每site视频latent网格21τ×29y×52x，row = τ×1508+y×52+x。τ不是4个step，也不是输出81帧。标准normal-forward边界为 preFFN V1[5120] → W1[13824] → GELU(tanh) h[13824] → W2 Y[5120]+b2 → gate e5[5120] × Y → BF16 product/add residual r → u → 真实Attention及后续block/step。按源码再确认preGELU a、post-attention及实际norm边界。

**每条原生记录**须绑定 source/imageSHA、promptSHA、seed、condition、arm、teacher或student、step/expert/block、τ/y/x/row、forward branch、step latent/noise SHA、真实数值格式/舍入路径、producer源码与protocol SHA。只把同source/step/block/τ/xy但不同model state的行看成对齐观测，不能称它们具有相同函数输入。

## 3. 第一批：2来源真实原生采集链路 pilot

先核对ord898、ord96的已成功ROI CPU EXIT与原Dense/新value10/dcac10视频及step state SHA，不以新候选视频运动指标反过来选择ROI。每来源取以下真实空间区域并集：① Dense预选高运动5×5；②预冻结画面中央5×5（主体位置是待验证而非假定）；③原选两个远距低运动负对照cell。保留区域来源标签及去重后原生row。最多52个独立xy×完整21τ=1092 rows/site/source，按原τ-major顺序，完整覆盖160 sites，含4 denoise step、两个expert。

采集三个arm：Dense真实normal-flow，value10与dcac10自己的free student normal-flow。重点保存V1、h、W2 Y（含bias）、真实g/r/u、step latent与关键真实attention消费者边界；可能加 preGELU a。对原始Y/g/r通过BF16先乘后加逐字节复原原u，按RAW tensor SHA及原生row校验。先核对被动probe是否影响same seed uninstrumented输出（若不一致，先定位probe干扰）。所有模型必须真实执行整个4×40，而不是以Dense h替代student输入。

**成本先冻结**：只算V1+h+Y+r+u的BF16原始行约68,608 byte/row；52xy×21τ×160site ≈11.99GB/source/arm，另有g/metadata/attention/latent。2来源×3arm可能达到约72GB原始张量量级，实际ROI重叠及流式统计使存储变化；这是设计级估算，不是磁盘许可。原先maxflow+两低流的567行/来源不能冒充本次含中央5×5的并集。制定真实HF/free disk、GPU显存、CPU RAM、IO、wall、调度与输出预算；必要时分成来源/arm的独立冻结part，但不因某些首末layer方便而放弃全160覆盖。

**执行权限**：实际session/RUN若规定最多2 GPU，则此次同会话并发≤2，通常一个已冻结part使用1 GPU；先检查Slurm/A100真实可分配状态。新scope必须单独在ACTIVE注册模型、源、code/assets、parent、预算、protocol SHA，实际RUN/EXIT后才能宣称运行。历史已闭合scope及失败收据不修改。

## 4. 第二批：形成真正跨来源时空统计

上述两来源只验证捕获链路与最小竞争观察。下一独立新scope按Dense确定的运动场景与source身份进行分层来源扩容：运动主体、静相机局部动作、相机平移、背景流动、遮挡/突发；建议首批至少12个与pilot不同的独立image clusters，再按source-level CI、极端尾部和源多样性滚动扩张。训练/开发/确认图像SHA严格分离，已有20来源视频均为历史暴露开发；不使用内部token数量充当独立样本量，也不把12定为采集上限。

结合真实连续空间ROI计算 time lag k=1..10、条件频谱/复相位、Temporal Gram有效秩、时间跳变/τ0、邻域对应与一致性，按step4、block40、source/scene/ROI分层。还须有真实完整step状态与运动语义、遮挡标注或可信离线flow oracle；解码帧Farneback/RAFT只能作候选对应参照，不能自动等同latent token运动。

## 5. 相同输入的离线反事实（另冻CPU/GPU scope）

令D(h)=真实原W2读出，Q(h)=候选r960 W2读出，同一step/block、相同可比较的gate/bias时，在连续算术下定义：
L = Q(h_D)−D(h_D)（原Dense输入上的局部算子误差）；
P = D(h_S)−D(h_D)（student输入/前缀状态漂移的读出差）；
I = [Q(h_S)−D(h_S)]−[Q(h_D)−D(h_D)]（算子误差随输入状态的交互）。
则 Q(h_S)−D(h_D)=L+P+I。若g_S≠g_D、r_S≠r_D，还要单独计算条件调制/残差差项及真正BF16 typed postwrite，不冒充同一恒等式的原生逐bit结果。对D(h_S)的Dense W2 GEMM显式计入离线oracle成本；比较M1092/M31668原生GEMM舍入且保留同状态控制。结合真实W1前/后/门控/Attention及step latent状态，按source与site给出L/P/I、有符号交叉、时间频带方向/相位、ROI运动对应及累计幅度。

## 6. 第三批：有向后缀因果干预（根据上一步定位另注册）

在所有160-site的被动分析定位高影响的step/layer/site与低影响对照后，选固定少量位置用正常模型后缀/完整4step重跑进行单次W2中性/局部恢复、student prefix置换等有定义干预，使用相同checkpoint/seed/源、原生BF16和明确写回时序。先判断是“局部拟合方向无法泛化”还是“prefix/下游放大”，再去设计新执行算法；真实full-video动作/相机方向、RAFT、闪烁、感知/语义、FFN+系统速度与显存仍必须独立验收。Dense恢复只是离线因果工具，不能自动转为最终的补算目标。

## 7. 成功交付、失败状态与下一判别

- P0-1：理论子Agent独立交付数学分解、至少三竞争机制的不同预测、原生计算图、数据需求/估算、源码及论文相似性审查、廉价判别，并由主Agent逐条答复。**理论子Agent不得直接改写正在执行的原生数据、RUN/EXIT，也不能自行审批GPU范围**。
- P0-2：pilot producer与运行协议提交，静态/合成/原始原生中性检查后由可实际分配资源的执行Agent按已获权限启动；每个source×arm×160-site有真实RUN/EXIT与byte-bound assets。
- P0-3：另交source扩容计划（包含运动/遮挡多层统计）、同状态反事实与必要真实因果控制，附全量source-level数据、图注/结论和明确下一种可执行函数表示候选。
- 故障、OOM、超预算保留原protocol和失败RUN/EXIT，不自动改写身份；下一part另冻且说明新增资源与来源。**本文件本身既不等于子Agent实际启动，也不等于native RUN已经执行。**
