# FFN-only 统一实验、实施与验收合同（G19 vs ICC/Chipmunk/TARDIS）

**2026-11-07 / DESIGN_ONLY / NOT_APPROVED / NO_RUN**。本合同细化 [实验前独立审查](../../agent_protocol/EXPERIMENT_PREEXECUTION_INDEPENDENT_REVIEW_20261029.md)，不替代原始治理。所有具体新 CPU/GPU、校准、回放、视频和性能实验均需在当前 `ACTIVE_RESEARCH.json` 登记、不同Agent正式复审后执行。

## 1. 科学问题与两个相反预测

- H1（函数预测优势）：在四步HHLL真实增量上，一次完整FFN折叠 `uA+C_s` 能以更小缓存/简洁规整GEMM保持主体和动态语义，具有更好的质量—延迟Pareto。
- H2（中间非线性必要）：两级低秩中间GELU（ICC）或选中激活精确增量（Chipmunk），在同计算预算下更有效保留动作方向、主体响应；G19因完全线性化引入运动/轨迹退化。
- H3（系统实际主导）：ICC/Chipmunk的校准、掩码、预取和缓存内存足以抵消主MAC优势，局部函数拟合更好但全系统延迟不改善。
- 直接反证：出现 `MSE下降但方向余弦/轨迹更差`、`rank更低但cache H2D主导`、`Sparse mask实际未减少合法GEMM`、`高光流幅值但文本动作失败`。这些都改变方法胜负结论。

## 2. 固定数值身份

- 模型：`Wan2.2-I2V-A14B-LightX2V-Distill-4step`，High/Low两专家各40层；step0/1为high，2/3为low；无CFG。
- `N=31668,d=5120,m=13824`；pre-W1输入是 `norm2→scale/shift→BF16 cast` 之后的当前原生 `U`；FFN为 `W1+b1→GELU(tanh)→W2+b2`，后接**当前** gate 和残差BF16写回。
- 视频832×464，81帧、16FPS；基础Attention `torch_sdpa`；LightX2V block offload；当前A100 80GB。除FFN外无变动。
- D0是 `20261107` 当前仓库冻结 checkpoint/scheduler/初态匹配Dense；G0为当前 `fused_post`；Q1r2可作为G19更保I2V基线，但在同机器正式速度需另测。不得借旧40step、双CFG或NVFP4/HiF4数字。

模型和外部作者源码、输入图片/prompt、随机数初始化、checkpoint shard、对应scope文件、torch/triton/cuda和实际GPU型号都需保存可检查的全SHA、软件版本、tag/commit，**没有真实文件不填编造路径或哈希**。

## 3. S0→S5 的操作协议

| 阶段 | 允许动作 | 必须产出 | 失败处置 |
|---|---|---|---|
| S0 资产盘点 | 只读代码、旧收据、版本/Git/原模型文件检查 | SOURCE_MAP, existing evidence crosswalk, hash inventory, missing material | 未找到资产→MISSING_ASSET，不启动旧job替代 |
| S1 接口与数值 | 不带模型科学结论的静态测试、以后获批的1层GPU数值中性 | 160-site counter, expert reset, FFN native off/on neutral, BF16舍入/偏置/gate | 差异为实现错误则STOP_AND_FIX，禁止视频 |
| S2 CAL同态回放 | 获批读取真实pre-W1配对CAL，rank/topk/区间扫描 | 逐源/层/step的FFN/gated SSE, l2, p95/p99, oracle paid/online分表 | 某方法离线失败或代价无余量则DROP_OR_REVIEW |
| S3 FFN与系统微基准 | 获批A100单FFN及缓存流量profile | anchor/hit, 选择/编码/decode, HBM/H2D/PCIe, allocated/reserved, warm/cold | kernel不可用归KERNEL_BLOCKED，不能拿PyTorch慢替代惩罚论文 |
| S4 限定自由视频 | 冻结3–4个诊断来源（含高动态和静态对照），学生自由生成 | paired MP4与sha, 8轴 + 连续RAFT, phase/end-to-end wall | 无实际质量/加速前沿则STOP；保留失败结果 |
| S5 7DEV与新CONFIRM | S4通过且新范围审批后，已暴露7DEV各2seed；以后冻结新独立来源 | source等权质量/成本Pareto+置信区间+完整方法身份 | DEV只能开发；置信区间/泛化证明须独立confirm |

**注意**：S0设计并不意味着后来各S1–S5自动得到运行许可。FFNDS原强制状态机：`IDEA → AUTHOR_SELF_AUDIT → INDEPENDENT_REVIEW_REQUESTED → REVISED_DESIGN → REVIEWER_RECHECK_APPROVED → SCOPE_FROZEN_IN_ACTIVE → PRELAUNCH_IDENTITY_RECHECK → ACTUAL_RUN → EXIT_AND_EVIDENCE`。必须使用真实不同 Agent reviewer 和原文反例，不能伪造评审票。旧 `review` 只支持其原冻结scope。

## 4. 实现范围/数值门槛

每个候选在 `Dense neutral` 下必须保持用户原生成路径一致：预W1输入、维度、BF16、GELU tanh、W1/W2 bias、current gate、step/expert身份、原native Attention；某方法自己前向改变后学生可自由偏离，但不能把Dense捕获U偷偷复用作学生输入。

核心站点：共160个, 高低专家各80，若使用锚点缓存方案，应先保证step0/2各40个full、1/3各40个hit。某方法选择性保留部分层Dense时，应明确真实full/hit数量、有效稀疏率和可省物理算术。缓存不跨视频/专家无授权共享；全站点范围内失败不能只报少量成功block。

数值对照：
- **精确等价控制**：no-op hook 与 native FFN，同seed/初噪/最后latent/解码RGB/MP4身份核对，存在异常先修；
- **实数算法**：每层/专家的校准输入与在线合法信号，明确先后与 bias/gate；
- **BF16导出**：FP32临时减加、各GEMM/激活舍入顺序必须对齐；数值变体建立不同arm；
- **后验oracle**：如使用真实当前全部h选择mask，只允许作质量上界，成本按真实W1执行，不能混入在线加速主表。

## 5. 局部与视频质量：不可只看 SSE/动态阈值

**必须报告8个现有分项**：
`subject_consistency, background_consistency, motion_smoothness, dynamic_degree, aesthetic_quality, imaging_quality(MUSIQ,0–100), i2v_subject, i2v_background`。

运动至少使用：RAFT全分辨率时间序列(两帧间隔40对，含全场平均、top5%)、对Dense的 `β=〈v_D,v_C〉/‖v_D‖²`、norm_ratio、cosine、正交残差；CoTracker须固定 **多候选和Dense共同可见同一组点** 再比轨迹幅度与方向。独立 flicker、相机物理运动分解、语义前景/背景运动、文本动作/状态改变、人工完整播放属于 `NOT_MEASURED` 直至真实执行，不允许把现有主体/背景一致性直接当这些维度已经通过。

动态二值6/14与4/14是**14个视频的阈值计数**，不能换算为运动减速率；更高RAFT范数不意味着正确方向。完整视频质量通过条件应在新confirm前预注册，包括各维最低允许差异与运动方向/事件指标，而不是看到数据后改阈值。

来源独立：七个开发图像源×2seed，先source内聚合、再来源等权；同源不同seed/81帧/128token块/40层不构成独立 source。跨版本额外CAL被多轮访问，不得把leave-one-source当独立盲测。

## 6. 真正公平的效率测量

两层主表：
- **FFN-only**：同一实际preW1形状、相同有效权重和device，分别测 Dense/锚点/命中、选择与cache开销；有效稀疏/真实rank和工作区/内存预算。不要仅写算术值。
- **完整generate**：固定Attention/scheduler/VAE/offload，记录gen wall、denoise wall、end-to-end cold-start及非冷启动、首次数据/矩阵准备时间。原G19旧11pair `1.0484×`仅作历史数字，新方法性能须各自重复同期计量。

公平的A/B/BA/ABBA预热重复、GPU同步、计算时间与host墙钟分离，记录每臂实际峰值 allocated/reserved、权重与缓存逻辑payload、全system读取、H2D/D2H实际可见收据；同机同配置同源同seed。跨节点按各节点内部配对取比，不能将不同节点绝对秒数拼接。compiler cold/warm分开，首轮显存准备和rank/SVD/codec校准离线成本另列。CUDA profiler观察器会改变开销，明确其身份，生产性能关闭heavy probe。

### 统一Pareto报表

| arm | 源/seed与模态 | FFN算量(包含选择) | FFN实测延迟 | gen_wall与冷/暖 | peak GPU | 8维 + 连续RAFT/CoTracker | 证据 |
|---|---|---|---|---|---|---|---|
| D0 | 当前7DEV | 100% | 新实测 | 新实测 | 新实测 | 已有历史，确认后复用 | SOURCE_HASH |
| G19 G0/Q1r2 | 当前7DEV | 68.5185%主MAC | 新实测 | G0旧pair仅描述性 | 新实测 | 最新归档 | G19_SHA |
| ICC-r SVD/CD | 当前7DEV | `0.5+r(d+m)/(2dm)` 主MAC | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | PENDING |
| Chipmunk旧P64 | 898/96 | mask/packing变动 | 已有局部结果 | 历史有observer | 旧质量不是新7源 | 旧结果（已验收） | HISTORICAL |
| Chipmunk新匹配 | 当前7DEV | 待实测 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | PENDING |
| TARDIS static/full | 当前7DEV | 真实predictor/选通 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | PENDING |

不得用“同一MAC预算”替代真正的相同GPU时间预算；可以同时绘制质量vsMAC、质量vsFFN延迟、质量vs完整gen wall、质量vs峰值显存。原作者H100/多步/attention+FFN性能单列 `AUTHOR_ORIGINAL`。

## 7. 最小首轮审查包（让独立Agent能直接审）

每项新scope提交：
- `SOURCE.md`：原论文、URL、upstream SHA、代码路径、license/依赖；
- `METHOD_DIFF.md`：FFN原函数和作者方法逐行映射、在哪里失真、为何合法；
- `RUN_SPEC.json`：实验对象/原checkpoint哈希、CAL/DEV/CONFIRM身份、rank/topk、编译/硬件、消融、实际可付费动作、预计缓存峰值/风险；
- `AUTHOR_SELF_AUDIT.md`：竞争预测、最强反例和会改变结果的观察；
- `INDEPENDENT_REVIEW.md`：真实不同Agent审查，给出APPROVE/REVISE/REJECT与回复；
- `EXIT/RESULT/VERIFY`：实际作业结束后建立，未运行则`NOT_RUN`，不得预制成功记录。

### 预定失败处理

- ICC 原四缓存 >80GB；先单层reference与代数2态；若缓存仍超显存，review要求主机流水或保留FUNCTIONAL_ONLY。
- Chipmunk `31668%128=52`, 尾块76 padding/valid mask，若错误立即BLOCK；Hopper-only CUDA核不可冒充A100可用，先测Triton/扩展。
- TARDIS oracle读当前全W1后判异常：它是诊断上界，不是在线预测器，不能算实测节时。
- 若局部误差更低而全视频更差，保留负例，停止后验调参；额外关注High/Low/动作方向/文本事件。
- 若benchmark只有短暂微小提速且跨repeat不稳，标 SYS_INCONCLUSIVE，不因理论MAC强行判胜。

**本合同由资料审查产生，当前没有真实新 scope、独立review批准或GPU实验收据。**
