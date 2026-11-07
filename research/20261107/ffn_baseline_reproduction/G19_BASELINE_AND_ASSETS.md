# G19 基线、已完成 FFN 研究与新复现缺口

**版本边界**：本次核对 `Qesire/FFNDS@6d7a52b7110f053b4b469c76cb0e8283620eda64`。这是资料crosswalk，不代表新验证。

## 一、与新复现绑定的 G19 对象

| 键 | 冻结值 |
| --- | --- |
| 模型/状态 | Wan2.2-I2V-A14B-LightX2V-Distill-4step；HHLL、无CFG |
| 层及计算位点 | High/Low各40层，4步×40=160；step0/2为锚点、1/3为命中 |
| 张量及激活 | N=31668, d=5120, m=13824；BF16、GELU(approximate='tanh')、W1/W2偏置、当前 gate |
| 其他模块 | torch_sdpa、LightX2V block offload、原生scale/shift、scheduler、VAE |
| 视频/硬件 | 832×464、81帧、16FPS，A100 80GB |
| CAL | chip10_02、04、07、09；seed42 |
| 暴露DEV | raft、bicycle、guitar、tennis、spoon、eagle、canoe_static_00；seed43/44 |
| CONFIRM | 新来源尚未确认；历史开发集及旧Chipmunk来源均不能充盲测 |

- G0：80次完整FFN＋160次A GEMM；FFN主MAC剩68.5185% Dense（节省31.4815%）；G0旧11对非冷启动生成wall 39.434518→37.613504s，1.048414×，是描述性配对，不是Q1-r2新加速。
- Q1-r2：I2V主体0.982884、背景0.983695，较G0 +0.002673/+0.001161，动态阈值4/14→6/14；但一般主体、美学、运动平滑度下降。
- RAFT：新Q1-r2相对Dense平均模长比0.9017（旧Q1r1 0.8951），方向余弦0.8083（旧0.8166），正交残差0.4585（旧0.4446）。大模长不等于正确运动，尤其CoTracker共同可见集合会变化。
- G19大矩阵打包和去重复W1预取均未观察到稳定SYS收益，不采用为默认。

[最新版方法及完整分数](../G19_METHOD_DERIVATION_AND_RESULTS_20261107.md)；[质量来源级JSON](../../../reports/20261107/g19/quality_summary.json)；[运动JSON](../../../reports/20261107/g19/motion_summary.json)；[工程JSON](../../../reports/20261107/g19/engineering_summary.json)；[独立数值复核](../../../reports/20261107/g19/INDEPENDENT_RESULT_REVIEW.json)。

## 二、仓库已有的竞争方法真实证据

| 家族 | 已有研究结果（不作为本轮新实测） | 还缺什么 |
| --- | --- | --- |
| **Chipmunk-MLP** | 898/96两个暴露来源的P64全160学生完整视频，原Dense/neutral对照、缓存80次建立/80次命中、8轴质量/RAFT，真实EXIT0+独立VERIFY | 最新7开发来源、2seed、同模型身份下的公平同源速度/质量表；是否应扩新匹配scope须审查 |
| **Chipmunk packed** | 两源高低专家真M=31668，packed BF16与同mask旧Triton输出相等；P64本地三测约23–24ms vsDense约35–39ms | 本地单算子试验不能代替新版系统部署/排名 |
| **Chipmunk同步** | 898/979原有ABBA，FFN事件约7.453→6.919s、phase约48.060→47.736s | 旧样本/观测时序不等同本轮Q1的非冷启动配对 |
| **TEAL** | 作者LLM单token kernel小测；Wan真M激活/行组支持研究 | 对Wan大M BF16 GELU的真正物理稀疏FFN不自动成立 |
| **SharQ** | 另一数值族的真实全160及质量 | 同G19 BF16数值族前不可直接比较质量 |
| **RT-Lynx inspired** | 一源BF16 2:4: 补偿后gate相对L2约0.24–0.27，重复M时约51–52ms vsDense36–37ms | 非作者完整原生内核移植，不能按它否定论文 |
| **ICC** | 作者DiT图像模型核心代码、数学比较 | 匹配Wan同配置FFN-only实验**尚未完成** |
| **TARDIS** | 原论文+旧FFN数学审查 | 区间/预测器/回退的真实忠实移植未做 |

### 本仓直接引用，先读避免重跑

- [Chipmunk Packed BF16](../../20261030/CHIPMUNK_FULLM_PACKED_NATIVE_RESULTS_20261030.md)
- [Chipmunk 全160学生视频](../../20261030/CHIPMUNK_GLOBAL_P64_V2_ALL160_VIDEO_RESULTS_20261030.md)
- [Chipmunk 质量/RAFT](../../20261030/CHIPMUNK_P64_V2_QUALITY_RESULTS_20261030.md)
- [Chipmunk 同机 ABBA](../../20261030/CHIPMUNK_SYNC_ABBA_RESULTS_20261030.md)
- [Chipmunk High/Low 消融](../../20261030/CHIPMUNK_P64_FACTORIAL_QUALITY_RESULTS_20261030.md)
- [TEAL 真M行组](../../20261030/TEAL_WAN_FULLM_ROW_GROUP_SUPPORT_RESULTS_20261030.md)
- [RT-Lynx启发的2:4](../../20261030/RTLYNX_WAN_BF16_2TO4_PAIR_RESULTS_20261030.md)
- [SharQ 全160质量](../../20261030/SHARQ_WAN_ALL160_VIDEO_V2_QUALITY_RESULTS_20261030.md)
- [2026-10-29作者代码独立评议](../../20261029/INDEPENDENT_METHOD_REVIEW_AND_FFN_BASELINE_20261029.md)

## 三、何种内容可以直接复用？

可直接使用已验收的公式、源代码身份、候选机制解释、失败证据与硬件算子诊断。不可将旧Chipmunk 898/96/979 的相对Dense结果和现行G19的chip10七源视为同数据集直接排名。先比checkpoint hash、输入图片、prompt、初始噪声、模型版本、调度、offload、GPU、kernel后端、舍入、当前gate与抽样群。

旧结果保留 **HISTORICAL_VERIFIED / NOT_MATCHED**，新ICC和七源匹配保持 **DESIGN_ONLY / REVIEW_REQUIRED / NOT_RUN**。无新scope执行许可，不能凭文档安排作业。
