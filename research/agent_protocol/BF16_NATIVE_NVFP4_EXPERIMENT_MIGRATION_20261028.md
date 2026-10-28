# FFNDS · 并行执行策略：A100 原始 BF16 × RTX PRO 6000 Blackwell 原生 NVFP4

2026-10-28 · CURRENT POLICY。**用户最新指令：RTX PRO 6000 Blackwell可以作为更适合的原生NVFP4目标，允许与A100 BF16科学研究并行推进。**此前“没有本地5090”仍有效，但不再意味着所有NVFP4任务延期。两个lane分别维护进展、算子身份、GPU与实验凭据；本文件不自动发起任何GPU运行或更改冻结实验。

## 1. 两条并行P0研究轨道

| 数值路径 | 设备 | 承担的问题 | 工作状态 |
| --- | --- | --- | --- |
| **BF16_ORIGINAL（P0）** | A100、原始BF16 high/low checkpoint | M-B/B-F/A0，跨深度范数自由度，低成本新函数，真实FFN算术和系统耗时、完整视频运动与语义 | 继续已开展工作，不等待FP4 |
| **NVFP4_NATIVE_W4A4（并行P0）** | RTX PRO 6000 Blackwell，SM120，96GB名义规格；**当前可访问性和MIG配额待验证** | 真正packed E2M1 GEMM、weight static与activation dynamic两侧scale、Dense baseline、候选FFN实算/速度 | R0只读准备READY，R1–R4按实际依赖与新scope逐次启动 |
| HIF4_FAKE_QDQ_DIAGNOSTIC | 旧模型及必要的合适设备 | HiF4 signed S1P2、64/8/4 scale耦合、历史数字与机制诊断 | 按需 |
| NVFP4_DECODED_BF16_GEMM | 同量化checkpoint | 解码权重BF16 GEMM对照，以隔离kernel和量化差异 | 专用控制组 |
| NVFP4_W4A16_FALLBACK | 具体软件后端 | Marlin/weight-only fallback识别 | 不算native W4A4 |

**资源事实**：用户没有本地RTX5090；RTX PRO 6000是另一个Blackwell目标。NVIDIA官方产品规格为96GB显存、CC12.0，实际可分配显存取决于型号、MIG及当前使用情况。历史 [SM120 CUTLASS原生基元](../20261026/E0_SM120_NVFP4_PHYSICAL_PATH.md)于**RTX PRO 6000 Blackwell**真实执行两个小shape packed GEMM并通过参考，**不是RTX5090，也不是完整Wan NVFP4模型**。[旧R0调查](../20261028/BF16_NVFP4_R0_READINESS_20261028.md)当时没有通过已批准入口重新访问那台机器、匹配完整NVFP4 checkpoint也未验证，历史access failure不推论当下无法访问。当前重查以真实设备管理入口为准。

## 2. 必须先区分源码中的两类NVFP4

- **NVFP4 Fake-QDQ研究实现**：[nvfp4_rtn_backend.py](../../upstream_audit/source_deltas/Wan2.2-I2V-A14B-W4A4/files/quant/nvfp4_rtn_backend.py)明确记录weight_storage为float_QDQ_weight、activation_storage为float_runtime_QDQ、packed_kernel为False、compute_backend为bf16_or_fp16_F.linear。它每调用重新量化group16激活，权重是离线RTN，提供数值近似，但**不能作为原生NVFP4 Tensor Core加速证明**。
- **NVFP4原生W4A4计算**：优先复用固定版本[NVIDIA CUTLASS SM120 79a](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu)或可核对的[LightX2V](https://github.com/ModelTC/LightX2V)成熟pack/scale/GEMM链路；weight packed码及静态scale离线生成，Q1和Q2激活码/每16元素局部scale在本次FFN调用在线生成，全局activation scale可能来自校准。实际需要记录两侧scale物理驻留、swizzle、量化producer及packed kernel名称、真实延迟。

原始BF16 checkpoint 与任何HiF4/NVFP4量化后再解码的BF16权重必须分开；NVFP4 group16不能不加论证直接沿用HiF4 group64闭包。Q1在线码在W1前才可能消费，Q2在线码在W1/GELU后只能作用于剩余W2。LightX2V官方四步NVFP4配置可能同时改变scheduler和sparse attention，**先做每种recipe内部Dense→候选的配对比较**，跨配置只做经确认身份后的质量Pareto。

## 3. 研究队列与真实执行门槛

**A100 BF16并行科学主线（持续）**：继续利用CAL24、B-H、同head A0、B-F学生、A0视频、原始BF16 Dense两视频、low39四角cross-state分解、fixed-norm转移。下一次设计以**跨来源/跨深度范数补全、独立source校准、视频运动与语义、低成本实际FFN物理算子**为中心；保留已暴露开发来源的证据边界。没有前置NVFP4依赖。

**RTX PRO 6000 NVFP4原生lane：**

1. **R0 / READY_STATIC：**按受许可入口核验当前RTX PRO 6000访问权限和可调用GPU资源，查询真实device name/compute capability/MIG/可用显存/驱动/CUDA>=12.8/torch/CUTLASS版本；复用已有SM120 primitive记录；pin与现有Wan原始BF16高低专家及四步蒸馏版本相容的**实际NVFP4 checkpoint或正式转换/校准生产者**。若暂时只有GitHub权限，写ACCESS_NOT_VERIFIED、CHECKPOINT_UNKNOWN，可继续代码级准备；不要声称当前分配了Pro6000。
2. **R1 / OPEN_NEEDS_REGISTERED_NATIVE：**实际可用SM120和新ACTIVE scope满足后，做packed W4A4非零kernel/数值no-op，核验W与A scale输入和编译架构，识别Fake-QDQ/W4A16 fallback；历史E0已经完成相同数值基元的情况下优先复用它而非无效重跑。
3. **R2 / WAITING_ON_EVIDENCE：**有匹配NVFP4 checkpoint、动态activation quantizer、原生GEMM执行路径后，跑Wan真实W1和W2全M Dense；量化、packed、scale swizzle、GEMM、offload及内存峰值分别计时。使用Pro6000可利用较大显存降低offload，但必须核验实际96GB驻留是否可用。
4. **R3 / WAITING_ON_EVIDENCE：**同一NVFP4 checkpoint/scheduler/attention/源与seed，Dense FFN对比实际candidate FFN；分别报告省掉的物理tile/MAC、前置量化观察成本和整个FFN真实wall，不把不同模型/步数/attention混成单一量化/稀疏收益。
5. **R4 / WAITING_ON_EVIDENCE：**R3条件通过后，进行真实四步两专家40block、完整视频source-disjoint与相同seed配对、运动幅值/光流、主体背景、camera、时序语义、失败尾例和端到端速度。

两轨同为科研优先级，不必等待彼此完成；各轨内部按producer→consumer依赖。当前全局资源政策为实际可调用量，不以历史固定卡数推导当前权益；任何模型/GPU运行必须按[ACTIVE](../ACTIVE_RESEARCH.json)已有会话限制、独立冻结scope、资产与运行收据授权。新增Pro6000真实GPU执行需重查授权并注册，政策文件本身不构成作业提交。

## 4. 下轮Agent可立即交付

BF16 lane 给出利用最新原始BF16结果的科学判别；NVFP4 lane 给出R0可重用source与硬件/权重就绪表，分别标注ACCESS_VERIFIED或ACCESS_NOT_VERIFIED、CHECKPOINT_VERIFIED或UNKNOWN、真实packed W4A4还是Fake-QDQ。等真正可分配Pro6000后独立安排R1–R4的新原生/VIDEO/SYS证据。保留旧R0报告、已冻结实测和哈希，不重写历史时间点。

资料：[RTX PRO 6000官方型号](https://www.nvidia.com/en-sg/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/) · [官方CUDA GPU CC](https://developer.nvidia.com/cuda/gpus) · [权重静态/激活动态与NVFP4时序](../20261028/WEIGHT_ACTIVATION_STATIC_DYNAMIC_NATIVE_FP4_TIMING_AUDIT_20261028.md)。
