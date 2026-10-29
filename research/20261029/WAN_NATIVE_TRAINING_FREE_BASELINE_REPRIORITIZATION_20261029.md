# FFNDS · 2026-10-29 用户最新指令：只复现 Wan 上的 FFN 稀疏基线

## 研究对象与执行门槛

**用户明确收紧范围：只研究 Wan FFN 的计算减少。** 目标是在本仓已有 G14/98篇+20补充的论文索引内，选择无大型训练、代码可审、能迁移至 **Wan2.2 I2V A14B、BF16或另列native NVFP4、四步 high/high/low/low、全部160处 FFN** 的现有方法，真正运行完整视频，评估视频动作/语义/时序/相机/背景与真实 kernel 和端到端时间，观察该类既有方法在 Wan 上的缺陷。

**不在当前复现队列中研究**：帧/Token/整Block跳过、注意力稀疏、整Transformer/denoising-step缓存、非FFN量化及单纯LLM/FLUX作者模型验证。FIS-DiT、MagCache、TeaCache、ProfilingDiT、Sparse-vDiT等不进入P0或P1 FFN实验；不按它们的全模型倍速作为FFN基线。LLM tokenizer/checkpoint/PPL、FLUX作者原模型复现不构成前置条件。已完成工作与原收据保留历史，避免重复投入。

## 依据已整理的论文列表/G14构筑的真正 FFN 候选

| 路线 / G14编号 | 原论文 Wan 与FFN关系 | 代码及真正算子 | 当前结论 / 排序 |
| --- | --- | --- | --- |
| **SharQ (L74)** | **原论文实际在 Wan2.2-T2V-A14B 演示FP4激活稀疏**，免训练；公开宣称与 SageAttention结合系统最多1.58×，**不可归为FFN独立加速** | 作者 [SharQ源码](https://github.com/actypedef/SharQ/tree/3cc584fe48e32e59c8b4e37323470dcdd606bb11) 提供 N:M sparse FP4 backbone + dense FP4 residual 双 GEMM，仓库当前集成主要是LLM；需要审核真正Wan FFN W1/W2入口、native NVFP4 scale与布局，作者视频T2V≠我们的I2V | **P0-1 直接 Wan + FFN论文正向基线**。先审作者Wan实现/公开代码，单算子与full160视频分开，分开比native dense FP4和原BF16 |
| **TEAL → Wan (C20)** | 论文原作Llama而非Wan，但我们**已有真正Wan2.2-I2V的FFN全160自由生成**；不是需要重新证明LLM效果 | BF16原 W1→GELU→postGELU幅值置零→**完整Dense W2**；4开发来源20视频及四来源RAFT/八轴已真实完成，25%运动比1.170，50%1.376但某source 0.423，50%出现新增碎片。缺真正大M sparse W2核 | **P0-2 最低迁移成本 Wan FFN 质量基线**：先独立评价原视频具体事件失真，再决定是否设计大M有效 sparse W2或W1前输入路由；**不能称当前已有FFN真实加速** |
| **Chipmunk FFN (L09)** | 作者 Wan2.1例**MLP默认关闭**，而真正FFN列差分主要在FLUX等原实现；没有原作者已完成Wan FFN-only正向成绩 | 我们已经A100 BF16一组paid proxy K5120/13824真实EXIT0，误差0.279554优于同K随机0.309–0.321，但相对Dense误差仍大。真实cache/W1 proxy/两kernel cost需全部付费 | **P1 FFN候选**。先扩来源及真实raster组；如跨来源有效，再审双专家四步全160原生视频；不花时间重复原FLUX/Hopper系统 |
| **RT-Lynx (L08)** | 原论文为image DiT而不是Wan，研究**FFN activation N:M**、norm compensation、轻量LoRA可选 | 需要仅选无需LoRA的真正FFN activation N:M 规则、重新实现到Wan GELU W1/W2与对应硬件N:M kernel，物理基线需比Dense native kernel | **P1 原方法物理FFN候选**，只有源码分支确实免训练且可用Wan，才申请实验 |
| **Activation Concentration (L12)** | 跨diffusion架构研究FFN列的实际零密度与可执行列比例；原论文并非可直接一键Wan算法 | 适合用已有 TEAL 真实全160 tensor 检查行元素零比例 vs 实际全列/whole block可跳比例；不必新GPU | **P1 成本可达性解释基线**，若物理全列数量过低，应优先改变合法物理行动单位 |
| **OBS-Diff (L11)** | diffusion一次性静态FFN神经元/权重剪枝；原paper不是Wan I2V，可能不需大训练 | 需要辨别纯one-shot后无需更新是否具有有效的Wan FFN/GELU/归一化原生权重切片路径 | **P2 静态结构对照**；优先复用本仓已失败的BF16低秩、全深度视频资产而不是盲目跑训练 |
| **Prox (L69)** | 原始LLM SwiGLU FFN无训练，**不是作者Wan结果** | paid INT4 preW1 proxy与Wan GELU不同；不当作Wan原生论文复现 | P2机制迁移参考，不执行LLM |

**L34/L36/L37/L51**虽涉及视频/Wan FFN、但依赖大的蒸馏/LoRA/路由训练，不满足本次“免大训练”的先决条件。**L19 DiffSparse** 包含Wan视频，但主要是学习/蒸馏token稀疏而非纯FFN。它们不入队。各条都需重新核对真实代码、硬件核与具体原生作用部位，不用标题/论文混合速度替代证据。

## 实际进度记录 / 本轮没有新科学实验

- [TEAL Wan 全160四来源视频](TEAL_WAN_ALL160_VIDEO_V3_NATIVE_RESULTS_20261029.md)：20完成视频、4part EXIT0、研究12/中性8；[四来源八轴和RAFT](TEAL_WAN_ALL160_FOUR_SOURCE_QUALITY_RESULTS_20261029.md)全部EXIT0。postGELU置零仍全W2 dense，未节省物理计算。
- [Chipmunk A100 paid-proxy](CHIPMUNK_PAID_PROXY_EQUAL_K_RESULTS_20261029.md)：一个曝光source/high block20/128非raster行，K5120误差0.279554，真实EXIT0、真实同mask检查；不代表四步视频效果与真实系统节省。
- [TEAL Llama检查点转换](TEAL_LLAMA3_AUTHOR_CHECKPOINT_CONVERSION_RESULTS_20261029.md)：历史已完成CPU资产，因用户收紧目标停止后续与Wan FFN无关的LLM校准/生成实验。

## 每项原生 FFN 复现的验收

1. **同一模型/数值族**：原BF16 Wan2.2 I2V A14B Dense、对应稀疏 vs BF16；若采用SharQ应另有native NVFP4 Dense**同格式**基线，且后补BF16质量参照；双expert跨步隔离，不把Wan2.2 T2V当I2V。
2. **合法 FFN 省算**：说明mask/selector信息在W1前还是GELU后、动静权重量化与在线激活scale、原生packing、GPU大M与group closure；记录有效调用skip、W1/W2实际kernel及端到端时间，扣除决定/重排/cache/补偿两GEMM成本。**如仍执行Dense GEMM，则只是质量模拟。**
3. **视频质量**：以原图像来源独立聚类、同noise与seed、全部160站点自由学生、完整81帧、连续RAFT、事件/主体轨迹/新增物体/相机/背景/八轴指标、每source差异和异常；不能把运动额外增加当与Dense语义相同。
4. **实验设计强制先独立复审**：主Agent具体自审、实际不同Agent提反例、修订及reviewer recheck、精确代码和模型SHA/ACTIVE冻scope/资源、真实RUN/EXIT。没有复审不launch；“source静态检查”不代表GPU或视频结果。

**当前科学排序**：先审并实现能在Wan FFN真正运行的SharQ物理路径；并行利用已完成TEAL视频定量诊断和设计可物理省算的FFN实施；Chipmunk单组提供对比但不自动扩大投入。所有新scope均需重审。**本次仅更新优先级/源级计划，不新增真实运行，不宣称新审核批准。**
