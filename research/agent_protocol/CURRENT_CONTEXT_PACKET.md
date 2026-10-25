# FFNDS · E0 当前科学上下文（2026-10-25）

**唯一活跃科学入口**：[FFN 本征机制重置](../20261025/E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md) → [首个 FFN 科学发现任务](../20261025/E0_FFN_FIRST_TASK_20261025.md) → [ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json)。本文件是阅读索引，不增加实验权限。**从 FFN 的输入系数与 W2 value vectors、有符号组合、时空变化、下游功能和可替代计算开始；量化尺度属于辅助幅值线索。**

## 真实对象与解释口径

- 当前已完成观测环境：Wan2.2 I2V 蒸馏四步，LightX2V D-Q/HiF4 QDQ→BF16 GEMM，5120→13824→5120 双专家 FFN，含 AdaLN、gate、残差和后续 attention；真实源码身份见 [C0 源绑定](../20261023/C0_coupling_offline_results.json) 与 [runtime 重建](../20261022/R12_E0_runtime_reconstruction.json)。
- 量化元数据在此研究默认视为可选的粗略幅值/范围统计；HiF4 现有 QDQ 并非原生 MXFP4/NVFP4 数据布局或 FP4 GEMM，方法发现不再预设 scale/code、K panel、64 hidden group 或 QDQ 是算法核心。
- 强调四对象分离：`|GELU(a)|` 强度；`Σ_j GELU(a_j)W2[:,j]` 的净作用；后续 norm/attention/多步生成的功能价值；可真实跳过或共享的运算和读取。
- 最近已保存的径向对齐 **19.794% raw 正确对 20.542% 错排、8.839% centered 正确对 3.590% 错排**及同 seed 六来源共有方向 cosine 0.99645，属于特定 32 行×5组 ideal epsilon-LN 表征结果。**不是 FFN 整体功能冗余率**；见 [E0_RADIAL_PAIRING.md](../20261025/E0_RADIAL_PAIRING.md) 与 [复审](../20261025/E0_SKILL_WEB_REAUDIT_20261025.md)。
- [原生四臂跟进](../20261025/E0_FOLLOWTHROUGH.md) 提示 BF16 写回与下游连续传播差异，属于可供未来功能机制分析的既有证据。star-null 是特定相关误差零模型，不是动作可加的充要条件。详见 [E0_SHARED_REFERENCE_MOMENTS.md](../20261025/E0_SHARED_REFERENCE_MOMENTS.md)。
- 历史 A5+HWG 发现主体外光照/细节不稳定；将其转为时间/位置/频谱重要性的竞争解释，研究目标仍是同等设置下全视频质量与实际 FFN 计算的 Pareto 改善。

## 当前第一组开放科学探索

**A FFN hidden→W2 的有符号贡献、superposition 与敏感子空间**：一个通道激活很小，不必然对应净输出或可删 tile 的价值；检查真实已存字段和能量/Gram/时空条件假说。

**B FFN token×空间×step×expert 的条件创新**：什么绝对输出和更新量能真正共享/复用？区分同坐标与同物体，4步双专家意味着有限历史可用。

**C FFN 对下游视频功能的有限影响与计算重构**：用残差/attention/去噪轨迹连接局部作用与背景光照/细小动态敏感性；提炼新的可省算自由度，不能将旧 dense completion 当成研究目标。

独立挑战者通过 [扩展研究文献地图](../20261025/E0_FFN_INTRINSIC_PRIOR_ART_20261025.md) 对照 Geva/FFN weighted-values、superposition、Chipmunk、RT-Lynx、AdaCache、VBench 与 GPU tiling 等原始工作，核对对象、信息时序、训练负担和实际 kernel。默认 AutoResearcher v2.4 的 mechanism-research、research-expansion、problem-reformulation、[WARD 深度检索](DEEP_WEB_RESEARCH.md)。所有 Agent 在 [E0 独立讨论](../method_discussion/discoveries/E0/README.md) 保存自己有实质差异的科学贡献。

## 可使用的证据与能力边界

[完整实验证据导航](../EXPERIMENT_EVIDENCE_INDEX.md) 与 [量化优先时代旧 E0 第一任务](../20261024/FIRST_TASK_E0_QUANTIZATION_FFN_DISCOVERY_20261024.md) 都继续保存，以服务确切新机制而非统领下一任务。已有六来源原生被动12次、有限四臂30次、跨节点技术重复10次，全都完成；无待运行的新模型任务。历史最高并发2张、授权最多4张的数字只是已完成范围。当前可以进行源码/原论文/数学/获准的离线保存资产复算；新的 native model/GPU/视频/训练范围需要用户再次明确授权。现阶段 `E0_OPEN_DISCOVERY`，未选择任何新算子，同行认可仍由真正的独立评议产生。
