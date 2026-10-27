**第三条探索：编码可用信息是否预测有符号功能（2026-10-27）**：[G11一手数学/论文审查](../literature/20261027/G11_QUANTIZED_SIGN_MAGNITUDE_INFORMATION_AND_FFN_FUNCTION.md)。weight的符号/scale可离线读、current input须先过Q1，W1 pair sign/cancellation及HiF4 Q2 group closure可能改变预先信息成本。竞争性解释为统计结构、认证负侧GELU、码本表示而非固定通道掩码，比较低位packed dense及有符号长时/视频功能。这是与FFN本征及G10实时路线交叉但独立的开放机制，零原生执行。

# FFNDS · 双线科研进程的数学对象、证据和下一条最小区分（2026-10-27）

**A线：内生FFN功能机制**仍以冻结的Wan2.2 I2V四步D-Q为真实对象，研究真实可省的FFN W1/GELU/Q2/W2工作、跨层完整视频质量与原生硬件成本。**B线：实时视频生成**作为独立源级研究对象，回答可由何种普通视频foundation经过何种训练/架构/系统流程转为连续交付，以及其FFN函数和GEMM成本的变化。两条文献/数学线同时推进，**所有新CPU/GPU/模型/视频运行仍暂停**（ACTIVE）。

## A线：把文献 G7/G8/G9 的竞争假说重新投射到原生实际计算

1. **冻结函数本身的有条件简化**（F04/F06/F08/F09/F12）：能否在相同 q 和双expert独立函数的前提下，找到比旧HiF4 whole64组更合适的低算力函数表示？真实全M tile满占、Q2 group闭包和后验signed W2抵消分别提供物理/代数的**已测约束**，不是固定新算子路线。
2. **跨步信息的价值**（G9 L²P / G8 SVD Cache/Chipmunk）：历史 \(H\) 用于预测新的 \(f(q)\) 是额外输入的近似**决策器**，不是固定FFN真函数本身的隐含输入；单一high/low expert仅各出现两次，且已有完整dense \`v_D(z_S)\`显示Student流状态偏移有实效。比较同一expert、不同seed/输入状态的冻结历史预测是否维持功能。观察时间早于W1还是晚于W1，决定所省实际GEMM范围。
3. **训练后重新分配函数**（Dense2MoE/FFF/SCD）：把函数本体重训练为route或解耦renderer能够超越原始后验mask的物理边界，但它是**另一个参数和架构的研究命题**；少量LoRA或轻后训练潜力独立保留。数学问题：在相同输入/下游函数质量约束下，新 \(F_\phi\) 是否以低复杂度表示 \(F_\theta\) 的重要功能？除拟合误差外，还须考虑多层/四步有符号传播和部署开销。

**竞争可区分性**：
- H-A1 冻结FFN本征函数有低成本条件表征：无需通过跨模块重新训练即可维持独立来源的全流函数，预测在source×seed held-out仍有稳定净省算；反解释为原模型denseW1所有列对真实动态/视频是有用的，只有配套训练的新函数结构才可能低成本。
- H-A2 新视频劣化源于多层/step受扰状态而非单site局部算子质量：已有保存的P23/P24支持区分算子和dense状态响应，不提供完整因果序列；继续设计全源/seed上的有限因果后缀判别（仅研究设计）。
- H-A3 可以低位密集或规则稀疏内核兑现速度：把真实 BF16 QDQ 与NVFP4-packed核区分，比较相同质量下W1输入依赖闭包、W2输出Tile、quant scale、metadata、selector、cache、weight residency与实际wall。

## B线：新的流式模型如何生成以及为什么FFN要求不同

**基本因果链**：[实时一手文献与源码](../literature/20261027/REALTIME_VIDEO_ARCHITECTURES_AND_FFN_METHOD_AUDIT.md)建立“bidirectional pretrained teacher→causal AR/few-step student→rolling KV and sink→新chunk内 K次去噪→frame/chunk decode→可见输出”。区别于A线一次性 81帧×四步全片模型，B线把生产单位从所有序列token缩到新frame或chunk，跨更长历史循环。

**全局外显任务指标**：\(T_\mathrm{TTFF}\)、chunk与display-frame deadline-miss、p95/p99/jitter、interactive response latency、steady output FPS、长时漂移/主体/光照/运动、audio-video同步（如适用）、每流GPU memory。已有FFNDS的 T_FFN/T_wall/VBench协议可作为设计参考，但对长期滚动系统须单独定义producer/consumer时间戳与源级输入变化。

**新科学机会**：学习两种正交时间尺度。
\[
\underbrace{C_{\mathrm{frame}}}_{\text{当前帧/外部运动/scene}}
+\sum_{k=1}^{K}\underbrace{C_{\mathrm{render},k}}_{\text{去噪细节}}
\quad\text{vs}\quad
K\cdot\underbrace{C_{\mathrm{full\,temporal+render}}}_{\text{旧全步重算}} .
\]
这不是可无条件等价的式子，而是SCD从作者模型观察提炼的功能分工问题。是否能映射到真实 FFN 的 gate-weighted signed W2输出（而不只是早层attention相似）必须通过原生分步算子观察与同质输入对照判别。

- H-B1 时间性FFN功能可较少刷新：SCD early/mid denoise-step重复观察提示尝试 \(\mathrm{FFN}_{\rm temporal}\) 和 \(\mathrm{FFN}_{\rm render}\) 的实际影响分离；反解释是attention缓存造成相似，FFN仍有密集高频作用。
- H-B2 结构/运动明确来自外部模型可减少状态推断（EdenLive），appearance可与motion的signed output功能分开；反解释是两类条件在全部隐藏空间仍高度耦合，无法便宜执行。
- H-B3 小 M chunk的性能主导因素可从算力转向权重访存/kernel launch。Low-bit dense (LongLive2 NVFP4 & EdenLive LightX2V) 是必须纳入的实际竞争对照；物理收益需要真实部署和严苛p99，不依赖理论MAC比例。
- H-B4 自生成历史使FFN失配具有正反馈/补偿依赖（SelfForcing/RollingForcing/CF++）：假定 local SSE 保持不变时长期状态仍可偏移；最小区别是同前缀/不同前缀、固定强度扰动、scene切换动作下的分时状态与视频功能，而非只在单帧或单block比较。

## 研究优先级与开放性

先按AutoResearcher WARD继续扩大 **原始FFN模块功能/时间冗余** 的论文查重链（尤其SCD消融中的MLP层、LongLive2 FFN原生算子与EdenLive实际LightX2V低位核），其次把每条文献按**模型训练身份、数学算子、决策时点、真实可省的W1/W2或整个网络调用、强比较器、视频功能/SLO** 关联到图。提出候选机制时可以超出现有K/S节点；图谱为已有证据索引，不是研究的边界或预定算法。各原生实验仍需未来明确授权，既有已冻结结果与质量认可身份保持原样。

**补充原始论文级别的双线联系（2026-10-27）**：[G11五篇深读](../literature/20261027/G11_FIVE_PRIMARY_PAPERS_EQUATIONS_CODE_AND_COUNTERFACTUALS.md)与[SharQ视频近邻源码/数据](../literature/20261027/G11_SHARQ_FP4_VIDEO_SOURCE_AND_FUNCTION_AUDIT.md)指出短chunk更接近少行 GEMV/GEMM，而原Wan完整M大矩阵并不能从CodeGEMM单vector查表/SpQt decode稀疏核直接获得收益。SharQ已有Wan2.2-T2V-A14B四RTX5090 FP4原生两路补偿，成为量化+稀疏竞品，仍不能直接代表I2V双expertD-Q和完整视频功能。
