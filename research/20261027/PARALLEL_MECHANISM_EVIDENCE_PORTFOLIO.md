# FFNDS · I/F/C并行机制证据与最小判别（2026-10-27）

> **性质**：E0开放科学问题地图，提供竞争解释及最便宜的辨别证据。[统一研究目标](../RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md)定义信息与表示(I)、函数与动力学(F)、计算与执行(C)三个互相促进的入口。用户已恢复离线研究推进，最多同时四张空卡；各实际范围先在[ACTIVE](../ACTIVE_RESEARCH.json)登记冻结。本页更新已完成判别的状态，未执行的预测仍保留OPEN。

## I-P1 信息与表示：联合W1行的有符号码信息

**已知**：[G12](../literature/20261027/G12_SIGNED_QUANT_CODE_INFORMATION_CERTIFICATES_AND_EXECUTION.md)按正尺度重排不等式可对**单个**W1行/组求pair-agnostic紧内积界；多行共享同一个输入排列，单行最紧并不意味着其区间积对所有行联合可达。真实GELU与[HiF4 Q2尺度闭包](../20261025/E0_HIF4_COLUMN_DEPENDENCY_COUNTEREXAMPLE.json)进一步耦合最终gate×W2输出。

**竞争机制**：I-a共享输入与符号组合能使多行联合界显著收紧，部分约束可以廉价复用；I-b联合求解或额外配对读取的成本抵消这些数学收紧。

**已完成判别**：[604个有限排列与联合支持推导](E0_SHARED_PERMUTATION_FUNCTIONAL_RESULTS.md)给出精确离散域的凸包和固定signed支持，证明逐行hist丢失跨行对齐。固定方向还允许离线`p=Wᵀc`、在线`pᵀx`精确计算，这成为证书的强比较器；GELU与Q2的合法扩展、方向适应成本及真实kernel收益仍OPEN。**状态**：I17/I18为ALG＋有限软件核对，未声称本模型加速。

## F-P1 函数与动力学：跨seed的有符号交互来源

**已知**：[source425跨seed](../20261026/E0_FRESH_FOUR_SOURCE_SIX_PAIR_RESULTS.md)中后续三步的cross符号存在方向变化，而部分局部FFN raw位置统计相同。[真实整步finite-map](../20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md)中 \(\delta_{t+1}=P+N\)，\(P=T_D(z_S)-T_D(z_D)\) 为已有状态经dense传播的有限差，\(N=T_S(z_S)-T_D(z_S)\) 为相同Student状态下的当前近似作用。

**竞争机制**：F-a实际FFN门控signed联合输出对q、expert与step敏感，参与这种条件性变化；F-b继承状态和attention/下游flow传播放大或抵消当前写回作用，局部FFN子结构尚不能据此归因。

**当前判别**：[G13 C02保存行研究](E0_Q2_DIRECT_COLLATERAL_FULL_SHAPE_RESULTS.md)已在八个条件上分出直接W2与Q2补集，恢复原M并通过严格Q2/W2基线；这并未归因source425的cross变号。后续针对cross-seed的[O/D/Q/P](G13_FUNCTION_LOCALIZATION_METHODOLOGY.md)需明确同状态donor、首个消费者及忠实后缀。**状态**：已有条件性与局部数值NATIVE-OBS，具体视频功能承担者OPEN。

## C-P1 计算与执行：原QDQ路径和packed低位物理成本

**已知**：原生Wan HiF4执行为fake-QDQ + BF16 Linear；packed NVFP4/MXFP4在码存储、尺度、解码/运算形状和硬件路径上不同。旧后验token×channel支持在特定M16/K64 tile布局中仍可能使全部tile占用。已有[原生权重驻留](../20261025/E0_DENSE_WEIGHT_RESIDENCY.md)结果也表明系统wall与FFN arithmetic可分离。

**竞争机制**：C-a编码域重写、合法内核融合或内存布局优化能减少真实decode/GEMM/访存；C-b强packed dense比较器可能已具有更优真实成本，增加selector/packing或格式变化反而亏损。

**当前判别**：[原M W2控制](E0_W2_SHAPE_CONTROL_RESULTS.md)将八组保存输出全部恢复到字节相同，M160的约0.292%差异因此获得具体数值控制。继续沿W1/GELU/Q2/W2/gate核算真实成本，并与[G11作者多token实际路径](../literature/20261027/G11_FIVE_PRIMARY_PAPERS_EQUATIONS_CODE_AND_COUNTERFACTUALS.md)比较。**状态**：SRC、SYS与局部形状NATIVE-OBS；新算法成本优越性OPEN。

## X-P1 跨线路：重新训练会改变功能承担结构吗？

**结构问题**：原模型中谁承担了某个功能，与新的参数化或轻后训练模型用什么结构维持该功能，是两个不同问题。若令原FFN为 \(F_\theta\)，新函数为 \(F_\phi\)，可以研究以更少的实际物理计算维持完整生成行为所需的条件及训练预算。

**竞争机制**：X-a轻量后训练能够将部分昂贵的时空职责重新分配给紧凑、共享或低频更新结构；X-b有限训练无法维持四步传播中的运动/背景/光照一致性，必须保留更大容量或付出额外训练成本。

**最小判别**：分开冻结原权重的因果解释和新模型的训练/函数表示；逐一查验[G7–G9的相关论文、实际训练和算子](FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md)，比较最强同预算先例与旧[完整视频失配](../20261026/E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md)。当前登记范围不含训练或新增视频。**状态**：OPEN训练后机制设想，尚无新方法验证。

## 科研记录路由

I/C线的数学或计算新发现直接进入[独立发现目录](../method_discussion/discoveries/E0/README.md)及源图ALG/SRC/OPEN关系；F线的具体行为–函数–结构研究使用[Function–Structure登记](../agent_protocol/FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json)和G13；实际具有算子设计与可比成本的候选可进入[方法提案](../method_discussion/TEMPLATES.md)。各路径可互相启发，不将任一路径作为另一条的准入条件。完整视频和真实物理收益主张仍须逐级提交原生证据，旧实验验收和授权保持原样。
