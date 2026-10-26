# FFNDS · 新工作证据审查与机制研究引导修订（2026-10-26）

> 当前源与执行状态见 [ACTIVE](../ACTIVE_RESEARCH.json)；本文是对已经合并入main的实验与独立评议作**科研解释**，不等于新增GPU/RUN、原生FFN候选或正式视频质量通过。始终以[AutoResearcher机制先行纲领](E0_FRONTIER_MECHANISM_FIRST_20261026.md)为科学起点，而非更换SSE坐标之后继续最优化。

## 1. 已获两种性质不同的结构证据

### F1：四输入投影非线性能力实测，跨神经元函数共享尚未识别

[固定基CPU实测](E0_FIXED_COORDINATE_BASIS_ORACLE.md)：high-noise/block10，原16个neuron与4个固定W1 anchor，来源0/12、step0/1共128已暴露行。共同常数85.50%、4坐标仿射65.34%、同rank随机75.34%、joint quadratic(15维)45.87%、separable nonlinear(15维)42.69%——全部是**样本内特定16-neuron输出**相对SSE，具有数值rank审核，非独立新source泛化或全FFN/生成质量。joint未战胜同预算separable，因此“联合二次交互是必要结构”没有获得独立支持。

四个anchor的连续GELU系数由这四个原W1投影本身解析给定；非线性基只要复现这四个已知函数，就能让full16的样本内损失下降。决定**是否存在可利用的共享机制**的是**另外12个neighbor**在相同输入坐标上的可预测性、独立功能以及跨source/step迁移。当前RUN并未单独记录这部分能力。

有价值的近端判别是现有[独立anchor-versus-neighbor竞争实验设计v9](../method_discussion/discoveries/E0/independent-anchor-versus-neighbor-transfer-discriminator-20261026_v9.md)：
1. 固定原16neuron、4个anchor、continuous/pre-Q2/post-Q2、two15列基和原W2 Gram，作**anchor4/neighbor12/signed cross**分块归因，区分“自重建”与“对别人有信息”。
2. 以source0→12和12→0两方向，用训练源自己的64行确定预处理与闭式readout，在另一source64行评价neighbor12，与同rank输入行错配、训练step均值和仿射进行匹配对照。这个训练/评价方向均使用**已经暴露的source**，是**本实验的label holdout，不是新数据集独立泛化**；seed42/54与内容身份绑定。
3. 正面信号是neighbor12在两方向均具有超越训练step均值/错配控制的条件函数信息，且具备推理时可获得、低成本输入坐标；负面信号也有解释价值，可以直接指向缺失输入轴、Q2变量或“许多独立系数”的机制。后续扩源应验证预先声明的条件结构，而不是继续提高多项式阶数。
4. 计费包含**闭式拟合/投影oracle**、真实anchor GEMM、共享基计算和完整W2读出。解析连续anchor bypass与原生HiF4 Q2不是同一个数值算子；16neuron的理想MAC下界不等于可部署吞吐。

**当前状态：这个分块/双向迁移实验是设计，尚未登记实际新RUN；原固定基CPU实验已完成。** 优先使用现有保存操作数和约束明确的CPU检验；不自动扩大到所有40层、更复杂函数族或模型运行。

### 物理分支：Blackwell packed NVFP4基元是真实执行能力，不是新的FFN机制

[已执行的SM120 packed原生基元](E0_SM120_NVFP4_PHYSICAL_PATH.md)在指定CUTLASS源码下完成两种随机packed E2M1矩乘配置，共5次真实GPU GEMM；官方host BF16参考逐元素相等。先前构建include故障与其后的修复已分别留存。**算量、真实Wan输入、在线FP4量化和端到端质量均不由基元成立推得。**

此项是**独立可替换硬件底座**。如果某个F1/F2/F3机制真正产生可打包的矩阵/路由形状，再在同一原生执行环境上分开量：真实tensor动态范围→online量化与scale/packing→packed计算→解包/布局与FFN输出→真正对比BF16及完整视频成本。旧HiF4原生路径属于QDQ→BF16，不能直接宣称已被NVFP4替换。保持多格式（HiF4、NVFP4、MXFP4、BF16）为有条件的实现比较，而不是论文研究目标。

## 2. 从新证据生成不同方向的三个研究问题

| 研究切面 | 新数学/计算问题 | 相互区分的下一证据 | 真正省算的必要条件 |
| --- | --- | --- | --- |
| **F1 结构单元** | 四个anchor的特征是否解释neighbor12的函数，还是只解释自己？哪些原函数单位可组合/共享？ | A4/N12/交叉项、跨source label-held迁移、step均值和错配负控；再检验真实Q2输入依赖 | 共享基数与readout实际比原W1/GELU/Q2/W2便宜 |
| **F2 条件/时空** | 计算不变性来自输入流形、step/expert、空间位置还是原权重几何？ | 在六已暴露source的合法相同步/同expert边界比较条件变化；未来真实独立来源交叉seed | 决策信号在昂贵算子前可取得；选择/重排/pack成本合理 |
| **F3 功能作用** | 可压缩单元是否与运动、背景光照、细节产生特定因果联系？ | 明确冻结候选函数单元与同物理费用的最小原生有限干预，完整自由生成与多维视频证据 | 视频保真且真实FFN和whole-wall双收益 |

**这些是平行的发现切面，不要求F1的一个小簇实验通过才能探索F2/F3。** 例如原Q2共享scale、原生tile闭包、跨专家时序、真实视频受扰状态都是独立的问题来源。避免因新N12判别有清晰步骤，就把整个研究缩成这16个neuron的函数拟合。

## 3. 当前Agent默认行动

**DISCOVERY：** 先提出最有解释力的正面计算机制，标明结构规律、真正未知的自由度、与最近论文/旧Wan实验竞争关系。优先利用实际源码、saved operands与真实小实验形成不同预测，形成机制提案，而不是直接指定一个SSE/频段/anchor/FP4格式作为下一步优化对象。

**DISCRIMINATION：** 对固定四anchor支路，优先分离neighbor额外信息、step均值和token错配；对其它支路选择其自己的最小证据和反例。跨source分组以真实source×seed轨迹为单位，重用校准池的source按exposed标签呈现。原生层/步/频带观测作为理解工具，不以描述性数值直接定义视频功能真值。

**EXECUTION：** 每次真实实验记录预先定义的source/seed、信息可用时点、权重/临时闭式拟合、GPU/CPU物理成本，证明候选动作实际产生不同真实计算行为。原SM120 primitive的GPU成功可用作硬件参考，只有在真实Wan计算接线测过才升级到FFN benchmark；正式视频与质量独立通过才允许算法acceptance。原有连续研究授权与G6实时暂停状态保持不变。

## 4. 上次引导的已修正事实

已完成：[full dense在Student latent上的有限反事实](E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md)、[两来源paired-native](E0_PAIRED_FULL_DEPTH_RESULTS.md)、[末步Dense视频桥](E0_SAVED_FLOW_DECODE_RESULTS.md)、[固定坐标CPU oracle](E0_FIXED_COORDINATE_BASIS_ORACLE.md)、[SM120合成packed执行](E0_SM120_NVFP4_PHYSICAL_PATH.md)。

开放：anchor→neighbor跨source条件预测、非线性表示是否能在真实Quan/GEMM省算、哪类FFN作用因果影响视频、跨未暴露source的迁移、真实Wan FP4定量/核函数成本、正式VBench和人工质量。

**单一最优方向尚未被证据确认。** 新机制以结构解释/信息时点/真实物理计算为第一评价；SSE、signed cross、频谱仅在对应问题下使用。两个source的有限动力学结果不等于总体规律。
