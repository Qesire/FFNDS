# 固定坐标函数空间的独立源码与数值秩审阅

日期：2026-10-26。作者：`whole_flow_native_review`。类型：独立数学／软件审阅；原有 v1–v6 评议保持原字节。

**DISCOVERY — 正向研究对象。** 四个预先固定的 W1 投影可以生成不同的低复杂度函数空间；把共同的 128 行样本与实际 W2 有符号值向量结合，能够区分“联合非线性交互解释了功能变化”和“额外的样本空间维数带来了普通的样本内拟合能力”。此次工具同时加入共同常量、四投影仿射、15 列联合二次、15 列可分多项式及按主数值秩匹配的固定随机控制，并报告减去同一 pooled 均值后的创新能量。这个问题比把任意非线性曲线称作一个低维坐标更可辨识。

**竞争预测与廉价判别。** 若联合交互有解释力，联合二次应在相同原始坐标、样本和有符号 W2 几何中提供超过仿射及同秩随机空间的增益，并与同名义宽度的可分空间呈现不同误差结构。若增益主要来自普通样本内容量或均值，匹配随机控制、常量基线和 centered-innovation 分母会解释相当部分改善。下一项信息量最高的动作是根代理登记这份精确协议后运行一次固定 CPU 描述；本审阅没有执行该真实数据研究。

## EVIDENCE：精确审阅对象

| 文件 | SHA-256 |
| --- | --- |
| `tools/e0_fixed_coordinate_basis_oracle.py` | `d46c128716f7cf76b2b7a22e80ea875039590e16e3223f4548f4c3cef57bd793` |
| `tests/test_e0_fixed_coordinate_basis_oracle.py` | `7dd5620e06d6483c61fe07e000264059af473921fc5cd2b132cfd57c0ed07be3` |
| `independent-fixed-coordinate-basis-oracle-plan-20261026_v1.json` | `8ad454c8dc359e99b6f10c9396da95e9981061ed981bd4566f550cf84efef2cd` |

我逐项阅读了驱动、合成测试、固定协议及前置 identifiability clarification，并核对原有 feasibility 工具的坐标、目标、Smooth2 与 shard 解析语义。本次不调用真实 operand 的分析函数或实验入口。

### 1. 正确的函数空间与 signed-value 几何

令固定特征矩阵为 \(\Phi\in\mathbb R^{128\times p}\)，16 个系数目标为 \(C\)，原始解码 W2 的 16 列为 \(V\in\mathbb R^{5120\times16}\)，则部分输出 \(Y=CV^T\)。理想精确算术中，给定特征空间的共同 raw-output 最小二乘残差为

\[
R=(I-P_\Phi)C,\qquad
L=\|RV^T\|_F^2=\operatorname{tr}(V^TVR^TR).
\]

对样本行的投影与固定 W2 右乘可交换；因此先投影 C 再用完整的 \(K=V^TV\) 计分是有效构造。实现保留非对角项；用每个系数独立能量之和替换该 Gram 会丢失有符号抵消。合成 cancellation 测试把 Gram 结果与直接输出平方范数比较。

这个 `OFFLINE_FUNCTION_CLASS_ORACLE` 是离线最小二乘能力参照：投影本身等价于对 readout 的样本内优化。代码的 `deployed_readout_fits=0` 与未持久化系数只说明没有产出部署拟合权重；它们不把该优化变成未经拟合的泛化结果。

四个 case 的当前 gate 分别形成 \(K_j=V^T\operatorname{diag}(g_j^2)V\)。代码评分的是同一共同 raw 投影所得 \(R_j\)，不是重新求 pooled gated 最优解。不同 case 的 gate 改变行／输出权重，无法把所有 gate 当作共同右乘拉出一个 pooled 目标；相应反例测试正确区分两者。常量基线与创新分母都使用同一个 128 行均值，没有逐 case 回归或重新中心化。

这些是 **FP64 部分解码权重函数的代数描述**。原始 FFN 全局 bias 被明确排除；16 个坐标没有覆盖全部 13,824 个 hidden 单元。连续 FP64 tanh-GELU／Smooth2、保存的 BF16 pre-Q2、保存的 BF16 post-Q2 是三个分开的目标。该工具不重演原生量化器、完整 BF16 GEMM 累积或全 FFN 数值行为。

### 2. 独立找到并已修复的数值秩反例

旧候选实现把所有浮点严格正奇异值都当作真实函数族的方向。取 `PCG64(3)` 的 128 个随机标量 t，并令四个 anchor 列全部等于 t：联合二次空间实际为 \(\operatorname{span}\{1,t,t^2\}\)，维数 3；可分空间为 \(\operatorname{span}\{1,t,t^2,t^3,t^4\}\)，维数 5。浮点 SVD 仍分别返回 14 个正奇异值，经 QR 得到 15 个数值正交方向。

原候选在联合二次真实空间之外增加了 12 个方向。选择其中一个与真实空间正交的单位 target，原 all-positive 投影残差范数约 \(5.7\times10^{-16}\)，真实三维多项式族的残差范数为 1。匹配 15 维随机控制并不能修复这个错误对象：匹配的是扩张的数值宽度。

最终实现把 `max(shape)*eps*smax` 以上的标准数值秩作为主前缀，用该前缀直接计算系数残差，并按这个主秩匹配随机控制；额外正奇异方向仅进入明确标记的 `EXPANDED_NUMERICAL_SPAN_SENSITIVITY_NOT_ACTUAL_FUNCTION_FAMILY_OPTIMUM`。重复 anchor 回归测试保留此反例。我的独立完整合成 `analyze` 检查得到主秩 3／5、随机控制秩 3／5、比较状态 `OPEN_RANK_OR_CONDITION_AMBIGUITY`；旧伪拟合 target 的联合二次主 SSE 为 1，扩张敏感性约为 0，二者已被正确分开。

反方向的风险也保留：真实但很小的独立特征可能需要巨大 readout 系数。现有 `1e-10` 独立方向测试仍被保留；条件放大超过阈值时不给家族比较资格。标准数值秩是披露的数值约定，不能把截断残差当作任意精确函数族的认证下界。所有阳性／阴性解释均限于本次预定坐标、基与暴露样本；扩张敏感性、单个随机种子或漂亮的样本内最小二乘残差不能建立普适排除或泛化结论。

### 3. 合成验证与资源记账

最终软件的 11 项合成 CPU 测试通过：无 NumPy／Torch 的模块导入、交互基区别、小独立方向、重复 anchor、可复现嵌套随机控制、signed Gram、raw／gated 目标区别、共同投影与同秩控制、大均值与创新分母，以及预扣 MAC 和完整预算。测试只验证这些软件／数学命题。

另用完全生成的 128 行、5120×16 W2 合成数组核对显式矩阵乘法账本：分析阶段 8,824,448 MAC，加固定 W1 重放上界 10,485,760，合计 19,310,208 MAC。协议保守上界为 19,719,936 MAC，低于 20,000,000；小特征 SVD／QR 的 LAPACK 工作另行披露。代码在乘法前扣账，保留失败 RUN，独占创建输出目录，并要求 ACTIVE 中恰好一个精确协议 SHA。30 秒数值阶段含导入和 mmap，前置哈希扫描明确不含在此计时中；RSS／时间／输出上限有失败门禁。这些检查没有建立真实数据运行已成功。

### 4. 实际阅读的一手数值来源

- [NumPy matrix_rank](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html)：浮点中精确线性依赖通常仍有非零 SVD 小奇异值，默认阈值处理这种误差；阈值也可能舍弃真实小方向。
- [LAPACK 线性最小二乘](https://netlib.org/lapack/lug/node27.html)：满秩与秩亏问题不同，SVD／完整正交分解可处理后者。
- [LAPACK SVD](https://netlib.org/lapack/lug/node32.html)：核对分解、正交因子与奇异值排序语义。
- [LAPACK 最小二乘误差界](https://netlib.org/lapack/lug/node83.html)：良好的后向误差并不消除条件数放大的解敏感性；把小奇异值置零是一种规定的正则化选择。

这些来源使本审阅同时挑战“所有严格正值都是真方向”和“所有小方向都没有功能价值”，而没有选择其中一个未经披露的极端。

## EXECUTION 与可用结论

执行边界：本审阅只做源码、数学、一手文档与合成 CPU 检查；零真实 saved-data 数值研究、零 GPU／模型／原生量化调用、零生成视频。根代理负责精确注册和一次固定执行。当前版本无阻塞此次有界 CPU 描述的剩余实质问题；这不是执行授权或方法／质量接受。

图锚为 F04／F06→F08R／F08／F09，当前边属于 posthoc／conditional。可识别的联合函数结构若在后续固定观察中成立，可为减少必要 W1 特征与 W2 readout 的新表示提供机制线索；真实在线 observer、readout、全 hidden 覆盖、原生 GEMM 工作和完整自由视频仍需分别验证。此次结果即使改善局部 SSE，也不自动改变已测全视频质量结论。
