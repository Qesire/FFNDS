# E0 · 全40层保存数据的非均匀误差：为何只测block0/30是不充分的

**数据身份**：纯读取已经完成的 [320-case四源校准、两持出来源、40层×4步](E0_ALL_BLOCK_RECONSTRUCTION.json)。分析值为 `by_block`/`by_step` 的 `affine_residual_g54_ridge → uniform_disjoint → relative_target_SSE.mean`（门控当前FFN输出SSE/当前输出能量）；每层是两source×四步=8个case等权平均、每步是两source×40层=80case等权平均。没有新模型/GPU/张量实验。**它是Teacher保存输入采样局部指标，而非Student自由轨迹的累计误差。**

## 对“只在两个block验证”的实证挑战

此前局部native partial test用 **block0、block30**（不是block39），四步×两来源16案例，另用128行重复至full-M计时；只有这些位置测了有限的native vs连续局部数值匹配。另一个全160位原生测试生成了全部视频，但没有记录各真实Student前缀的配对native局部误差。

现有320-case全层保存数据按局部Affine54平均SSE降序：

| block | gated相对SSE均值 | 该层8案例的最大值 | 同预算whole94均值 |
| ---: | ---: | ---: | ---: |
| 5 | 15.78% | 18.58% | 18.35% |
| 6 | 13.99% | 21.73% | 17.13% |
| 7 | 13.68% | 21.80% | 16.46% |
| 32 | 13.49% | 21.49% | 20.54% |
| 35 | 13.17% | 24.01% | 18.03% |
| 4 | 12.76% | 19.66% | 14.60% |
| 12 | 12.04% | 15.72% | 16.66% |
| 10 | 11.83% | 19.04% | 14.89% |
| 0 | 0.85% | 1.18% | 1.27% |
| 39 | 0.34% | 0.53% | 0.48% |

端点`0/39`特别容易被此方法重构；不能把最容易层的局部结果作为全层数值/误差传播的代表。**单看局部数值，严重偏差更常出现在内部的5–7、32、35等层**；其绝对下游功能重要性仍未知，需实际FFN门控写回、后续attention、head和scheduler来判别。

按step的全40层均值（每step两个来源×40层）：

| Wan去噪step | affine54均值 | whole94均值 | scheduler相关 |
| ---: | ---: | ---: | --- |
| 0 | **12.66%** | 15.60% | high expert、首次真实前缀被更换 |
| 1 | 8.81% | 11.24% | high expert、同专家第二次调用，但latent已可能离轨 |
| 2 | 6.73% | 10.44% | low expert切换，旧high历史非同算子 |
| 3 | 5.97% | 9.42% | low expert、最后去噪 |

它不能证明“step0最伤视频”——真实scheduler步长、flow方向、残差gate和终端敏感度并未纳入这些数。它支持优先在全160处记录step/depth/来源的实际student输入和signed跨层变化。

## 具体的第一轮区分路径（设计，不是自动试验）

1. **全160处 native paired 输入**：每source、每step、每block捕获`x_D`/`x_S`的来源/shape、同输入`f_D/f_S`标量/必要分层抽样值、实际gate后残差，以及本次Block输出差；先登记老师输入直接函数误差与学生同输入函数误差，再计算前缀函数响应。记录每个step结束的`head`、`latents`向量差，不仅保存hash。
2. **负例/强异质层匹配**：仅在完整表出现后再冻结有限的后缀干预窗口，至少覆盖低误差首/末与5–7、32、35等困难层、两专家四step及不同source；选择规则预先登记，若按测得影响大小选出极值，要在真正未参与筛选的source检验，不冒称来源泛化。
3. **两类真实因果分量**：只在`same x_S`上比较native`f_S(x_S)-f_D(x_S)`，可隔离算子差；`f_D(x_S)-f_D(x_D)`衡量输入变化经原FFN传递。两者有signed交叉项，不能把相对SSE简单相加得到总误差。再透过原生BF16 gate/residual和后缀有限运行观察增益/抵消，不做默认Jacobian光滑假设。
4. **与质量指标联动**：每条真实自由视频完整帧指标、运动幅度与运动错位、静态背景照明、细节/边缘、官方视频指标状态及同卡物理成本一起报告；Source0运动冻结、Source12额外时变可能对应不同机制。必须以完整终端质量结果约束下一算法，而不是仅优化本表局部数。

[统一指标/全深度政策](../agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md) · [诊断设计模板](../../config/e0_full_depth_quality_discriminator_TEMPLATE_20261026.json) · [前轮全部六视频](E0_RECONSTRUCTED_VIDEOS.md)。

**状态**：`E0_OPEN_DISCOVERY`；全40层保存统计已完成，但student-prefix全160位置因果观测及新全指标视频报告**未运行**。没有新GPU授权、视频生成/训练或新方法验收；实时场景依用户指示暂停。
