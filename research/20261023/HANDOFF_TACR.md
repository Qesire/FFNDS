# TACR审查交接 · 2026-10-23

**研究结论：推荐探索“补偿后净通道动作的条件轨迹价值”。首版用少量完整动作直接预测价值；条件协方差是最有潜力的机制解释。**

主文档：[查重、推导与方向取舍](../../docs/TACR_FFN_查重与机制推导_20261023.md)。研究状态：[research_state_TACR.json](research_state_TACR.json)。源提交：27f86cd3f8821b2771f1be633baf32d4cc79ebb0；输入提案SHA和原始来源见[TACR_source_manifest.json](TACR_source_manifest.json)。

## 已完成的科学工作

- 读原提案、最新R16/R17、R4/R7/R11及RB2/HP1/TC1原始记录，核查原生行分派与补偿投影代码。
- 外部最近先例是PulseQuant；内部R4已经包含补偿投影、Gram交互和双组组合，RB2已经使用四个支持bank。
- 推导实际补算的净动作 \(D_g-\Delta C_g\)，修复原提案只传播裸通道贡献的目标错位。
- 推导 \(A(a\mid z)=E[2e^\top Qv-v^\top Qv\mid z]\)。均值向量代入漏掉条件相关性；零条件均值的动作仍可能显著降低损失。
- 完成10项合成代数检查与1项历史标量复算；结果见[TACR_cpu_deductions.json](TACR_cpu_deductions.json)。新模型调用0，训练步数0。
- R16局部SSE改善6.12757%，RGB MSE相对最佳固定高8.22575%；R17旧来源每臂五次测速中位Dense50.85824秒、UB147.27993秒、动态48.68239秒，均复算一致。

代码覆盖post-Q1输入、现有R256、实际54/108/108/162组基础支持、alpha/BF16与folded bias顺序。新增通道动作、mask相关补偿和它们的下游响应仍为设计对象。

## 新机制与最近关闭路线的关系

| 模板字段 | 本轮内容 |
|---|---|
| 拟议研究对象 | TACR完整补偿动作的条件下游价值；处于探索设计 |
| 最近历史路线 | R4E1、RB2、HP1/HP2、TC1、R11T1、R12D0、R14N0 |
| 结构变化 | 原生完整补偿净动作；下游损失差标签；区分条件均值和二阶相关；以少量规整动作保证可执行 |
| 历史障碍如何处理 | 现有R7/R11/R16支持重新定义价值目标；新目标上的条件收益尚须直接测量。增加bank、rank、q、来源或重复C-score预测不构成此变化 |
| 可区分预测 | 直接价值/条件矩模型超过均值向量与幅度排序；来源留出条件策略超过同预算最佳已选固定 |
| 最便宜的下一步 | 先冻结每个动作的补偿、bias、舍入、区域与原生边界，再设计同前缀有限响应资产；现有局部样本不能代替新下游响应 |
| 负结果分支 | 新动作被UB1支配则改动作/补偿；仅位置固定价值则形成静态对照；强非线性交互则保留整动作库而非扩大线性组合搜索 |
| 计算 | 两侧GEMM、全行R256、选择、读出、QDQ、分组与搬运都计入；预算档比较同实际时间/质量 |
| 数据 | 已有24校准、R16及旧source0计时属于已暴露证据；新训练/选择/最终来源分别固定 |
| 原生候选晋级 | 动作实现与有限响应支持之后，再建立tracked candidate和完整四步自由视频比较，接续官方视频质量与实际系统时间 |

当前模型执行状态由仓库CURRENT_DECISION和R17交接管理；本轮作为附加研究审查发布，历史关闭范围及模型资格保持原记录。

## 尚待验证的机制

1. 新完整动作在同成本固定比较中是否有可利用的条件价值。
2. 传播后的补偿残差相关结构是否能由W1前廉价信息预测。
3. 有限动作的作用是否近似可加；低维表示是否保留动作决策。
4. 这些收益是否迁移到候选自身状态与完整自由视频。
5. 新算子和选择成本是否形成更好的实际质量—时间点。

## 复现与验收

在仓库根目录运行：

```text
python tools/check_tacr_deductions.py
python .agents/skills/mechanism-research/scripts/validate.py research/20261023/research_state_TACR.json
```

第一条只使用Python标准库及已提交R16/R17摘要。第二条检查研究状态格式与引用一致性。数学例子与历史模型测量在结果文件中分别标记。评价指标沿用学术指标族，实验时按问题选用，保留来源聚合、配对参考、实际计时边界。


## TACR v2 sampled net-action experiment, evidence reuse and next practical move

The updated forward method and experimental scope are in [docs/84](../../docs/84_TACR_净动作价值与修正采样_20261023.md), [the sampling plan](TACR_sampling_scope_v2.json), and [real interrupted R17 source audit](TACR_sampling_evidence_v2.json). Saved R17 quality evidence has five complete paired subject source groups (source indices 0..4), plus only dense/stride for subject source5; 27 source-level media files are not independent video-source samples. No background or camera source has full five-arm quality; official VBench calls are zero. R16 time bin0 receives 66.97% dense rows versus 22.90% later20; investigate benefit, do not assert cause. New TACR source-disjoint native intervention phases across four steps, layer depth and temporal zones are prospective, with pilot 12/expansion 24 positions illustrative, not rigid pass/fail conditions. Use full compensation-aware native action, true downstream finite response and actual W1/pre-step observable for value prediction. Reuse old artifacts before authorizing selected new full-free video. No modification of R17 caps or execution eligibility.

## ACTIVE next-research routing after R17/R16 review (2026-10-23)

The scientific follow-up is now [coupling / pre-W1 information / positive literature synthesis](../../docs/85_耦合机制_前置信息_积极查重_结果不佳转向任务_20261023.md), with executable research checklist [COUPLING_PIVOT_TASK.json](COUPLING_PIVOT_TASK.json). R17's incomplete official evaluation remains UNKNOWN, not a negative quality outcome. Begin source/saved-evidence/CPU S0/E0/E1 now, then decide whether actual compensated whole64-group actions require a separately authorized native capture. Where the newest completed candidate is unsatisfactory against matched Pareto, switch from repeating identical sweeps to the new coupling mechanism task. Compare upstream information conditional on current post-Q1, norm scale and attention residual, with native action-response and time/space coupling. Search similar and complementary papers positively; only substantively complete duplicates become credited baselines instead of new independent claims. No new model execution entitlement.
