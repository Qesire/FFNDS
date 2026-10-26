# FFNDS · 当前上下文（动态科研交接入口）

本页是E0_OPEN_DISCOVERY的最小Agent上下文，**不存储固定的“最新GPU作业状态”**。当前stage、授权、执行、失败、采集进度和质量结果必须从[ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json)、固定protocol与实际RUN读取；不得把先前`NOT_STARTED`或某次运行中的`RUNNING`当作永久事实。

## 核心科研判断

[唯一权威机制/质量科学审查](../20261026/E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md)已经合并旧post-affine方向评议、Affine54方法族Skill审查、完整视频方向审查及追踪偏移复核。其判断为：

- tanh-GELU实数奇偶与完整非线性余项的代数关系成立，固定54组截断、回归拟合和BF16/QDQ部署不是该恒等式的精确推论。
- 当前Affine54在两个已暴露来源的全40层×四步固定替换下实现真实FFN与warm生成加速，同时造成明显的视频偏离。这一张力支持NUM/CAP/SHIFT/TRANS/STATE的竞争性判别研究；8.54% teacher输入局部SSE是有用的局部指标，其对终端质量的因果贡献由原生配对和下游干预确定。其它Affine/非线性余项与独立表示继续开放探索。
- 已有六视频共486帧/480组帧间核心CPU指标补评，但正式VBench-I2V/VBench2、语义背景分析和人工全片未通过完整验收；旧四份候选的Dense参考model_digest元数据存在勘误，原始资产不可篡改。
- **最新全深度机制观测（NATIVE-OBS）**：[v3两来源320处配对结果](../20261026/E0_PAIRED_FULL_DEPTH_RESULTS.md)实测318/320处Student-prefix raw FFN绝对误差较Teacher前缀小（其余2处相等），post-Q1状态能量/方向改变且latent差在四步传播中扩大。后3步flow差与已有latent差高度反向，给出有符号传播线索。完整dense在Student latent上`v_D(z_S)`仍待有限原生测量，因果主因保持OPEN；官方视频质量尚未验收。

## Agent 工作模式

**A｜已测原型失配判因**：用同一真实输入的数值差、Student受扰前缀、postgate写回、40×4 signed flow/latent与受控后缀区分竞争解释，形成可迁移的函数与传播认识。

**B｜开放机制发现**：来自信息可得性、条件函数表示、signed W2、时空运动、实际运算/访存和可证伪数学问题；允许完全不同的新方法。与Graph-Agent的[节点注册表](GRAPH_AGENT_BINDINGS.json)双向交接：P21局部OBS、P22真实视频失败、P23/P24 OPEN、P25质量政策。G6实时方向暂停。

新视频以[质量/物理证据规范](VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)形成完整可复核结果包；分别标注 `ALG`、`SRC`、`OBS-TEACHER`、`NATIVE-LOCAL`、`VIDEO`、`SYS`、`OPEN-CAUSE`。实验范围、新模型调用与训练权限以活动状态中的独立明确授权为准。

**工作起点**：[唯一第一任务](../20261025/E0_FFN_FIRST_TASK_20261025.md) · [权威科学审查](../20261026/E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md) · [实验发现与证据](../EXPERIMENT_FINDINGS_ATLAS.md) · [Graph-Agent协议](GRAPH_AGENT_COLLABORATION.md)。

**原生运行证据索引**：[v3原生全深度结果](../20261026/E0_PAIRED_FULL_DEPTH_RESULTS.md)为当前实测起点；[v2后处理路径失败及DD视频核对](../20261026/E0_PAIRED_V2_FAILURE_AND_RECOVERY.md)仅记录其历史失败范围。执行权限与实时进度从ACTIVE及原生RUN读取。

**研究工具与基础**：[默认WARD深度检索](DEEP_WEB_RESEARCH.md) · [FFN本征纲领](../20261025/E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md)。
