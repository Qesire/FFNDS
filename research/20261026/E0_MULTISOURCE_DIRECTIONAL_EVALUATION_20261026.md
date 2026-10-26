# FFNDS · 六来源起步、10+来源迁移与有符号功能评价（2026-10-26）

> **阶段：研究设计已经登记，新增原生GPU任务尚未执行。** 本研究方向由现有两来源完整native paired和全模型Dense反事实出发，增加独立来源与可区分的多维终端质量。正式来源身份、指标与执行状态由 [机器可读计划](../../config/e0_multisource_directional_evaluation_20261026.json) 和 [ACTIVE](../ACTIVE_RESEARCH.json) 区分；旧原生RUN、质量SHA和协议不改动。

## 科研对象：必要输出方向与实际少算

现有 320-site paired 原生实验中，318处Student-prefix局部raw误差减小，但两个来源的最终latent偏差增加。[完整网络有限反事实](E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md)实测六个非零状态对的Dense能量增益3.63–7.72。[末步解码对照](E0_SAVED_FLOW_DECODE_RESULTS.md)中，Source12平均RGB RMSE改善13.58%，Source0仅改善0.45%，运动/细节代理各有不同变化。[频域分析](E0_SAVED_FLOW_SPECTRUM.md)发现占实维4.79%的固定频带含末态偏差69.99%/53.47%，但这个频带的全域cross符号与局部cross甚至可能相反。

**新目标是：预测并保留完整视频必需的有符号FFN功能方向，选择可以廉价获得并实实在在减少W1/W2/QDQ/数据搬运的表示。** 全局SSE仍是严谨的量，但转为辅助诊断量；单独优化它、或仅用低频能量占比排序候选，均不能构成方案的科学验证。

定义原生step映射 \`T_D,T_S\`。在同一个已测Student状态：
\[
P=T_D(z_S)-T_D(z_D),\quad N=T_S(z_S)-T_D(z_S),\quad \Delta'=P+N.
\]
在事先冻结的正交子空间 \`B\`，保留三种可辨识贡献
\[
\|\Pi_B\Delta'\|^2=\|\Pi_BP\|^2+\|\Pi_BN\|^2+2\langle\Pi_BP,\Pi_BN\rangle.
\]
也保留 \`||Π_BΔ||²\` 以明确输入状态偏移；此分解对已测有限state对成立，不声称全局Jacobian或者将某一方向认作终端画质因果。先前20个嵌套累积mask只有经差集改为互斥频带后才能相加，推荐固定DC、低频非DC、中频新引入、其余高频；报告每个band的P、N、signed cross和剩余频段。**latent的21个时间位置不是81帧视频，频谱低频不是主体/背景的语义标签。**

## Source 扩展：第一批6个，第二批10个以上

依据已核验 [六来源原生回执](../20261025/E0_NATIVE_OBSERVATIONS.json)，固定来源 \`[0,4,8,12,16,20]\`（实际类别：3 subject、3 background、0 camera），源SHA和seed见计划JSON。当前Native覆盖0和12，每来源160个site，故已测320；新增4/8/16/20可把同一固定bank的全深度测量扩展至**6×160=960个site**（新增640），预计新增**8条完整Dense/candidate视频**。这些都是**拟议预算，当前新运行计数为零**。原4/8/16/20已参与Affine54参数校准，因此只能用于已暴露source稳健性复验，**不属于独立held-out视频泛化测试**。

第二批至少加入**4个真正不曾参与拟合、择优、指标设计或原生判别的不同source**，总数至少10。实际index/image/prompt必须先从\`.runtime/inputs/calibration24/canonical.json\`和历史全部来源索引核验；若CAL24候选没有真正未暴露来源，应获取外部新的合法I2V输入并重新冻结SHA。真实存在的相机运动类别优先补充；没有可靠标签时写UNAVAILABLE，不能按背景流动猜测为camera。至少两个独立source交叉使用公共seed和另一个seed，从而比较内容差异与随机轨迹。此阶段还需要按实际source-level异质性估计是否进一步扩样，10只是探索下限。

**统计独立单位是source×seed的视频生成轨迹，160个layer×step site属于嵌套重复观测。** 测试时保留每个source的完整区间、step、expert和层段分布；不会从320/960 site推断320/960个独立样本。

## 多维评价及真正的判别机制

| 证据域 | 新标准（逐来源、逐seed、逐step保留） | 对应竞争机制 |
| --- | --- | --- |
| 局部函数 | teacher/student同输入FFN误差、raw/gated/writeback、post-Q1范数和方向 | 数值偏差/局部幅度收缩/函数容量 |
| 原生完整状态 | P/N/正负cross、频带内输入偏差和输出差，四步变化 | 已有Dense状态响应与当前候选注入，来源依赖的交互符号 |
| 固定频带与区域 | **互斥**频域分区、帧时窗、固定几何区域；有凭据的mask再对应实际主体/背景 | 时空输出方向是否解释非一致的视频变化 |
| 视频输出 | 完整81帧逐帧RGB/luma和p95/max、时序差、流幅度/位移、motion-compensated差、曝光/背景亮度、细节 | 静态画面好看但运动冻结、背景光照闪烁、细小物体失真 |
| 质量和部署 | 真正运行的VBench-I2V/VBench-2.0、LPIPS/DISTS、人工全片及原生FFN/wall/memory | 终端质量与实际物理省算的Pareto边界 |

**判别方式**：每个候选先报告按source聚类的指标**向量**及有符号结构，再用预登记的视频分项质量门禁和Pareto比较；不随实验结果再调一个全局加权SSE。若想将某一频段或函数值权重变成新训练目标，先要证明其对未暴露source视频指标的预测价值，且说明该信息在真正推理时是否可获得。原本两个source可用于构造假设，新的独立source用于反驳或支持其迁移。

所有新视频需按 [全视频完整指标政策](../agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)留存来源、实际帧、模型SHA、视频、真实成本和官方/人工结果，未运行评分维持NOT_RUN；FFN/WAN是否省算依然需要原生kernel及whole-wall观测。

## Web research：为什么采取分层评价

- [VBench / CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.html) 明确拆开subject/background consistency、temporal flicker、motion smoothness、dynamic degree和imaging。
- [VBench-2.0 官方实现](https://github.com/Vchitect/VBench/blob/master/VBench-2.0/README.md) 进一步区分物理、可控性和固有真实度；[VMBench / ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/html/Ling_VMBench_A_Benchmark_for_Perception-Aligned_Video_Motion_Generation_ICCV_2025_paper.html) 聚焦运动与人类感知。
- [LPIPS / CVPR2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html) 和 [DISTS](https://arxiv.org/abs/2004.07728) 提供感知、结构和纹理补充，不可用来替代时序与原生操作成本。
- [He et al. / ICML2026](https://proceedings.mlr.press/v306/he26q.html) 论述diffusion score误差逐步累积，是机制对照，而非Wan此处有限映射的定理。
- [Beyond FVD / JEDi](https://arxiv.org/abs/2410.05203) 指出FVD与样本量和时间敏感性的限制，小样本优先做配对source级诊断。

## 执行入口与冻结条件

优先完成四个已核验新增source的哈希预检与明确预算；独立冻结新source级native协议后，由具备A100运行权限的执行器实际提交。已有两来源的v3和whole-flow仍提供相同的强Dense比较器，但**旧协议只绑定原两来源，不能直接复用旧SHA冒充新执行**。新模型范围按[ACTIVE](../ACTIVE_RESEARCH.json)中持续授权与分范围登记要求执行；保留失败计数，不自动扩大预算或声称质量验收。本次更新的是研究设计、数据契约和静态检查器，**未启动新GPU作业**。G6实时/streaming仍暂停。
