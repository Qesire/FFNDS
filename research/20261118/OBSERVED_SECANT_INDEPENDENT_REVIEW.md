# 当前响应驱动 FFN 算子：独立审查与实现记录

日期：2026-11-18。基点：f89e4fae868ac8d21b3b8f0da47616f43b0c954a。

> 版本说明：以下原审查记录和文件哈希属于a2a3f7e发布时的9+6方案，保留其历史身份。用户随后质疑局部MSE筛选，本次据此撤回旧门槛；当前12+3修正的独立复核另列于文末。旧审查通过不证明MSE门槛能够预测视频质量，也不自动覆盖改写后的文件。

## 原9+6方案：历史结论与范围

**DESIGN_REVIEW_PASS。** 两种构造的数学/数值定义、合成FP64参考核心、局部可辨识性、作用范围和9+6视频预算已由真实独立agent复核。该结论不覆盖尚未实现的CAL producer或BF16/Wan端口，不表示模型、视频、性能实验已完成。

实际分工：

- 主协调者：恢复既有源码/结果，确定方法规格、实验树、机器清单，整合并复核。
- conditional_operator_reference：独立实现NumPy参考核心和合成测试。
- conditional_plan_assets：只读核对学校资产、六位置/全80范围、坐标和预算。
- conditional_operator_peer_review：独立提出反例、审查公式/代码/实验树，修订后recheck，并独立执行测试。

三个子任务均由协作接口实际启动；本记录不把主协调者自审当成独立审查。

## 实质修订

| 问题 | 修订与验证 |
|---|---|
| 仅当前Gamma正则在近零位移下仍可拟合固定舍入噪声 | 加入tau² Gamma_CAL；冻结一次CAL定标；新增小位移噪声衰减反例 |
| ridge=0在白化坐标判秩、原尺度lstsq再次截断 | 判秩和求解统一在白化空间，新增跨尺度测试 |
| static控制不包含非零先验或换用了不同缓存 | 同一C0/Phi0与V_stat；同一U0/Y0、编译顺序与theta_stat；关闭adaptation时不调用observer |
| 截断当前支撑后错误丢掉完整静态预测 | 先扣除X_raw V_stat，再在保留坐标拟合delta，并重写误差恒等式 |
| full512拟合被误称query下界 | 每fold在query256标签上求连续类oracle；量化后候选单列，不冒充BF16离散全局最优 |
| CAL训练轨和部署目标不明确 | 结构选择FP64；最终prior、lambda、实际准入使用冻结BF16/FP32路径与实际probe |
| mode训练可能成为开放ALS搜索 | 固定来源级PCG80迭代/容差、作用Gram top4、正则和退化阈值 |
| 六位置失败可能关闭完整范围 | 0392三个六位置臂只诊断；随后固定执行primary全80六视频 |
| 人审待定会被当作失败触发备选 | HUMAN_REVIEW_PENDING不触发；只有完整判断NOT_MET才打开另一family |
| P512 probe、canonical P64与student query身份混淆 | 分别保留ID/producer/batch形状；student static/online共用query ID，不称其为full-M shadow |

封稿同时明确了零尺度配对列的sigma=1/inactive处理、T_CAL实际作用于特征、FP32 Phi缓存的1.25%状态成本以及混合基线A的准确范围。这些是实现消歧，不扩大实验网格。

## 已实际执行的验证

实现agent、独立审查者和主协调者分别执行：

    python -m unittest discover -s tools/ffnds_observed_operator/tests -v

三方均得到7/7通过、退出码0。主协调者末次测试显示0.032秒。只使用小型合成张量；没有读取模型张量拟合、真实回放、GPU或视频任务。

覆盖内容：

- 满秩观测对全部token响应的恢复；
- 输出方向随状态改变；
- 零位移、未观测方向与零秩；
- CAL度量随可逆换基协变；
- 固定观测噪声在近零位移下的衰减；
- observer只收到当前选中的p行；
- 全局token ID与行重排的一致性；
- native模式直接收缩与合并编译等价；
- 共享gate改变模式系数最优解；
- 非零静态先验与零observer调用的匹配控制。

主协调者另外执行机器清单的算术/范围检查：9/15 D4、36/60 NFE、1440/2400主链调用、246/486完整probe、student query上界、CAL/DEV不相交和token网格均闭合。

## 开发预算

| 项目 | 主路径 | 条件上界 |
|---|---:|---:|
| 新D4 |9|15|
| 新NFE |36|60|
| 主链FFN位置调用 |1440|2400|
| 当前完整64-token probe调用 |246|486|
| 对应token-FFN |15744|31104|
| 生成后query token-FFN上界 |31872|62592|

P64-6另有6次64-token W1/GELU观测。静态控制不支付无用probe。没有独立确认、额外Dense捕获或专用系统重复隐含在本表内。

## 封稿文件身份

以下SHA256由主协调者在整合修订后计算；实际执行仍需绑定生成者所用完整参数和原生端口身份。

| 文件 | SHA256 |
|---|---|
| research/20261118/OBSERVED_SECANT_OPERATOR_DESIGN.md | 4236fb01cabf9ed2e119172faf9596a65b954ca86c1966d341b7f58c961d3c6f |
| research/20261118/OBSERVED_SECANT_EXPERIMENT_PLAN.md | 69f954f70d95e3673160b5b89e31f305fe12fb4523cb6dffb7b8dca9977347cb |
| research/20261118/observed_secant_experiment_spec.json | 514476fa4126aef8c18161f83c68e341fa4d4da5066e2e16e29c599763515141 |
| tools/ffnds_observed_operator/core.py | e9e83ae3e0de05f1e0e6d96c712606597a81a9576b7c519658fbf2bdfba337f0 |
| tools/ffnds_observed_operator/__init__.py | 18e4199549e5dd0138ab43b0e89296bc3cb2040a96ece86258a2a0ec7c4cccd2 |
| tools/ffnds_observed_operator/README.md | 829d945ab16ef5af18e89453a9eb9caf300b3c6555a338d613ea690a0bbe5a7c |
| tools/ffnds_observed_operator/tests/test_core.py | 5160e8ae54bd100766802835adec21e8f5e5c1a5db38c7c88ac2f797603892da |

## 尚待执行的工作

1. 按设计完成CAL参数生产及原生BF16/Wan port。
2. 复用学校现有132对六位置端点与已登记880对全80 CAL端点，完成有限绑定和实际部署数值验收。
3. 将确切producer/参数/来源/预算按仓库既有ACTIVE流程登记后执行冻结实验树。

本次没有修改ACTIVE，也没有申请或启动GPU任务。上一轮已关闭结果及其原始实验身份保持不变。

## 同日修正：MSE角色与12+3实验设计

用户质疑继续用MSE筛选后，主协调者重新使用operator-tensor-analysis与evidence-analysis，提交以下自审结论：旧2%/8of11/10%门槛没有验证视频预测力；两类不同表示都应获得一次完整功能比较；CAL欧氏选择器保持受检身份，不临时扩展loss搜索；局部query与oracle不能代替下游或完整视频。

实际独立分工：

- mse_evidence_reassessment：只读核对P1/P2、strict RESULTS与旧P64误差分解，发现并限定同源指标反转及ICC/P64跨构造错排证据。
- mse_operator_reassessment：独立推导目标拉回度量、跨位置项、公共输出权重消去；随后读取实际修订文件、复核原始结果、静态JSON/预算，并提出封稿反例。
- mse_plan_amendment：按明确修正要求编辑实验方案与JSON，独立核对12/15预算。
- 主协调者：交叉核对原始数值、编写MSE重评、修订设计限定与当前结果导航，逐项修复复核意见。

独立评议要求并已修正：

1. 明确latent时间token轴21、解码帧轴81、去噪步轴4；误差张量采用latent时间坐标，差分例子不混同这些轴。
2. 幅度/夹角恒等式声明非零前提，并给零向量处理。
3. 12/15是成功完整配置数，不能解释成含任意失败重启的总调用上限；自动额外完整重启额度为0，必要重试另冻结有限范围并记录全部消耗。

独立agent实际recheck后给出 **AMENDMENT_DESIGN_REVIEW_PASS**，并在状态元数据更新后再次确认。覆盖范围为MSE角色、数学说明、历史证据解释、两类全80固定12视频、最多3条范围诊断及成本记账。它不覆盖尚未实现的CAL producer/BF16-Wan port，不证明任何新构造的真实效果。

封稿身份：

| 文件 | SHA256 |
|---|---|
| research/20261118/MSE_SELECTION_REASSESSMENT.md |6467c97029a682749bbe74bbf1fa0254bbd84bb62ee393063a029a157c1fe111|
| research/20261118/OBSERVED_SECANT_EXPERIMENT_PLAN.md |f5ed0a5c38a8d7d561aa846e6125f64e386b15e6c949d8c0bfc6a72b6388c1db|
| research/20261118/observed_secant_experiment_spec.json |ad6e1faefbd75d40cf56bba47870d9cceaaab9a8323a02ecb363384cb05cdf06|
| research/20261118/OBSERVED_SECANT_OPERATOR_DESIGN.md |7af404ad68cde2761e0a7e8daa56de2bc96c428a44c9a5478bbd978922216925|

本次静态预算为12/15 D4、48/60 NFE、1920/2400主链FFN位置、480/486完整64token probe、30,720/31,104在线probe token-FFN、61,440/62,592离线query token-FFN；P64 W1/GELU probe为0/6。独立评议和主协调者均核验闭合。

四份参考核心/测试文件SHA与原表完全一致，本次没有改变算法代码，也没有重跑或新增算法测试。模型、校准、回放、GPU、视频运行仍为0。已完成历史结果保持原始身份；本次仅修正下一轮设计。

