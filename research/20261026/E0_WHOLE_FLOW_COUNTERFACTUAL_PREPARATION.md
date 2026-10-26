# 整步 dense 响应与候选注入：新的判别入口已备好

**新的研究对象是实际离散更新的两个有限作用。** 对同一真实 student latent，补求完整 dense flow 后，可以把下一状态差拆成 dense 对当前状态偏移的响应，以及当前 candidate 的净注入。这项观察将决定更值得研究函数重构、状态响应还是低成本输出校准。另一条独立线提出共享 W1 投影的向量值非线性曲线，作为不同的 FFN 计算单位。

## DISCOVERY：用整步映射区分同一个已观测流差的两种解释

已有 `δ=zS−zD`、`d=vS(zS)−vD(zD)`。新增完整网络 `vD(zS)` 后定义

`R=vD(zS)−vD(zD)`，`e=vS(zS)−vD(zS)`，`P=δ−hR`。

理想实数更新满足 `δ_next=P−he`，能量为

`||P||² + h²||e||² − 2h<P,e>`。

因此观察 dense 整步增益 `||P||²/||δ||²` 与候选净效应 `h²||e||²−2h<P,e>`，比只看 `δ·d` 的符号更有信息。已有相同 `δ,d` 可以对应收缩或扩张的 dense 整步映射，见[独立源码/反例审查](../method_discussion/discoveries/E0/independent-whole-flow-native-source-review-20261026.md)。各项相互有 signed cross，能量占比不充当因果百分比。

**原生 BF16 版本同时保存。** 利用新增 dense flow，在 scheduler clone 上按原 `step_post` 的 FP32乘加顺序和 BF16 cast 求一次 shadow update。记 `Pnat=TD(zS)−TD(zD)`、`Nnat=TS(zS)−TD(zS)`，则真实 `δ_next=Pnat+Nnat`。这只多做向量更新，不增加模型 NFE。理想增益和原生增益分别列出，所问对象是固定状态对上的有限响应；初始致错层及长期视频因果作用仍须进一步区分。

## EVIDENCE：保存数据已经进一步缩小解释空间

[可复算 writeback 结果](E0_NATIVE_WRITEBACK_GEOMETRY.json)绑定原 v3 RUN/TRACE/OBSERVATION/strict bundle：来源0与12各159处 student-input 原生 writeback 算子误差较小、各1处初始相等、0处更大。两来源 student/teacher writeback 算子SSE比中位数分别0.4311/0.4575。这把318/320的观察从 raw FFN 延伸到实际 gate／residual／有限精度写回；该同输入差仍小，并不代表 student 视频更好。prefix项含 residual 基底及 dense 有限输入响应，不能据其能量宣布 SHIFT 占比。

复算（只读已保存 JSON，不调用模型）：

```sh
.venv/bin/python tools/analyze_e0_paired_writeback.py \
  --paired-result research/20261026/E0_PAIRED_FULL_DEPTH_RESULTS.json \
  --hf-derived /home/xiang/.cache/huggingface/ffnds-derived \
  --out research/20261026/E0_NATIVE_WRITEBACK_GEOMETRY.json
```

[独立六步骤标量推导](../method_discussion/discoveries/E0/independent-whole-flow-control-and-value-geometry-20261026_v7.md)还得到沿真实偏差方向的理想反馈界：若 `u=γδ`，最小下一状态SSE为 `h²(||d||²−<δ,d>²/||δ||²)`。来源12的step1下界555.174，高于输入差474.421。故这一个状态对上，纯沿δ控制即使有teacher oracle也无法使理想整步收缩。δ在线不免费可得；改变flow后的BF16缺陷与四步修正视频未测，结论只约束该控制族和这一步。它积极提出下一问题：可在线取得的少数flow/state方向覆盖多少必要修复作用？

## 文献如何改变本轮选择

2026-10-26实际检索 flow/sampling-aware compression、correlated quantization error、trajectory calibration、neuron merging/ridge函数；沿原文与作者代码迭代。亲读的关键来源：

- [PTQD v2 §4.1–4.2](https://arxiv.org/html/2305.10657v2)有输出相关误差、bias及方差校准；确定性采样没有自动可吸收的随机方差预算。它提供已知低成本竞争机制，不能由当前负 `δ·d` 直接推出应缩放flow的方向。
- [Q-Drift v1 Appendix A.1](https://arxiv.org/html/2603.18095v1)的flow扩展明确为heuristic；边缘分布保持与固定seed轨迹重合是不同目标。独立审阅另亲读作者scheduler/collector，代码事实及commit见其[原记录](../method_discussion/discoveries/E0/independent-whole-flow-control-and-value-geometry-20261026_v7.md)。本协调者未独立复核其完整作者实现。
- [Sampling-Aware Quantization v2 §3.3–4.3](https://arxiv.org/html/2505.02242v2)对齐混合阶采样方向，并有重构／QLoRA阶段；[作者仓库](https://github.com/TaylorJocelyn/Sampling-aware-Quantization)本次可读页面只有README与待发布说明，未将其当已核验runtime。其跨阶目标不是本次固定四步模型的免费替代。

这些来源使下一动作从“直接试一个flow缩放”转为先测整步有限分解；通用gain/bias、轨迹校准与ridge/feature bundling属于已知原语，本轮没有算法新颖性认定。

独立机制线的[向量值响应曲线](../method_discussion/discoveries/E0/independent-signed-neuron-curve-coordinates-20261026.md)给出条件表示 `Hc(t)=Σ_i(v_i/s2_i)GELU(α_i t+b_i)`。若方向共线且低阶曲线可逼近，可从计算图中缩小W1和W2；近共线神经元的相反W2可能保留重要差分，Q2共享scale也可能使响应多变量化。这两项是便宜的否证条件，Wan簇复杂度与速度尚未测。该独立方向无需等待Affine54归因结束。

## EXECUTION：可审查的固定范围，尚未执行

[v1固定协议](../../config/e0_whole_flow_counterfactual_20261026_v1.json)绑定新入口、全部原生源码、原80个pack与v3历史结果。拟用单张空闲A100串行两来源0/12、seed42/54：四条完整四步**latent路径**，另8次完整40层dense flow（每来源4步，step0为中性控制），保留320处原配对观察和8次shadow step_post。**0新视频／VAE解码，0训练／拟合／分解／官方评分，HF新增≤0.5GiB；失败即停，无自动重试。** 本页和协议均不授予新模型执行权；当前权限以ACTIVE为准。

入口是 `tools/run_e0_whole_flow_counterfactual.py`，默认只读预检；新controller复用原配对数学文件。实际预检已通过全部源码/80个pack/历史视频严格bundle，输出 `model_scope_authorized=false`、`GPU_initialized=false`。9项CPU合成测试覆盖完整调用顺序、原四步latent保持、流/原生更新恒等式、不可微无关的有限反例、失败释放hook及raw保存、预算账本与授权拒绝。**这些属于软件/来源验证，新增原生中性与显存测试尚未发生。** 原v3的75.01GiB仅为先前峰值，新缓冲费用仍需实测。

审查提前发现并修复了 BaseRunner结束hook读取未初始化`gen_video`、最终monitor错误漏检以及数值归约失败前原张量保存的问题。源审查记录绑定最终SHA。原v1/v2/v3协议、原RUN/TRACE/视频及四个原配对数学组件字节不变。

预计真实工作账：24个完整模型flow（16用于四步双轨重放，8为反事实），1600个主／shadow／反事实FFN，11520个原生目标GEMM、10880次QDQ，另1920个32行连续FP32诊断GEMM。八个完整状态包原张量356708352字节，余量供JSON/源码快照；这些是观察费用，非部署时延。原16个逐步latent指纹全部匹配才接纳该范围；已有四视频strict bundle仅作为历史绑定，正式质量仍未验收。

图锚点 `F08/F09→F12/F13→P23/P24→G21/D09`：writeback及原轨迹为posthoc NATIVE-OBS，新的完整flow边保持conditional/未执行。共享投影曲线沿 `F04/F06/F08/F09/P14` 为条件ALG＋OPEN执行假说。
