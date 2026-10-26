# E0 · 两来源320处原生双轨：局部误差下降与状态发散并存

**新的实测结果**：来源0/12各160处真实配对FFN及四步状态记录完整保存，四条视频、全部16个历史latent指纹和四视频指纹一致，核心全帧与严格身份门禁通过。**318/320处student前缀上的raw算子误差小于teacher前缀，另两处初始site相等，没有更大者；但两个来源的latent差持续增大。** 因此这两次已测失败不能解释为“student离轨使每处局部FFN误差增大”。这也不表示student更好、误差更小充分保证质量或失败原因已被识别。

[可复算结果](E0_PAIRED_FULL_DEPTH_RESULTS.json)绑定每组RUN、NATIVE_RUN、OBSERVATION、TRACE、BUNDLE、STRICT_BUNDLE及逐帧报告SHA；320个8行全宽BF16向量文件也逐一核对SHA。来源是先前已暴露的0/12、seed42/54；不是320个独立视频样本。旧v2的两条DD与失败记录另见[原始故障/恢复](E0_PAIRED_V2_FAILURE_AND_RECOVERY.md)，不补造其缺失trace。

## 1. 原生观察与信息口径

对每一真实site，以`e_T=f_S(q_T)-f_D(q_T)`、`e_S=f_S(q_S)-f_D(q_S)`记同输入算子差。比较绝对SSE或共同teacher能量分母时，两个来源都是159小/1等/0大。按step跨40层中位数，`||e_S||²/||e_T||²`分别为：

| 来源 | step0 | step1 | step2 | step3 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.4688 | 0.4411 | 0.3643 | 0.3869 |
| 12 | 0.5312 | 0.4977 | 0.4261 | 0.4261 |

这不是两个误差用了不同分母。与此同时，post-Q1的student/teacher能量比的各step跨层中位数约0.638–0.681，方向cosine也显著偏离1。输入幅度与方向改变是竞争解释；目前未独立保存full-M `f_D(q_S)`能量，不能由较小绝对误差宣布其归一功能风险更低或FFN输出塌缩。完整block输出的范数亦不等同于post-Q1范数。

raw FFN及真实writeback四臂的有符号重组均闭合，能量恒等式相对误差约1e-15。当前同step/block两轨gate相同：同一modulation与经断言相等的embed0产生gate。writeback的prefix项仍含residual基底、dense FFN输入响应及有限精度乘加；它不是纯FFN输入漂移。

## 2. 四步状态的反向流差与末步增长

令`δ=z_S-z_D`，`d=v_S(z_S)-v_D(z_D)`，实际记录的理想更新为`δ_next=δ-h*d`，另列真实step_post缺陷。这里的flow是完整40层后、原生head/post生成的实际四维flow。

| step | 来源0 latent SSE（更新后） | 来源12 latent SSE（更新后） | 来源0 cos(δ,d) | 来源12 cos(δ,d) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1731.58 | 474.42 | 初始δ=0 | 初始δ=0 |
| 1 | 11125.33 | 3283.15 | −0.8703 | −0.7901 |
| 2 | 56392.66 | 17278.61 | −0.9050 | −0.8008 |
| 3 | 454530.65 | 137447.95 | −0.9418 | −0.8849 |

后3步`<δ,d><0`，所以`−2h<δ,d>`是正的误差增长项。末步`h=0.625`；来源0的平方项`h²||d||²=198723.13`与signed交叉项`199401.58`接近各半，来源12为`62163.48`与`58000.43`。因此末步增长同时关联较大步长与流差方向，不能只说“最后步长大”。实际step_post与该理想更新的直接能量差很小；这不排除网络内部QDQ/BF16对flow的作用。

当前逐FFN shadow并不产生完整dense模型在student latent上的`v_D(z_S)`，故上述观察尚不能区分“近似模型在偏移状态上的直接失真”与“原dense对该状态偏移的有限响应”。

## 3. 不能跨口径比较的旧分数

旧8.54%来自两个来源的teacher保存panel、按gate加权的FFN误差、native `q2_post`特征与FFN标签上的FP64重构；新全M raw FFN指标未经gate加权，且使用实际部署BF16 pack。来源、平均/中位数、token支持、gate度量及执行形式均不同，不能直接称为8.54%涨到20%–28%。旧0.85%还是block0八case均值，不是本轮首site的同条件分数。

新32行native–continuous相对SSE中位数约0.249%/0.252%，参考函数是**同一部署BF16系数升FP32、去掉Q2和中间BF16**。它不测缺失的原始FP64拟合系数，也不能与full-M raw误差直接比较判定NUM主导与否。[独立口径审查](../method_discussion/discoveries/E0/independent-paired-metric-domain-audit-20261026.md)保留了这些反解释。

## 4. 实际费用与证据状态

v3单张空闲A100串行两组：4完整生成、320site、8320个原生目标GEMM、7680次原生QDQ，另1920个32行FP32诊断GEMM。物理采样峰值两组均76815MiB（75.01GiB），同卡隔离通过。HF新增278488412字节，低于0.5GiB。

来源0/12的整个双轨观察wall为220.94/198.38秒、setup301.59/203.04秒；FFN主调用＋shadow边界为31.44/31.43秒。CPU补评前回执仍记录6012407296字节（约5.60GiB）GPU分配；Python对象删除不作为即时物理释放的证据。这些费用包含两条轨迹、影子与观察，**不是候选部署延迟**。四视频完整324帧/320帧间对已评价，正式VBench及人工全片仍NOT_RUN；候选与旧有损视频逐字节一致，本轮没有画质验收或新方法提速结论。

[独立两来源复核](../method_discussion/discoveries/E0/independent-native-input-contraction-and-flow-error-20261026_v6.md)还给出解析反例：常量teacher flow与有偏但收缩的student更新就能同时出现局部student误差降低及outward流差，因此不能由当前符号现象反推teacher动力学不稳定。该反例未拟合Wan；[图注解](../method_discussion/graph_annotations/E0/E0_GRAPH_PAIRED_NATIVE_STATE_DIVERGENCE_20261026.json)仍为PROPOSED。

## 5. 下一项最有辨识力的检查

在非零`δ`的一步，额外求一次完整dense模型的`v_D(z_S)`，作有限差分解：

`d = [v_S(z_S)-v_D(z_S)] + [v_D(z_S)-v_D(z_D)] = e_flow + R_flow`。

比较`<δ,e_flow>`与`<δ,R_flow>`，即可区分哪部分推动已有状态差；无需假设QDQ可微。当前per-FFN四臂不能代替它。该检查应选当前实测的非零δ状态、复用现有bank并保持强dense参照；不是继续增加固定组数或按局部SSE排名。**本报告不运行该后续模型检查，不授予新范围或训练权限。**

复算命令（CPU JSON/hash读取，无模型调用）：

```sh
.venv/bin/python tools/summarize_e0_paired_results.py \
  --study /home/xiang/.cache/huggingface/ffnds-derived/e0_paired_full_depth_20261026/v3_attempt1 \
  --hf-derived /home/xiang/.cache/huggingface/ffnds-derived \
  --out research/20261026/E0_PAIRED_FULL_DEPTH_RESULTS.json
```
