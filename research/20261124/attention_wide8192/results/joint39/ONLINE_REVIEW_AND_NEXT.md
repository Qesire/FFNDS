# 三臂成片视觉近同之后：以 C 推进 attention 与 wide8192 的整体扩展

日期：2026-11-24。最新执行证据固定于 [FFNDS@832e9eae827364d49900232f439b9d74daabb624][head]，学校 A100 质量作业194188、完整周期计时作业194228。采用 evidence-analysis、operator-tensor-analysis 和 active-abstraction；本轮执行真实收据复算与源码分析，新增模型/GPU执行数为0。

> 第 6 节记录的是上一轮 C0 提案，现已完成同源配对。0392 在 C0 成片中出现运动质量失败，1013 主观变化较小，原 C 继续作为工作候选。[C0 配对与运动复核](C_SCHEDULE_REPORT.md#逐来源人审与帧间复核)；[最新联合状态分析与推进顺序](C_SCHEDULE_REVIEW_AND_NEXT.md)取代本页后续提案。下文保留当时的证据与提案状态。

## 1. 当前结论

用户本轮原话：

> 结果已同步，三个方法的视频对比，肉眼几乎没有差别

这条反馈作为本轮三臂整体视觉判断原样保存，见[人审记录](HUMAN_REVIEW_20261124.json)。对应实验包含0392、1013，用户未另行提供逐来源或动作/纹理分项评分。

**C应作为attention扩展的下一工作候选。** 本轮用户观察三臂成片近同，C相对同批wide8192＋原生attention基准，完整周期配对时间下降12.01%，四步GPU配对时间下降15.42%。这给最初问题提供了完整系统证据：逐位置状态、当前有限变化、稀疏原生教师和结构化恢复，已经在self-attention与后续wide8192 FFN的实际组合中取得质量—时间收益。

这次比较的A已经使用wide8192，所以本轮结果是attention扩展带来的额外收益。C的额外峰值约17.85GiB，与收益一同记录。下一步优先沿C改变high step0的attention时序，直接扩大可替换计算；B的精细有限矩与方向耦合保留为机制分支。

## 2. 完整周期的真实结果

三臂共享原BF16 wide8192 FFN；B/C的high均为OUT-ABS，low分别为FIN-C0和OUT-D。

| 臂 | attention配方 | 整片墙钟中位数 | 配对墙钟时间比 | 配对四步GPU时间比 | 峰值分配 |
|---|---|---:|---:|---:|---:|
| A | 原生 | 38.768s | 1.0000 | 1.0000 | 47.57GiB |
| B | high OUT-ABS / low FIN-C0 | 40.977s | 1.0307 | 0.9675 | 66.37GiB |
| C | high OUT-ABS / low OUT-D | 34.809s | 0.8799 | 0.8458 | 65.41GiB |

比值按同来源、同重复编号先配对，再取六对中位数，因此不是两列单臂中位数相除。两来源各三次重复属于重复测量。

C的整片配对节省中位数为4.658s，GPU配对节省中位数为4.721s；GPU在6/6对中更快，整片墙钟在5/6对中更快。来源内整片时间比分别为0392的0.90824、1013的0.85915。

B也保留了约0.992s的GPU配对节省，GPU在6/6对中更快；完整墙钟的配对比中位数为1.0307，六对平均绝对差约−0.027s。合适结论是B获得了部分GPU计算收益，而当前整片时间没有形成C那样的收益。[正式报告][timing]、[独立数值复算](ONLINE_NUMERIC_REVIEW.md)

### 成本收益来自哪一段

取六对GPU事件的平均有符号差，四列可以相加：

| C−A | step0 | high命中 | low建锚 | low命中 | 四步合计 |
|---|---:|---:|---:|---:|---:|
| 秒 | −0.017 | −2.867 | +1.114 | −2.890 | −4.659 |

C在两个命中步分别节省约2.9s，承担约1.1s的low建锚成本后，仍获得明确的完整四步收益。这个结果已经把锚点、恢复和调用频率放进同一个周期。

## 3. C保留了原方法的哪项具体机制

对一个query/head，设锚点完整输出为A_a，选中支持的归一化输出为O_a^S、O_s^S，锚点未选中质量为ρ_a。代码保存

\[
u_a=A_a-(1-\rho_a)O_a^S,
\]

命中底座为

\[
b_s=(1-\rho_a)O_s^S+u_a.
\]

在实数计算图中重写：

\[
\boxed{
\widehat A_s
=A_a+(1-\rho_a)(O_s^S-O_a^S)+X_s\Theta_h.
}
\]

X_s由selected输出投影的两端BF16差与常数项构成，Θ_h用当前逐头P64全键原生响应拟合；P行最后覆盖为原生教师。u_a与投影的实际BF16舍入按源码执行。

这个形式包含明确的作用关系：

- A_a保存位置自身的起点；
- 当前O_s^S−O_a^S提供真实已计算的有限变化；
- P教师观测剩余作用；
- 逐头恢复把观测约束传到其他位置；
- 后续W_O、cross、归一化和wide FFN继续在候选自己的状态上执行。

FFN中的“锚点＋当前选中非线性变化＋模型方向/核恢复”与这里的attention形式，保留了相同层次的关系，并按算子结构采用了不同坐标。C的完整结果支持这一层机制迁移。

B进一步引入有限矩、质量变化与当前value方向的乘性关系。B/C是两套low恢复方案的比较，包含特征、耦合求解和实际统计路径的变化。当前视觉反馈与周期成本支持优先选择C；精细耦合的研究价值可在更紧计算预算需要它时继续检验。

## 4. 成片近同与内部轨迹差异共同带来的认识

C/A最终latent相对L2在0392、1013上分别为23.302%、7.701%，RGB PSNR分别为19.03、26.68dB。用户对本轮三臂视频的整体判断仍为“肉眼几乎没有差别”。这些数值说明候选确实改变了轨迹，而本轮成片观察给出了这种变化的任务作用。[质量报告][quality]

由此，下一阶段可以采用更直接的目标：

\[
\min_m T_{\rm cycle}(m;c)
\quad\text{s.t.}\quad
Y_m(c)\in\mathcal A_c,\qquad
M_{\rm peak}(m)\le M_{\rm budget}.
\]

c为固定输入图、prompt与初始噪声，A_c为任务允许的成片集合。本轮人审提供了三个已生成实例的判断。局部张量误差帮助解释变化，实际选择由成片表现和完整成本共同决定。

若将一组任务观测记为可微函数φ，最终解码为\(\mathcal D\)，则实数域有限差异可写成

\[
\phi(\mathcal D(z+e))-\phi(\mathcal D(z))
=\left[\int_0^1\mathrm D(\phi\circ\mathcal D)(z+te)\,dt\right]e.
\]

该式说明误差方向经过实际读出后的作用比单独范数更贴近目标。现有人审给出结果判断，并未测量这个导数或指定是哪一模块产生补偿。因此本轮可保留的认识是：这些真实轨迹的内部变化，具有比逐坐标复现要求更大的成片容许范围。

### B/C的共同前缀让比较含义更清楚

两臂初始噪声、step0—2的flow/latent和80个支持路由指纹完全相同，最后一个low命中才开始分开。最后flow相对差为1.550%/1.455%，latent相对差为1.580%/1.372%。内部缓存只记录部分字段身份，不能据此称所有缓存数值完全相同。

令x_3表示最后一步入口，末步步长为−5/8，则实数域中

\[
x_4^B-x_4^C=-\tfrac58(v_3^B-v_3^C),
\]

实际FP32计算另有舍入。这里v_3已经包含全部40层attention、cross、FFN和head的共同响应。因此B/C是同一末步入口下的完整low方案比较；其较小输出差异与用户当前近同的视觉判断，一起支持以成本较低的C继续推进。

## 5. B的额外成本已经定位，研究顺序可以据此调整

暖机组件事件取第4–39层单层中位数，各列存在包含关系：

| low命中 | B | C |
|---|---:|---:|
| selected归约与统计 | 66.987ms | 19.455ms |
| P64全键教师 | 0.407ms | 0.405ms |
| Schur拟合 | 1.122ms | 0.603ms |
| 命中总计 | 81.640ms | 32.250ms |

selected＋统计差47.532ms，占总命中差49.390ms的96.24%。原生P教师在B的low命中总时间中约占0.50%，已具有低成本。B的主要代价来自获得和组织廉价特征的实际计算链。

前一轮提出的原生O/LSE共享归约、selected V144增广统计已经实现。B路径还包含全长FP32 Q/K/V、中心化矩、全键混合矩、query投影和特征映射；当前profile把这些工作与稀疏kernel合在一起，尚未分出各项成本。[执行器核查](ONLINE_IMPLEMENTATION_REVIEW.md)

当前值得保留的研究顺序是：以C扩展可替换计算；当更紧的预算需要额外恢复能力时，再考察B能否用其结构维持质量。这样，精细统计的价值由它带来的质量或预算空间决定。

### 容量代价也有明确来源

C的40层low显式缓存包含u0 BF16约12.0804GiB、ρ FP32约0.1888GiB、31维anchor BF16约2.9257GiB，合计15.1949GiB，另有路由和工作区。真实峰值65.41GiB相对A增加17.85GiB。改变high的锚点时序保留这份low状态需求。

## 6. 下一主比较：以C为起点，只改变high step0

high OUT-ABS只需要当前selected输出、固定CAL变换、当前P教师及由Q/K均值建立的支持。它不读取完整high attention anchor输出。因此下一明确构造为：

| 步 | attention | FFN |
|---|---|---|
| 0 high | 当前OUT-ABS＋P64，保存本层路由供step1使用 | 原生FFN，在候选当前输入上建锚 |
| 1 high | 当前OUT-ABS＋P64 | 原wide8192命中 |
| 2 low | 原生attention，建立OUT-D锚点 | 原生FFN，在候选当前输入上建锚 |
| 3 low | OUT-D＋P64 | 原wide8192命中 |

保留当前C的P64、CAL、selected密度与路由规则、low OUT-D及FFN时序，只改变high step0 attention。后续输入和锚点值按新轨迹重建。先在现有两源各新增一条候选，复用匹配的C作为参考；完整计时与视觉评价沿用本轮边界。

现有CAL来自命中步，step0的恢复表现由这个配对检验。该构造尚未执行。当前C的后续来源确认与这次调度变化分别记录，最终选定配方再开展独立来源确认。

### C对应的静态预算

两种输出坐标读出均使用V128。令α为selected密度、β=64/31668，以四次完整attention core的QK＋PV为分母：

\[
r_C=\tfrac12+\tfrac34\alpha+\tfrac12\beta,
\qquad
\boxed{r_{C,new}=\tfrac14+\alpha+\tfrac34\beta.}
\]

α=1/8时分别为0.5947605和0.3765157；新构造的理想core MAC约为当前C的63.31%。这是一次完整attention变成一次selected＋P恢复的预算差：

\[
r_C-r_{C,new}=\frac{1-\alpha-\beta}{4}=0.2182448.
\]

具体调用由80次原生attention＋80次恢复变成40次原生＋120次恢复，P全键query行总数由5120变成7680。全周期还包括投影、路由、恢复、FFN、解码和状态，以上比例不直接换算成新墙钟。[可复算算术](c_extension_arithmetic.json)

### 给后续实现的两个必要语义变化

1. step0先建立route，再用该route执行OUT-ABS；route保存至step1消费。计数区分“路由建立”和“完整原生attention建锚”，不要仍要求80次native anchor。
2. 当前runner要求候选step0 flow/latent与A相同。新调度显式改变step0，因此该新分支保留初始噪声一致，改为保存step0实际结果；原A/B/C分支原有等价检查继续使用。FFN建锚输入来自新分支自身计算。

这一单因素变化直接检验模块各自的状态依赖能否带来更多整体收益。D176和B的后续机制研究继续作为已有资产，不作为推进这项比较的前置条件。

## 7. 本轮核验与资产

独立复算已验证质量与计时使用相同实现SHA256；18次正常计时的initial、全部flow/latent和video记录均匹配对应质量运行或A归档。正式计时关闭了轨迹采集。用户这次视觉反馈与实际计时方法有明确的指纹联系。

- [人审原话及范围](HUMAN_REVIEW_20261124.json)。
- [完整数值辨析](ONLINE_NUMERIC_REVIEW.md)、[标准库复算脚本](recompute_online_evidence.py)、[逐次和配对结果](online_evidence_recomputed.json)。
- [计算图与实现辨析](ONLINE_IMPLEMENTATION_REVIEW.md)。
- [C新时序和缓存算术脚本](derive_c_extension.py)、[结果](c_extension_arithmetic.json)。

在本目录可复算：

```sh
python recompute_online_evidence.py
python derive_c_extension.py
```

[head]: https://github.com/Qesire/FFNDS/commit/832e9eae827364d49900232f439b9d74daabb624
[quality]: https://github.com/Qesire/FFNDS/blob/832e9eae827364d49900232f439b9d74daabb624/research/20261124/attention_wide8192/results/joint39/ONLINE_QUALITY_REPORT.md
[timing]: https://github.com/Qesire/FFNDS/blob/832e9eae827364d49900232f439b9d74daabb624/research/20261124/attention_wide8192/results/joint39/ONLINE_TIMING_REPORT.md
