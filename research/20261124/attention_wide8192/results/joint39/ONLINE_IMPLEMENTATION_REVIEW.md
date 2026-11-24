# 三臂自身轨迹后的计算图与执行判断

依据：FFNDS@832e9eae827364d49900232f439b9d74daabb624，学校A100质量作业194188与计时作业194228。用户在2026-11-24给出的原始人审结论为：**“三个方法的视频对比，肉眼几乎没有差别。”** 本文不将此扩写成尚未提供的逐片细节评分。

使用operator-tensor-analysis的收缩、状态与完整周期工具，以及active-abstraction的作用观测与状态闭合判据。核查实际461行在线执行器、组件计时器、面板检查、真实后端收据、逐步轨迹和报告。本文新增模型执行数为0；只做源码推理与明确标记的形状/MAC算术。

## 1. 这次已经运行了真实组合方法

三臂为：

| 臂 | attention | FFN |
|---|---|---|
| A | 原生self-attention | 原BF16 wide8192 |
| B | high OUT-ABS，low FIN-C0 | 原BF16 wide8192 |
| C | high OUT-ABS，low OUT-D | 原BF16 wide8192 |

在线执行器钩住真实`infer_self_attn`内部的SDPA入口，Q/K/V均来自当前模型，断言真实形状为31,668×40×128且为BF16。正式候选路径`audit=None`，四步每步40层全部执行。只有同输入审计模式限定0/13/26/39层，并在候选计算后返回原生输出，以保持归档A轨迹。

每条正式候选执行80次原生attention建锚、80次近似命中，以及80×64=5,120行全键原生教师查询。真实QKV投影、归一化、RoPE、W_O、gate、cross与wideFFN保持原计算图位置。P64 attention在拼接前覆盖为当前原生教师，后续FFN按这条候选路径的当前输入建锚和恢复。

因此，这次结果已经推进到全层四步组合。早期`ONLINE_PANEL_194153.json`中的非面板query填充只是后端/面板检查；正式质量路径直接使用全长真实Q/K/V，没有使用这些填充数据。

## 2. 锚点与自己的轨迹确实闭合

`slots[layer]`在step0/2建立，在step1/3通过`slots.pop(layer)`消费。step0只存高噪声路由，不存不使用的完整输出状态。step2保存低噪声路由、rho、u0和投影后的anchor特征；B另存vstar。各专家40层的状态在各自命中过程中释放，结束时断言slots为空。

路由由当前anchor Q/K block均值建立，使用稳定argsort选31个128-key块。最后一个块只有52个有效keys，所以各query/head的selected键数为3892或3968。缓存与生成状态来自该候选自己的路径。

质量记录显示候选与A的初始噪声、step0 flow和latent精确匹配。B/C的step0/1/2 flow和latent以及80个路由指纹精确一致，差异从最后一次low命中开始。内部anchor数值因读出种类不同并未全部逐元素记录；字段收据给出shape/dtype，不应称B/C所有anchor值相同。

正式计时关闭面板审计、轨迹tensor保存和anchor指纹记录。组件事件仅放在暖机。数值复算agent另逐项核对：全部18条正式计时的初值、四步flow、latent及video记录，均等于对应质量/归档运行。因此这次性能与用户审看的方法身份对应。

## 3. C延续了FFNDS的哪项具体机制

对一query/head，selected输出为O_s^S，锚点完整输出为A_a，锚点未选中质量为rho_a。代码存储

\[
u_a=A_a-(1-\rho_a)O_a^S,
\]

并在命中时构造

\[
b_s=(1-\rho_a)O_s^S+u_a.
\]

故在实数域可重写为

\[
\boxed{b_s=A_a+(1-\rho_a)(O_s^S-O_a^S).}
\]

OUT-D的设计由selected输出在固定投影后的两端BF16差与常数项构成，最终

\[
\widehat A_s
=A_a+(1-\rho_a)(O_s^S-O_a^S)+X_s\Theta_h.
\]

Theta_h由该头当前P64全键原生输出拟合。它保留了**逐位置锚点、当前真实有限输出变化、稀疏原生教师与跨位置残差恢复**。这是FFNDS结构化恢复机制在attention上的具体实例，当前成片与速度证据可以支持这一层扩展。

FIN-C0进一步用有限矩坐标与当前value方向耦合约束恢复；OUT-D则用输出坐标有限差。用户当前视觉结论给B/C同等可见质量，后续工程基线可优先采用C。有限质量/矩耦合的独立视觉收益属于后续机制问题，已有C的结果不需要等待该问题解决。

数值执行中u_a和投影有明确BF16舍入，上式是其计算图含义；实际值由代码保存和写回规则决定。同输入面板中P教师、selected输出和H恢复最大相对差分别为0.0974%、0.1364%、0.4320%。

## 4. 当前成本已定位到有意义的操作层

暖机profile取第4–39层单层中位数，组件存在包含关系：

| 低噪声命中 | B：FIN-C0 | C：OUT-D |
|---|---:|---:|
| P64全键教师 | 0.407 ms | 0.405 ms |
| Schur拟合 | 1.122 ms | 0.603 ms |
| selected归约与统计链 | 66.987 ms | 19.455 ms |
| 完整attention命中恢复 | 81.640 ms | 32.250 ms |

因此，P64的原生信息获取及小系统求解在当前实现中已具有低成本。进一步减少probe数或微调岭求解，不对应当前主要执行差距。

实际代码已经实现前一轮提出的两项构造：

1. 低噪声原生Flash SDPA同次返回O与自然对数LSE；不另做全行完整QK获取LSE。
2. B把selected所需8维key均值、8维VK混合矩并入144维value，额外16维以BF16进入FlexAttention。

B的`selected()`还物化FP32的全行Q、K、V、中心化K、投影value，计算全键未加权混合矩、query投影、中心化LME及64维原特征，再映射成31维坐标。当前66.99ms覆盖了144维稀疏kernel与这些统计链的全部作用。它没有分离二者，因而不能仅凭这一个数把差距归因于144维padding或某次全局矩乘。

C同路径只需普通V128 selected attention及输出投影。它同时保留ANCH底座，因此依然需要低噪声原生O/LSE、rho、u0和anchor投影。

完整配对结果由正式周期计时给出：C/A配对墙钟比中位数0.879941，C四步GPU配对比0.845794。B的四步GPU也在6/6对中略快于A，但墙钟配对波动抵消了这部分收益；B不是所有计时口径都慢于A。C同时具备当前人审结果与清楚的完整速度收益。

## 5. 稀少LSE修复与显存的实际含义

低噪声anchor先计算rho=−expm1(LSE_selected−LSE_full)。对于子集单调性偏差超过5e−5或非有限的query/head，改用同一全键logit最大值归一化，直接累加未选中键的正权重，除以全键和。这样保留了有限精度下的质量定义。

两来源修复比例分别约0.0173%和0.0325%，B/C数量几乎相同。该路径确实有CPU同步、逐head与小行块调用的成本，但现有计时没有将它隔离，不能以此解释B/C的主要差额。

从真实cache shape计算，40层低噪声显式状态为：

- u0 BF16：12.0804 GiB；
- rho FP32：0.1888 GiB；
- anchor31 BF16：2.9257 GiB；
- 三项合计15.1949 GiB，尚未计BlockMask及工作区。

单层[head,query,128]为0.3020 GiB BF16或0.6040 GiB FP32。当前C峰值65.41 GiB较A增加约17.85 GiB，因而大部分新增容量有清楚的状态来源。减少完整attention锚点数量与减少low状态容量是不同的操作；下一节改变high时序，并不会自动消除low的u0/特征缓存。

[静态形状算术](c_extension_arithmetic.json)不作为真实显存计时。

## 6. 下一项整体扩展应以C为起点调整模块时序

high OUT-ABS只读当前selected输出、CAL变换与当前P64教师。它的route可由step0当前Q/K均值建立，独立于完整attention输出。因此已运行的C为后续构造提供了更简洁的起点：step0/1都用high OUT-ABS，low仍原生建锚于step2、OUT-D命中于step3；FFN保持原0/2建锚、1/3命中，但按新轨迹输入重建状态。

这将attention的原生完整调用从80次降到40次，恢复命中从80次增到120次。C采用普通V128，两次high selected与两次low selected，加三次P教师，使理想四步core MAC比例为

\[
r_{C,3hit}=\tfrac14+\alpha+\tfrac34\beta,
\quad\beta=64/31668.
\]

当alpha=1/8时为0.376516；当前C对应0.594760。这些是core静态MAC，未包含其他操作和实际利用率，不是新墙钟结果。上一轮C0的新时序式1/4+33alpha/32+3beta/4含144维low矩，不能直接套到OUT-D。

该构造只改变high step0，不改变FFN建锚时序、P预算、selected支持规则或读出参数。其质量由完整新轨迹决定，因为现有CAL来自命中步，而step0分布尚未实测。它直接扩大可替换计算比例，比重新扩展D176终端字典更贴近这次已经得到的质量—成本关系。

## 7. 具体执行来源

- [在线执行器](https://github.com/Qesire/FFNDS/blob/832e9eae827364d49900232f439b9d74daabb624/research/20261124/attention_wide8192/results/joint39/attention_online_structured.py)。
- [组件计时器](https://github.com/Qesire/FFNDS/blob/832e9eae827364d49900232f439b9d74daabb624/research/20261124/attention_wide8192/results/joint39/profile_attention_online.py)。
- [正常周期报告](https://github.com/Qesire/FFNDS/blob/832e9eae827364d49900232f439b9d74daabb624/research/20261124/attention_wide8192/results/joint39/ONLINE_TIMING_REPORT.md)。
- [质量与状态报告](https://github.com/Qesire/FFNDS/blob/832e9eae827364d49900232f439b9d74daabb624/research/20261124/attention_wide8192/results/joint39/ONLINE_QUALITY_REPORT.md)。
