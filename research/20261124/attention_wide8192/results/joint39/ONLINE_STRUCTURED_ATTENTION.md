# 从联合诊断进入结构化 attention 的实际组合

依据：FFNDS@cc01675282db463fc8983f9113ea3a4ff8c309ef 的固定 P64 可达空间报告；此前封存的 attention_joint39.py、attention_panel_fit.py、exact_port.py 与原实验方案。采用 operator-tensor-analysis 的带轴计算图、有限变化、目标误差与执行成本，结合 active-abstraction 的状态闭合和任务保持判据。

## 1. 整体路线中已经存在的候选

当前被称为“独立读出”的路径，已经是 FFNDS 向 attention 扩展的一份具体构造：high 使用 OUT-ABS，low 使用 FIN-C0，后续 FFN 使用 wide8192。它有当前 P64 attention 教师、逐头读出、原生后续 cross，以及当前输入上的 FFN 锚点/非线性/核恢复。

D176 检验的是进一步合并恢复对象和校准位置的探索。最新 P 可达空间结果已经定位其表达与系数差距；它的进一步改善无需成为检验上述已有结构化 attention 候选的前置条件。

下一主比较应是同一 wide8192 背景上的：

\[
\text{原生 attention}
\quad\text{与}\quad
\text{逐头结构化 attention 恢复}.
\]

两者使用各自的当前状态、anchor、P64 标签和完整四步轨迹。现有末层固定输入实验尚未覆盖后者的全层自身轨迹，这项验证才直接回答扩展能否成为真实可用方法。

## 2. 已有 B11_fixed 的 H 输出可以对应标准在线执行

这里需要厘清一个很具体的实现关系。固定入块 X、已有 cache、当前 Q/K/V、attention 拟合标签与校准资产。设完整 attention 为 A*，读出为 Ahat。P/H互斥。

标准在线路径先覆盖 attention 的 P 行：

\[
\widetilde A_i=
\begin{cases}
A_i^*,&i\in P,\\
\widehat A_i,&i\notin P.
\end{cases}
\]

本模型的同块 tail 是逐视频 query 的映射：

\[
(R_i,T_i,U_i)=\mathcal C_i(X_i,A_i),
\]

其中 cross 使用固定文本 K/V，LN/AdaLN逐query。因此：

\[
\widetilde U_P=U_P^*,\qquad
\widetilde U_H=\widehat U_H.
\]

wide FFN 的当前拟合只读取 P 行输入/anchor/原生FFN标签和固定probe metric。标准在线系数就是：

\[
\theta_F=\operatorname{Fit}
\big(\psi_F(U_P^*;s_P),F(U_P^*)\big).
\]

这与真实分叉代码：

    weights = ffn_fit(obs(n["U"]), ynative[P], prepared, inv_metric)

一致；H预测仍使用 a_op，即候选 attention 后的实际 FFN 输入：

    ffn_predict(a_op, weights, k)

所以，固定同一状态与数值实现时：

\[
B^{\rm online}_H=B^{11,\rm fixed}_H.
\]

原实验在全512面板保存未覆盖的候选P支路，是为了观察P影子误差和联合残差；它没有改变H路径的上述等价关系。单列 B11_refit 才使用未覆盖候选P输入及其原生FFN标签，属于另一种条件反事实。这里的等价是逐query计算图及相同算术定义的H函数等价；面板与全行GPU的batch/kernel舍入应按既有回放容差验证，尚未产生两者逐位一致的在线结果。

这使已有 B11_fixed H 结果可以直接服务在线候选设计。它仍只是同输入局部证据；自身轨迹中的新X、当前K/V、anchor以及所有层的状态将由在线执行决定。

### P 覆盖后的块输出

在线还会把 FFN P 行覆盖为当前原生 F(U_P*)。同块P因此得到原生块输出，而H保持结构化候选输出。下一层self-attention读取这个全行混合状态。这里保留了正常P覆盖语义，无需为全层候选额外设置终端flow校正。

## 3. 在线化的真正工作是把面板计算接成全行当前计算

当前 attention_panel_fit.py 接口读取512行已封存面板；它不是31,668行的生产执行器。要形成真实候选，需要把以下已定义对象放入在线路径：

1. 正确归一化和RoPE后的当前Q/K/V，以及anchor定义的每头query-block key支持。
2. 当前 selected attention 与必要的softmax统计。
3. 原生全键P64 query教师；当前每头32维拟合与全行读出。
4. low FIN-C0 的逐位置anchor状态；high OUT-ABS主要使用当前输出坐标。
5. P覆盖、真实W_O/门控/cross，以及正常wide8192 FFN。
6. high→low时释放旧expert状态并构建当前expert的anchor。

原CAL变换在同一module/expert内对层和head共享，故它在算法上可以应用到40层；当前直接实测的观察层仍为0/13/26/39，未观测层的实际表现会进入完整在线结果。

### 有一个可节省工作、又保持原公式的实现辨析

FIN-C0 的质量监督权重为0。代码的实际求解只需要P64原生O输出，不使用当前完整质量标签的右端项。因此，当前P的完整LSE可留给诊断，而无需被当作C0拟合的必需监督。

但low anchor的 \(\rho_a=1-\exp(L_{S,a}-L_a)\) 需要所有query的完整anchor LSE。这需要从真实attention的在线softmax归约中取得。采集代码对512个面板query另做全键FP32 LSE，不能原样扩大到全部query后仍按原来成本预算解释。应先核验实际attention后端能够返回所需normalizer及其约定，再使用一致数值路径。

### selected 矩统计可以与当前 attention 共用一次归约

原FINITE特征需要选中softmax下的key投影均值与8对value/key混合矩。令

\[
k_j^{(8)}=P_k^{8\top}k_j,\qquad
v_j^{(8)}=P_v^{8\top}v_j,
\]

构造增广value：

\[
\widetilde v_j=
[v_j,\;k_j^{(8)},\;v_j^{(8)}\odot k_j^{(8)}]\in\mathbb R^{144}.
\]

用相同Q/K分数与softmax权重计算一次加权和，即同时得到：

\[
O_i^S,\quad
\mathbb E_{p_i^S}k^{(8)},\quad
\mathbb E_{p_i^S}(v^{(8)}\odot k^{(8)}).
\]

而 \(\mathbb E v^{(8)}=P_v^{8\top}O_i^S\)，所以对应协方差由已有结果相减获得。这是精确代数重排，不是新近似；144维value的kernel支持、布局和速度由实际后端决定。

全键未加权混合矩是每head的8×128统计：

\[
C^{VK}
=\frac1N\sum_j v_j^{(8)}k_j^\top-\bar v^{(8)}\bar k^\top.
\]

先归约一次，再计算 \(C^{VK}q_i/\sqrt{128}\)。因此原有限特征不需要保存全部query-key概率，也不需要保存完整anchor K/V。

### 成本应绑定实际候选

原方案的15.241 GiB是FIN类40层静态attention状态预算。high OUT-ABS不需要同样的逐位置创新缓存；low FIN-C0仍需要相应anchor字段。显存峰值还包括wide FFN缓存、当前Q/K/V、selected输出、特征投影、求解与后端工作区。

原方案四步core比例 \(\tfrac12+\alpha+\tfrac{64}{2N}\) 是四步selected、普通V128的历史交互估计，尚未计入FINITE矩增广。按实际high OUT-ABS/low FIN-C0依赖，可省去high anchor的selected统计：high OUT-ABS只读当前selected输出，经固定变换和BF16舍入后拟合；high anchor仅需原Q/K block均值路由，保留支持和tie规则即可。

若当前selected key密度为各步 \(\alpha_1,\alpha_2,\alpha_3\)，P比例 \(\beta=64/N\)，以每步常规QK128＋PV128为单位，low的增广value归约系数为 \((128+144)/(128+128)=17/16\)。四步core的理想MAC比例为：

\[
r_{\rm C0}=\tfrac12+
\tfrac14\alpha_1+\tfrac{17}{64}(\alpha_2+\alpha_3)+\tfrac12\beta.
\]

三处密度相同时：

\[
r_{\rm C0}=\tfrac12+\tfrac{25}{32}\alpha+\tfrac12\beta.
\]

这一步只剪去未被实际读出读取的high anchor统计，当前输出所需的CAL变换、BF16投影、完整query支持和模型计算保持不变。

若low使用OUT-D，仍需要ANCH底座的u/rho及anchor完整LSE；但无需FINITE混合矩，selected core比例变为 \(\tfrac12+\tfrac34\alpha+\tfrac12\beta\)。如果后端不能从low原生anchor取得所需完整LSE、必须额外重算一次完整QK，四步core比例需再加 \(1/8\)。high OUT-ABS不需要这一额外LSE。

上述算术只描述core MAC：Q/K/V与W_O投影、全键统计、读出、状态和布局均另计。144维value的kernel选择也可能改变实际常数。全行实现按正确数据流完成后，在关闭shadow/capture的同一版本测完整调用周期。

### 模块的锚点时序可以由其自身信息依赖决定

high OUT-ABS的公式只读取当前selected输出、当前P64教师和固定CAL变换；原anchor Q/K均值仅用于建立支持。Q/K均值在step0原生attention core之前已经可算，构建该支持并不需要step0完整attention输出。

因此，从信息依赖看，存在一个明确的后续调度候选：high的step0和step1都使用OUT-ABS；low仍在step2原生attention建立FINITE anchor、step3用FIN-C0。FFN继续在step0/2执行原生FFN并保存它自己当前输入上的anchor。attention与FFN各自的cache定义保持自洽，attention无需继承FFN的两次native-anchor约束。

若采用该新调度，则有一次原生attention anchor、两次普通high selected、两次low增广selected和三次原生P查询。相同密度假设下：

\[
r_{\rm new\ schedule}
=\tfrac14+\tfrac{33}{32}\alpha+\tfrac34\beta.
\]

这是从真实依赖图推导的可执行候选和静态MAC预算，尚无该调度的质量或速度证据。现有CAL变换来自命中步，step0分布尚未验证。它改变step0及后续自身轨迹，应作为清楚登记的新调度，与原冻结HHLL attention候选分开比较；FFN保持建锚/命中时序，锚点值按候选当前输入重建。当前优先完成已有候选在线闭环；其后若质量允许，这个构造可以检验扩展是否真正增加可替换的计算。

## 4. 若后续确实需要联合恢复，一个与 FFNDS 连续的具体构造

以下是可实施的后续构造，不作为现有attention在线候选的前置条件。

### 4.1 保留逐头 attention，重定义 FFN 读出恢复的对象

以下每个 query 的激活采用行向量。先用现有逐头P教师得到 Ahat。保留一个覆盖前P影子支路，并计算全行真实cross和FFN便宜操作：

\[
(\widehat T_i,\widehat U_i)=\mathcal C_i(X_i,\widehat A_i),
\]

\[
b^B_i=\widehat T_i+b_F(\widehat U_i;s_i)G_F^\top.
\]

q16与激活核都在这个实际候选FFN输入上计算：

\[
q_i=q_F(\widehat U_i;s_i),\qquad
K_{ip}=K_F(\widehat U_i,\widehat U_p;s).
\]

与此同时，原生P查询链得到当前块教师：

\[
B_P^*=T_P^*+F(U_P^*)G_F^\top.
\]

用同一条教师链，不需要再计算原生 \(F(\widehat U_P)\)。

构造块出口恢复：

\[
\widehat B_i
=
b^B_i+
q_i\operatorname{diag}(g)V_G+
\tfrac12\sum_{p\in P}K_{ip}C^B_p,
\qquad
V_G=VG_F^\top.
\]

按本段行向量约定，\(V\in\mathbb R^{16\times5120}\)，\(G_F\)为5120维对角gate；\(C^B\in\mathbb R^{64\times5120}\)。拟合对象是完整块教师残差 \(B_P^*-b_P^B\)。保留原FFN的直接kernel与probe metric，可以沿用其16方向gains＋64探针核的结构求解；在此构造中，C的含义由FFN输出剩余变化变为完整块剩余变化。

这保留了attention每头teacher、真实cross、当前2560选中激活及模型方向。它同时把attention与FFN未恢复的共同作用交给原FFN读出所在的结构，而不是在旧终端读出上再叠加一份任意字典。

### 4.2 与 D176 的具体区别

- FFN坐标用 \(\widehat U\)，即已经经过逐头attention恢复的当前输入；D176使用SP/ANCH的cheap_U。
- P/H的核坐标针对同一种候选路径，teacher则是同输入X下完整原生块；目标定义吸收两者差别。
- 使用原直接激活核，而非把核列归一化后再统一构造一次核列Gram。
- 输出仍为真实块的5120维残差，并保留16个模型方向；终端flow用于评价所恢复的作用。
- 该读出可以替换原FFN在线读出，避免把原独立成本和一份新终端拟合永久叠加。

C直接写在gate后的块空间，避免除以可能很小或为零的gate。若选择把C限制到gate可写空间，则需要明确额外约束；不能默认gate可逆。

### 4.3 这份构造的精确边界

这里的公式在实数域定义了新方法；BF16 cast、gate融合和累加顺序需按实际实现登记。它没有自动继承原独立路径的全部函数族：核的P中心从原生U_P换为候选影子U_P，拟合对象也发生变化。因此真实配对将它作为一个新候选，与正常P覆盖的独立路径比较。

所需新增操作主要是64行覆盖前attention后的shadow cross/LN/FFN便宜坐标，以及块出口teacher残差；无需第二次P64原生FFN调用。全行kernel correction的形状仍是N×80乘80×5120，和原FFN读出的主形状相同。实际常数、融合、额外P影子及anchor开销需计入。

它适合在已有独立attention自身轨迹建立之后、且仍需更好联合恢复时开展。当前新增可达空间结果并不要求立即实施这条分支。

## 5. 最小而直接的下一验证

主路线先把已有high OUT-ABS/low FIN-C0接入全部40层，在wide8192上完成少量匹配自身轨迹。以0392和1013的原prompt/seed为起点，现有原生attention＋wide背景轨迹可复用；新增结构化attention候选轨迹保留全帧视频、各步状态和缓存身份。

这一比较直接观察：逐层当前教师与自己anchor能否在闭环中保持空间关系、动作与纹理；完整调用周期是否给出成本收益。原生逐头恢复、原FFN以及P覆盖语义已经明确，继续终端字典调参并非它的必要先决条件。

若之后开启4节的新块目标构造，最小辨析是同一当前X/cache、同P64教师和实际候选U的独立FFN目标对照块出口目标，保留点/时空边与真实flow评价。先改变一个有明确语义的恢复目标，不同时改变P、路由、字典宽度和正则。此前两源均为已使用DEV，后续确认需另按冻结候选选择数据。

## 6. 当前证据带来的整体认识

固定P可达下限已经低于原独立误差，说明当前特征与探针位置诱导的恢复空间含有可改善目标的分量；它由H标签识别，在线系数还需通过实际P教师与结构先验获得。原独立结构在low上显著改变可恢复残差，说明结构化当前计算本身是信息压缩的重要组成部分。

方法扩展因而有两层互相支撑的对象：

- 构造层：沿attention和FFN各自的计算关系，用当前P教师建立结构化恢复。
- 验证层：在真实串联、全层与采样闭环中，判断共同作用及成本。

目前第一层已经存在具体候选，第二层的全层自身轨迹应成为下一主证据。进一步的联合求解构造可以据该结果推进，并按清楚的计算对象与成本比较。

## 固定版本依据

- [实际attention读出与FIN-C0求解](https://github.com/Qesire/FFNDS/blob/cc01675282db463fc8983f9113ea3a4ff8c309ef/research/20261124/attention_wide8192/results/attention_panel_fit.py)。
- [四臂分叉、B11_fixed与FFN接口](https://github.com/Qesire/FFNDS/blob/cc01675282db463fc8983f9113ea3a4ff8c309ef/research/20261124/attention_wide8192/results/joint39/attention_joint39.py)。
- [D176固定P空间实测](https://github.com/Qesire/FFNDS/blob/cc01675282db463fc8983f9113ea3a4ff8c309ef/research/20261124/attention_wide8192/results/joint39/D176_OBSERVABILITY_REPORT.md)。
- [整体推进与四步成本](GLOBAL_METHOD_SYNTHESIS.md)、[成本程序](compute_extension_cost.py)、[代数核验](verify_online_statistics.py)。
