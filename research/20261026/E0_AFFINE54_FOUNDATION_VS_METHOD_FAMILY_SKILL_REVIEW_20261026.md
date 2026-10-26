# E0 · Affine54 具体原型失败 ≠ Affine/非线性余项方法族被否定

> 2026-10-26｜AutoResearcher v2.4 方法审查与科研路由。审查前仓库基线 `30806238b156f1db1223f280a13a5f491d1e6e9e`。**本文件为源码/数学/已冻结实验的二次判断，不是新实验、正式独立同行认可或新算子推荐。** 当前仍为 `E0_OPEN_DISCOVERY`，不追加GPU/模型/新生成授权，实时G6仍暂停。

## 0. 经过的 Skill 路径与审查纠正

已读取本仓库绑定的 [AutoResearcher 源manifest](../agent_protocol/ACTIVE_SOURCE_MANIFEST.json)，以及 `mechanism-research`、`problem-reformulation`、`research-expansion`、`web-research-discovery` 四个实际 skill；按 **NOTICE → MAP_FLOW → DERIVE → EXPAND → DISCRIMINATE → RECORD**、**OPEN → TRANSFORM → FIND FREEDOM → PREDICT → GROUND** 和 **evidence→counterexample→minimal decision-changing check** 复查。仅修改科学判断与路由，不把技能的使用本身当作方法有效性的证据。

**最重要的修正：** 早先的口语判断“局部 SSE 优化目标错误是主要原因”“当前的基础不对导致Affine54不可行”强于证据。实际数据只反驳了 **「较低的teacher前缀局部输出SSE足以保证四步自由视频质量」** 这一充分性推理。局部SSE仍是有用的数值诊断与可能的辅助损失；目前并未识别它是不是本轮失败的主因，更没有证明全部Affine表示不可行。

**积极但有边界的新解释：** 本轮找到了一个**真正可以执行的函数重写/缩小W1/W2宽度**，并使局部近似优于两个同预算固定比较器；失败的终端视频现在是定位「部署数值、函数表达、输入离轨、有限后缀响应与四步状态目标」的判别数据。它为新函数单位/时空作用设计提供实证约束，而不是给固定54组续调参发通行证。

## 1. 源码对象与数学关系：精确性究竟在哪

实测对象为 Wan2.2 I2V 四步蒸馏版，两个expert（high/high/low/low），(D=5120)、(H=13824=216\times64)，LightX2V Wan FFN调用点保留原调用者的norm/condition modulation、tanh-GELU、gate、残差与真实QDQ→BF16执行。审查来源： [原生六视频](E0_RECONSTRUCTED_VIDEOS.md)、[全40层匹配回归](E0_ALL_BLOCK_RECONSTRUCTION.md)、[折叠公式及有限原生](E0_FUNCTION_RECONSTRUCTION.md)、[原生/连续边界审计](../method_discussion/discoveries/E0/independent-matched-calibration-and-runtime-audit-20261026.md)。

1. **实数级 tanh-GELU 身份（ALG）：** `phi(a)=a/2+e(a)`，`e(-a)=e(a)`。其依据是tanh近似的内部多项式为奇函数；[GELU原作者实现](https://github.com/hendrycks/GELUs/blob/master/README.md)写明该tanh表达。这只证明激活函数的代数分解，不证明经过Smooth2、Q2量化/反量化、BF16舍入后仍满足该分解。
2. **任意系数的完整残差恒等式（ALG）：** 对**实际post-Q2系数** `q(a)`，定义 `r(a)=q(a)-kappa*a-beta`，则 `q(a)=kappa*a+beta+r(a)` 总成立；保留所有组的 `r` 只是重写原函数，并未省算。真正的Approx从「仅付费求选中组的原非线性Q1/Q2」以及重新拟合读出和偏置开始；静态其他组残差没有精确保留。
3. **原生部署的具体表达（SOURCE/MEASURED）：** 当前affine54并非简单 `a/2`；校准产生逐坐标 `kappa,beta`，FP32合成折叠后cast为新BF16参数。原生执行形态为 `Ahat*x+V*q_S(x)+Bhat`，其中 `q_S` 是经过所测Smooth2/Q2的原生选组系数，`V` 为新的BF16读出；所选54个完整64组没有与原HiF4后量化权重等价。详见[实际合成式](E0_FUNCTION_RECONSTRUCTION.md)。因此“GELU有奇偶结构”既没有推导出本轮54组足够，也没有推导出固定54组必失败。
4. **gate/residual/trajectory 是另一计算语义：** 输入层的 `fhat(x)-f(x)`、原生postgate写回、下游block输出和跨step的`flow/latent`并不是一个误差对象。全40×4与输入双轨的 [静态工具](E0_FULL_DEPTH_TRACE_TOOLING.md)已定义四臂和signed cross term，真实Wan producer **尚未实现**。

注意 `q2_post` 才是W2实际系数；pre-Q2 `GELU(a)` 不得当作W2输入。以激活/hidden稀疏能量直接推断带符号W2组合或最终视频功能仍不成立。

## 2. 哪些事实已成立，哪些解释尚未成立

| 结论或命题 | 证据类别 | 可成立的范围 |
| --- | --- | --- |
| 全40层四校准来源256行匹配，affine54相对SSE均值8.54%，whole94为11.67%，仿射锚定输入ridge为15.35% | OBS | 两个历史已暴露held来源、320个step×layer案例；同为观察到的匹配比较而非总体统计独立样本、函数最优性下界 |
| 同等理想投影MAC约94/216，affine54局部指标优于whole94的305/320案例 | OBS/ALG预算 | 在所测支持/正则/校准下证实**含非线性特征的重构值得研究**；不是物理成本完全相同或所有其他函数类都劣 |
| 原生160处全部替换，热态source12 FFN9.707→4.449s、生成50.750→45.313s；setup139.188→237.089s | NATIVE/SYS | 真实有损情况下局部宽度减小和warm性能优势；不证明每任务总成本占优或质量通过 |
| source0/12联系图可见明显质量偏离，486帧工程补评确认来源间不同的motion/帧差模式 | VIDEO/POSTHOC | 当前两个固定全模型原型在这两来源**未达到视频质量目标**；光流、边缘能量等不是可直接代替人工/官方的感知保真真值 |
| 四候选报告 `reference_identity.model_digest` 的dense身份字段有勘误 | METADATA | 原视频/SHA与像素计算未改；核心gate通过不保证引用身份字段语义正确 |
| 当前bank与离线连续目标完全匹配，因而失败一定是student漂移 | **OPEN** | 只有0/30两层有限partial same-input；160处未原生配对。不能给传播提前归因 |
| 8.54%平均误差已足够小，任何下游都应保真 | **反例/无依据** | `8.54%`是case-mean局部能量比，单例最高24.86%，中间层可很差；方向/后缀作用无法从case-mean决定 |
| 方法家族被数学/物理证明不可能兼顾质量和省算 | **无此证明** | 本实验排除的是固定bank/支持/拟合/全位置替换组合的本次已测视频表现，不是任意分层/条件/表示/轻后训练的最优前沿 |

当前全帧质量结果：[补评](E0_EXISTING_VIDEO_QUALITY_BACKFILL.md) source0 affine/dense光流均值比0.091、相邻RGB变化比0.368；source12相应0.846、1.463。这说明两来源的时序异常**不应被统一命名为冻结**；但这些proxy亦不能单独判定哪个FFN功能方向丢失。正式VBench-I2V/VBench2与人工全片审阅仍未执行，`quality_accepted=false`。

## 3. 区分“本次基础不合适”与“表示家族不值得继续”的竞争解释

| 机制假说（均OPEN） | 若为真，预计会看到什么 | 最小高辨识检查与反预测 | 研究后果 |
| --- | --- | --- | --- |
| **H-NUM：原生部署数值差异** | 同一个真实 `q1` 上，BF16/实际Q2部分bank与离线连续预测有显著偏差，且在失败层集中 | 先静态检查pack、合成和Q2/GELU边界；拟议native四臂在全部160 site同时记连续和原生same-input，若数值误差始终远低于目标拟合残差则削弱H-NUM | 修部署/数值标定，而不是贸然改变研究目标 |
| **H-CAP：固定54组的函数能力/拟合缺口** | 在Dense teacher相同输入上，真实部署算子就留下大量signed的、受gate/后缀敏感的误差；student离轨并非必要条件 | 即使bank与连续一致，测 `f_S(x_D)-f_D(x_D)`，再以受控固定后缀试验验证其功能效应；若teacher敏感方向差很小而student分叉很大则削弱H-CAP主导 | 考虑新函数坐标/条件支持/分层容量/补偿目标，**不是自动增加组数** |
| **H-SHIFT：Student前缀输入分布偏移** | `f_S(x_S)-f_D(x_S)` 或 `f_D(x_S)-f_D(x_D)` 相比teacher配对出现明显变化 | 同site四臂原生测 `f_D(x_D),f_S(x_D),f_D(x_S),f_S(x_S)`，signed分解且记录真实postgate写回；若teacher/student误差相近则削弱“输入偏移主导” | 异步/受扰状态校准可能比teacher SSE更相关 |
| **H-TRANS：下游方向及跨层/跨step交互** | 同量级局部误差在真实gate/attention/调度器后呈现显著不同的有符号增益、抵消或传播 | 40×4全点signed/latent状态观测先做范围发现；选择小规模预冻结hybrid/suffix真实后续对照。若终端不随该扰动变化则削弱所测方向归因 | 学习/保持有限后缀敏感方向，SSE仍可作辅助指标 |
| **H-STATE：四步受扰状态目标不匹配** | 近似flow虽可接近teacher，却不能修复已有latent偏差；step更新含signed cross term | 记录每步真实step_pre/post、dtype/cast、`delta`与flow差内积；比较teacher-flow误差和实际下一状态误差 | 研究小型后训练或有界闭环目标；CTEC是近邻而非本仓库的新颖性 |
| **H-COST：表示在本实现中的系统价值受限** | warm FFN显著更快但setup/显存/磁盘/无额外融合造成净收益缩水 | 已有setup和FFN wall可直接复算；新方法再核 W1/W2/QDQ/读出、迁移、存储、真实端到端成本 | 若其他成本压倒收益，转向不同算子/时序可得信息，而不是仅调近似损失 |

这些解释不是互斥答案；先解决最容易击穿某种归因的观测，而非一次性为全部假说建新系统。**全点paired trace不等于因果归因；要归因终端视频仍需预冻结有限的真实后缀干预。**

### 一个直接的数学反例：局部 SSE 不决定终端功能

取两个大小相同的误差方向 `e1,e2`，且 `||e1||²=||e2||²`。设后缀对局部有限扰动的响应近似为 `T`，可存在 `||Te1|| ≫ ||Te2||`；如果多个site误差经后缀变换后方向相反，还可能抵消。数学上多项误差到最终输出的能量包含 `2< T_i e_i,T_j e_j >`。这说明**teacher局部SSE不能充分决定视频质量**，但**不能反推出**本轮必由传播造成——真实 `T` 未测。这个反例只属于条件数学，不是Wan测得的Jacobian，更不能假定QDQ处普通微分可用。

四步轨迹进一步给出理想表达 `delta_(k+1)=delta_k-h_k*(v_S(z_D+delta_k)-v_D(z_D))+rho_k`；`rho`承载原生数值更新及dtype舍入。只有在额外条件下才谈某种理想受扰纠正目标，不能把 `v_D(z_D)` 当成免费在线oracle。详见[轨迹来源](../method_discussion/discoveries/E0/transported-function-and-trajectory-target-20261026.md)与[step_pre修正](E0_FULL_DEPTH_TRACE_TOOLING.md)。

## 4. 文献比较：对问题空间的真实改变，而非照搬配方

- [GELU作者实现](https://github.com/hendrycks/GELUs/blob/master/README.md)：支持tanh-GELU的原始实数公式，**不支持对post-Q2使用未修正的奇偶恒等式**。
- [TARDIS，arXiv:2501.10054](https://arxiv.org/abs/2501.10054)：原有部分线性化/热区与动态fallback，说明Affine折叠不是新颖性本身；其语言模型token分布、选择器开销与Wan full-M不同，不能移植性能结论。
- [BRECQ，arXiv:2102.05426](https://arxiv.org/abs/2102.05426)：块重构与跨层依赖/泛化权衡的邻近问题；不推出更大block目标总是更优。
- [Q-VDiT，ICML2025](https://proceedings.mlr.press/v267/feng25q.html)：从视频时空关系出发区分局部量化误差与时序一致性目标；它是**研究目标的相关竞争视角**，并未验证本仓库54组函数重写。
- [CTEC，AAAI2025](https://ojs.aaai.org/index.php/AAAI/article/view/34039)：已有跨步误差校正训练；本仓库若考虑受扰state目标，应以实际Wan四步与真实开销区别于CTEC，而不是重复给一个新名字。

**本次WARD实际覆盖：** 原作者GELU公式、TARDIS arXiv HTML、ICML Q-VDiT正式入口、AAAI CTEC正式入口、BRECQ摘要与原仓库历史已读数学记录。未声称独立复现这些方法的全部代码或完成最新全领域系统综述。

## 5. 可执行的研究引导：区分两种并行工作线

**A｜已失败原型的原因判别（有边界）**：复用现有源码/已保存tensor先检查same-input数值与bank身份；列出原生四臂producer缺失组件、输入与post-Q1边界、真实gate/残差和step_pre/post数据；再在独立明确GPU授权下按冻结的40×4接口采集所有site的轻量观测、先观测后小量预登记后缀干预。目标是给H-NUM/H-CAP/H-SHIFT/H-TRANS/H-STATE分配真实证据，不是继续固定54组筛选。

**B｜开放的新FFN必要计算机制（创造性，非Affine优先）**：可从pre-W1可得信息、W1/GELU/W2 signed composition、时空运动/纹理功能、专家/step异质、低成本条件表示、真实MAC与内存访问、MXFP4/NVFP4 native数值、少量后训练等出发；不要求先解决Affine54才能提出新的数学结构。任何新方案须给出「区别于最近文献/旧实验的机制 → 与竞争解释不同的预言 → 最小可复用资产检验 → 最终视频/成本验收路径」。

两线相互交换证据，但不相互劫持。**禁止从固定支持失败推出固定支持族不可行，也禁止从代数恒等式推出可直接保质加速。** 反例用于推进机制，不是防御性停止研究的理由。

### 持续执行规则（由Agent自行应用，不是新审批）

1. **表示标签**：ALG-EXACT / SRC-CONDITIONAL / OBS-TEACHER / NATIVE-LOCAL / NATIVE-FULL-VIDEO / POSTHOC-PROXY / OPEN-CAUSE / QUALITY-NOT_ACCEPTED 不可混用。
2. **结论裁剪**：拒绝把两段视频当作总体统计、把最小SSE直接排名成画质、把flow proxy当主体运动真值、把平均数遮住中间层单案例24.86%最大误差。
3. **前缀/费用**：`q2_post`而非`GELU_preQ2`是W2输入；所需observer的全W1/GELU代价、FP4格式依赖与native BF16 bank不免费；保存的teacher panel≠student自然轨迹。
4. **保持计算图状态**：G5/P22属于VIDEO失配与已测速度，P23/P24属于OPEN测量，P25属于POLICY，G6实时暂停。图的标签不是原生执行记录。
5. **当前授权**：新GPU/模型/视频/训练=NONE。现有六视频完整核心分析已结束，正式质量仍未验收。行为改进以更多科学辨识力为目标，非自动把Prompt拉长。

## 6. 现在应当如何回答“基础不对还是整个思路不可行？”

> 这次Affine54不是“只剩一条已证明无效的思路”，也不是“换个局部loss马上可救”。实数分解成立，受限实验显示真实非线性特征与仿射折叠有局部预测价值且原生省算可行；但当前固定54组及拟合/全层部署产生了无法接受的现有视频偏离。已经失败的是**具体原型与「局部SSE足以推导视频保真」**，尚未识别数值、容量、前缀和传递哪项主导，也未证成方法族的可达质量—计算前沿。下一步先用同输入bank与student双轨/有符号后缀观测分清因果，再决定是否修改表示或转向别的计算机制。

这段判断是当前科学交接意见，不自动改变注册实验、冻结评价标准、全局stage或Agent职责。
