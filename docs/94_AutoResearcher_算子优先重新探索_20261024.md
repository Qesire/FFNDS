# AutoResearcher 再探索：FFN 计算算子本身（2026-10-24）

## 研究对象与发现

研究目标仍是：降低真实 FFN 执行成本，保持四步 Wan 视频生成的质量。Dense D-Q 是质量参照，UB1 是历史静态基线；补算与 R256 完全可选，不能成为候选方法生成的默认动作。

源码定位：Wan2.2 I2V distilled HiF4 D-Q，FFN 主投影为 D=5120 → H=13824 → D=5120，GELU(approximate=tanh)，主 W1 前 Q1、GELU 后 W2 前 Q2，原有 gate 和 residual 写回。单 token 两个 dense GEMM 的理论 MAC 为 2DH=141557760。四步、两个专家、每专家40个block；量化/内存/算子调度未包含于以下理论数字。所用源码为 tools/packed_ffn_reference.py 与 tools/ub1_row_dispatch.py。

### A：双侧低秩保留非线性的算子（第一条独立计算图）

不先删除组，也不恢复 dense：W1 ≈ U1 V1，形状 U1:[H,r1]、V1:[r1,D]；W2 ≈ U2 V2，U2:[D,r2]、V2:[r2,H]。

    xQ = Q1(x); z = U1(V1 xQ)+b1
    h = GELU(z); hQ = Q2(h/s2)
    y = U2(V2 hQ)+b2

按上述数学表示，MAC/token = (r1+r2)(D+H)。r1=r2=1024 时是 38797312，即 dense 的约27.41%，而直接保留一半 hidden channels 的两个主 GEMM 约为50%。此为**代数成本可能性，不是加速或视频质量结论**。两次 GEMM 变成四次，hidden 13824维的 GELU 和 Q2 仍存在；分解后的权重 QDQ/精度及操作顺序应作为新算子测试，不声称与原 HiF4 逐位等价。

真正重要的问题不是“SVD 有没有用”，而是四步蒸馏 Wan 的两个矩阵是否允许在原生执行和视频质量下形成有竞争力的双侧低秩配置。先比较**固定**分解、只分解W1或W2、同预算通道剪枝，再讨论是否有必要用状态选择不同的 r1/r2。不得预设 router。

**积极查重：** 2026-09 [SVDtrunc](https://arxiv.org/abs/2609.04646) 已对 DiT 跨块低秩分配研究；2026-10 [LRCC](https://arxiv.org/abs/2610.08858) 已研究嵌套低秩路径与按 token 的条件路由。双侧SVD或嵌套rank本身不是独立新颖性。与历史 R8/R9 的区别在于两侧权重因子化、原 GELU 仍完整，不采用其“仿射骨干＋非线性余项”；但 R8 的新 HiF4 编码8/8失败提醒量化可能直接否决实现，不能忽略。

### B：从输入和输出权重关系证明可直接删去成对计算

两单元贡献 p(x)=a GELU(ui(x))+b GELU(uj(x))，恒等地重写成

    p = (a+b)GELU(ui)+b*(GELU(uj)-GELU(ui)).

如果 GELU 的导数绝对值上界为L（应使用真实 tanh 近似的界），则

    ||p|| <= ||a+b|| |GELU(ui)| + ||b|| L |uj-ui|,
    uj-ui=(w1_j-w1_i)^T x+(bias_j-bias_i).

这提供“输入侧相似、输出侧反向”时整对隐藏单元**可不算**的候选条件。先读权重检查成对关系是否存在，静态删除已足够有效就不强造动态性。实机 HiF4 的64通道量化分组、重新打包、Q2和BF16求和需要另行计费并观测。跨组单元对若无法有效物理打包，代数对消不等于计算节约。动态条件判别不能先执行被删通道的GELU。

**既有工作与范围：** R4 已探索 Gram 有符号联合交换，不能将其改名为新发现；FANG（AAAI2026，https://ojs.aaai.org/index.php/AAAI/article/view/40021）和 2026 antipodal GELU 机制分析（https://www.kanikona.com/blog/weight-identifiable-antipodal-gelu-populations-mediate-compensating-responses-in-pythia）是结构启发，不据此认定 Wan 中真实存在有利对消。新主张只能是由W1/W2同时可检查的结构预测真实native删组效应，而非现有输出 Gram 再调参。

### C：不同输入区域独立执行一条窄 FFN

在条件区域 k=k(z)，只计算 Fk(x)=Vk GELU(Uk x+bk)+ck，**不补dense，不加R256**。对比单一固定窄网络的活跃MAC，以及等**总参数/存储**固定网络、LRCC相同激活成本、路由/打包费用。不同区域是否需要不同非线性算子是待验证假说。之前本地未入仓工具声称的0.00682/0.69387无法核验；尤其半宽m=6912>D=5120，输出秩m<D的理论下界不能外推到 Wan。若固定算子在同成本上已同样好，优先固定算子。

## 历史实验精确限制

- R8F1：仿射骨干BF16代数精度表现好，但原生新HiF4折叠矩阵门槛8/8未通过；R9A1具体固定affine remainder失败。**并未测试 A 的双侧factor + 完整GELU**。
- R13 已保存193个near row-pairs，直接full-row复制作为与两个UB1半hidden的同理论主MAC替代：193对里复制均未赢，pooled SSE为34661.06 vs 8910.64。不要重新包装“直接复制邻近行”为新机制；此负结论仅对应原动作/样本。
- R4已做Gram联合支持交换；R12的无补偿控制混合改变alpha、folded bias、R256；均不能证明新的直接算子必须补算。
- 现有 C2 dense例外行/UB1标签仅定义那一动作家族，绝非所有低计算算子共用的训练目标。

## AutoResearcher 的决策推进

**OPEN→MAP_FLOW→E/T/R→DISCRIMINATE→CARRY**：使用上游 AutoResearcher 最新 5bd7ec53 的 compact harness（https://github.com/Qesire/AutoResearcher/blob/main/core/AGENT_HARNESS.md），一次聚焦一个能改变路径的判别，不将历史C2草案自动路由到新实验。

A0：审查W1/W2因子化输入和权重是否已保存、列出实际Q1/Q2/HiF4编码和GEMM边界；如缺实测谱，只记未测，禁止假定r=1024必然够用。
B0：只读检查已存在的W1/W2结构条件，特别是原生64通道整组而不是跨组任意列匹配；如不存在成对对消，直接放弃此具体路径。
C0：复核本地未提交脚本的完整输入/训练验证样本/容量/路由，建立同时约束活跃计算和总参数的公平比较；不能使用原toy秩下界。
只有之后出现独立机制证据，才考虑一次明确注册的原生数值对照和至少一条完整四步free video；实际质量/速度由同来源/精度/算子条件下的完整质量—wall/FFN时间Pareto确定。

**权限边界：** 用户本次重新发起理论/源码/文献探索，并未授权恢复全部实验。research/20261024/EXPERIMENT_PAUSE.json 所列CPU/GPU/模型实验、重试与自动后续仍暂停。本文件是源码与数学推导，不声称新增模型前向、CPU实验、官方质量、真实视频或硬件加速。
