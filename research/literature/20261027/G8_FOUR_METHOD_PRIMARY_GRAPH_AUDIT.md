# FFNDS · G8 新增四组直接近邻的原文计算流（2026-10-27）

**资料身份**：本文是原始论文与作者代码的算法研究，非新增 FFNDS 模型实验。与[真实 Wan G0–G5 / 文献 G7](../../../docs/98_端到端计算图与证据注解_20261025.md)并存，G8 中全部新节点是外部 **LIT**。当前[ACTIVE](../../ACTIVE_RESEARCH.json)明示 fixed scope 已完成、**模型执行暂停**。现有 10 source、12 条 source×seed、1920 paired sites 为先前完成的原生资产，不是本文运行结果。

## 1. RT-Lynx（L08）— 当前 GEMM 的 activation N:M（K13–K15）

**原始出版**：[ICML 2026 / PMLR306](https://proceedings.mlr.press/v306/cong26a.html)（作者摘要）；[论文对应 arXiv](https://arxiv.org/abs/2605.26632)。[作者在 Hugging Face Papers 的公开解释](https://huggingface.co/papers/2605.26632)进一步确认 **online N:M activation sparsification + norm-based compensation + lightweight LoRA branch**，并主张 sparsify/compress/sparse-Tensor-Core 合并 CUDA kernel。本文**未取得原始 ICML 正式 PDF 方法段，也未核查 RT-Lynx 官方实现**；norm 补偿公式、LoRA具体层数、训练预算不能标为逐公式一手确认。

必要执行图：
- **K13**：从 Q1 后输入 W1 的 X∈R^(M×5120)，或在 W1+GELU+原HiF4 Q2 后消费者 W2 的 V∈R^(M×13824) 选择要 sparsify 的**当前 GEMM 输入**。W1 前选择与 W2 前选择可省的前序计算不同。
- **K14**：对某一 GEMM 的输入沿K轴构造合法 N:M（论文主述2:4）metadata/packed activation，执行对应 sparse GEMM；W2 输入已经来自完整 W1 的情况下，这个动作**不能追溯地省掉已经计算完的 W1**。
- **K15**：在此稀疏结果上加论文所称误差补偿（作者公开说明包含norm支路、轻量LoRA）；再进入G20/F12 gate残差。分层kernel速度与全视频质量是不同证据分母。

**与 Wan HiF4 的结构差异**：原Q2的whole64→8→4共享scale树依据完整hidden参与max决定；将原post-Q2 decoded activation改成2:4必须区分“先量化后压缩”与“先mask再量化”，一般并不交换。2:4 Tensor Core kernel和SM120 dense packed NVFP4 primitive又是不同实际执行路径。原论文linear层最高平均1.55×不是WAN整体视频提速；未经模型同一QDQ、pack和正式质量对比不能升级。

## 2. Dense2MoE（L10）— 训练后 FFN 容量与activated参数分离（K16–K19）

**原始全文一手**：[arXiv 2510.09094 v1，10页](https://arxiv.org/pdf/2510.09094)，§3.1 Eq.(1–5)、§3.2 Eq.(6–10)、§4消融。当前首次直接读到全部主要方法方程（前一版文献表错误地只标摘要/二级解读）。[正式ICC V2025](https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_Dense2MoE_Restructuring_Diffusion_Transformer_to_MoE_for_Efficient_Text-to-Image_Generation_ICCV_2025_paper.html)。

- **K16：离线专家切分与训练**。论文先按原FFN intermediate neurons的Taylor累积贡献 \`I_i=|(∂L/∂w_i)w_i|\`（论文Eq.6）选 rs×h 单元为**共享专家**，其余均分到普通专家；先只训练shared，再冻结shared、训练normal experts和router+负载均衡，最后蒸馏group of blocks。Taylor是初始化排序，不是自动证明原Dense FFN存在天然分工。
- **K17：同层 FFN Top-K 路由**。原FFN MLP(x)=W2 GELU(W1 x+b1)+b2。Eq.(2–3)改成 \`y_token=MLP_shared(x)+sum_(i in TopK) normalized_softmax_i(Wg x)*MLP_normal_i(x)\`。专家总宽度 \`r=rs+n·rn\`，每次激活宽度 \`ra=rs+k·rn\`，理论activated-FFN参数比例 \`ra/r\`；**shared对所有token计算，normal只在按token路由时参与**。router的Wg x、TopK、token dispatch、weight gathering和融合写回要真实计价。论文的FLUX试验不是本Wan2.2 I2V。
- **K18：Mixture of Blocks (MoB)**。不是在40个blocks任意独立0/1跳过；将连续m个blocks分为group，每次根据输入feature与AdaLN condition embedding \`y\` 选择**连续的κ个blocks**。论文Eq.(5)是 \`TopK(α Wx([x,c])+(1−α) Wy y, κ)\`，用它控制合法连续区间（完整选择约束见Eq.4），group只执行选定子串，省下m−κ个block。condition在router所在层之前必须可读；Wan已有 AdaLN e1…e6，但不能视为旧router现成输出。
- **K19：三阶段蒸馏与质量**。Eq.(7)完整输出KD，Eq.(8–9)归一化加权block features，Eq.(10)加load balance（论文λfeature=1，λbalance=10^-2）；MoB再匹配原teacher group的最后block特征。论文Fig.4与Tab.5（Taylor切分/独立训练消融）、Tab.3（同等激活block数量比较）是可核实的训练恢复证据，**不是未训练FFN隐含MoE或FFN零SSE重构结论**。

关键对照：原Wan high/high/low/low 是**两套完整主网络权重按去噪step选用**，与Dense2MoE在**单block内按token选FFN专家**不等价。它的router、蒸馏预算及新版参数容量构成论文方法主体。

## 3. SVD-Cache（L39）— 主特征预测、残差复用（K20–K23）

**原始全文一手**：[arXiv 2601.07396 v1，10页](https://arxiv.org/pdf/2601.07396)，§3.2–3.5 Eq.(4–11)、Fig.2；[CVPR 2026 正式页](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Forecast_the_Principal_Stabilize_the_Residual_Subspace-Aware_Feature_Caching_for_CVPR_2026_paper.html)。本文首次核对其真正的**投影与预测**实现，而非只用作者摘要。公开 GitHub 当次检查未发现可运行实现，因此标 PRIMARY_METHOD_PDF 而非 CODE_CHECKED。

- **K20：离线参考prompt做SVD**。对选中DiT block feature \`F_ref∈R^(N×D)\` 求 \`F_ref=UΣV^T\`，以累计平方singular values达到τ选rank k；缓存右基 \`V_c∈R^(D×r)\` 和 \`σ_c\`。依据论文观察基跨prompt较稳定，是论文实验现象，不能直接推断原Wan所有层、expert、step与source之间固定。
- **K21：新prompt有完整feature时，复用右基做投影**。Eq.(8–9) \`U≈F V_c diag(σ_c)^(-1)\`，\`F_k=U_k diag(σ_(c,k)) V_(c,k)^T\`，代数上在非零σ位置等价于 \`F_k=F V_(c,k) V_(c,k)^T\`。这里依然要读取**已计算出的 F**，所以是cache refresh/decomposition，**不是本次FFN W1之前直接得出的hidden选择**。
- **K22：按两个子空间分别维护时间状态**。Eq.(10) \`R=F−F_k\`；Eq.(11) principal EMA \`EMA_t=β EMA_(t−Δ)+(1−β)F_(k,t)\`，论文设β=0.9；在未来未计算step预测principal，残差用上一refresh的 \`R\` **直接复用**。
- **K23：后续被跳过step的feature合成**：\`F_hat_(t+Δ)=forecast(F_k,t)+reuse(R_t)\`，供后续网络消费者使用，跳过整块重算。作者强调低能量residual虽然不易预测，但不能因此舍弃；时序抖动与低能量≠无功能。注意论文还以多步FLUX/HunyuanVideo为主测试，FLUX.1-schnell只作为可兼容场景，不是已在本四步Wan2.2 I2V取得收益。

**研究边界**：本项目之前计算的signed Fourier latent bands分解的是**两套网络运行差值**，SVD-Cache分解的是**某DiT block的实际feature跨step轨迹**，数学对象不相同。谱能量k选择属于此论文的cache representation工具，不是可泛化的FFN重要性评分。四步和双专家引发缓存次数、W2权重身份和history memory的新问题。

## 4. Chipmunk（L09）— 与SVD-Cache不同的MLP系数增量（K24–K27）

**原始作者技术正文及代码**：[arXiv 2506.03275](https://arxiv.org/abs/2506.03275)、[作者Part II](https://sandyresearch.github.io/chipmunk-part-II/)、[原MLP源码](https://github.com/sandyresearch/chipmunk/blob/master/src/chipmunk/modules/mlp.py)与[本仓库既有源码审查](CHIPMUNK_SOURCE_READING.md)。已核源tree为 \`177fdefc4803e621fd8a02aad491812e9d1ecc94\`，非新的源码复现。

- **K24**：当前W1之前输入 \`x_t\`，上次同一**权重**FFN缓存post-GELU或对应数值状态 \`h_(t−1)\` 与 \`y_(t−1)\`。同expert条件下数学形式 \`y_t≈y_(t−1)+W2 S(h_t−h_(t−1))\`；已变Q2 scale或不同专家W2时不直接成立。
- **K25**：先在缩小token网格做 \`blockmean(x_t)→fc1\` 生成预估激活与历史blockmean对比，并Top-K选择候选hidden indices。这项W1子计算、block mean和mask/reorder**有真实在线成本**。源码对**复用过去选择的mask**还有 \`inference_step≥10\` 的guard；不能说“四步全盘不可能稀疏计算”，只说明该复用分支在四步默认场景不能自动触发。
- **K26**：自定义column-sparse mm1/gather+activation产生少量新系数变化，再与cache对照形成delta。
- **K27**：column-sparse W2 scatter-add到历史output cache，写回真实layer输出。对Wo/GELU/Q2的因果依赖必须逐个核对；原代码BF16/某FP8，不是当前HiF4的已测算子。

Chipmunk关注**跨步变化稀疏**，RT-Lynx关注**当前GEMM的激活N:M**，SVD-Cache关注**主/残差时序不同可预测性**，Dense2MoE关注**训练形成的条件容量**。四者各自的时间状态、实际省算颗粒与训练费用不同，不能按局部SSE统一选胜者。

## 5. G8 与 G7 的关系及后续扩展格式

| 外部 LIT 分支 | 一手来源状态 | 在本Wan中的映射位置 | 未闭合的关键科学/物理边 |
|---|---|---|---|
| K13–15 RT-Lynx | ICML原始摘要+作者公开解读；论文方法PDF未完整读 | F03→F04或F08→F09 | N:M实际合法dtype/pack、Q2依赖、补偿训练和代码 |
| K16–19 Dense2MoE | arXiv v1原文公式Eq1–10, Fig4, Tab3/5 | G18→F04/F06/F09、G04–G06 | 原Wan无训练专家分工、token dispatch及组block连续router |
| K20–23 SVD-Cache | arXiv v1原文Eq4–11, Fig2 | G02、F13/G21、历史缓存 | 额外full-step刷新、expert状态身份、skip-step完整视频 |
| K24–27 Chipmunk | 作者实际MLP Python/CUDA源码已核 | G02→D01/F04/F06/F08/F09 | HiF4 Q2 scale、四步缓存及patch reorder实际费用 |

**图更新政策**：读新论文以后同一轮更新节点、公式、形状、信息时点、上下游、source tier及与旧论文的**竞争机制**，不是只把参考文献插在某个节点。FFNDS原生SRC路径和现有配对视频/实测费用永远由独立RUN证据确认；本轮所有新节点都是外部LIT，GPU运行保持暂停。
