# Whole-FFN 表示：原论文/作者源码竞争性审阅

2026-10-29；独立理论子 Agent；研究性质 LIT/ALG/SRC，非新实验、质量通过或实现审批。入口是 [whole-FFN 独立提案](../../method_discussion/discoveries/E0/s2-whole-ffn-even-group-and-quadratic-units-20261029.md)。检索前查询 `PRIMARY_LITERATURE_INDEX.json` 与 `PAPER_METHOD_FOCUS_GRAPH.json`，未发现下列三标题/ID；现登记为 C17–C19。进一步全库查重发现 TARDIS 和 Whipp 曾在 `research/20261021/research_state_R2.json` 引用，Neuron Merging 曾在 `research/20261021/method_search_R10.json` 引用，且 TARDIS 在旧 affine 报告已有原文深读；这些旧引用保留，本次不是项目首次发现。检索包括 GELU 符号/神经元合并、FFN 折叠、输入低维二次/多项式、替代/负结果；社区摘要仅用作检索线索。完整原文与真实作者源码在 HF `ffnds-research-sources/s2_whole_ffn_primary_20261029/`，不是 Git 内论文副本。正式来源身份见同目录 `S2_WHOLE_FFN_REPRESENTATION_SOURCE_LEDGER_20261029.json`；G14 已登记，未改 ACTIVE 或冻结协议。

## C17：TARDIS，对整个 FFN 线性折叠的直接竞争

[Hu et al., arXiv:2501.10054v1](https://arxiv.org/html/2501.10054v1)，§4–7、§9 已核；本次未定位明确作者实现仓库，**CODE_NOT_LOCATED**，论文所说的 PyTorch/CUDA/vLLM 实现及性能是作者叙述，不能标 CODE_CHECKED。原文 HTML 本地 SHA `6c3156de0b58dfd1e1fb7dbb39a47bf41ab5824f86c842fca22e4817b26346fa`。

对每隐藏单元离线用校准输入估计高频区间和线性回归 `φ_n(a)≈α_n a+β_n`；式(5.2)将 `α_n W1[:,n]W2[n,:]` 汇总为 `C∈R^{d×d}`，在线 `xC+B`。当前 x 在 W1 前可得；在线另用 2-bit GPTQ W1 预测各神经元区间，选择性读原始 W1/W2 以减去近似、补回真实输出。原作者默认 C4 八条、各2048 tokens；范围/KDE及线性拟合离线，推理预测不是免费、也不是原生 BF16。Falcon 等模型 RTX 4090D 文本生成，作者报告回退索引/加载占时、长 prefill 因各token不同使稀疏优势减弱；其压缩/加速数值不能迁移至 Wan M=31668。在生效前先核权重总常驻量：C、2-bit predictor、回退原权重及其读出/选择/GEMM/输出融合。论文 §5.4 同时称只存需校正神经元的原始权重与按在线预测修复，二者对全输入动态支持的存储/访问需求需源码或精确系统证据，不由摘要百分比解决。

**区别与反例**：C 的整体线性折叠已有先例；所提 parity 分解为所有输入的实数恒等式、斜率固定1/2，并对保留的偶非线性做组联合，而非在高频区间将整激活近似成线性后按离群预测回退。TARDIS 隐藏 a 离群预测会漏判且输入分布漂移；parity组也可因偏置、W2有符号消解及BF16换算失效。若都只保留线性部分则它们均受非线性残差约束；完整 K 并不令偶项消失。建议 T04/F04、T05/F06、T07/F09、T13/P12、T16/P20；anchor 是 Wan 自身 W1/GELU/W2 而非量化Q1/Q2。

## C18：低秩双线性残差的负结果与其能力边界

[Whipp, arXiv:2606.19379v1](https://arxiv.org/html/2606.19379v1)，§3–6、附录复现实验路径已读；[作者 PolyWeave 仓库](https://github.com/Notabot123/polyweave) 实际检视 HEAD `7c1f241a7477100a09d01d86ecf771d3322952ef` 的 `polyweave/layers/poly_linear.py`、`polyweave/distill/regression.py` 和 `research/run_residual_gain_clean.py`；HEAD 是检索日代码，不冒称论文发布时固定版本。两个关键源码文件 SHA 分别 `5520150a39a6aecd73ab200f26697b23ef3f7d147994fec977c30a1451b9fdcb`、`614795de54ea8759722eea90e979766ae56f6478eb4ec04479b300656fefb766`；原文 HTML SHA `a3405372913b315e1031ff4332cc1dc2b50529ff3fba85347b31473c6aa951b7`。

作者将整层输入输出分解 `g(x)=W*x+b*+ρ(x)`，训练样本上闭式最小二乘拟合线性项，在 held-out 激活上测 R²；以 rank16 的 `(Ux)⊙(Vx)` 加输出 mixing 去拟合残差，并在选定层插回模型测 PPL，另有少量 swapped-only healing。源码实际是 `linear(x)+exp(scale)⊙mix((Ux)⊙(Vx))`；一个输出间共享的16维乘积特征，而不是完整 k(k+1)/2 两两交叉特征，也不是由原W1/W2解析折叠。原论文报告三种12层小LLM、WikiText-2及对应测点 rank16 residual gain中位数小于0.01、最大0.13；这对宽非线性不足的竞争解释有力，但不能据此判定完整二次或Wan失败。源码用固定尾部20%验证、以同验证集选择最佳训练迭代并将 best 下限钳在线性基准，因而无独立第三方测试集时该 gain 有选择偏差；生成任务、视频source与自由学生状态均未验证。

**实际成本细节**：论文 §3.4 近似写 `d²+2dr`，当前代码 forward 是两次 `d→r`、一次 `r→d` 加线性 `d→d`，名义 `d²+3dr`（外加逐输出scale、逐元素乘法与累加；对称权重绑U=V才可复用一投影）。需核论文结果运行时确切 commit 才能将该差异归于版本变动或报告漏项。闭式训练解不是 held-out 分布的数学最优解；其数值还受条件数、样本量和 FP64拟合→运行dtype转化影响。建议 T04/F04、T05/F06、T07/F09、T09/F13、T16/P20，source-cluster held-out/gated-output、signed cross以及真实下游自由状态是Wan额外辨别。

## C19：ReLU 正比例神经元合并的精确前提

[Kim et al., NeurIPS 2020, arXiv:2010.13160v1](https://arxiv.org/html/2010.13160v1) §3 theorem、算法与实验已读；[作者仓库](https://github.com/friendshipkim/neuron-merging) commit `d25cf3e39ecf376a1ffc8b3cbf326a38b6db19cf`，`decompose.py` 已读并保存在 HF，SHA `c4bc0dac56c380ae7791473d4a8f5bc0c2225c5b7bde1b378bbc5ad6e22ea85f`；原文 HTML SHA `71966bb1f9cef93fe175acb6f565576fd21c370b144885149f26b92e60968177`。原文 theorem 的核心为 ReLU 非负、每输出只取一个正输入的缩放矩阵才能与激活交换；所选代表和合并系数使两侧有效宽度同步减少。代码 `create_scaling_mat_ip_thres_bias` 以权重+偏置的拼接向量选余弦最近保留单元，再用范数比给正缩放，`Decompose` 用矩阵乘合并下一层。作者用 CNN/LeNet 等分类、一次性 data-free 合并；无作者 Wan/video 或完整后缀证据。

不同于正齐次 ReLU，GELU 既不是正齐次也不是偶函数：`f(-a)=f(a)-a`。完整 `f(a)v₁+f(-a)v₂=ψ(a)(v₁+v₂)+a(v₁−v₂)/2`；必须保留线性差项。单纯最近权重余弦无法保证真实联合输入上 a 与 ±z 同余，额外偏置 mismatch 或学生 x 沿小差方向偏离都可造成大误差。强对照应包括普通同号合并、近权重但偏置破坏、分组输出符号抵消、以及不同 source/student 条件。建议 T04/F04、T05/F06、T07/F09、T16/P20。

## Root 独立提出的 full-K refinement：复核及失败边界

以下补充由 **root 提出**，本人独立核算，不重写原提案身份。令 `K=0.5 W1W2` 完整保存，且正确并入 `0.5 b1W2+b2`，奇线性分支为一次 D×D GEMM；G 个镜像偶组需要 D→G、G→D 两次 GEMM。实数计算名义 MAC 与原两个 dense GEMM 之比是 `D/(2N)+G/N`。`D=5120,N=13824,G=4608` 时为 `0.5185185`，约少48.15%；完整 K BF16 存储 `52,428,800` B，两个偶组矩阵共 `94,371,840` B（未包括 bias、alignment/工作空间和原状态）。原两权重合计 `283,115,520` B。K 低秩不是此分支的必要条件；低秩 K 是另外的近似和成本选项。`G≥N−D/2=11264` 时算术不再少于原两个GEMM，还未计 ψ、合并、离线构造、内存、launch或typed rounding。

精确±镜像组可能极少，常见近邻的 bias或在student x上的差异会破坏关系；尤其 group误差与线性/原生GEMM舍入余项存在 signed cross，相消与放大均可能。`ψ` 与线性合并改变 BF16 逐步舍入，gate、原 residual、typed postwrite和后续 attention 也要比较。针对有限候选最小判别：先固定源与step/block、同一Dense及学生x收集真实native full-M `a=xW1+b1` 的选中行和 full-M BF16 `Y/g/r/u`，同输入离线观察镜像条件及分组误差，用 native full-M 与 small-M bridge 区分GEMM形状舍入；输出 linear/even/误差的有符号交叉，匹配G和成本比较同号合并、随机组及 source-held split。此采集是建议，不是已注册或执行；只看已有h无法稳妥反演非单调GELU的a。

## G14 写回与静态验证（实际执行）

C17–C19 已登记于 `PAPER_METHOD_FOCUS_GRAPH.json` 的 supplemental papers/links 和 G14 Mermaid，机器 source ledger 保留原文与实际核读代码 SHA、commit 和旧发现链接；`GRAPH_AGENT_BINDINGS.json` 中 G14 supplemental 计数更新为19，其他研究状态不由本次改写。`.venv/bin/python -B tools/graph_agent_bridge.py validate` 实际返回 `STATIC_GRAPH_BRIDGE_VALID`，98 primary、19 supplemental、16 focus、313 paper-to-stage links；`paper C17/C18/C19` 查询均返回实际原生锚点。

附加运行 `.venv/bin/python -B -m unittest tests.test_graph_agent_bridge`：21项中19通过、2失败。两项都是现有旧规则断言：`test_full_literature_index_is_linked_to_actual_native_focus_nodes` 硬编码10个supplemental（写回前已有16），`test_research_reading_priorities_have_one_current_method_authority` 要求旧导航项；本轮未改导航或测试。静态图一致性不代表作者方法在 Wan 运行或本地视频通过。
