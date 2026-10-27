# UB1 方法重建与文献查重｜2026-10-27

**对外科研交流用独立技术审查。** UB1是FFNDS内部实验编号，仓库未定义其正式英文全称。本文件依据已冻结的UB1源码、配置、原生RUN及已公开2024–2026论文的官方记录和作者论文；不是新的模型实验或算法验收。

## 1. 结论

UB1是在LightX2V Wan2.2 I2V A14B、四步蒸馏D-Q/HiF4 fake-QDQ+BF16 Linear上实施的**按时间步固定FFN宽度**基线。它把C24的四步50/50/50/50%主FFN宽度分配改为25/50/50/75%，按模型真实原64通道组裁剪W1输出行和W2输入列，沿用输出贡献选组、中心化增益/偏置校准及rank-256辅助低秩支路。

原20来源官方三维VBench逐项损失<2%且完整FFN 1.5461倍加速；另20个不相交来源imaging损失3.166%，未过原门槛。**方法作为可复现静态系统比较器成立；时间步异质计算本身已被DyDiT、DIFS、TAPE等先例覆盖，不适合直接声称新颖算法。**

## 2. 原生对象、真实预算与成本

四步调用high/high/low/low两个独立expert，各40block，共160个FFN位置。代表性几何M=31668、D=5120、H=13824=216×64。

| Step | Expert | 原64组数量/216 | FFN隐藏宽度 | 主GEMM密度 |
| --- | --- | --- | --- | --- |
| 0 | high | 54 | 3456 | 25% |
| 1 | high | 108 | 6912 | 50% |
| 2 | low | 108 | 6912 | 50% |
| 3 | low | 162 | 10368 | 75% |

原Dense两层FFN线性主MAC是2MDH，每步缩窄后为2MDH_t。平均主密度(54+108+108+162)/(4×216)=50%。原生调用保留norm2/AdaLN、Smooth1/Q1、W1+b1、tanh-GELU、Smooth2/HiF4 Q2、W2+b2、e5-gate/BF16 residual。HiF4 Q2具有64→8→4共享scale，所以按原64组裁剪有明确数值意义。未把fake-QDQ+BF16的计时说成真正packed FP4。

rank256补偿另有两次BF16线性层，理论MAC占原Dense FFN主MAC的256/13824=1.85185%；因此主+辅助线性计算约51.85185%，而QDQ、GELU、搬运、gate、setup等费用要另计。原原生RUN在全160处留下320次主GEMM的实际shape，核对全局MAC确为一半。[执行器](../../tools/step_budget_runtime.py)；[主GEMM原生审计](../../reports/worker_20261020/UB1_formal20_main_compute_audit.json)。

配置的step索引是[1000,750,500,250]，实际shift5后的model timestep为[1000,937.5,833.333313,625]，sigma为[1,.9375,.833333313,.625]。同输入的scheduler连续flow更新系数绝对值为[.0625,.104166687,.208333313,.625]。末步系数为首步10倍是**数学事实**，但不能由此单独证明每步FFN的质量敏感度比例，亦不能证明UB1是全局最优调度。[原生scheduler审计](../../reports/worker_20261020/native_shift5_scheduler_audit.json)。

## 3. 选组、校准和低秩残差的确切实现

**后验组贡献**：针对step×block的calibration样本，读取真实保存的post-Q2 hidden a及原量化有效W2、门控g，计算每个原始64组j的
\[
C_{ij}=\|g\odot(W_{2,:,G_j}a_{i,G_j})\|_2^2.
\]
从24个校准来源每来源256行的平均贡献C_mean中选取k_t个组（稳定排序，组编号并列时确定性；最终升序）。这是真实隐藏值上的后验对角输出能量排序，**不是在推理W1之前花零代价可计算的selector，也不是有符号联合输出最优集合**。[打分源码](../../tools/generate_proxy_pr2.py)；[静态fit](../../tools/fit_cstatic_centered.py)；[UB1 refit](../../tools/fit_step_budget.py)。

**固定标量增益和折叠偏置**：定义不含原b2的dense和所选组输出信号分别为y_i、s_i，经验均值mu_y、mu_s。选择非负alpha最小化centered gated输出平方误差：
\[
\alpha=\max(0,\frac{\sum_i\langle g\odot(s_i-\mu_s),g\odot(y_i-\mu_y)\rangle}{\sum_i\|g\odot(s_i-\mu_s)\|_2^2}),\quad
b'=\mathrm{BF16}(b_2+\mu_y-\alpha\mu_s).
\]
部署在缩窄实际W2之前对其输入做FP32乘alpha后BF16 cast；b2折叠且原bias只加一次。[中心化拟合](../../tools/centered_signal_fit.py)。

**R256辅助**：以真实post-Q1输入V1预测目标FFN门控输出与当前紧凑主分支门控输出之差，在6144行上拟合lambda=1、rank=256的白化reduced-rank ridge，最终表示为A:(5120→256)、B:(256→5120)两次BF16 GEMM及其bias。拟合发生在FP64，BF16最终参数采用gate逆变换以便在caller gate之前加入残差；运行时R256可在当前W1之前从V1发起预测，但它的实际成本仍需计费。[RR算法](../../tools/reduced_rank_residual.py)；[UB1拟合](../../tools/fit_step_budget.py)。

**继承结构**：C24四步皆108组。UB1保留step1/2原完整C24参数，仅为step0和3各40block重拟合，共80位置。拟合前先在block20按108组严格复现旧C24增益、偏置、RR的参数SHA。原C24是24来源×每位置256行；用于修正的训练标签保留了原Dense post-Q2切片，部署却执行缩窄W1/Q2，这一训练/部署数值路径差异需要真实native对照而不能被公式自动消除。[冻结协议](../../reports/worker_20261020/step_budget_ub1_registered_protocol.json)；[policy](../../reports/worker_20261020/UB1_policy.json)。

## 4. 实验与失败范围

| 三维VBench | 原20 Dense | 原20 UB1 | 相对下降 | 新20 Dense | 新20 UB1 | 相对下降 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Aesthetic | .629235296 | .623736723 | .874% | .616151106 | .604236497 | 1.934% |
| Imaging | .744562166 | .734549666 | 1.345% | .726293683 | .703299460 | **3.166%** |
| Motion | .983922636 | .983850171 | .0074% | .985400287 | .986328447 | **−.0942%** |

原20包含10个subject及10个background，已经在先前开发多次复用；新增20从不同来源冻结，不改变候选参数。原20有5个来源的Aesthetic和7个来源的Imaging个体下降>2%。新20中的群鸟source27 Imaging下降16.784%，雪夜公交及暴雪车辆约12.033%、11.560%；总体新20失败不是同一批正式结果的复算误差。[原20 CSV](../../reports/worker_20261020/formal20_UB1_paired_sources.csv)；[新20 CSV](../../reports/worker_20261020/disjoint20_UB1_paired_sources.csv)。

原完整160处FFN隔离测试：Dense三次10333.253/10098.097/10101.356ms，均值10177.569ms；UB1三次6577.826/6583.760/6586.480ms，均值6582.689ms；**1.546111倍**。R17另外在**一个来源每臂5次技术重复**测完整生成Dense50.847s→UB1 47.315s，即**1.075683倍**，不能替代40来源的端到端性能分布。[工程验收RUN](../../reports/worker_20261020/UB1_joint_engineering_acceptance_RUN.json)；[R17真实生成时间](../20261023/R17E0_system_timing_prefix.json)。

真实80条Dense/UB1配对生成均有历史路径和结果记录，其中40条为UB1原20+新增20；20条原UB1驻留等价验证的4个latent/视频字节身份对齐。Git仓库提供截图/JSON/CSV/视频SHA，但不包含mp4视频本体；历史worker/HF派生缓存需另核存续。[原20生成](../../reports/worker_20261020/formal20_UB1_generation_RUN.json)；[新增20生成](../../reports/worker_20261020/disjoint20_UB1_generation_RUN.json)。

补充机制对照：固定相同总体主预算的UB2改为25/25/50/100%，在同Student状态末步FFN恢复完整Dense也没让整条视频通过；新20 Imaging损失约4.652%，原20约2.218%。因此“给末步越多就越好”的单调关系，在这两份具体固定策略上受到反例挑战；不能扩展为全部预算策略的全局结论。[UB2报告](../../docs/38_固定预算下恢复末步完整FFN.md)。

## 5. 文献查重：最强同构先例

此表分辨先例的真实作用单位/训练身份，不用跨模型/跨硬件原论文分数与UB1的1.546倍直接排名；外部论文核验深度为官方出版页及可公开访问的作者文章正文或源码账本，未逐项复现其全部实验。

| 先例 | 与UB1重复的机制 | 实质区别 | 查重判断 |
| --- | --- | --- | --- |
| [DyDiT, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a44a70acd5d0abc1a252ada9719dd06d-Abstract-Conference.html)；[DyDiT++，TPAMI 2026](https://pubmed.ncbi.nlm.nih.gov/41538341/) | Timestep-wise Dynamic Width，按去噪step改变MLP宽度；后续扩展到视频和flow | 学习timestep掩码/空间token和可选TD-LoRA，UB1是固定原64组/无新训练 | **极高概念重叠** |
| [Dynamic-in-Few-Step，2026-07](https://arxiv.org/abs/2607.06631) | 四步Wan视频模型step-specific结构计算，精简不同step容量 | 联合蒸馏、Progressive Training及Output Rollout成MoM，UB1是固定distilled checkpoint上的离线校准 | **同研究对象高度接近** |
| [TAPE，2026-05](https://arxiv.org/abs/2605.17837) | 视频扩散早高噪步多剪、后保真阶段少剪的预算方向 | 动作是时间平滑token选择/注意力而不是FFN hidden channel | **时间预算方向高度重叠，执行对象不同** |
| [MosaicDiff，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Guo_MosaicDiff_Training-free_Structural_Pruning_for_Diffusion_Model_Acceleration_Reflecting_Pretraining_ICCV_2025_paper.html) | 时序分阶段结构剪枝，非均匀预算 | 训练动态映射和非单调阶段重要性，与固定Wan四步C_mean/R256不同 | 高层预算重叠 |
| [SlimDiff，2025-09](https://arxiv.org/abs/2509.21498) | activation-guided、training-free、timestep-aware的结构化attention/FFN压缩 | spectral/协方差与跨模块rank分配，而非固定门控C_mean和RR | 选择/校准范式重叠 |
| [E-DiT，ICML2026](https://proceedings.mlr.press/v306/wang26dd.html) | block级MLP宽度与稀疏推理 | online sample router决定skip及宽度，UB1完全source-independent | 条件宽度先例，动作不同 |
| [PARE，2026-05](https://arxiv.org/abs/2605.27336) | Wan2.1-14B I2V、按denoising stage分配模型执行预算 | 注意力head宽度/深度router，配合训练，而非静态FFN组 | 视频Wan近邻 |
| [PPCL，CVPR2026](https://openaccess.thecvf.com/content/CVPR2026/html/Ma_Pluggable_Pruning_with_Contiguous_Layer_Distillation_for_Diffusion_Transformers_CVPR_2026_paper.html) | 结构化宽度压缩与质量恢复 | depth+width配合teacher/student蒸馏 | 同压缩大类 |
| [TARDIS，2025](https://arxiv.org/abs/2501.10054) | FFN近似和校准/残差恢复 | 分段线性化GELU+在线回退，UB1保留真实GELU子集及线性RR | 低秩恢复不是独有创新 |
| [Adaptive Time-Stepping，UAI2024](https://proceedings.mlr.press/v244/chen24c.html) | 时间轴资源问题 | 优化采样时间节点而非每步计算宽度 | **不同优化变量** |

**结论**：UB1步骤预算25/50/50/75本身是一个预注册固定实验点，不是“首次发现”或能证明优于其它已知调度的独立机制；C_mean支持、中心化一维拟合、R256辅助也属于已有静态剪枝/回归恢复方法族。项目独到的可复现对象是**真实Wan HiF4量化双expert少步算子、整160位置物理shape/FFN+生成成本、两批不相交实际视频质量与后续signed动力学资产**。实验对象的严谨性不自动转化为算法首创性。

## 6. 科研延续建议

把UB1列为强静态步骤预算基线，与C24、UB2、真正packed低位dense和resident dense比较。未来新贡献优先研究：信息何时可得、真实HiF4组scale闭包、联合gate×W2功能及其在下游真实normal-forward中传播的有符号作用；也可另从code-domain代数、真正tile-kernel或轻量后训练函数重组起步。任何算法性宣称都应与最强公开先例有**机制不同预测**，并在独立新来源以完整视频质量、FFN wall和生成wall共同验证。

本文只整理已有资产和公开文献；模型/CPU/GPU/训练/新视频执行权限、原始RUN及一次性交接不变。