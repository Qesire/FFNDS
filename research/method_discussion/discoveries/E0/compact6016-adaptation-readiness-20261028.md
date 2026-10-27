# A：6016窄FFN恢复的最小执行准备

**独立方法提案 / SRC+ALG / 尚未登记执行**。Whole94 提供一份已经走过真实缩窄 Q2 路径的强初始化：选94个完整64通道组后保留原 W1 与 b1、Smooth1 和对应 Smooth2；输出 W2 与 bias 是用4个历史来源拟合得到的BF16 readout。新问题是：在同一训练预算与真实学生Q2下，这个初始化能否比普通6016学生获得更好的来源外恢复？相同成本的随机/另一合法组初始化、相同参数更新和步骤是竞争对照；如果差距消失，支持的是训练预算而非原组选组的独有价值。

## 可复用身份与真实量化边界

固定源 [bank协议](../../../../config/e0_reconstruction_bank_20261026.json)、[bank结果](../../../../research/20261026/E0_RECONSTRUCTION_BANK.json)、[构建代码](../../../../tools/build_e0_reconstruction_bank.py) 与 [原生hook](../../../../tools/e0_reconstructed_runtime.py)：两个专家各40函数；high在step0/1、low在step2/3，历史4来源4/8/16/20各两个专家step×32行，总256行/函数，基于组方差和gate加权W2范数选择94组。W1是原QDQ权重选行后BF16 `[6016,5120]`，b1 `[6016]`选行；v是原W2选列加闭式ridge修正后BF16 `[5120,6016]`，bias `[5120]`也经拟合，并非原W2/b2直接切片；s1原尺度 `[1,5120]`、s2对应选列 `[1,6016]` 为FP32。新readout是BF16拟合值，未重新HiF4权重量化。历史bank raw locator 为 `e0_function_basis_20261026/prototype_banks_attempt1/whole/`，逐pack SHA在结果中；当前本地常用HF根该locator不可见，必须在实际worker确认原始80 pack存在并逐SHA核验，不能据文档声称已实物预检。

运行时先 `BF16(x.float()/s1)`→原native Q1→BF16 W1+b1→GELU(tanh)→`BF16(hidden.float()/selected_s2)`→原native HiF4 Q2→BF16 W2+bias；调用者负责原gate和残差。由于组为完整64，选组次序重排仍保持每组64边界，该固定模型具有自己的实际窄Q2调用与选组尺度；但**拟合更新后的学生**不能自动继承量化一致性。须冻结scale沿用或重估规则、参数更新何时合并及BF16舍入，实际执行学生Q2并记录码/尺度、对原钩子及全M/QDQ数值测试。QDQ的训练梯度策略（例如STE）的实现与偏差尚未确定，不能把推理hook当现成可微训练器。

## 历史捕获与缺失资产

旧 CAL24 四位置的一个实际文件 header 和 [采集源码](../../../../tools/collect_ffn_operands.py) 显示 `V1 [256,5120]` 是实际post-Q1、`V2 [256,13824]` 是GELU后的**post-Q2**（不是未量化post-GELU）、`Y [256,5120]` 是W2+b2 pre-gate，`gate [1,5120]`、`b2 [5120]`；五者均BF16，具有原source/seed/row/位置身份。24×256是来源内采样，不是6144独立来源；完整范围和文件SHA按旧scope逐一核验。teacher V2不能当6016学生post-GELU或其实际Q2输出，更不能当学生自由运行状态。Q1之前modulated状态、pre-Q2 GELU、学生Q2码/scale/梯度、6016学生自己的输出及rollout都须单独生产或确认实际现存。

[on-policy teacher capture](../../../../tools/native_teacher_capture.py) 有保存真实学生输入V1、同状态完整teacher Y、StudentY/BaseY、gate/b2及来源行的机制；它服务既有UB1轨迹和其producer，不自动覆盖新6016学生。能复用的是已有精确UB1版本和已保存行，新的6016状态、相同坐标teacher监督及完整学生轨迹均 **NOT_PRODUCED**。原CAL24为历史已暴露校准；所需新heldout image clusters需在训练/模型选择前排除并冻结，不能把原视频开发来源充作盲确认。

## 参数与保守资源账本

单专家block两个主矩阵共 `2×5120×6016 = 61,603,840` 参数，80函数共4,928,307,200；BF16主矩阵约117.5 MiB/函数、9.18 GiB/80函数，bias/scale、对齐和加载开销另计。W1、W2均加rank32 LoRA，每函数 `2×32×(5120+6016)=712,704` 参数，80函数57,016,320；未合并BF16 adapter约1.36 MiB/函数、108.75 MiB/80函数。若adapter的权重、梯度和Adam两矩全FP32，训练持久状态约10.875 MiB/函数、0.850 GiB/80函数；不包含master base、optimizer额外buffer、QDQ实现、梯度检查点、模型或通信。合并后须重新计算及验证BF16权重与Q2行为，不能将unmerged/merged速度视为相同。

每次M256前向仅一个FP32 `[256,6016]` 中间量约5.875 MiB、`[256,5120]`约5 MiB；反向通常同时持有多份输入、GELU/QDQ、输出及梯度，加上约117.5 MiB冻结BF16基底、10.875 MiB LoRA状态、FP32临时与BLAS workspace，建议首个四位置串行以**每位置至少2 GiB实际峰值上限候选**做预检，另加加载、碎片和上下文余量；这不是测得上界，需真实峰值和runtime失败处理决定预算。主投影前向为每token61,603,840 MAC；训练反向、LoRA两路和fake-QDQ皆额外计入，不可仅用推理MAC估训练总成本。

## 最小四位置判别

固定 `(step,block)=(0,0),(0,39),(3,0),(3,39)`，以24个旧CAL来源的真实同坐标V1、Y与gate构造source-balanced局部监督；先按来源冻结训练/heldout划分，不按分数挑样本。先做未更新Whole94与同宽随机/另一合法whole64选组的原生Q2基线，继而相同rank32、可训练范围、优化器、步数、行数、种子、量化梯度及实际算力预算下适配。保留原Dense和历史Affine54/UB1作为不同成本及训练历史参照；匹配训练的普通6016学生才是强对照。单独比较是否使用原拟合W2而非纯选列初始化，避免将历史ridge预算误归为组选择收益。

预先冻结heldout按source等权gated output误差、尾部和真实学生Q2状态；训练集局部改善若无法在来源外保持，预测是初始化/teacher-state过拟合。下一阶段另收同noise学生轨迹与完整视频行为及实际全FFN成本，区分teacher-state逼近与自由运行分布转移；本四位置试验本身不证明视频或部署收益。该稿未训练、注册、下载或执行模型。
