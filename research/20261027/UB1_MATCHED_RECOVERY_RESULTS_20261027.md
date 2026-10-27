# UB1 匹配来源恢复：支持与消费者的探索性交叉验证

**已测现象（NATIVE-OBS，保存状态）**：24个历史已暴露来源、4个嵌套位置的来源均权 held-out 统计中，C-mean 支持的 scalar+R256 相对 SSE 均值为0.004701，改成 ridge consumer+R256 为0.002864；FLAP式支持对应为0.005113→0.002776。四位置里 step0/block39 的 ridge+R256 两个支持均略差于 scalar+R256，其余三个位置均更好。该现象为消费者重构提供明确的竞争解释，也提示不同位置可能需要不同恢复机制；它不是完整视频质量、全 M 数值或速度结果。

## 对象、身份和执行

每源每位置256个 teacher-saved 行，来源是24个已用于既有研究的校准来源，偶/奇来源分为互不重叠的12训练/12 held，各折各含6 subject 和6 background；两个方向交叉后每个来源在每个位置恰作为 held 一次。4位置为 step0/3 × block0/39，**96条 source×位置是24个来源的嵌套观测，不是96个独立来源**。每个来源 index、source_group、source_id 和实际校准 seed 见[机器结果](UB1_MATCHED_RECOVERY_RESULTS_20261027.json)，已与冻结来源身份逐条核对。

冻结协议 SHA256 `a07cf64e62633aeedfa0f18ebe456504acaaa0875d64b67a1c5c4fedfa86f70c`；原 RUN SHA256 `d21216e83a3566499062dd7327e18d5147aa4ad8ceb25d2acf5463fd1f2513e3`；launcher 收据 SHA256 `6dbea4029800a7469eda47f3f3c093184f24533fc366334d9de2da334fb46fdf`，计算尝试退出0。独立只读复核12个源码快照和123项HF资产 SHA/大小均匹配，另权重镜像由执行前检查、未在本次独立复核重新哈希。计数为32次 R256 拟合、16次 ridge consumer 拟合和2304次 BF16 prediction linear；还发生4次 group weight Gram、8次 activation Gram、192次 FP32 scalar signal 和96次 FP32 dense linear。拟合包含32次 Cholesky和随机 SVD，ridge另有16次 Cholesky，故2304不是全部 CUDA 运算账本。无 W1、QDQ、全模型、视频或 optimizer step。GPU监测10次采样、无违规，记录峰值6156 MiB；原 RUN 计算耗时28.996秒。

此前一次 launcher 在资产准备完成和授权登记之前被 `authorize` 拒绝，退出1，未创建 ATTEMPT、未导入 Torch、无 GPU 计算或拟合；该事件保留在 ACTIVE 的 `pre_execution_guard_events` 和计算 launcher 收据中。随后登记的冻结协议才产生上述一次计算尝试。

## 配对结果和函数解释

| 支持与consumer | held main 均值相对 SSE | held main+R256 均值相对 SSE |
|---|---:|---:|
| C-mean scalar | 0.0240170 | 0.0047007 |
| C-mean ridge | 0.0028717 | 0.0028645 |
| FLAP式 scalar | 0.0203440 | 0.0051130 |
| FLAP式 ridge | 0.0027835 | 0.0027764 |

均值先按每个 source×位置记录相对 SSE 等权汇总，每源恰有4位置，因此等价于先对4位置取平均再对24来源等权。两类 ridge 后，R256 的平均增益约为0.0000072和0.0000070相对 SSE；它在这些训练和评价条件下很小，并非对所有数据或其他拟合结构的证明。ridge consumer 围绕选中 W2 的先验，以固定0.01迹比例岭惩罚拟合并折叠 BF16权重和偏置；它本身不按 gate 加权求解，gate 用于支持、scalar、R256及最终评价。FLAP式打分是 pooled variance × gated W2 对角能量的组级适配，不是完整 FLAP 实现。

同源同位置比较，ridge+R256 比 scalar+R256 更低的 C-mean 有75/96、FLAP式有77/96；每来源先对4位置平均后为24/24来源。FLAP式 ridge 相对 C-mean ridge 为78/96位置来源对更低、22/24来源均值更低。这些方向计数是描述性，位置嵌套、两折共享训练数据，不是独立显著性检验。每 site/fold 的 main、combined 和逐来源统计保存在 JSON；两支持的重合：step0/block0 两折34/54、35/54，step0/block39 均43/54；step3/block0 为147/162、145/162，step3/block39 为144/162、141/162。支持重合受到保留比例影响，不能单独解释最终视频质量。

step0/block39 是明确反例：C-mean scalar/ridge 的两个折 combined SSE 分别为0.003838/0.004228与0.003655/0.004040；FLAP式为0.003804/0.004153与0.003683/0.003950。相反，step0/block0 两折 C-mean scalar→ridge 为0.013062→0.005834、0.012410→0.005881。训练来源、位置、目标与固定岭系数共同作用，尚不能从该实验识别哪一个原因主导。

## 独立审阅与下一判别

源码确认支持选择只使用训练折；固定整64组54/162，采用实际 gate 的 group Gram 或 pooled variance；每个 arm 都独立拟合自身 R256，held 按来源独立 M256 BF16 前向并计算 gate 后 SSE。JSON 保留每折支持、alpha、consumer诊断、主支路及组合的逐来源指标和来源均权统计。原始 Q2/hidden 作为已保存 teacher 状态，故该研究未验证学生自由生成状态、原 M kernel 数值、完整 FFN 延迟或视频质量。

下一项有辨识力的检验是在预先冻结的新来源和同一实际输入状态下，保留原 M 条件及真实 W1/GELU/Q2，对同支持的 scalar/R256 与 ridge/R256 做配对；同时明确 ridge 的额外消费者权重、拟合成本、物理 GEMM 与 gate/残差后的误差。step0/block39 的反例应作为竞争预测保留，不按本次分数挑选新确认来源；最终须由完整自由视频和端到端真实成本独立评价。
