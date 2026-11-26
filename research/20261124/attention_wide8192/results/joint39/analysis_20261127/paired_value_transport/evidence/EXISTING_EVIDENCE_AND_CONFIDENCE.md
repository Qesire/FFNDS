# 对 SparsePR v2 比较的现有实证与可计算范围

日期：2026-11-27。核对研究分支 head 为 `89fd1ec46a65e0a53bd0be85edca28b8625a42f0`。本轮只做真实归档数据的CPU复算；未执行模型、GPU或SparsePR作者实现。

## 1. 最有价值的现有证据

从归档 `replay_evidence.zip/site_metrics.csv` 的4608行中提取192条相关记录，重新比较11/24的 OUT-D 与 OUT-ABS。两者均为原冻结CAL的32列恢复器，共用同一稀疏支持、固定P64和448个留出query。

OUT-ABS用当步selected输出特征恢复当步dense−selected残差；OUT-D使用逐位置历史底座及selected输出的跨步特征。它们同时改变了底座与特征，因此这组对照测量的是二者组合。OUT-ABS是当前项目的受限CAL32估计器，不是SparsePR v2的全128通道、当步标准化、coverage加权估计器。

以下为 DEV 0392、0800、1013，每来源4层的结果。先在各来源内对四个逐站误差平方和之比取log平均，再对三个来源等权平均，最终取指数。

| 背景/专家 | attention head | W_O＋gate点误差 | 空间邻边 | 时间邻边 |
|---|---:|---:|---:|---:|
| wide8192 / low | **0.3464** | **0.3835** | **0.4537** | **0.3929** |
| wide8192 / high | 1.1245 | 1.4958 | 1.5648 | 1.6127 |
| Dense / low | 0.7691 | 0.9428 | 1.2420 | 1.0563 |
| Dense / high | 1.6180 | 2.1864 | 2.3009 | 2.3915 |

wide8192/low四项均12/12站改善，三个来源的点误差比为0.3556、0.4250、0.3731。该结果支持把历史机制的主要希望放在low专家，且说明已有wide8192轨迹背景对这种比较很重要。high更适合当前响应恢复，与现在的专家分工一致。

这些是平方误差能量之比，不是总池化能量比、相对L2比或视频质量分。0.3835对应的几何L2比约0.6193。point位于W_O和self gate之后、外层残差与后续FFN之前；空间/时间指标分别为224条H内邻边在同一读出后的误差差分能量。12站嵌套在3个反复使用的开发来源中。

## 2. 表示能力与probe估计有两个独立读数

同一CAL32函数族在全部H448标签上进行无正则拟合，得到的head表示下限比为：

| 背景/专家 | OUT-D / OUT-ABS 的H448同特征表示下限 |
|---|---:|
| wide8192 / low | **0.3449** |
| wide8192 / high | 1.1549 |
| Dense / low | 0.7343 |
| Dense / high | 1.6515 |

wide8192/low的表示下限与P64误差比相近，支持“历史底座与差分字典改变了当前任务的可表示性”这一解释。该下限是在旧的两个32列函数族中计算，不能据此推断完整128通道SparsePR函数族的下限。

## 3. 与真正full-S估计器的条件追平阈值

在这些相同旧面板、支持、P64和度量上，full-S当前响应估计器若要追平旧OUT-D，需要相对CAL32 OUT-ABS达到：

| 度量 | full-S / CAL32 OUT-ABS 所需误差比 | 所需平方误差降低 |
|---|---:|---:|
| head | ≤0.3464 | ≥65.36% |
| W_O＋gate点误差 | ≤0.3835 | ≥61.65% |
| 空间邻边 | ≤0.4537 | ≥54.63% |
| 时间邻边 | ≤0.3929 | ≥60.71% |

这是可用于下一次实际重放的条件阈值，提供了低噪方向较有希望的具体理由。full-S实际能缩小多少误差尚未测得。原论文改变分组、路由、probe、coverage和标准化后会改变比较条件，完整SparsePR系统的胜负需要另行评价。

旧OUT-D不是当前centered。11/24固定面板的轨迹背景与201500的固定C轨迹不同，上述比值不能再乘以新centered相对raw的改善比例。

## 4. 当前centered有真实方向证据，但幅度调参空间较小

从作业201500原始Gram独立计算：八个固定站点attention误差均下降，两个来源low0下降约38%；同输入Dense块出口七处改善。一处1013/low26的attention比为0.9672，Dense块出口比为1.0200，直接说明下游复合状态仍需单独评价。

令两种已测响应的误差为e_0、e_1，d=e_1−e_0，可由Gram计算离线直线上的最优幅度

\[
\eta_*=-\langle e_0,d\rangle/\|d\|^2.
\]

八个attention站点的eta_*均为正，约0.683–1.280；在该已测方向上，选择oracle幅度相对已执行eta=1最多仅能再省约1.30个基准误差百分点。这里是两个观测输出场的后验插值，不是修改attention后重新执行40层的真实轨迹。

该结果支持已经采用的运输方向，同时提示进一步机制应改变所能解释的变化或保留的信息，而非主要依赖调节一个全局幅度。Dense块出口的方向作用可以不同，例如1013/low26的后验最优幅度约0.110；它不是可直接部署的层系数，也不能由attention误差单独选定。

## 5. 现有资产能否立刻重算SparsePR v2？

当前本地/仓库可读资产包含：CAL变换、P_v投影、指标CSV和若干小Gram。没有找到完整P/H的当前128通道S和原生A数组，不能从31/64维投影或2×2/3×3 Gram恢复它们。本轮没有把受限OUT-ABS当作已执行的SparsePR v2结果。

但原始采集格式已经存储所需输出：`attention_panel_fit.py` 的 `Assets.prepare` 直接读取 `o_native`、`o_selected`，然后才用P_v投影。原始轻面板含两个时刻的完整40-head×512-query×128-channel输出；`CAPTURE_AUDIT.json`记录192个面板、总10,147,109,376字节，以及索引/协议哈希。

因此可以复用学校现存面板，进行**相同S/P下的SparsePR v2恢复器重放**，无需重新生成整片，也无需重新计算全部attention。最小输入为：完整S_P/H、A_P/H、probe/held索引、用于声明覆盖权重的group信息；若比较历史，还需要S_a/A_a及相应状态统计。point/邻边评价再读取W_O、当前gate和固定边索引。其全系统响应分组和probe选择不能仅由这些输出数组恢复。

原始文件通过面板根目录的 `RESULTS.sha256`、index中的 `sidecar` 和 `tensor_file` 定位。本轮无学校文件系统访问，未验证这些外部路径的当前可读性。这里识别的是可复用的数据契约，不是新的运行收据。

## 6. 信心应如何表达

对low专家、wide8192轨迹背景、相同稀疏支持与probe的机制比较，可以给出**中等偏积极**的研究判断：历史方案已有跨三个开发来源的较大、一致的留出改善，且同特征表示诊断方向一致。

高噪段的现有证据支持当前响应回归。对完整SparsePR v2在同质量下的端到端速度/显存胜出，现有局部数据仍未覆盖必要分母：原生锚点、完整历史状态、动态分组与执行内核的成本均要加入。

## 原始身份与复算

- 数据来源：[replay_evidence.zip](https://github.com/Qesire/FFNDS/blob/89fd1ec46a65e0a53bd0be85edca28b8625a42f0/research/20261124/attention_wide8192/results/replay_evidence.zip)，Git blob `9cdc01b79c483ef8abb39cdf3a2780dd890e617a`。
- CSV成员SHA256：`5ab4e8d518cd8a0ef7b35b01f63b538f6f24bc0d3bc1a4ef9559391d5b5973c9`。
- 原始代码：[attention_panel_fit.py](https://github.com/Qesire/FFNDS/blob/89fd1ec46a65e0a53bd0be85edca28b8625a42f0/research/20261124/attention_wide8192/results/attention_panel_fit.py)，`Assets.prepare`、`metrics`、`evaluate_site`定义输入与指标。
- 当前centered原始数据：`MASS_CENTERED_201500.json.gz`，Git blob `79c2a06b1f7e54720c16f774d065c9ef8abee8cf`，路径与解压哈希保存在输入JSON。
- 本目录：`EVIDENCE_INPUT.json`、`recompute_confidence.py`、`EVIDENCE_VERIFIED.json`。

运行 `python recompute_confidence.py` 可重算全部表格和方向诊断。
