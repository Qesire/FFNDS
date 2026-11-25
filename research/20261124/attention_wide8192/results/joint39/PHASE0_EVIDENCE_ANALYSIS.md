# 首步换基完成后：实际恢复类与任务结果

日期：2026-11-25。证据固定于 FFNDS c5e4ee8168708a8e57cdb981103c4dbf715e3bfa，作业 196471。本分析读取真实执行代码、校准矩阵、面板误差和逐转场运动数组，运行 NumPy 的 QR/SVD 与 SymPy 符号化简。没有运行生成模型、解码视频或加载学校归档的大型 trace 张量。

## 1. 最有价值的认识

**本次实验实际改变了恢复坐标，完整视频仍表现为原有运动收缩。由真实代码可以把下一层问题明确为：当前 OUT-ABS 是以 selected attention 输出作为 query 特征、由当前 P64 拟合的逐头共享仿射通道变换。首步换基调整的是这一恢复类内部的子空间与正则几何，没有改变 query 信息的来源。**

这使本轮成为一次有明确停止意义的辨析。阶段坐标已经按既定规则构造和检验，接下来值得研究的是昂贵 attention 中哪些位置关系和状态需要保留，以及如何让廉价计算与稀疏观测获得这些关系。原 C 的后续确认可以继续承担成片质量与实际收益的判断。

## 2. 实际执行符合单一阶段坐标变化

比较旧 attention_online_structured.py 与新 attention_online_structured_phase0.py，新增参数 out_current_basis 只在 step0 的 OUT-ABS 调用传入；step1 继续读取原 high fit。底座、P64 教师、lambda=0.1、原输出尺度、BF16 映射顺序、选中支持规则及 low OUT-D 均保留。

attention_phase0_cal.py 从每个原 CAL self/high 面板的 o_selected[0] 读取 step0 anchor，计算 raw64=selected@Pv。每源两个背景与四层共八个面板，三个来源共 24 个；按原 source/site 权重统计 RMS 与相关 Gram，再取 31 列。0392/1013 的 16 个 held 面板仅用于固定基之后的描述性评价。

完整运行记录为 160 个 attention 站点、40 次原生 attention anchor、120 次恢复、7680 个全键 probe query；wide8192 FFN 为 80 次建锚和 80 次命中。初始状态与旧 C0 一致，四步 flow/latent 及 MP4 指纹均与旧 C0 不同，证实新基实际进入了生成路径。指纹差异只检验身份，不度量误差大小。

## 3. 局部变化很小，运动后果持续

原始面板逐层能量经重新相除、按四层等权几何平均：

| 来源 | phase0/step1 基的 H448 误差比 | 变化 |
|---|---:|---:|
| 0392 | 0.99829457 | −0.1705% |
| 1013 | 0.99849016 | −0.1510% |

评分对象是原生背景首步面板的逐头输出，位于 W_O 与 self gate 之前。Dense 与 wide8192 在每个来源、每一层的两项能量均逐值相同，符合首步两背景均为原生 FFN 的路径。因此 16 个面板是两个来源、四层、两个背景的嵌套记录；本次该项评分只形成八组不同的来源/层数值。

0392 的 layer0 误差比为 1.01909，其余三层分别为 0.98481、0.99166、0.99795。1013 对应 1.01720、0.98736、0.99653、0.99312。这是很小且随层有正有负的变化，本轮未出现“大幅改善局部恢复而成片完全无效”的情形。

逐转场运动重新聚合得到：

| 0392 | C | C0 | C0-phase0CAL |
|---|---:|---:|---:|
| 全片灰度相邻帧变化 | 8.01730 | 1.37384 | 1.39143 |
| 后 20 转场灰度变化 | 10.11400 | 1.22233 | 1.23858 |
| 光流幅度代理 | 0.44182 | 0.03849 | 0.03888 |

新基相对旧 C0 的灰度变化增加 1.28%，光流代理增加 1.00%；相对 C，两项仍只有 17.36% 和 8.80%。仓库保存的用户判断是“仍然没有动作”。依据事先设定的停止条件，0392 检验后关闭首步换基分支，1013 完整视频与候选正式计时均未执行。

单条记录中的 114.269 秒属于带 trace 的首条质量生成，不与旧 warm 计时作为性能对照。此处 motion 文件以作业和报告对应视频，没有内嵌 MP4 哈希；本轮复算了其数组汇总，没有独立解码新 MP4。

## 4. 真实矩阵计算表明：这次基变换在几何上确有变化

已核对新基 safetensors SHA 与执行回执一致，也核对旧 self_high_transform.npz SHA 与原冻结 CAL 一致。

令新旧变换为 T0、T1，均为 64×31。以 QR 得到两列空间的正交基，再用 SVD 计算主角；同时计算 raw64 空间中 K=T T^T 的变化。

| 几何量 | 实际结果 |
|---|---:|
| 31 个主角的中位数 | 17.748° |
| 最大主角 | 83.792° |
| 平均 cos² 主角 | 0.753912 |
| ||T0T0ᵀ−T1T1ᵀ||F / ||T1T1ᵀ||F | 0.681418 |
| 通道 RMS 比值的最小/中位/最大 | 0.99013 / 1.03848 / 1.10943 |

这些量使用同一个固定 raw64 坐标中的欧氏度量；它们描述校准变换的几何，不是运动敏感度或实际输出误差。K 的计算也位于 BF16 量化和常数列之前。它们共同确认这次干预进行了实质的特征改变，而开放循环输出与成片运动对这一改变的响应都很小。

新旧各自相关 Gram 的前 31 方向分别保留其自身标准化特征能量的 62.50% 与 59.89%。这两个分母来自不同阶段，所描述的是 selected 特征方差；它们不表示原生剩余响应、任务信息或运动能量的保留率。

## 5. 用实际读出重写恢复类

固定一次调用及一个 head，query 行数 N=31668，输出通道 d=128。定义：

- B∈R^(N×128)：当前 selected attention 输出；
- Pv∈R^(128×64)：固定投影；
- T∈R^(64×31)：CAL 变换；
- X=[1, B Pv T]：实数域设计；
- 当前 P64 拟合的系数分块为 Θ=[b;W]，b∈R^(1×128)，W∈R^(31×128)。

则未覆盖 P 行的实数读出可精确写为

\[
\widehat A
=B+X\Theta
=B(I+P_vTW)+\mathbf 1b.
\]

已用 SymPy 按实际形状验证等式差为零。增量通道矩阵 PvTW 的秩至多 31。系数在每个 layer/head/current call 独立拟合，同一次调用中在 query 间共享；由于系数本身依赖当前 P64 的真实响应，整个输入到输出算法仍是条件自适应的。

实际 apply_transform 先对 B Pv T 作 BF16 舍入，因而更准确的连续值记法为

\[
\widehat A
=B(I+P_vTW)+\mathbf1b+
\bigl[Q_{\mathrm{BF16}}(BP_vT)-BP_vT\bigr]W,
\]

随后还有最终 dtype cast，以及 P64 的直接原生输出覆盖。这里没有将实际有限精度实现说成严格的仿射映射。由实际计算仍可得到一个精确的关系：对两个非 P query，若同一 head 的 selected 输出相同，则其特征、舍入值及预测也相同。

这条关系指出 high OUT-ABS 的信息接口：所有非 P query 的区别通过当前 B 进入恢复器。phase0 更换 T 会改变它保留的方向、拟合几何和当前系数，却不会引入 B 之外的新 query 特征。仅在这个恢复类里继续换坐标，面对的是同一信息来源。

low OUT-D 具有不同的底座：

\[
\widehat A_s
=A_a+(1-\rho_a)(O_s^S-O_a^S)+X_s\Theta_h.
\]

其原生锚点 A_a 携带每个 query 自身的历史完整响应，当前 selected 变化和 P64 观测用于更新。两者的差别落在明确的信息与计算过程上：绝对首步重建需要从当前有限信息形成响应，带锚更新可以继续利用完整位置状态。

## 6. 从这一轮可以推进的核心问题

这一轮支持收束“单独改变 phase0 CAL 能恢复 0392 运动”的具体构造，并把下一层问题写清楚：

1. 当前廉价输出 B 已经保留哪些 query 间区别，哪些完整 attention 作用需要其他可用信息？
2. 逐位置锚点、当前选中计算与 P64 观测分别承担什么作用，哪些关系由后续 FFN、残差与采样放大或保留？
3. 需要恢复的是单个 attention 输出、带缓存的阶段状态，还是对最终任务有用的某组响应？

这些问题与实际恢复类、观测接口和状态依赖对应。本轮没有加载全行原生/候选响应与最终任务导数，因而没有实测“0392 的运动方向位于该仿射族之外”。合适的肯定判断是：已经找到一个可精确描述的结构限制，且一次实质坐标改变没有改变当前失败类型。研究精力可以转向改变信息接口或计算对象，而原 C 保留作为完整模型中已经可执行的组合方案。

## 复算与来源

运行：

    python recompute_phase0_evidence.py --old-transform ../replay_evidence.zip

也可通过 --source-dir、--old-motion、--old-schedule 指定现有回执。脚本只依赖 NumPy/SymPy，输出 phase0_evidence_recomputed.json。旧 CAL 变换从 replay_evidence.zip 读取，不需要重新拟合或重新生成。

固定证据版本下读取：

- [phase0 运行与面板原始回执](https://github.com/Qesire/FFNDS/blob/c5e4ee8168708a8e57cdb981103c4dbf715e3bfa/research/20261124/attention_wide8192/results/joint39/PHASE0_CAL_RESULT_196471.json)
- [phase0 逐转场运动数组](https://github.com/Qesire/FFNDS/blob/c5e4ee8168708a8e57cdb981103c4dbf715e3bfa/research/20261124/attention_wide8192/results/joint39/PHASE0_CAL_MOTION_196471.json)
- [用户人审](https://github.com/Qesire/FFNDS/blob/c5e4ee8168708a8e57cdb981103c4dbf715e3bfa/research/20261124/attention_wide8192/results/joint39/PHASE0_CAL_HUMAN_REVIEW_20261125.json)
- [phase0 拟合与生成执行器](https://github.com/Qesire/FFNDS/blob/c5e4ee8168708a8e57cdb981103c4dbf715e3bfa/research/20261124/attention_wide8192/results/joint39/attention_phase0_cal.py)
- [阶段隔离的在线实现](https://github.com/Qesire/FFNDS/blob/c5e4ee8168708a8e57cdb981103c4dbf715e3bfa/research/20261124/attention_wide8192/results/joint39/attention_online_structured_phase0.py)
