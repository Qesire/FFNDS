# Wan FFN 基线：完整质量、运动为何部分保留，以及与 UB1/S2 的真实计算差异

**状态**：2026-10-30 科学分析/既有已验证实验的只读综合，不是新GPU/模型/人为视频验收。输入基准 pinned at Git commit `ba67e76d251ec45d60353218de1762fe50ba22ac`；资产见同目录 [876条原生源SHA保留登记](WAN_FFN_CROSS_METHOD_ASSET_PRESERVATION_MANIFEST_20261030.json)。最新用户范围：只看 FFN W1/GELU/W2 的真实省算；不能用运动幅值代替完整质量。该登记**只保存既有HF文件路径+SHA及RUN身份，不是把MP4/权重备份进Git**，现有HF文件是否仍存在尚未独立查验，必须在可访问原集群时复制/校验。已添加[只读SHA校验与不覆盖备份工具](../../tools/verify_preserve_wan_ffn_comparison_assets.py)，尚未在原HF节点上运行。

## 1. 方法不能混淆的共同对象

Wan2.2 I2V A14B 四个denoise steps，high/high/low/low两独立expert、每expert40 block，共每视频**160 FFN位置**。每位置（简化列向量约定）
\[
x\in\mathbb R^{M\times5120},\quad
h_t=\mathrm{GELU}_{\tanh}(x_t W_1+b_1)\in\mathbb R^{M\times13824},\quad
Y_t=h_t W_2+b_2,\quad
u_t=g_t\odot Y_t,
\]
随后加回残差并经过下游注意力、FFN和flow更新。门控\(g_t\)和student输入\(x_t\)随视频、step及上游误差变化；同初始noise ≠ 同中间student状态。这里的 \(t\) 是去噪调用；**视频81帧/latent21个时间槽不是四个denoise steps**。

各实验使用数值/hardware不同：UB1是HiF4 fake-QDQ+BF16并带R256；S2和Chipmunk/TEAL为原BF16（S2 SM120与新A100设备配对各以自己的Dense）；SharQ是实际SM120 native sparse FP4，与原BF16/同权重Dense32都有对照。**跨cohort的均值只是内部质量缺陷的参照，不是方法优劣的公平排名**。

## 2. 全八轴质量：保存全部，不以RAFT或Motion Smoothness冒充运动质量

单位：MUSIQ Imaging是0–100，其他轴按原评分；dynamic_degree为二值在来源上的平均；RAFT是stride-2相邻视频帧对top5%光流幅值，**不是主体/相机运动分解，也不是第九个官方指标**。

### 2.1 UB1 consumer：20个曝光来源、单seed（HiF4数值家族）

| 维度 | 同cohort fresh Dense | UB1 consumer | 候选−Dense |
|---|---:|---:|---:|
| Imaging | 72.629 | 71.626 | -1.003 |
| Aesthetic | 0.6162 | 0.6160 | -0.0002 |
| Motion Smoothness | 0.9854 | 0.9847 | -0.0007 |
| Dynamic Degree | 0.300 | 0.000 | -0.300 |
| Subject Consistency | 0.9602 | 0.9809 | +0.0207 |
| Background Consistency | 0.9700 | 0.9774 | +0.0074 |
| I2V Subject | 0.9822 | 0.9870 | +0.0048 |
| I2V Background | 0.9872 | 0.9874 | +0.0002 |
| 连续RAFT（原始幅值均值） | 6.894 | 1.152 | -5.742 |

I2V两轴**各只有10个相应来源**，不能当成20个来源的共同八轴均值；来源簇bootstrap见 [UB1 20源质量与区间](../20261027/UB1_CONSUMER_DISJOINT20_QUALITY_RESULTS_20261027.md)。这种“主体一致性上升而dynamic_degree归零、RAFT显著下降”的组合说明**一致性可以奖励静止/重复外观，不能证明运动语义得到保留**。

### 2.2 S2 fixed-rank 全W2及前后10层Dense消融：12个曝光来源、单seed（BF16 SM120）

| 维度 | 同cohort Dense | 全W2低秩 | 前10层Dense | 后10层Dense |
|---|---:|---:|---:|---:|
| Imaging | 66.462285 | 61.006727 | 62.072089 | 61.561194 |
| Aesthetic | 0.612033 | 0.546368 | 0.560533 | 0.551470 |
| Motion Smoothness | 0.988193 | 0.986546 | 0.986583 | 0.985422 |
| Dynamic Degree | 0.416667 | 0 | 0 | 0 |
| Subject Consistency | 0.964272 | 0.967304 | 0.972992 | 0.963414 |
| Background Consistency | 0.963250 | 0.961875 | 0.964505 | 0.960027 |
| I2V Subject | 0.983892 | 0.955930 | 0.965623 | 0.956614 |
| I2V Background | 0.988024 | 0.971199 | 0.977272 | 0.972025 |
| 来源等权RAFT/Dense | 1 | 0.258195 | 0.275486 | 0.289902 |

来源979保留后10层Dense时RAFT/Dense约**0.929**，但动作时间和终态仍偏离原Dense，不能将幅值恢复认作事件重现。[48视频八轴/RAFT](S2_NEW12_SM120_FIXED_DEPTH_QUALITY_RESULTS_20261029.md) 记录整个cohort、完整81帧缩图及独立审阅。20源的 value10/dcac10 还有 [不同样本的八轴](RAW_BF16_TEMPORAL_VALUE_DCAC_VIDEO20_QUALITY_RESULTS_20261029.md)：imaging 73.720→71.319/70.366，aesthetic .625→.570/.562，dynamic .35→.10/.05，I2V subject .984→.965/.958，RAFT平均幅值比 .451/.439，亦无质量验收通过。

### 2.3 Chipmunk global P64：两个曝光来源、完整四步视频，High/Low 2×2对照

| 维度 | NONE=原Dense | HIGH_ONLY | LOW_ONLY | BOTH=P64 |
|---|---:|---:|---:|---:|
| Imaging | 74.868816 | 75.821042 | 73.066913 | 73.810414 |
| Aesthetic | 0.595626 | 0.600305 | 0.582189 | 0.591506 |
| Motion Smoothness | 0.977357 | 0.976338 | 0.978120 | 0.977541 |
| Dynamic Degree | 0.5 | 0.5 | 0.5 | 0.5 |
| Subject Consistency | 0.939142 | 0.937893 | 0.923771 | 0.925259 |
| Background Consistency | 0.956964 | 0.956805 | 0.956195 | 0.955237 |
| I2V Subject | 0.972233 | 0.973884 | 0.962528 | 0.964359 |
| I2V Background | 0.974181 | 0.972958 | 0.965253 | 0.967403 |

所有均值按**source898与96各权重1/2**。BOTH相对Dense八项分别是 [-1.058402, -0.004120, +0.000184, 0, -0.013883, -0.001727, -0.007873, -0.006778]。**High-only的少量指标改善不是全方向通过**：在来源898其主体一致性下降0.00325、motion_smoothness下降0.00153；仅来源96的imaging上升1.912，来源898几乎不变。Low-only在两源均引起imaging下降（-2.120、-1.483），而且 I2V subject/background 及主体一致性总体更明显下降。

| 来源（同Dense） | HIGH_ONLY RAFT比 | LOW_ONLY RAFT比 | BOTH RAFT比 |
|---|---:|---:|---:|
| 898 鸟 | 1.0155 | 0.9588 | 0.9503 |
| 96 河景 | 0.9751 | 0.8262 | 0.8257 |

BOTH mean-of-source-ratios 0.888；该值不是全部视频中的实际主体动作保真率。AI缩图见 [P64四臂视觉](CHIPMUNK_P64_FACTORIAL_AI_VISUAL_REVIEW_20261030.md)：鸟翼变化仍在，BOTH的帧48–65可见边缘软化、拖曳。实验实际生成与八项评分均EXIT0，但**运动事件级、相机路径、flickering、人类视频验收、official每条件5样本均未测**。源 [四臂全质量/RAFT](CHIPMUNK_P64_FACTORIAL_QUALITY_RESULTS_20261030.md)、[两源全160视频和原生FFN成本](CHIPMUNK_GLOBAL_P64_V2_ALL160_VIDEO_RESULTS_20261030.md)。

**扩展的4来源 High-only H-history对照（898/96/1040/979）**给出完整八轴来源等权对照：
| 维度 | Dense | High-only P64 | 变化 |
|---|---:|---:|---:|
| Imaging | 67.190556 | 65.791571 | -1.398985 |
| Aesthetic | 0.580626 | 0.568103 | -0.012523 |
| Motion Smoothness | 0.984361 | 0.984231 | -0.000130 |
| Dynamic Degree | 0.250 | 0.250 | 0 |
| Subject Consistency | 0.943743 | 0.942958 | -0.000784 |
| Background Consistency | 0.949081 | 0.948476 | -0.000606 |
| I2V Subject | 0.977330 | 0.978803 | +0.001473 |
| I2V Background | 0.976706 | 0.978689 | +0.001984 |

来源级 High-only RAFT/Dense：898=1.0155、96=0.9751、1040=1.1318、**979=0.7876**。来源979 MUSIQ下降6.382、美学下降.04546，却主体一致性略上升.00323。这是“运动幅值、主体一致性与感知质量不同行”的强反例。H-history和从X重算old h的两套候选在四源的**40状态+4视频逐字节相同**，只是不同缓存表示，不能当8个独立候选视频样本。参见 [4来源完整质量](CHIPMUNK_INPUT_VIDEO_QUALITY_RESULTS_20261030.md)、[H/X逐字节恒等](CHIPMUNK_INPUT_VIDEO_H_X_BYTE_EQUIVALENCE_20261030.md)。

**只复用先前完整Y（HIGH_Y_REUSE）是重要负对照**：两源RAFT/Dense为0.851/0.920，且鸟帧46–63出现重叠半透明轮廓；相比之下HIGH_P64分别约1.015/0.975。它支持“局部增量更新确实发挥作用”，但没有证明哪个通道/输出方向负责真实运动。[Y-only八项/RAFT/视觉](CHIPMUNK_HIGH_Y_REUSE_QUALITY_RESULTS_20261030.md)。

### 2.4 SharQ：两曝光来源、BF16、same-W Dense32与真实SharQ

| 维度 | BF16 Dense | same-W Dense32 | SharQ |
|---|---:|---:|---:|
| Imaging | 58.126929 | 57.599310 | 58.342499 |
| Aesthetic | 0.553718 | 0.552100 | 0.543414 |
| Motion Smoothness | 0.993525 | 0.993317 | 0.993425 |
| Dynamic Degree | 0 | 0 | 0 |
| Subject Consistency | 0.956427 | 0.946367 | 0.949923 |
| Background Consistency | 0.949609 | 0.936876 | 0.939801 |
| I2V Subject | 0.984083 | 0.983328 | 0.983412 |
| I2V Background | 0.986710 | 0.985142 | 0.983817 |
| RAFT原始两源等权值 | 4.75653 | 4.32132 | 4.38873 |

SharQ总体**imaging高于BF16但aesthetic低于BF16**；相对same-W Dense32，subject/background略好但I2V background略差，**不是全维改进**。来源1040的主体/背景优于Dense32，来源979的主体/背景低于Dense32。录像缩图显示1040两潜水员均游动但姿态/遮挡分叉，979的双手落下时间偏移；不能把动态程度全部为0解读成无运动，也不能把RAFT回升解读成语义轨迹准确。

[两源完整八轴与RAFT](SHARQ_WAN_ALL160_VIDEO_V2_QUALITY_RESULTS_20261030.md) · [作者核双路径的真实全160时间](SHARQ_WAN_ALL160_VIDEO_V2_NATIVE_RESULTS_20261030.md) · [逐帧接触表](SHARQ_WAN_V2_AI_VISUAL_REVIEW_20261030.md)。

### 2.5 TEAL：四曝光来源，BF16，前后激活计算边界明确

TEAL q25/q50 的四来源等权 **candidate−fresh Dense** 完整八轴：
| 轴 | q25−Dense | q50−Dense |
|---|---:|---:|
| Imaging | +0.151683 | +2.070802 |
| Aesthetic | +0.002033 | +0.001517 |
| Motion Smoothness | +0.000128 | -0.001643 |
| Dynamic Degree | 0 | +0.50 |
| Subject Consistency | -0.004428 | -0.015232 |
| Background Consistency | -0.001863 | -0.010392 |
| I2V Subject | -0.001704 | -0.011499 |
| I2V Background | +0.000582 | -0.008381 |
| RAFT来源等权比例 | 1.170 | 1.376 |

q50 source979新增大量彩色碎片，可能**人为提高全帧RAFT及binary dynamic_degree**，不能判定保持了原动作。此路径完整W1→GELU及完整Dense W2；只在h上置零，**没有实测物理算力收益**。两源真实M16组column-union仅理论省0.87% (q25) 或2.71%(q50)，全部测试K16/K64块无全零块，物理省算空间很有限。[视频八轴](../20261029/TEAL_WAN_ALL160_FOUR_SOURCE_QUALITY_RESULTS_20261029.md) · [大M组可达性](TEAL_WAN_FULLM_ROW_GROUP_SUPPORT_RESULTS_20261030.md)。

**质量共同边界**：上述八轴均为原函数的有限自定义调用。camera_motion、temporal_flickering、真实事件/对象分割轨迹、人类连续视频评分与官方独立seed足量评估未覆盖；不要把“八轴”标成完整九/更多指标全部合格。所有来源均已开发曝光；不能当盲测泛化。

## 3. 为什么Chipmunk **可能**在运动上优于UB1/S2：从计算图重建，而非RAFT归因

在同一专家（high:step0→1；low:step2→3），每个Block先于首次调用保存旧 `h_0=GELU(x_0W1+b1)` 与 **旧的完整 `Y_0=h_0 W2+b2`**。在第二次调用，先用64行当前输入计算全部H=13824 hidden的代理变化统计，选取**全局K=5120通道**，实际只为这些通道重新执行当前W1→GELU→W2 delta：

\[
\hat Y_1 = Y_0 + \bigl(h_{1,S}-h_{0,S}\bigr)W_{2,S},\qquad \hat u_1 = g_1\odot\hat Y_1.
\]

Dense恒等式为\(Y_1=Y_0+(h_1-h_0)W_2\)。因此**在忽略数值舍入、缓存身份及门控之前**，P64的遗漏项明确是

\[
Y_1-\hat Y_1 = (h_{1,\bar S}-h_{0,\bar S})W_{2,\bar S}.
\]

其行为效应又乘当前门控并沿完整下游模型和flow更新传播。**关键不是“只剩37%神经元仍能生成运动”，而是“旧的完整FFN输出一直保留，只有新增变化的一部分被省略”**。P64保留的老输出中已经有来自全部H通道的内容；W2投影仍保留有符号方向的旧贡献。**40层深度中的每层都经过真实student新输入、当前gate和其它Dense attention**；没有把整个视频框架冻结。新近的High-only、Y-only和H/X字节等价对照与此计算图相容。

### 3.1 同样说“50%稀疏”，物理计算对象完全不同

| FFN动作 | 当前step是否完整重算W1/GELU? | 当前step的W2作用 | 留存来自何时的函数信息？ | 被遗漏/压缩的函数 |
|---|---|---|---|---|
| **UB1** static HiF4组裁剪，25/50/50/75%，R256 | **仅固定选中组**；其它组当前h从未求出 | 仅固定组、再线性R256预测修复 | **没有旧完整Y时间缓存**；R256根据当前V1预测偏差 | 每step当前未选组的非线性输出、跨组有符号协同、下游时空作用 |
| **S2** 固定低秩W2(r960) | **完整原W1与GELU** | `W2→低秩替代`，所有位置有效值被投到较小输出函数 | 不复用完整旧Y，所缺方向持续失去 | 当前h在W2值方向上的高秩残余、符号/相位功能 |
| **Chipmunk P64** | t0/t2完整；t1/t3 **64行proxy全通道 + 选中K5120通道真实W1** | t0/t2完整W2；t1/t3 old完整Y + 选中\(\Delta h\)的W2 | **同expert上一完整FFN h和Y** | 当前未选通道的变化\(\Delta h_{\bar S}W2_{\bar S}\)，而非其旧值本身 |
| **SharQ 4:8 + Dense FP4残差** | 每一步都有FP4稀疏主通路**和**密集残差读出（W1/W2均需核实实现） | 所有通道信息仍通过full-K dense residual部分进入结果 | **同step当前输入**，不存前一步Y | 主要是低位表示与分解/稀疏数值近似误差，而非强制舍弃一半所有W2输出方向 |
| **TEAL q25/q50** | W1/GELU完全支付 | 将h的元素置零后**仍全Dense W2** | 当前输入，无完整Y跨步复用 | 当前小幅h的W2贡献；高元素稀疏不自动形成整tile可跳计算 |
| **RT-Lynx启发2:4** | W1/GELU完全支付 | 物理2:4稀疏W2 + 行范数补偿 | 当前输入 | 当前被删掉h×W2贡献，Norm标量难以恢复向量方向 |

### 3.2 预算和条件化是两个独立解释变量

**Chipmunk P64不是与UB1相同预算的公平对照。** 在四步160个FFN站点中，P64对于每个expert第一次调用，完整支付Dense W1/W2；第二次只选K=5120/13824=37.037%隐藏宽度。仅从两次主W1/W2乘加的*理想比例*考虑，四步的平均宽度比例为\((1+0.37037+1+0.37037)/4\approx68.52\%\)，**另计两次64行×13824全宽代理、缓存写读、h/Y克隆、Gather、index、gate和数值路径**。原UB1的主宽度比例约50%，外加rank256等补偿。比例不同意味着运动保留的差距可能只是**更多实际执行与历史完整Y支撑**，而不等于P64 selector更懂运动。

P64首次采用**当前student输入**上的少量行测 \(\Delta h\) RMS，mask随视频来源、step、block和本次student轨迹变；UB1根据校准时的离线完整h/W2/g贡献挑固定原64组，推理时不重新识别当下场景变化。P64更可能覆盖当前强变化方向，但其RMS并未看W2有符号投影和gate，故不保证覆盖对视频语义/动作真正重要的低能量方向。选中K=5120已大于原UB1第一步3456宽度，但不是相同动作位置的公平比较。

### 3.3 现有真正反事实的解释力

1. **HIGH_P64 vs HIGH_Y_REUSE**：同样旧完整Y、高阶段处；加上本次稀疏\(\Delta h\)后，相比单纯Y复用，两来源RAFT幅值更接近Dense，鸟边缘鬼影较轻，支持“补上当前变化是必要功能之一”。但还不能证明**保留的是正确主体动作**；需对象运动与语义事件观察。
2. **HIGH_ONLY vs LOW_ONLY (898,96)**：当前两源低阶段动作带来更大MUSIQ/I2V/RAFT下降，提示少步低噪FFN的当前增量更难缓存替代。**979 high-only也显著降成像6.38/RAFT至78.8%**，排除了“High必然安全”的泛化结论。High/Low在Wan是噪声范围的两套expert，不得与其它论文semantic vs detail双专家概念等号。
3. **H-history vs X-old-input**：相同student全40层四源候选输出40组状态和4条视频**逐字节一致**，但历史逻辑状态约44.70→24.23GiB（另有重新计算成本）。这是历史表示等价的工程证据，**不是对Dense函数或质量的改进**。
4. **逐step误差积累**：P64两来源末尾latent relative SSE来源等权从step1约0.00099、step2约0.00728至step3约0.09479；误差能量主要分布在空间AC成分（末步≈98%）。这种分解说明“静态外观仍可识别而空间/时序细节与末态明显偏差”的现实可能，并不等同直观运动，也不能由单个SSE直接判定主体运动。
5. **SharQ vs same-W Dense32**：SharQ的质量部分提高并非时间缓存带来，而是当前dense残差投影使FFN在量化后输出更接近某些BF16函数方向，同时成本更高。真实两源每视频FFN CUDA事件约5.74s vs Dense32 4.16s，同格式不存在目前可证明的净收益。
6. **已执行额外 RT-Lynx启发 2:4**：A100实际大M重复形状，top2 row-norm后 gate L2仍≈.243–.269，51–52ms vs Dense36–37ms；说明二值结构合法省算与完整函数动态保真之间有实测缺口，不是结构化稀疏天然更优。

### 3.4 三种竞争机制与可证伪预测（尚未验证成定律）

- **H1 历史完整Y是最大保护源**：旧所有通道分量让静态与部分历史动态得以延续。预测：删除旧Y但固定相同本次K、mask、当前gate会大幅改变视频；与Y-only比较还需要等成本/输入对照。
- **H2 当前变化选择对动作区域更有效**：P64聚焦较大的\(\Delta h\)，为主要动态区域做当前计算。预测：固定K、成本、已知token邻域下，用真正当前\(\Delta h W2\)或对象条件事件贡献排序优于RMS；并且*对象/镜头独立动作*而非仅全帧光流改善。需严格paid oracle与可计算前置量区分。
- **H3 主要由宽度/step分配和低噪末步敏感性解释**：未必selector懂运动，而是step0/2完整计算且总体68.5%主乘加较宽。预测：与同数值家族、同step位置、同MAC/访存预算的静态宽度及无缓存强基线相比，P64优势可能缩小甚至消失。
- **H4 有符号输出/空间局部函数遗漏造成拖影**：即使\(\Delta h\)能量proxy高，门控后\(\Delta h_{\bar S} W_{2,\bar S}\)可能在物体边缘/摄像机运动相关方向突出。预测：在真实学生前缀全40层、全部四步采 signed residual和gate后真实写回并按对象运动ROI分层，低能量的高影响方向与明显边缘/动作异常相关；已存在fullM signed对照仅证明单位置有界输出改善，尚无终态验证。

**最短且高信息增益的下一动作**：复用本次已保留的4来源完整Dense、P64、High-only、Low-only、HIGH_Y_REUSE及SharQ视频，对标注事件段落和相机背景做完整81帧联合语义/轨迹/闪烁审阅，**先不新生成视频**；然后独立评议注册“同预算、同source、同数值路径、P64历史Y vs 普通静态宽度、当前高低expert固定动作”的有限全160生成，以区分H1/H2/H3。历史P64内部fullM已经有后验raster分组同K有效改进，但那是用全部当前h的**付费oracle**，不能直接当在线可用方法。

## 4. 资产保全和下一Agent读取要求

**必须保留**同source的原始输入图/提示词、原权重指纹、model/recipe、BF16 Dense及neutral、各种候选完整MP4、初始noise、四步flow/latent状态、FFA及W1/GELU/W2/有符号gate观测、完整逐帧40组RAFT、八维逐视频原值、contact sheets及视觉评论、kernel时间/系统总成本、独立review、RUN/EXIT/旧失败身份。零平均图像和视频截图不能替代原MP4。

- [876项SHA绑定manifest](WAN_FFN_CROSS_METHOD_ASSET_PRESERVATION_MANIFEST_20261030.json) 从**现成六份原始生产者/质量文件**抽取 HF-relative path/sha/bytes，包含Chipmunk主线及因果四臂、SharQ、TEAL、旧UB1 20来源、H/X状态与RUN/EXIT；**GitHub不存HF二进制**。只保留哈希索引尚无法证明物理备份，须由可接触原HF的Agent先核全部文件实际存在并复制至不会被工作目录清理的位置；用相同SHA逐字节复验。
- 旧失败也作为资产：SharQ V6 stream/cache非有限值及修复诊断 [V6 FAIL](SHARQ_WAN_ALL160_V6_CACHE_NEUTRAL_FAILURE_20261030.md)、[SM120安全流诊断](SHARQ_SM120_STREAM_DIAGNOSTIC_RESULTS_20261030.md)；Chipmunk V7 allocator门限 [V7 FAIL](CHIPMUNK_GLOBAL_P64_V7_FAILURE_20261030.md)；TEAL V2失败 [V2 FAIL](../20261029/TEAL_WAN_ALL160_VIDEO_V2_RESOURCE_FAILURES_20261029.md)。**原运行资产原地不删除、不覆盖、不重命名，不把reviewed失败改称成功。14份S2/UB1/当前关键质量及失败报告另有Git blob SHA冻结，保留旧格式以供之后再解析。\n- 在确有HF源文件与独立归档盘的节点，先运行 `python tools/verify_preserve_wan_ffn_comparison_assets.py --mode inventory --role video` 检查登记范围，再 `python tools/verify_preserve_wan_ffn_comparison_assets.py --mode verify --hf-root /home/xiang/.cache/huggingface --role video` 只读验SHA；真实复制时明确指定单个 `--family`、已经存在的 `--archive-root` 和 `--mode copy`，程序不覆盖已存在不同内容的目标。**这只是可执行接口，尚未实际执行或验证物理备份。****
- 同一来源的不同方法若训练/精度/模型/M/step/seed不一致，只能作为机制解释背景，不自动构成控制变量一致的正式横向指标排名；标记数值族+GPU+model identity。
- 最新发现是**有条件的运动与内容保持机制**，不是“证明动态稀疏普遍保留运动”。后续每个实验在视觉和八轴外，增加\(camera\_motion\)（可运行时）、temporal flicker、主体动作事件、相机分解和局部写回；获得前这几列应明确为`NOT_MEASURED`。

**本报告为重用已有实测后的推导与文档，不是新增独立同行支持，也未执行任何GPU/CPU新实验。** 
