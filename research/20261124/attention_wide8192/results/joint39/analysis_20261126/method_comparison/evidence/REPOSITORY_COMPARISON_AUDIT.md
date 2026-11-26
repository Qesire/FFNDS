# 仓库证据核查：Chipmunk、ICC 与当前 FFNDS 的得失

日期：2026-11-26。只读核对时 main 为 `13ffb127e71a5d4d1519ecb0df9c7b4d03efe601`，研究分支为 `154fe8620724bf1fbe6f0f9b6c5950e154468af4`。本轮新增执行是原始 JSON 的 CPU 复算与源码检查，没有新增模型调用。

## 1. 首先分清三代对照

| 证据 | 实际方法身份 | 可以回答的问题 |
|---|---|---|
| 10 月和 11 月上旬的 Global-P64、普通 SVD512/2R | FFNDS 的全局真实行支持选择变体；ICC 的早期分解/缓存变体 | 这些明确变体的行为与历史失效机制 |
| 11 月 17 日 strict DEV11 五臂 | 作者式 ICC-FFN CD r1408/4S；作者 Chipmunk FFN 安全端口；旧 Q1r2；Global-P64；同四步 Dense | 作者 FFN 机制经过明确 Wan 适配后，在四步目标域的开发结果 |
| 11 月 22–26 日 wide8192、完整 C、centered | 当前 FFN 预测＋当前 P64 输出恢复，进一步联合 attention；已有真实稳态计时 | 当前完整方案相对同四步 Dense 的有效收益 |

**11 月 17 日已经完成较完整的作者 FFN 模块复现和适配核验。** 静态设计文档仍保留当时的 NO_GPU 标头，应以实际完成报告、参数冻结和作业回执判定执行状态。它们的作用范围是 FFN；attention 原生。Chipmunk 原论文在 Wan 的完整视频系统使用 sparse attention，ICC 原图像系统也含 attention，因此仓库这份五臂表不能代表两篇论文整系统的相对速度或完整质量。

固定主来源：
- [完成报告](https://github.com/Qesire/FFNDS/blob/13ffb127e71a5d4d1519ecb0df9c7b4d03efe601/reports/20261117/ffnds_strict_reproduction/README.md)
- [逐源质量与成本](https://github.com/Qesire/FFNDS/blob/13ffb127e71a5d4d1519ecb0df9c7b4d03efe601/reports/20261117/ffnds_strict_reproduction/QUALITY_AND_COST.md)
- [原始结果 JSON](https://github.com/Qesire/FFNDS/blob/13ffb127e71a5d4d1519ecb0df9c7b4d03efe601/reports/20261117/ffnds_strict_reproduction/output/RESULTS.json)

## 2. 忠实性已经做到什么

### ICC

实际运行的主配置是 **CD r1408、四状态、专家内 A/H、全部40层**。原作者统计函数、缩放/截断/分解、四状态缓存、重置通过小型 oracle；真实 high/04 又执行六次作者分解和八个探针动作对照。强制 Dense 后，两来源逐步 flow、latent、解码结果与原生一致。

校准实际采用11个 PE Video真实片段，以其完整clean latent作加噪目标，首帧及文本另作条件。CA是fresh noised clean上的一次原生调用；CD是fresh noised起点→真实一步Euler→同专家第二次调用。完整保留四边界统计、先abs后token mean、逐样本通道min–max、来源等权、sqrt、1000尺度上限。原始校准的旧rollout替代已经纠正。CA/SVD/CD三族先在六站点局部比较，CD在11来源都优于CA和SVD，因此冻结CD进入全80处。

明确的域适配是：目标四步时间网格和双专家；Wan原生归一化mean latent替代DiT posterior；作者0.1条件dropout映射为真实T5空字符串，保留I2V条件；BF16推理及FP32统计。CAL3对posterior和文本条件做了配对敏感性诊断。校准 source 与旧Q1的CAL11并不配平；开发选择与视频评估同用DEV11。

[ICC作者校准合同](https://github.com/Qesire/FFNDS/blob/13ffb127e71a5d4d1519ecb0df9c7b4d03efe601/research/20261107/ffn_baseline_reproduction/strict_reproduction_20261116/ICC_AUTHOR_CALIBRATION.md)；[ICC参考定义](https://github.com/Qesire/FFNDS/blob/13ffb127e71a5d4d1519ecb0df9c7b4d03efe601/research/20261107/ffn_baseline_reproduction/strict_reproduction_20261116/ICC_STRICT_REFERENCE.md)。

### Chipmunk

采用作者通用SparseDiffMlp、compiled selector与官方Triton。保留128-token块均值preactivation代理、top_keys=.3、random=.05、period10、mask-cache=2、每专家前两层Dense，以及选中隐藏激活与输出的持续缓存更新。四步且每专家只有两次调用，落为8个fixed-dense位点＋76anchor＋76hit。

真实13824列对抗输入暴露selector合法计数/未写入问题；主臂使用经相同输入核验的安全修补，名为Chipmunk Author Safe。Wan适配另含专家/请求缓存隔离、时钟接口、voxel排列与逆排、52个尾token默认Dense、CUDA流双向事件及生命周期。尾52另有valid52三来源消融。生产selector补齐支持存在运行间变化，模块一致性检查使用同一实际mask区分选择随机性与算术。

[上游审计](https://github.com/Qesire/FFNDS/blob/13ffb127e71a5d4d1519ecb0df9c7b4d03efe601/research/20261107/ffn_baseline_reproduction/strict_reproduction_20261116/CHIPMUNK_UPSTREAM_AUDIT.md)。

### 原生attention的源码检查

`wan_comparison_job.py`只进入`bind_native_ffn`上下文；`wan_ffn_port.py`用AST定位pre-W1后的W1→GELU→W2部分并替换返回值，保留宿主前序归一化/调制和其他模块。没有在该五臂执行中装载sparse attention。实际forced-dense轨迹核验进一步验证了该边界。

[运行器](https://github.com/Qesire/FFNDS/blob/13ffb127e71a5d4d1519ecb0df9c7b4d03efe601/research/20261107/ffn_baseline_reproduction/strict_reproduction_20261116/wan_comparison_job.py)；[FFN接口](https://github.com/Qesire/FFNDS/blob/13ffb127e71a5d4d1519ecb0df9c7b4d03efe601/research/20261107/ffn_baseline_reproduction/strict_reproduction_20261116/wan_ffn_port.py)。

## 3. 同批质量证据：能肯定什么

495条原始逐视频指标经重新聚合，全部均值与源报告在1e-12内一致。对象为11个已暴露开发来源、seed42；数值如下：

| 历史FFN方法 | 主体一致性相对Dense差值 | 背景一致性相对Dense差值 | I2V主体相对Dense差值 | MUSIQ相对Dense差值 | 连续RAFT均值，px/两帧 |
|---|---:|---:|---:|---:|---:|
| Dense | 0 | 0 | 0 | 0 | 3.33085 |
| 作者式ICC CD r1408/4S | −0.09653 | −0.04196 | −0.04222 | −13.97915 | 1.54068 |
| 作者Chipmunk安全端口 | −0.04894 | −0.03265 | −0.04076 | −1.91349 | 1.75923 |
| 旧Q1r2 | −0.01607 | −0.00797 | −0.00641 | −1.20045 | 2.44379 |
| Global-P64变体 | −0.00796 | −0.00853 | −0.01447 | −0.11665 | 2.57620 |

归一化一致性指标的原始差值乘100才是百分点；MUSIQ保留原分。RAFT描述运动幅度，不能当作“越高越好”的质量分。ICC与Chip的主体一致性和美学都在11来源下降；旧Q1与Global-P64在该批若干维度更接近Dense，且两者互有取舍。

两源同输入局部MSE是ICC 0.008394、Q1 0.014400、Chip 0.018402、Global-P64 0.011273。**ICC的局部拟合优势和自由生成顺序确实不同**，说明研究最终目标应保留闭环轨迹、运动与多模块的共同作用。该结果给出了改用当前真输出约束的研究动机，但它并不孤立证明“在线校正”这一单因素的因果贡献。

当前wide8192＋attention的1.227×GPU、1.156×完整请求来自另一份六来源对Dense结果。历史表的Q1和Global-P64不是当前方案，来源也不同；不能将两张表拼成当前方案对Chip/ICC的正式胜负表。

用户在当时的对话反馈中还指出：Global-P64与全80在线读出在0392仍存在全画幅色块同步运动中的抖动和解离，虽比其他近似较轻。该观察与上表并存，是后续逐位置锚点与当前响应恢复发展的依据。当前视频获得正面反馈属于后续方法阶段，不能回填成旧Global-P64已经视觉通过。

## 4. 新增成本计算：三家的FFN主要算术预算其实接近

使用真实N=31668、d=5120、m=13824，原生每位点F=4Ndm FLOPs：

- 当前wide8192主要命中F_H=2Nd²+4Ndk，k=2560；完整周期比0.685185。纳入上轮已列P64与恢复GEMM为0.68903。
- ICC4S r1408主要命中F_H=4Nr(d+m)；完整周期比0.688426。
- Chip真实11片的完整块平均选中约4741–4765列，加入前两层全Dense、76anchor/76hit、52尾token与当前blockmean-W1代理，完整周期比 **0.690282–0.691082**。

ICC与Chip这一栏使用11月17日实际运行配置；wide8192一栏是11月22日后当前方法的静态账本。该近似预算对齐不把旧质量表升级为当前方法的匹配质量胜出。

Chip逐片精确统计计算式为：
\[
F_C=84F+4(128)d\sum_{\text{hit blocks}} K_b+76\cdot4(52)dm
+76\cdot2(247)dm,
\qquad r_C=F_C/(160F).
\]

这里第二项来自实际18772个hit完整块的selected列数之和，第三项是hit尾52，第四项是每hit247块均值的W1代理。该账本不包含非线性、统计、索引copy、权重packing及访存。它是一份可计算对照，不是等时延证明。

**有辨识力的结论是：当前四步内，三类FFN都在大约31%的周期主算量节省附近；方法差异主要在同类预算下获取什么信息、如何恢复变化，以及执行形态能兑现多少。** 我们的额外当前输出观测与恢复应以质量保留和预算分配为核心价值，不能仅依据稀疏列数较少来声称FFN算得更少。

## 5. 缓存与执行成本的真实得失

当前FFN缓存是u0、y0、selected z0k，共2d+k维状态；一个专家40层BF16约30.201 GiB。ICC四态需要2d+2m维，约89.395 GiB；同因子两残差变体约44.697 GiB，但锚点多执行低秩投影。Chip持有full hidden/output以及代理缓存；该严格端口38层单专家缓存序列化约42.635 GiB。前两项是静态有效张量预算，后者是实际序列化字节；都未计模型参数、工作区和当前方案attention附加状态。当前完整方案的attention状态需另加，不能将FFN缓存优势当作完整系统显存优势。

历史诊断墙钟中位数：ICC1445.36s、Chip440.30s、Q1225.11s、Global-P64387.37s、Dense38.70s（Dense只有9条该口径）。这些路径每步执行磁盘cache序列化、hash校验、同步与观测；例如ICC每片写/读各178.79GiB、Chip各85.27GiB，且FFN事件区间包含这些等待。它们证明该具体诊断部署的成本，不代表两篇论文原生GPU内核速度，也不能与当前34.8秒稳态请求直接比较。

当前方法已实现常驻状态、按探针拟合和规则矩阵分块执行；这一工程完成度是有效成果。公平的速度主张可以直接用现有对Dense增量，近邻之间的系统速度则需相同存储/计时边界下的原生执行对照。

## 6. 最有价值的比较对象与简洁表述

- Chipmunk：比较“局部代理选择哪些隐藏更新”与“结构预测后用当前原生输出校正尚未解释部分”；它的成熟稀疏内核与动态局部支持是应学习的优势。
- ICC：比较“两级低秩因子作用于真实增量”与“直接约束最终FFN/attention响应”；作者式校准后的强局部基线已经存在。
- 历史Global-P64：其P64用于估计隐藏激活并选择全局支持；当前P64用于观测原生模块输出、拟合恢复。二者同叫P64，实际作用不同。

现在已有证据可以支持：在相近FFN主算术预算上，跨步差分、全局当前观测和输出恢复提供不同的质量/资源交换；当前联合attention方案已经取得同四步蒸馏基线上的真实端到端增益。当前方案对Chipmunk整系统的优胜仍由同批完整系统对照定义。

## 复算与来源

- `recompute_strict_comparison.py`：重新聚合495条逐视频记录，并以真实selected计数计算Chip完整周期。
- `STRICT_NUMERIC_INPUT.json`：从原始RESULTS Git blob提取的分数、身份、缓存与工作量字段。
- `STRICT_COMPARISON_VERIFIED.json`：本轮运行输出。
- `SOURCE_IDENTITIES.json`：17个正文/源码文件逐一通过Git blob SHA-1核验。
- `research_tree.json`：研究分支7511个tree条目，truncated=false。

