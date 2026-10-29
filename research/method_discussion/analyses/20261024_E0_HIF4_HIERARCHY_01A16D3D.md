# E0 发现档案：HiF4 层级舍入与 FFN 输出方向

作者：codex-ffnds-method-01a16d3d  
日期：2026-10-24；固定 FFNDS 源：ff6942763d8ce3e1823a4814905ad5ee61da7c31  
方法来源：AutoResearcher 5bd7ec537ae63f2b6a6dd579f0e886aff58a177c  
阶段：OPEN_DISCOVERY / SOURCE_MECHANISM；本轮没有算法实现、数值实验或模型运行。

## 新发现与当前问题

**源码允许一种“数值集合不变、未移动通道的舍入结果却变化”的现象。** HiF4 的 64/8/4 层级把通道的邻接关系写进量化函数：一个数值跨越 4 元素子块，可以改变同伴的有效舍入间距。这个机制不是全块 RMS 或 amax 的函数，却仍然完全由更细的局部幅度及排列决定。

本轮给出与实际 hifx4 参数相符的手工反例：交换第 1、5 个数值，使第 2、6 个数值的解码结果分别变化 +1/4、−1/4。经 W2 后，差异为两列有效权重之差的 1/4；如果这两列相同，量化差异完全消失。**格式确实引入耦合，不等于这种耦合在真实模型中具有功能意义。**

新的科学问题是：真实 Wan 的层级舍入差异主要在 W2 中抵消，还是与其敏感方向有系统对齐？这个问题可以区分量化布局效应与 FFN 权重几何，尚不选择稀疏、缓存、重排或代理算法。

## 1. 按最新原则重新安放此前工作

直接依据本次 main 新增的 [方向复位](../../20261024/OPEN_DISCOVERY_DIRECTION_RESET_20261024.md) 与 [E0 第一任务](../../20261024/FIRST_TASK_E0_QUANTIZATION_FFN_DISCOVERY_20261024.md)。根 AGENTS 与 ACTIVE_RESEARCH 的总目标、Q1/Q2/Q3 耦合视角保持一致；这两个新文件明确了当前探索入口和阶段，没有替换研究权威。

| 前一轮材料 | 保留的结果 | 本次整理 |
|---|---|---|
| 本人 QCell tile 提案，PR9 | 条件量化组等价与共享构造；命中率和代价未知 | 作为历史种子保留，不把精确相等作为新探索前提 |
| G6C v1 评议，PR12 | 固定状态部分和恒等式、E2M1 轨道容量公式 | 保留限定 SUPPORT；撤回据此“优先推进码字轨道”的工作排序 |
| G6B v2 评议，PR12 | 完整状态稳定的编码等价、GELU 负瓣推导 | 保留限定 SUPPORT；不先寻找证书来决定研究对象 |
| G6A v2 评议，PR12 | Smooth 坐标、完整重编码误差需说明；固定平台上的线性折叠限制 | 保留具体修订意见；平台限制不否定量化响应关系，缺少实现本身也不是拒绝探索的理由 |

原提案、原评议和原版本不改写；此处记录的是本人的解释修正和新工作入口，没有增加支持票或替他人改变判断。PR9：https://github.com/Qesire/FFNDS/pull/9；PR12：https://github.com/Qesire/FFNDS/pull/12。

本轮选择 research-expansion 寻找未解释现象，problem-reformulation 的对称性表述分离布局与功能，mechanism-research 重建关键数值路径。交付对象是一份发现档案，不是新算子提案。

## 2. 信息—计算流地图

以列向量记号描述单 token，矩阵采用 W1=[13824,5120]、W2=[5120,13824]。实际运行可以采用行批张量；这里不改变轴意义。

\[
r=x+\mathrm{attention}(x),\quad
q_{\rm in}=\mathrm{AdaLN}(\mathrm{norm2}(r)),\quad
v=Q_{\rm H}(\mathrm{BF16}(q_{\rm in}/s_1)),
\]
\[
z=W_1v+b_1,\quad
h=\mathrm{GELU}_{\tanh}(z),\quad
u=\mathrm{BF16}(h/s_2),\quad a=Q_{\rm H}(u),
\]
\[
y=W_2\,\mathrm{BF16}(\alpha a)+b_2,\quad r_{\rm out}=r+g\odot y.
\]

alpha 是特定参考/历史动作中的后 Q2 gain；基准可取 1，不能把它误记为所有原生 dense FFN 的额外训练参数。各步的实际 cast、原地加法与 GEMM 仍按源码；上式仅表示依赖顺序。

| 位置 / 信息 | 产生者与消费者 | 时间、依赖与实际保存 |
|---|---|---|
| residual、attention、norm2/AdaLN | LightX2V infer_ffn 366–407；post_process 418–425 | residual 绕过 Q1；OP2 schema 没保存 pre-norm residual 或 attention increment |
| Smooth s1/s2、已 QDQ W1/W2、bias | WanPreparedHiF4Linear 初始化、forward 375–389、cache export 850–873 | 模型/离线缓存中的持久数据；Smooth scale 不是动态量化 scale |
| Q1 输入、输出 v | adapter 275–286 → hif4_backend → QTensor | 最后轴 5120，80 个完整 64 组；Smooth 后先 BF16，force_fp32 只改变量化内部精度，返回原 dtype |
| Q1 的 lv0/lv1/lv2 | CUDA hifx_quant_cuda_inner 229–286 | 每组 64/8/4 局部归约与临时变量；传回的是 QDQ 数值张量，没有独立导出的 scale/code 缓冲区 |
| z、GELU、Q2 前 u | W1、GELU tanh、Smooth2 | 13824 隐藏轴；Q2 后信息不能作为“尚未计算 W1 时已有”的信息 |
| Q2 的层级状态与 a | 同一 HiF4 kernel，216 个 64 组 | 组内共享状态、组间独立；a 供 BF16 GEMM 消费。当前 OP2 schema 没有 u/a 配对 |
| W2、gate、残差写回 | 已解码 BF16 权重、LightX2V post_process | gate 的 5120 输出轴不同于 Q2 的 13824 隐藏轴；当前组误差和下游视频价值不等同 |

由 adapter、backend、QTensor 和 CUDA 四层源共同确认：当前归档对象为 HiF4 QDQ → BF16 GEMM。source 中存在临时尺度不等于它已成为可免费读取的运行时字段。

时间标识还须用真实 step/expert/block/row/grid。R12 runtime 重建记录的配置坐标为 [1000,750,500,250]，对应已保存实际 timestep [1000,937.5,833.3333129882812,625]；不能将配置坐标直接当实际模型输入。OP2 的 sampled row IDs 不同，也不能将跨记录行号当作配对的时空位置。

**格式对照，仅是规范/文档信息：** MXFP4 使用 E2M1 与每 32 元素 E8M0 scale；TE 所述 NVFP4 使用每 16 元素 E4M3 与全局 FP32 factor。它们没有在本轮 FFNDS 原生数据中被观测，HiF4 的 4/8/64 结论不能移植。[OCP MX 规范](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)，[NVIDIA NVFP4 文档](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)。

## 3. 从源代码推导：层级状态是“分段精度分配”

backend 使用 QType("hifx4")。固定 QType 52–63 行给 man_bits=3；QTensor 62–64 行调用 CUDA 时传入 N=man_bits−1=2。**不能把 HiF4 名称直接解释为 E2M1。**

对有限、非 NaN 的一个 64 元素向量 u，设 A=max|u|、M8_l 为各 8 子块 max、M4_j 为各 4 子块 max。源码依次形成：

\[
c=R_{\rm BF16}(1/7),\quad
s=R_{\rm E6M2}(R_{\rm BF16}(\mathrm{clamp}(Ac,2^{-48},49152))),\quad
r_s=R_{\rm BF16}(1/s),
\]
\[
b_l=1+\mathbf1\{M8_l\,r_s\ge4\},\qquad
e_j=b_{\lfloor j/2\rfloor}\bigl(1+\mathbf1\{(M4_j/b_{\lfloor j/2\rfloor})r_s\ge2\}\bigr).
\]

这里子块下标取 0 起点。最终 N=2 的逐元素操作为：

\[
n_i=\min\!\left(7,\left\lfloor4((|u_i|/e_j)r_s)+\tfrac12\right\rfloor\right),\qquad
Q_{\rm H}(u)_i=\operatorname{sgn}(u_i)\,e_j(n_i/4)s.
\]

式子保留关键括号；一般数值不得随意改变乘除顺序。外部还有返回 BF16 的 cast。源码采用 floor(x+0.5) 及饱和，不能直接替换为 E2M1 nearest-even。

因此动态状态既包括全组幅度 s，又包括由局部相对 maxima 决定的 e_j∈{1,2,4}。它比单个全组 RMS/amax 更细；给定完整 4/8 maxima 及源码后又没有新增信息。所谓“量化信息优势”应具体限定为相对于什么可见统计、以什么读取代价，而非声称创造了信息。

### 3.1 用对称性把局部幅度与布局效应分开

若 Π 仅在同一 4 子块内部置换，所有共享状态不变，逐元素函数相同，因而 Q_H(Πu)=ΠQ_H(u)。整块交换同一 8 内的两个 4 子块、交换 64 内的完整 8 子块、交换完整 64 组也保留相应层级。结论针对同一数值路径的解码值；没有声称重排 GEMM 后 bitwise 相同。

跨 4 边界移动单个值一般破坏这个性质。定义分析对象：

\[
\delta_\Pi(u)=\Pi^{-1}Q_{\rm H}(\Pi u)-Q_{\rm H}(u).
\]

Π 保留全向量数值多重集合、RMS、amax 和直方图。若它只交换同一个 8 内两个 4 子块的元素，M8 和 s 都不变；差异只能来自这两个 4 子块的精度状态和舍入。影响限于这 8 个坐标，而不是整个 64 组。

这是数值函数的对称性分析。Π 不是建议部署的通道重排算法；没有修改 W1/W2 或重新量化权重。

### 3.2 一个实际 N=2 路径的手工反例

以下编号采用 1 起点。取合法 BF16 量化输入：

\[
u_{1:8}=(3,\tfrac58,0,0,\ 1,\tfrac58,0,0),\quad u_9=7,\quad u_{10:64}=0.
\]

这只是允许输入的源码反例，不声称真实 Wan 的 W1/GELU 已产生它。

A=7。由源码 c=73/512，Ac=511/512，BF16 的 nearest-even 步骤将其变成 1；随后 E6M2 与倒数仍为 1，所以 s=r_s=1。第一组 8 的最大值为 3，因此 b=1；两个 4 子块的 e 分别为 2、1。

| 未移动的元素 | 原值 | 原 e | 原 QDQ | 交换 u1 与 u5 后 e | 交换后 QDQ |
|---|---:|---:|---:|---:|---:|
| u2 | 5/8 | 2 | 1/2 | 1 | 3/4 |
| u6 | 5/8 | 1 | 3/4 | 2 | 1/2 |

3 和 1 本身都精确编码，其他值不变。故逆置换回原坐标后：

\[
\boxed{\delta_\Pi=\tfrac14(\epsilon_2-\epsilon_6),\qquad
\|\delta_\Pi\|_2^2=\tfrac18.}
\]

epsilon_i 为第 i 个标准基。以上所有中间值在这条计算中可逐项按二进制有理数核对；本轮没有运行 synthetic CPU demo、枚举或 CUDA 检查。

历史 R12 对组内 offsets 0/4 交换保存了 401 个 BF16 mismatch；它提供历史 Python counterpart 的存在反例。本次源推导更具体地定位了“全组 scale 不变，但子块精度改变”的原因，既不是新的真实分布测量，也不把 Python 记录当 CUDA 比特验证。

## 4. 同一量化变化为何可能没有功能意义？

令 B 为当前已经 QDQ、实际用于 W2 的有效权重矩阵；不对置换后的权重再次量化。在理想解码算术、alpha=1 下，把逆置换作为回到原坐标的分析操作，有：

\[
\Delta y=B\delta_\Pi=\tfrac14(B_{:,2}-B_{:,6}),\qquad
\Delta r=\tfrac14\operatorname{diag}(g)(B_{:,2}-B_{:,6}).
\]

这已经给出两种不同的解释：

- 若 B 的两列相同，非零量化差异完全抵消；
- 若两列相反且非零，差异加强；gate 又能改变它被写回的方向与幅度。

保持局部变化范数不变并不能确定功能变化。实际 BF16 GEMM、post-Q2 gain/cast 与后续传播还需要其自身数值路径；上述关系不冒充原生输出 bitwise 等式，也不推导视频是否可见。

这个联系与旧 G6C 的码字相等、G6B 的免算证书、G6F 的 K-panel 干预不同：这里没有指定省略动作，而是在辨别**固定原函数中布局耦合与有效输出几何的关系**。分组量化的排列依赖及线性投影本身是已知数学性质；本记录贡献是对冻结 HiF4 数据流的具体反例、影响域与待区分问题，不作论文新颖性主张。

## 5. 事实、竞争解释和反解释

| 证据层级 | 能说什么 | 不能说什么 |
|---|---|---|
| SOURCE_DERIVED | 动态 scale 未独立导出；N=2；4/8/64 层级；上述对称性与反例 | 真实 Wan 中的出现频率、读取 state 的实际代价 |
| ARCHIVED_REFERENCE | R12 保存完整组/行可分离检查及组内置换反例 | 新 CUDA 实测、MX/NV 原生分布 |
| CONDITIONAL_ALGEBRA | Bδ、gate Bδ 区分局部差异与功能方向 | BF16 bitwise 输出、完整视频质量 |
| ARCHIVED_NATIVE_SCALAR | C1 的写回/nextnorm 排序不完全一致；C0 同 q 反例有合成构造 | 此次层级舍入是否造成那些差异 |
| UNKNOWN | 真实 Q2 边界/精度状态与权重方向的联合分布 | 不可从旧 V1 cosine 或轨道容量补造 |

**解释 A：幅度压缩足以解释实际现象。** 全局统计不足以描述所有合法输入已经被反例证明，但真实输入可能很少跨子块精度边界；在匹配细粒度 maxima 后，层级标签不再提供有用区分。可观察分歧是：实际 u 的跨 4 重标号是否经常产生非零 delta，以及该差异是否只是局部幅度范围的函数。不能将 A 偷换成“量化一定毫无用处”。

**解释 B：布局耦合在实际输入中常见，但大多被 FFN 几何吸收。** 预测 delta 非零而 Bdelta / gated Bdelta 小；只数码字变化会高估意义。

**解释 C：布局耦合与有效权重方向相关。** 预测在相近 delta 范数、相近全组幅度的情况下，原坐标的 W2 列关系会系统地区分写回方向。这个结构仍可能被后续 norm/attention 吸收；不能直接提升为最终视频敏感性。

A/B/C 可以在不同 step/expert/block 上分别占优，也不必立即合成算法。C0 合成残差反例与 C1 历史标量差异仅提醒存在更远的因果层，不为 C 提供直接支持。四步蒸馏或时空结构是否改变这种对齐，仍是未回答的比较问题。

## 6. 原始资产审查：缺口位于哪里

历史 OP2 完整性记录确认 3,840 payload；C0 的原始 header audit 给出的唯一 schema 为：

- V1、BaseY、TeacherY、StudentY：各 [256,5120] BF16；
- b2：[5120] BF16；gate：[1,5120] BF16。

当前 capture 源的 values 构造也只持久化这些字段。norm_input、teacher hidden 等中间值曾在生产过程中存在，不代表已落盘。V1 是 post-Q1，不能反演出原始 pre-Q1 值与完整舍入状态；更不能代替 Q2 前后的 u/a。

本轮只读取 Git 原始文本、header 审查记录和源码，没有访问或声称本环境持有 HF 的完整 tensor bytes。OFFLINE_SAVED_ASSET_QUANT_STRUCTURE_R1 也明确未读取原始 payload。因此现有可读资料足以证明存在机制，**不足以区分真实 Wan 的 A/B/C**。

## 7. 唯一下一步建议

**寻找或请求一个真实 Q2 的最小配对切片，回答“层级舍入差异是否沿 W2 有效方向保留下来”。**

先查已有归档是否存在完整原顺序 64 组的 Q2 输入 u（Smooth2 后 BF16）和实际 QDQ 输出 a，并绑定其已 QDQ W2 的对应 64 列、gate、source/step/expert/block/row/grid、源码与 Smooth/gain 身份。如果没有，仅将缺失字段作为后续研究请求；本轮不采集、不运行模型、不创建作业。

一条记录的最低量级是 1 source × 1 原生位置 × 1 row × 1 完整 64 组。u/a 共 256 bytes BF16；对应 B[5120,64] 为 0.625 MiB、gate 为 10 KiB，可从已验证权重资产复用。此量级只能确认机制实例，不能估计分布、源间普遍性或视频因果效应。

该记录支持一项后续对照观察：在保持数值集合不变的坐标重标号中，区分保层级置换与跨 4 边界置换，观察 delta 以及同一固定 B/g 下的方向响应。保层级对照应为零；跨边界若非零而投影消失，则转向 B；若方向性保留，则深入权重几何，再考虑传播。这里不把这个观察设计包装成新算法、评分器或实验启动请求。

## 8. 冻结来源与可复核位置

所有 FFNDS 路径均相对于 ff6942763d8ce3e1823a4814905ad5ee61da7c31，blob 为 Git blob SHA，另列的运行时 SHA256 不与 blob 混用。

| 原始来源 | blob / 固定提交 | 本轮用途 |
|---|---|---|
| tools/packed_ffn_reference.py | 28653fd6098b03e8a5ea92083d8b95620590c730 | Smooth/QDQ/GELU/alpha 次序；不是任意候选的基准实现 |
| upstream_audit/source_deltas/HiF4_Sp/files/HiFloat4/ptq/adapters/wan_i2v.py | 8e8b04e544610003fac6f21095dfe1c9cc66ae13 | 275–286、325–338、375–389、850–873 |
| upstream_audit/source_deltas/Wan2.2-I2V-A14B-W4A4/files/quant/hif4_backend.py | ee4bb583adb5f1a449abbfd90e0090c3a6136e33 | hifx4、force_fp32、返回 float QDQ |
| upstream_audit/source_deltas/HiF4_Sp/files/HiFloat4/hif4_gpu/quant_cy/base/cusrc/hifxg_quant_cuda.cu | 031b7fb0065517ca9c8532f4d73f250735a8eb5d | 186–286；归档运行时 SHA256 07120a5d2519e3ed3c8782acb8897a44e31103f93dd84d353ae7bb86b043e00c |
| alvinshao0313/HiF4_Sp：QType.py、QTensor.py | commit 9aa2d3be6de265910731aa7c0d053e9c773cf556；blobs e02ff5cd1476f0e39eaa0c6586daf575db0b13b5 / a98adea3c0ab07fc3a77028dd301c488eaadc2fd | N=2 的调用链、内部 FP32、返回原 dtype |
| ModelTC/LightX2V：lightx2v/models/networks/wan/infer/transformer_infer.py | commit ed146db64d0b6130c669285fc1aa0b280a6a4823；blob 6fa4614d517b8ee5c8da07b476f8204cd5d03150 | 366–425 的 norm、FFN、gate/residual；不把其他 mxfp8 分支当目标路径 |
| research/20261022/R12_HiF4_group_separability.json | 2d6c74b6754c72e3c4724c73134f02468c7a2e31 | 历史存在反例及其 Python scope |
| research/20261023/C0_coupling_offline_results.json | d24d695c56fbdc58050c152073637d3b3e63fa5a | 原始 source hash 绑定、OP2 header schema |
| research/20261024/C1_boundary_target_audit.json | 1cf5b7b8e245aaf5cf19bd59e342f3c4871c6cfb | 写回与 nextnorm 目标边界 |
| tools/native_teacher_capture.py | 129b7d0eca997aa01df83eae776f6319d92bb5c4 | 当前持久化字段 170–192；历史 schema 以 C0 header audit 为准 |
| reports/worker_20261020/OP2A_complete_dataset_integrity_RUN.json | f4ec90f87c3e884495d7aceb8128090bbb6eabe4 | 原 payload 完整性记录，不冒称本轮读取 |
| research/20261024/OFFLINE_SAVED_ASSET_QUANT_STRUCTURE_R1.json | 8174b4370d97ac174e574745b81cc0c7383689b5 | 既有离线统计的实际可达资产边界 |

根/局部 AGENTS、OPEN_DISCOVERY 及 E0 指令均已读取。没有修改共享 QUEUE、endorsements、ACTIVE_RESEARCH、原提案/评议或实验授权。既有有限离线恢复记录不被解释为本轮模型/GPU执行许可；本次 E0 按最新任务只做源码、数学和记录整理。
