# Chipmunk high-step1：独立学生轨迹观测与可辨识性设计

作者：`/root/chipmunk_student_discrimination_terra`。状态 **DESIGN_DRAFT / SRC / ALG / OPEN**；没有新实验、运行授权、独立支持票或因果结论。基于[原发现](chipmunk-979-competitive-function-units-20261030.md)与[主 Agent 独立评议](chipmunk-979-competitive-function-units-root-review-20261030.md)，不新增文献；L09/C20 的既有原文账本身份保持不变。此处是 Wan2.2 原 BF16 的 P64 迁移研究，不是作者 Wan2.1 配置复现。

## 关键判别与现有缺口

[VIDEO] 四个已曝光开发源 H/X 完整视频字节相同；979 候选/Dense 连续 RAFT 0.787582、MUSIQ −6.382，运动估计与画质差异提供问题，不能据此定位原因。H/X 相同也仅限定已测路径和条件。若同一学生状态下，实际 mask 在不同时间/空间区间遗漏的 W2 投影模式不同，我们可以分离 **公共 mask 的异质性** 和 **有符号投影组合**；但两者都可能共存，且局部误差不能识别 **后缀传播** 的因果贡献。

| 假说 | 预先规定的相反预测 | 需要的真实量和明确弱化条件 |
| --- | --- | --- |
| H1 局部异质性 | 固定同 K、真实学生输入、固定坐标分区，分区独立 mask 相比共享 mask 的 held-out 分区尾部门后误差下降；与同大小随机分区和共享 mask 拆分对照相比仍有额外下降。 | 真实 old/current h、W2、gate、实际 mask、独立评估行和分区坐标；若空间分组不优于随机组，或 held-out 改善消失，空间特异解释受削弱。组后验 oracle 仅是上界。 |
| H2 signed W2 cross | 固定实际 mask，遗漏残差向量的交叉项显著正/负且在分层评估行稳定；相同样本预算的 signed 动作改善 held-out 投影误差，超过随机符号及零补偿。 | 同一行 signed Δh、静态 W2、gate 及真实输出；仅交叉项大不证明可预测/可修复。若大交叉项但 signed 动作不胜对照，只支持投影相互作用，不支持该动作有效。 |
| H3 后缀传播 | 控制前缀、注入位置、方向及门后范数后，固定扰动在不同层/方向导致不同最终状态或视频；配对干预与单点响应和不同，提示有限交互。 | 稳定前缀、可固定扰动及多次真实后缀/视频。被动观测即使有误差与视频相关，也无法辨认因果放大、共同原因或多个层交互。 |

两个在线动作的胜负不能排他归因：分区内可能有 signed 组合；64 行漏采可以让两动作同时失败。Dense 和 student 的相同行号是相同网格坐标，**不是**相同内容或相同输入。分别计算真实各自状态及同状态反事实，禁止把跨轨迹输出差当作单次局部动作误差。

## SRC：真实 producer 接点和数值对象

[`chipmunk_input_video_driver.py`](../../../tools/chipmunk_input_video_driver.py) 对 31668 行用 CPU seed42 `randperm` 首64行，五臂每臂实际生成，step0/1 high、step2/3 low。[`chipmunk_input_video_tap.py`](../../../tools/chipmunk_input_video_tap.py) 以 `infer_ffn` 包裹事务和 `mm_call` 拦截；step0 保存同 block old h/Y，step1 在 W1 拦截时计算64行完整 W1/GELU RMS 并排序 K=5120，保存的仅为 mask SHA。随后全 M 执行 selected W1/GELU、delta BF16 舍入、FP32 W2 matmul、与 old Y FP32 相加并转 BF16；sentinel 使原 GELU/W2 分派不再产生当前 full h。`infer_ffn` 的 gate 参数及 `post_process` 写回还在原生路径，故必须获取原 gate 和确认实际写回时的 dtype/顺序。旧 y 含 W2 bias，delta 投影不再重复加 bias。[`readonly_action`](../../../tools/chipmunk_input_state_math.py) 是具体运算权威。

[`e0_chipmunk_wan_fullm_raster_capture.py`](../../../tools/e0_chipmunk_wan_fullm_raster_capture.py) 已演示在 `mm_call` 获得实际 V1/h/Y、在 `infer_ffn` 拿 gate、在 `post_process` 对象身份和 grid 校验；当前硬编码 source898/96、block20、四步、Dense，不能直接声称支持 student 或40层。新 producer 应新建并组合这些思路，不能直接复用原 tap 的 sentinel 当作 h，也不能修改正在使用的运行器。明确定义 live active W1/W2/bias 的 buffer 身份、stream、tensor stride、权重 SHA 与异步生命周期。

对同一行定义 `d = float(h_current) − float(h_old)`，遗漏列 `U`。理想投影 `r = (d_U W2_U) ⊙ gate` 的平方由对角项 `Σ_j d_j² ||gate ⊙ W2_j||²` 与跨列项 `Σ_{j≠k} d_j d_k ⟨gate ⊙ W2_j, gate ⊙ W2_k⟩` 构成。此恒等式是 ALG；实际误差以原生 BF16 当前 dense Y 和实际 candidate Y 相减、分别依原生写回算式加 gate/残差为准。还应报告理想投影与实际差，包含 BF16 delta rounding、FP32 accumulation、Y 舍入、bias 和 gate 写回的路径差；不能把理想 cross 直接当实际误差因果分解。

## 最小可执行的观测范围：40层、匹配坐标、额外评估行

先选 **979 与对照898，seed42，原 checkpoint/recipe，两个完整自由轨迹 Dense 与 high-only H student**。两者都是暴露开发源，仅用于机制诊断；不拟合、不阈值调参。冻结每个源图像/prompt/noise、checkpoint/backend、4步、高专家、grid `[21,29,52]` 和原始64行、code SHA、运行顺序和硬件预算。必要时扩到96/1040或新图像簇与新seed；单源×多行不构成独立来源。

每源、每臂记录全部40个 high step1 site：

1. 精确保存原64个决策行、step0 与 step1 的同坐标 post-GELU h、实际 mask **索引本体**（排序/sha）、sample score、gate、同坐标 old Y、actual candidate Y，以及原生 dense-reference Y 和必要的 V1。记录行号及 `(τ,y,x)`，`row=τ*29*52+y*52+x`。在 Dense 轨迹运行同一个 P64 observer，仅影子算 mask，绝不改变 Dense forward；学生轨迹读取真实原动作 mask。step0 原始 h 必须来自相同轨迹、同一 block，同步复制时不得改变 forward。
2. 原64用于选 mask，不能又作为无偏质量估计。额外预先固定与原64不重叠的 **256个观测行**，按21τ与空间均匀分层并保留抽样概率；这256行只用于评估，不参与 mask 或动作拟合。以源为独立单位，报告每层及分区/尾部、按采样概率权重的统计，256行无法保证覆盖局部极小物体；不足即增加行或做 full-M，不能把未知尾部说成已排除。分区/随机对照和拟议 signed 动作仅使用原64或各组内分得的原64；若将评估行用于选择，则它们失去 held-out 身份并需第三组行。
3. 在学生事务取得当前 V1 和旧 h/Y 后、真实动作完成且不替换其输出，另对256评估行调用完整 W1+b1、`gelu(tanh)`、完整 W2+b2，以参考同一 student 当前输入；64行的当前 W1/GELU已有 selector 输出，保存时确认可复用且未修改。记录真实 candidate Y、当前 gate 与必要 pre/post residual，复核 same mask/全部 K 和 no-action 控制。Dense 轨迹本身在同位置产生 current h/Y，可按行收集；对 step0 的 h/Y 存同坐标副本。不同轨迹的观测行完全匹配但各自基准分别计算。
4. 原生 full-M 对分区盲区最有价值：预先按层深冻结 block 0/13/26/39 四层，学生和 Dense 分别保留 old/current h 与 old/current Y；在学生端当前 full-M h/Y 需额外 full-M dense W1/GELU/W2 shadow，不得接回模型。还需保留实际 mask、gate、行映射和当前 candidate Y（若要 full-M 真实误差）或只保留预先冻结的分区统计；不能从256行断言 full-M 尾部。分层位置为设计选择而非已证明最敏感层，发现位置性异常时重新审查扩容。

此 producer 是 **新增设计**：运行器当前没有这些全部观测，也没有40层 Dense/student 配对；采样现有输出不能恢复 mask 索引、979 的全 h 或 student 当前 full h。`post_process` gate 可能广播，必须按真实 shape 校验，而不能预设每行独立 gate。

### 全40层流式统计的增强选项

每个学生 site 的 packed candidate Y 完成之后、旧缓存 release 与活动 W1/W2 buffer 换出之前，单独执行 full-M `W1(current V1)+b1 → GELU → W2+b2`，用原生规则构造门后同状态误差；直接按预先冻结的 `τ×空间` 分区归约总体、分位/尾部、候选与参考能量，保存摘要后释放 current h/Y，主路径仍返回原 candidate Y。这样只需四层持久化完整操作数，但 **40层都要支付一次 full-M Dense W1/W2 shadow**。要得到真实 signed cross，对每层另把 full-M `d_U` 投到 W2_U，按 gate 求投影总能量；对角项按 `d_U²` 与 `||gate ⊙ W2_j||²` 归约，二者之差为交叉项。门为全行共享时可先计算一次列权重；若 gate 逐行变化，须按实际广播定义计算，成本更高。该投影及对角运算是额外收费；只算 `candidate Y − dense Y` 并不能精确分离 cross。用分块流式实现可降低瞬时内存，但因 GEMM 形状及舍入改变必须与已冻结参考核对，不能假定数值相同。

H 路径已经保存 step0 完整 old h/Y，实际 current V1 在 W1 输入可得，因此 **H 状态的最少 shadow 不需要 old x 或重算 old W1**；X 路径只有 old x 和64行 h shadow，若选X则必须付费重算全M old h，不能从64行恢复。为了留有复算能力可在四个持久化层保存 old/current V1，但它们是增强资产，不是 H 路径计算全M统计的必要输入。Dense 轨迹的 step1 全M h/Y 本来就产生，可直接观察；如需同状态对应的实际 P64 输出，还须对 Dense 当前输入、同臂 old h/Y 另付费生成其 shadow mask 与 packed action。不可把 Dense 轨迹的 mask 当 student 实际 mask。

该增强将40层真正的总体与预设分区统计从样本估计变为实际全M观察；有限分区仍可能掩盖小对象，而这批统计依旧不能证明H3。全M 张量可以不全持久化，但临时显存、写出与统计、CUDA同步、转移、在线 observer 对路径的影响必须在新scope单列，既有512 MiB记录不构成新scope预算约束。

## 数据量、额外动作和干扰

BF16 每64行一份 h 为 1,769,472 B，一份 D=5120 向量为 655,360 B；每层保存两份 h、三份 D（例如 old Y、current dense Y、实际 candidate Y）、一份 gate 10,240 B、int64 mask 40,960 B，**64行共 5,556,224 B**，不含 V1、score、索引、metadata、额外写入和序列化。320行(64+256)约 27.7 MB/层、40层约1.11 GB/臂/源，仅是近似上限模型；gate/mask只需每层保存一次、若存 V1 或 postwrite 则更大。Dense 臂的 candidate Y 为其影子动作输出，需额外计算；亦可冻结不采集该量以节省预算但须修改对比定义。

full-M 一份 h 为 875,556,864 B，一份 D 为 324,280,320 B；每层 old/current h+old/current Y 为 **2,399,674,368 B**，四层约9.60 GB/臂/源；另存 candidate Y 每层再加324,280,320 B。实际全量 producer 要计入 dense reference GEMM、shadow GELU、D2H、同步、写文件、hash、重复读取验证和峰值显存；两源两臂总量、临时副本、文件格式及完整视频均另计。现有 H 路径已保留40层 old h/Y，采集时可以从活缓存按行读取，full-M 历史值需在 release 前复制；不可假定零额外峰值。

学生每层至少增加一次256行完整 W1、GELU、W2 参考（若重复64行则另计）；H2 的理想遗漏投影还需额外同列 W2 GEMM，交叉项对角统计需权重列范数及行级乘加。四个 full-M shadow 每层各增加一遍完整 W1/GELU/W2、临时 h/Y 和输出拷贝；同一 stream 排序、offload 权重缓冲在算完前不能被覆盖，D2H/CPU hash 和同步可改变时延及显存。观测计时绝不能报作原算法速度。

强 neutral：同 source/seed/checkpoint 的无 observer baseline、observer-on，逐状态 raw hash 和完整视频字节比对；若差异出现，先查同步和算子扰动，按预先冻结数值阈值复核，不自动宣称字节中性。64行 actual mask 与原运行索引 SHA、完整状态、old cache 身份也应逐层验证；未通过即停止解释。Dense 影子和 student full-M shadow 的候选输出需确保未接入真实 forward；`all-K` 可作为独立付费参考而不保证由于运算顺序与 Dense 字节相等。

## 本次判别的边界与下一动作

此采集能量化实际学生状态中 H1/H2 的行级条件差异，以及原64采样对256独立行/四层full-M的代表性；它无法单靠观测判定979的画质变化是否由任何层局部误差 **造成**，也不能区分下游敏感与不同源的误差方向共同变化。想检验 H3，下一独立scope需要冻结同一前缀和实际误差向量，在早/晚层用同 norm 不同方向及 neutral 作固定单点、配对和联合干预并完成后缀；逐层的当前误差本身会受前面学生动作影响，不能直接作为可自由移动、未混杂的干预。

推荐下一步：先把上述 **两源、40层、64决策+256评估、四层full-M** 构建为独立 producer 设计与资源账本，逐scope主审→真正独立 reviewer 反例→修订/recheck→ACTIVE冻结→身份复核，再真实 RUN/EXIT。任何 full-M 追加、后验动作或干预均需覆盖在经过独立审查的明确范围，不能继承已有 video/ABBA 权限。若先要更小 scope，可去掉full-M，但明确其无法排除漏采区并须依结果再审扩容。

最强反例：979 的罕见关键对象占少量行，原64及256评估行均未命中；四个全量层亦未覆盖真正敏感层。此时 H1/H2 观测均可看似平稳，而少量早层误差经后缀非线性传播造成视频差异。反向例子是各层都存在大 signed cross 或分区 oracle 改善，但真实后缀对这些方向不敏感；局部可修复性不能替代固定干预和多来源完整视频。
