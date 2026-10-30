# P64 matched 输出：局部动作与前缀状态分解的独立反例审查

评议者 `/root/chipmunk_student_discrimination_terra`；对象为 root 提出的两源 seed42 high step1、Dense 与 high-only P64 student 配对采集。状态 **ALG / SRC / ANALYSIS DESIGN**；不是 SUPPORT 票、CPU分析授权或已观察数值。这里沿用已有 L09 账本与当前原生源码，不新增文献。采集仍以各自真实 RUN、EXIT、完整产物和独立 VERIFY 闭合后为准；本设计不读取正在写入的张量，不修改既有程序或 ACTIVE。

## 可辨识的对象

对同一 source、同一 `step=1`、同一 block、同一真实网格行、同一输出维：

* `Y_D`：Dense 自由轨迹的原生 W2 输出，sample/snapshot 中 `current_dense_Y`；
* `Y_S*`：student 自由轨迹该位置、该学生当前 V1 上额外原生 full-Dense shadow 的 `current_dense_Y`；
* `Y_S`：原 P64 真实 student 路径 `candidate_Y`。

[SRC] [`sink`](../../../tools/chipmunk_student_observation_sink.py) 将64个真实选择行置于文件前部，随后256个与之不重叠、按21τ×四空间区分层的评估行；两个轨迹分别持久化同名 multi-tensor sample，四层另存独立 full-M 文件。`Y_S*` 由 [`observe`](../../../tools/chipmunk_student_observation_core.py) 的同学生输入额外 W1/GELU/W2 计算；student 自身返回 `Y_S`，影子输出不进入生成。Dense 模式在相同 Dense V1 上验证 shadow h/Y 与原生 Dense h/Y 逐字节一致；这一门的通过依赖对应真实 RUN 和独立验证，不能先假定已成立。

[ALG] BF16 数先各自精确转 FP64，定义 `L=Y_S−Y_S*`、`P=Y_S*−Y_D`、`T=Y_S−Y_D`：

`T=L+P`；`||T||² = ||L||² + ||P||² + 2⟨L,P⟩`。

先在每行/预设组累计非负能量和有符号 cross，再作来源、层、分区汇总；报告 identity 绝对闭合差、预定数值容差和零分母。BF16 值可精确转换为 FP64，但不同指数的 BF16 值相减未必能在 FP64 精确表示；极端指数跨度和抵消可令计算得到的 `T` 与 `L+P` 不逐位相同。向量和能量归约都应按冻结容差审计；有非有限值即停止该 scope 解释。推荐 FP64 作为基准，避免仅用 ratio 隐藏小分母。`L` 是给定实际学生前缀/缓存/所选mask的同状态输出差，`P` 是两个自由轨迹当前输入映射出的 **状态差的描述**；它既包含先前所有已生效动作经模型传播的结果，也可能包含舍入、实现和测量中性未通过的差异。不能把 `P` 称为一步误差、单层贡献或后缀因果放大。

## 能改变什么判断，以及强反例

| 观察模式（每源逐层、先验规定） | 有用的下一决策 | 不能推出的结论 / 反例 |
| --- | --- | --- |
| 979与898同层相比，独立256行的 `L` 及尾部持续较高，且四层 full-M 与抽样一致 | 优先检查同一学生状态的真实mask遗漏、signed投影和随机/分区等预算替代；原学生mask和gate必须保留 | 不能说局部误差造成运动下降；它可能与视频方向无关，且 Y 的大范数不等于真实写回大。 |
| `P` 远大于 `L`，或 block0后逐层增长 | 优先选择少量固定前缀重放、同一误差不同位点/方向的后缀响应；同时检查归一化分母和以前各层 `L` | 这不证明H3放大。非线性状态变换、不同输入参考幅度、动态mask/方向改变及以前动作之和即可导致大 `P`；也可能是上游偏差经正常输出映射产生。 |
| `2⟨L,P⟩` 大且负，使 `T` 小；或大且正使 `T` 大 | 报告有符号cross和余弦，保留L/P绝对能量；后续须固定前缀比较替代动作是否改变方向 | 将 `||L||²/||T||²` 当贡献份额会超100%，`P` 也可能与L抵消。大小/符号均不能解释动作因果贡献。 |
| 一个已冻 full-M 层的256行和全M统计有明显偏差 | 扩充空间/时间分层或对额外层做真实全M分析，再决定是否需要新采集 | 256独立于选择mask，并非独立来源；比例加权只能估计除64选择行以外总体，不能保证稀有对象命中。 |

H1 token/time 异质性、H2 signed W2 列间组合和 H3 后缀传播不会被此分解单独识别：L/P 是输出级向量而非遗漏通道项；同一大的 L 可以来自全局共同mask或局部遗漏，P 可由完全稳定的后缀上的前缀状态差产生。即使 `P=0`，以后后缀仍可能对小而特定方向的 L 高度敏感；即使 `P` 很大，最终视频也可能对其不敏感。固定 `L`、等范数相反方向、early/late block 与零扰动经同一前缀的后缀干预才提供 H3 的不同预测，并需另行独立审查、登记和真实运行。后缀响应的定位若以这两开发源选层/方向，另须未曝光来源及独立seed确认。

**最强原生身份反例**：同源两轨 high step0 本应相同，若起始 latent/flow 与所有对应 step0状态已过逐字节门，但两个 `old_h` 或 `old_Y` 在某block样本/快照 raw SHA 不同，不能直接把后续 P 解读为候选动作引入的状态差：先查行顺序、offload buffer、转储、历史 neutral/dispatch 与原始版本。反向地，old h/Y 相等只锚定此缓存和有限采样，不足以证明整个 high step1 当前状态相同。若 block0 的 step1 Dense 与 student V1 同字节、shadow身份门通过，`Y_S*` 与 `Y_D` 应同字节；不通过先追数值路径及输入，不能把第一层 P 自动解释为此前FFN传播。若没有 block0 的 full-M V1，样本可检查相同坐标而非宣称全M相同。

## Gate 与实际残差写回

先逐源×block比对两轨 metadata：真实 gate 的 BF16 shape `[1,1,5120]`、stride `[30720,5120,1]`、dtype 与平铺5120个有限值。这里 `.float()` 对有限 BF16 数值是精确提升；应比较对应 BF16 bit-pattern，连 `+0/−0` 也区分，并拒绝无法无损还原的 JSON 数据。只有相同 gate 才报告在公共 gate 下的 `G⊙L`、`G⊙P`、`G⊙T` 及其同一恒等式；如果不相等则分别保存门值与差异，暂停共门假设。仍可定义 `G_S Y_S−G_D Y_D = G_S L + G_S P + (G_S−G_D)Y_D` 的 FP64 **理想门乘** 分解，但这是数学比较，不是实际 BF16 write。

当前采集未保存 Dense 与 student 的 write 前 residual `R_D,R_S`。真实 [`post_process`](../../../.runtime/code/LightX2V/lightx2v/models/networks/wan/infer/transformer_infer.py#L418) 是 BF16 `R.add_(Y*gate.squeeze())`，乘法及加法各自舍入；`G⊙T` 不是实际两轨 residual-write 总差。student 既有80/84分区的实际 write 对比也仅是固定 **student** residual 的 `Y_S` 对 `Y_S*`，不能用其推演 Dense/student 两轨 write。若需要真实跨轨写回分解，另须捕获相同坐标的 `R_D`、`R_S` 及原生写回输出，含插桩 neutral 与新scope，才能在 BF16 实际顺序下比较；两轨 residual 差仍不是因果归因。

## 可实现的最小 CPU 读取 producer（待复审，不运行）

**输入与单位**：先锁定 source898/979 各自真实完成的两 phase RUN、原protocol/code/ckpt、EXIT、独立VERIFY及manifest SHA；检查所有40 site、对应seed/step/expert、历史视频和10状态的 neutral/byte门。只有完整并闭合的产物可进新分析。独立单位是 image/source，row、层和同一视频内τ皆嵌套；两个已曝光 source、单seed是机制开发而非泛化确认。因同源 Dense/student 分别生成，视频变化差异不是本分析的重复样本。

**主范围**：每源40个样本文件对，解析 safetensors 8字节头长和header，以实际 `data_offsets` 寻址 Dense `current_dense_Y`、student `current_dense_Y` 与 `candidate_Y` 三个 `[320,5120]` BF16 片段；按生成时原有行序配对64+256并各自逐 tensor raw SHA 验证。三张量每源 `40×3×320×5120×2 = 393,216,000 B`，两源 **786,432,000 B** 数据载荷；不含160个header、元数据、计数文件与原生RUN/VERIFY读取。冻结额外可选四层 full-M读取，每源 `4×3×31668×5120×2 = 3,891,363,840 B`，两源 **7,782,727,680 B**；它们为单tensor文件，可整文件SHA和rawSHA同一次连续流读取同时验证。两种范围叠加数据载荷 **8,569,159,680 B**，head/metadata/receipt及操作日志另行预扣，禁止把逻辑读取等同实际磁盘物理I/O。

**原始缓存身份**：先比各对应 sample 内 `old_h` 与 `old_Y` 的 manifest `raw_sha256`、dtype/shape、行顺序；四个 snapshots 同比已有单tensor raw SHA。此为已闭合父验收的 hash 身份读取，**不读取 h/weights**，但仅在先确认父独立VERIFY对每张量raw SHA与文件绑定真实通过时才可信。如果父VERIFY尚未闭合、需要本分析重新独立计算h哈希，则按新scope额外预算全部 h/old_Y 读入，不能声称只读Y完成此重新验收。无需载入旧 h 即可执行Y分解；可选择仅检查报表列出的old h/Y哈希，不把 hash 等价升级为历史全层全M identity。

**样本内文件与完整性**：multi-tensor样本每个还包含 V1、old_h/current_h、old_Y 等数据；要重新计算整个 safetensors **文件SHA** 必须读取这些非Y部分，所以“只读Y”与“重新完整复算文件SHA”不兼容。推荐依赖真实已闭合独立VERIFY的完整文件SHA身份，分析中先核对 manifest 绑定、路径、文件 lstat/打开fd身份、可信完成状态、所有读取前后 inode/size/mtime/ctime，按header校验 dtype/shape/合法且不重叠的offset，对三条目标片段逐字节一次性读出并核 raw SHA；随机seek和页缓存带来的物理读放大标 UNKNOWN。要求本scope重新全文件hash时另预扣完整80文件、将其他张量字节列为完整性验证读，并在前述Y片段读取以外避免双计或额外再读；不能写成未读h。四层full-M单tensor文件可一遍同时核文件/raw SHA。文件在验证到分析之间的替换和范围逃逸须在打开fd后重查，禁止仅信路径字符串。

**行与统计**：metadata 的 selector列表及256个坐标、stratum、eligible_population、sample_count、inclusion_probability 必须 Dense/student 精确相等并检查与选择64行无交集，grid行号 `τ*29*52+y*52+x`。分别输出 selector64、evaluation256及全M（仅四层），不将前者当独立 holdout。evaluation 的总体估计按每行 `1/π_stratum` 加权，并将64 selector行作为与其互斥、已观测的确定集合；全M总体能量可用两部分的估计合并，且说明这是对随机设计的条件估计，不从单次抽样推断精确尾部。预先冻结 mean/sum、每层加权及每τ/象限统计、尾部阈值/分位数定义；比值报零分母、源向量和分区样本量，两个source不作来源总体置信结论。full-M按预设 `[0,13,26,39]` 且逐chunk关联同一 `row`，在各轨 gate byte一致后再做理想 gated；不要以多层能量之和充当各层对视频贡献。

**资源与账本**：预扣每个单独的tensor片段读次数、精确载荷字节、header上限、核哈希/seek/文件stat、manifest/metadata和RUN/EXIT/VERIFY读取上限；逐次写在可恢复操作journal中，失败的当前读取标UNKNOWN。按固定chunk例如256行，将三路BF16读取各2,621,440 B，转换FP64和构造 L/P/T/平方/cross临时内存依据实际实现预扣峰值上界并设置 RSS、磁盘空余、总墙时与CPU核预算；原生视频I/O期间不调度。算法不落盘任何 tensor，输出仅source×block×分区摘要、闭合误差、有效计数、hash/控制证据及机器可读source路径/receipt，独立计入JSON原子临时副本和sha、journal和supervisor输出。要指定可执行的具体数值RSS/CPU wall/输出上限，必须先独立审该分析实现的精确allocator/中间数组和采集产物file headers，不能从当前 design 假装已测量。新producer及小规模CPU软件fixture完成后由主Agent自审→独立Agent反例→修订/recheck→ACTIVE注册→身份复核；本文件不代表review/执行。

## 后续动作

采用上述 **40层×320行优先，四个全M层为同一分析scope的预算明确选项**。成功结果将给出两暴露来源在局部 L、已有前缀 P 与对齐方向的具体观测，并能检验采样对四层全M的代表性；这可决定固定状态 mask竞争优先，还是把预算转向冻结扰动后缀，但不能给H3投票或宣称质量恢复。若 gate/缓存/neutral任一身份失败，输出失败收据并先追溯；若上述分解不足以区分竞争机制，继续采用对照明确的固定前缀干预，而不挑选单一“最重要层”。
