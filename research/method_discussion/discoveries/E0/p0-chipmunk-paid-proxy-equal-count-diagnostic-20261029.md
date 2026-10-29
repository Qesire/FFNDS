# Chipmunk：付费代理与等选中数对照的下一项局部判别

提出者 `/root/s2_theory_design_independent`，2026-10-29。**SRC/ALG 与待审设计；无新增数值读取或运行，REVIEW_REQUESTED。** [已闭合单组结果](../../../20261029/CHIPMUNK_A100_SELECTED_DELTA_PROBE_RESULTS_20261029.md)提供可工作的算子起点：固定前4096列 gate 输出对当前 M128 Dense 的 relative-L2 约0.3462，而同mask独立算子参考约0.00151；全更新对当前Dense约0.00299。这证明当前局部实施与其参考接近、固定子集在该组有明显遗漏，**不检验作者付费选择规则**。下一问题是：同一实际数量下，作者式可获得的组均值信号是否选到比随机或静态列更有用的更新？

## SRC：真实作者 proxy 与选择器

固定原仓库 commit `177fdefc4803e621fd8a02aad491812e9d1ecc94`；[MLP source](https://github.com/sandyresearch/chipmunk/blob/177fdefc4803e621fd8a02aad491812e9d1ecc94/src/chipmunk/modules/mlp.py) 首个full步执行 `mid=fc1(x)`、`pa=activation(mid)`、`out=fc2(pa)`，缓存 `pa`、含b2的out以及 `block_mean(mid,MBM)`。选择步把当前x先做128行均值，再执行 **`fc1(mean(x))`，含b1且发生在GELU前**；与先前缓存的 pre-GELU `mean(mid)` 相减取绝对值。若 BM/MBM>1，先按作者reshape对多个子均值的绝对差求和；[FLUX作者配置](https://github.com/sandyresearch/chipmunk/blob/177fdefc4803e621fd8a02aad491812e9d1ecc94/examples/flux/chipmunk-config.yml)两者均128，故一组一个值。精确实数下b1从差值抵消，但平均/加bias/GEMM写回BF16及归约顺序不保证机器相等；缓存的是历史 mean(pre-GELU mid)，**不是mean(h)、不是当前逐token误差**。选中后 `copy_indices` 仅更新所选列的均值缓存；未选列参考继续停留在它们上次更新时刻。第一次选择步由真实首次full缓存提供，不能用当前完整Dense h或未来状态决定mask。

[作者 `topk_indices.cu`](https://github.com/sandyresearch/chipmunk/blob/177fdefc4803e621fd8a02aad491812e9d1ecc94/csrc/indexed_io/topk_indices.cu) 实际排序前1024个分数样本、按 `sparsity=0.7` 选sample cutoff，随后遍历全部H，用 `score >= cutoff` 或 `curand_uniform < 0.05` 接受；BF16 ties可明显改变数量。随机5%是逐列额外接受机会，并非保证另外恰好5%独立列，也不是全H精确top30%；实际count再向256倍数补齐。内核seed由block/thread位置及第一个activation的四字节决定，非标准可指定统一seed；atomic顺序可能不同。H=13824时 count、实际唯一集合、mask顺序和padding成本必须实测记录，不能把0.35当实际比例。对完整count=H和量化ties应单独考虑。

**静态安全反例：** 作者padding段中，所有拥有 `last_invalid_value` 的线程都 `atomicAdd` 后写indices，没有按 `remainder_shared` 限制写者数量；有效 `counts` 只取补齐后的前缀，但多余写入可能超过索引buffer容量或与边界冲突。是否在特定输入发生未实测；不可为追求忠实直接在当前设备运行潜在不安全kernel。作者sample、threshold、随机状态、padding、tie和有效count语义与安全实现逐项对照；保证有界写入但改变其余写入行为的端口明确记为修补。若无法逐项重现其随机/并发顺序，则称 **作者启发的安全受控selector**，不用“作者原核/原随机流忠实复现”。本次额外阅读源码保存在HF `ffnds-research-sources/p0_chipmunk_teal_20261029/chipmunk_topk_indices_177fdef.cu` SHA `05407a642dcb5f201b605547d53b91effb15bb9b38fce3344207c8f4204ae251`；`chipmunk_copy_indices_177fdef.cu` SHA `7b3725a28082dfff98f18ad70e29537ad68bf6f93265f518328798386542128f`。两者是 SRC，非执行证据；建议补L09 source ledger及G14静态焦点，边不改成native。

## 可证伪竞争解释

令 `Δh_i,j=h_t,i,j-h_c,i,j`、`e_S=Σ_{j∉S} Δh_:,j B_j,:`。实时门控误差为 `g_t⊙e_S` 加上BF16算子/缓存rounding；不同通道在W2上可有正负抵消。H1：组均值preactivation变化与可见逐token gated输出变化足够关联，作者式选择在实际相同count下的同source、同初始化、同W1/W2执行中小于固定和随机对照；完整付费选择边界后在较大组/全M有可测试的省时潜力。H2：组内正负相反变化使均值≈0但每个token的Δh明显，或GELU曲率、不同W2通道范数/方向与符号交叉支配输出；于是代理的较大分数并不保证低 gated Y误差，同count随机或静态可胜出，即使partial kernel与独立参考高度一致。另一个反例：两个未选通道输出贡献方向相反，丢弃两者比只更新一者误差更小，因此所选单通道变化能量越大不保证输出误差单调降低。当前一个组无法确定这两种机制的群体频率。

## 最小有限诊断，待作者实现与独立复审

复用已闭合source898 seed42 Dense high/block20 step0→1 两个已有1092行包的**首128匹配坐标**和真实checkpoint，身份及原M31668捕获Y/h列于[资产核查](p0-chipmunk-spatial52-paired-input-probe-20261029.md)；读取前核全payload SHA、稳定stat、原checkpoint header/SHA、row/专家/step、原RUN/EXIT及执行环境。**本次不需新模型、视频或重新拟合。** `M128` 是从21τ×52xy混合抽样行打包的离散组，不是原raster连续组；同样来源重复不产生独立source。行打包及输出回散计价；FFN逐行置换代数不使依组均值选mask的算法对置换不变。

初始化采用相同原M128 Dense step0重新计算 pre-GELU mid、h、Y，缓存mean(mid)并保留b2；step1计算 `fc1(mean(x1))` 和差分分数，生成**安全、版本化作者机制受控移植**，冻结BM=MBM=128、样本前1024、0.7 cutoff、random概率0.05、256有效count粒度，具体CURAND、ties、并发、padding保真能力必须在源码及小型合成软件测试中交待。若作者随机流未实际复刻，必须记录seed并标为“替代固定seed RNG”，而不是声称作者exact RNG；所有mask固定先算再hash，完整记录真实`K`及有效索引、padding和去重。预计K为256倍数但不可预先设4096；若合法K=0或H使对照退化，则原样记录并判定本组不辨识，不换source/seed或阈值。

各臂使用独立克隆相同h/Y缓存、同step1输入和current gate、已验证两个BF16 sparse kernels：A安全代理+随机选集；B **相同实际K** 的均匀随机全集，冻结三个种子并保留逐seed数值；C 相同K的固定前K通道；D full与empty作为neutral和误差锚；可另保留原4096固定集用于原观察复核，但不得与不同K的代理直接作等预算胜负。若比较纯proxy而无其random，另给同K纯proxy组，区分选择与随机保底贡献。若研究多个K，以固定seed预生成随机排列的前缀构造嵌套对照，明确所选大小且不以误差选K；本scope的K来自A的在线合法状态，不从后验输出选择。所有mask分别记录；若padding引入新通道，它们是**真正更新列**而非空操作，按真实有效K比较。

每臂逐元素检查未选h不变、kernel delta、与相同mask独立参考的h/Y/gate误差及finite；分别比较step1 M128 Dense当前teacher、原fullM保存的对应h/Y gate（报告与M128 Dense桥），保留逐元素signed cross、绝对误差与相对L2；step0初始和step1当前gate/偏置不混用。必要的 `h_t` 和 full Dense teacher只用于评测，不参与A/B/C选mask；以它构造的oracle只能是另标已付费离线上界，不作为有效在线方法。可用按通道Δh能量或W2加权范数作oracle解释，但完整计算和忽略cross的局限必须披露。

计时分别隔离CPU/设备pack、缓存初始化、当前组均值、proxy W1+b1、阈值/随机/padding/选集构造与拷贝/同步、两个稀疏kernel、h/Y缓存更新和回散；另报一次完整行动边界的同步wall，预热/编译/数据输入各分开；参考和oracle开销另记，不混入实际selector。不能以先付费完整 W1或全teacher计算的oracle代替实时开销，也不能把一次含监测/冷编译的耗时当系统速度。按实际mask K及tile计W1/W2、索引和缓存读写；物理HBM、全M时延未测则 UNKNOWN。

可提交的预算草案：单A100 GPU、CPU≤4、GPU reserved≤24GiB、host peak≤64GiB、worker≤900s、outer≤1020s、HF新输出≤512MiB；全局当前最多4 GPU，由协调者保障。按 **1个组、1个初始化、3个随机种子、A/B×3/C/full/empty共7次独立臂**（如添加旧4096或纯proxy即先修订计数），预声明各GEMM、selector、mask和kernel调用、reference、完整哈希读/拷贝、同步、compile/warmup/repeat的精确上界与预扣/完成/失败收据；统计每臂独立副本。64GiB RSS若只是高水位/边界检查须如实说明非连续硬限制，所有资源超限失败保留且不自动重试；无实际物理launch观测不假称launch计数。此表只是审查草案，不授ACTIVE。

**接受性验收：** 作者与真正不同Agent逐项完成自审、对抗评议、修订和精确指纹recheck，再冻结代码/资产/recipe/阈值/资源并注册。缺真实proxy行为或安全mask合法性、错误缓存/bias、无neutral或独立mask数值参考、未计时的选择/拷贝、资源违反/异常，均不进入完成结果。成功也只说明一真实离散组的局部对比；原raster连续组、跨更多来源/层/两个expert、自由student后缀和全M性能需要新证据，必要时单独申请同block相邻步完整行采集；不得基于该局部结果提前称全视频或作者系统复现。
