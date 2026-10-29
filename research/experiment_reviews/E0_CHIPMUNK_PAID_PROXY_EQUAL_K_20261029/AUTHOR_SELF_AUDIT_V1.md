# Chipmunk 付费代理与等 K 对照：作者自审，待独立评议

**状态 REVIEW_REQUESTED。** 固定已闭合 source898、seed42、high block20、teacher step0→1 的首128个匹配坐标和同 checkpoint；这些是空间/时间混合非 raster 行，不是完整视频、连续原生分组或独立来源。已闭合的 A100 固定子集两核作为依赖引用，未修改其文件。此处没有新 GPU、模型、权重、科学 tensor 读取或执行。当前全部4 GPU用于视频，设计本身不注册也不调度。

## 研究问题与不同预测

上次固定前4096的局部 gate×Y 相对下一步 Dense 误差0.346，而同mask算子参考误差0.0015；未检验作者均值代理。本次问：同一实际更新数量 K，使用合法当前输入的 pre-GELU mean 代理是否比三个固定seed无放回均匀随机集或固定前K减少该组误差？H1：均值变化与重要通道一致，代理优于三随机集及固定集；H2：相反token变化抵消均值、GELU曲率和W2有符号交叉造成代理不优。未选两个方向相反的通道同时遗漏甚至可能比只更新一个更小。该单组仅提供具体反例或有界观测，不估计跨来源优胜概率。

## 选择与作者差别

首次 M128 Dense step0 保存 BF16 pre-GELU `mid0`、h、含b2的Y；paid proxy 先 `mean(mid0)`，step1 计算 BF16 `mean(x1)` 和完整 `fc1(mean(x1))`（W1+b1），两者 BF16 差取绝对值。BM=MBM=128，不调用作者 quantized/Hopper 核。固定作者源码 commit，先排序前1024分数，取第 `int(1024*0.7)=716` 索引 cutoff，全部 H 遍历以 `score >= cutoff` 或额外 random 0.05 接受（含 ties），然后将有效数向上补齐到256并实实在在更新这些列。Python MT19937 seed46829 和按升序挑未入选列的有界 padding 是**作者启发的安全适配**：不等同 CUDA 作者的 CURAND 状态、线程/atomic顺序或潜在越界 padding。作者两份 CUDA 选择/复制源码及旧实际 RUN/EXIT 字节绑定。选择后只更新选中列的 pre-GELU mean 缓存并检查其他列数值不变；没有后续自由轨迹。

mask计算后立即冻结全部真实索引、K、threshold/base/random/padding 数与 SHA；同时保存 BF16 分数值，closed 核验器独立重放确定性选择并检查结果和 SHA；三个 random 种子101/202/303分别对全 H 无放回取实际 K，固定集选前K；另有全H和空集，总7臂。未用当前完整 h/Y、oracle或结果误差选mask：完整 step1 M128 Dense 在全部 mask 确定后才计算。K=0或H按原样执行/标记单组不辨识，不换 seed、source或阈值；本 q=0.7 的有限非空分数中 K=0 实际上不预期，边界仍经软件测试。padding列视为真实更新通道，K重复/越界和 SHA闭合均严格检查。

## 数值与成本

全部臂用相同 step0 M128 h/Y 的独立克隆、同输入和step1 gate。复用原两核：当前选列计算、BF16 delta、FP32 tensor-core归约加含b2旧Y、BF16写回。每臂另用 M128 Dense 当前h构造独立同mask BF16 delta与禁TF32 FP32 Y参考；选中 h relative-L2 ≤0.015，delta ≤0.05，Y及gate×Y ≤0.015；全选相对 M128 Dense h/Y/gate ≤0.03；空集和未选坐标按 `torch.equal` 数值逐元素相等（不声称+0/-0逐字节相等）。这些均在试验前固定、失败留证、不按结果调阈值。另记录对当前 M128 Dense 和原 M31668捕获参考的 h/Y/gate 最大差、relative-L2、signed cross；只用后者评价，不给 selector 看。

完整代理分段计 `mean(mid0)`、`mean(x1)`、完整W1+b1、分数GPU→CPU、样本排序/RNG/padding、只更新选定mean cache；其总单次同步wall含账本。对照mask生成单列，不混入代理成本。每臂报告选集GPU复制、cache克隆、A/B两个核、128行index_copy回散及独立参考开销；代理加动作报告两个分别计时区间的和，含账本、分配和可能编译，**并非统一连续端到端时间或公平热速度**。全权重预加载并保持 contiguous，核直接 gather，未预打包按选择的W1/W2列；检查和输入加载、编译、selector CPU复制均收费或显式另报。回散只写128行、余下未初始化，非完整模型输出。没有真正的硬件HBM或物理launch计数，均 UNKNOWN；single-shot微成本不外推full-M/系统加速。

## 预算、证据和运行门禁

2 payload hash/load、1 checkpoint hash/load、2 M128 dense、2 mean、1 proxy W1、1 safe selector、1 mean cache update、1 control-mask构造、7选集GPU复制、7回散、7独立参考；A/B核上界各6，K=0时各1、K>0时各6。每项预扣并完成或保留 inflight；原子级物理launch未知。逻辑输入≤30GiB，完整 checkpoint SHA和两个payload SHA按块预扣、失败记录已完成字节及最多8MiB在途不确定，文件读取前后稳定stat；全部payload和源闭包绑定；checkpoint tensor逻辑读取与物理磁盘读分别报告。单GPU、CPU4、GPU reserved 24GiB、host RSS 64GiB、worker900s/outer1020s、HF512MiB；host RSS和输出为边界/最终高水位检查，非连续硬限制。并发由协调者按当前全局≤4且每scope GPU1分配，不继承旧2卡。网络禁用、fresh attempt、失败无自动重试。

真正独立于理论提案者和本作者的reviewer需检查源码复制语义、安全padding与ties、固定seed等K、mask因果信息时点、缓存/bias/gate/两个参考、资源与失败闭合，提出反例。Torch/Triton环境身份、精确 worker/supervisor spec 和资源还需在修订后冻结并 recheck；没有ACTIVE时不启动。数值与耗时仅通过之后单次实际RUN证明。

待审 draft SHA `e93e5d4e40bdb5cbb82810657fd07dab489a5c10a502d3d789bf705891f3108b`、fingerprint `add96c24adcdbb9f9bfc658deae187891719ff9b5c217dd95302e1397f44790e`；producer SHA `a0a69c8bda134e1cc3fa968961a3681cf9785e2a7bfd9df218d7d77312c8c1c1`，selector SHA `34e7f27aff437020e057e8fd12e822e9a9d4eab5a86ea86f7e83e53045001e6e`，tests SHA `ba7eef23379cfda6bf76a8f4a5acf797d548a15f3a68fbba8f3e4549b5661adf`，复用kernel SHA `ec2994a7b446d31b7aa967b85a13a528791678c72eb0e93df35e356e2cbfc0d0`。8项非实验性单测通过，包括tie/样本cutoff、补齐、重复/越界、K0/H、三种等K随机、数学反例及失败账；不等于GPU正确性测试。
