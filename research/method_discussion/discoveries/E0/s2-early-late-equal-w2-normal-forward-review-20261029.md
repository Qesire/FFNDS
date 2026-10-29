# 全深度早段／晚段等次数原W2：可解释的策略对照与实现边界

2026-10-29 · `/root/s2_theory_design_independent` · 独立科学设计评议；未改执行器或协议、未运行模型/GPU，不是代码审签、支持票或运行授权。本人实现过已完成的New12 Dense V1核心，此处依源码重新检查该计算路径，不声称实现独立性。

固定已有value10 r960 factor、同checkpoint和BF16 recipe，对12个新开发图分别配对完整四步Dense与三条自由生成：全160位置factor（C）、每step block0–9原W2其余factor（E）、每step block30–39原W2其余factor（L）。E/L都在自己的当前h上执行相应真实算子，完整W1、GELU、b2、gate、残差与四步调度保持原生。E和L各40原W2/120 factor位置，能给出**两种确定性全轨策略的配对因果差**；这个差同时包含各层局部误差、门、状态、后缀响应和跨step闭环，因此单靠终端视频不能分解出“早期传播”与“晚期读出”。

只读核验两路 [Dense RUN/EXIT part0](../../../../../../.cache/huggingface/ffnds-derived/e0_s2_new12_dense_baseline_v2_20261029/part0/attempt1/worker/results/RUN.json) 与 [part1](../../../../../../.cache/huggingface/ffnds-derived/e0_s2_new12_dense_baseline_v2_20261029/part1/attempt1/worker/results/RUN.json)：各7条完整四步Dense、旧898/96 neutral十状态通过，实际EXIT0；12条新图Dense作为已生成开发参考，2条旧源是重复控制。新图的真实动作分层仍须另据Dense视频与人工审查建立，不能由静图或prompt断言，也不能把已观察结果用于反复选臂和选源。

## 固定问题与竞争预测

先冻结三臂、12图、seed42、原prompt和factor SHA。主比较以source为单位，记录 `E−C` 与 `L−C` 的配对最终视频质量向量及各step flow/latent方向误差；给出每图值、source等权汇总和不利尾部，先声明主指标、方向、对照和有意义效应量，再看新混合臂输出。空间、帧、step不当新source。新12和先前两图均是开发，正式未曝光图及独立seed确认另行安排。

- **上游状态传播占优**：如果恢复block0–9后，自block9起的学生state差相对C发生一致变化，随后block30前、该step flow、下一step输入和末latent仍保留方向，并在source级视频指标上较L更常改善，则支持早段行动的后续传播价值。反例：E虽改变早段state，但其后影响消失或相比L无优势。
- **晚层局部读出／后缀灵敏度占优**：如果L在block30之后和该step flow产生更大稳定改善，而E作用早消失或晚期状态改善较弱，支持本范围内晚段行动价值。反例：L显著改善局部读出而flow/后续不变，或E跨step作用更大。
- **投影过滤／局部取消**：因原W2恢复重新放出候选丢弃的状态方向，E或L可以令局部相对Dense误差增大，即便使用更精确的原W2；也可能随后带来视频改善。反例：若各来源的实际typed恢复总沿减少误差方向，且影响主要由当地恢复本身预测，则被滤状态的有益取消在这些位置较弱。此前gY投影近似不能直接推广到真实u、rowLN和完整视频。

每组是对当前学生轨迹的行动；E与L的各自状态和当地h不同。动作点的位置也决定其gate、权重谱、W2局部误差、同step剩余深度、下一step上下文和后缀函数；开销相同并不令这些因素相同。即使E明显优于L，也不能归因于“更长传播距离”而排除E组本来较难或较重要的局部函数。step0/1 high expert、step2/3 low expert重复同样层段，使层效应与step、expert及后续去噪闭环交织。

## 更有辨别力的有限扩展

优先方案是在相同三臂加第四臂 EL：每step block0–9和30–39均用原W2，其余factor（80原/80 factor）。对任何预定向量读出z，计算 `J=z_EL−z_E−z_L+z_C`；它衡量两个固定集合行动的总体交互，包含h变化、非线性、门、调度、BF16等，不能单独识别某条传播路径。按同等40位置算力比较仍仅是E与L，EL是增加预算的因果控制。四臂每图总共新增4个视频（另复用现有Dense），比三臂多12个；若预算不容许，先保留三臂和中间观察，再另冻EL，不在看到结果后从12图中挑最有利图补跑。

可增加一次预冻结的小型正常前向A/B单点实验来解开组内层混杂：在固定step0与step2各选 block10(A)、block30(B)，S00/S10/S01/S11以学生自身h作全行原W2恢复，其他位点保留C；捕获A后的中间state、B前后的变化和下一step状态。这里用全行是真正物理W2动作，不需借Dense hidden；额外只读同轨指标，单个介入不代表部署速度。预定被恢复的所有token而非后验挑ROI，或另冻等row、同xy的τ0/τ10/预定随机τ；后者不能外推全M，并须核验小M到实际full-M的该臂gY、typed后写回signed bridge。无论组还是单点，恢复是付费因果工具，必须记录真实计算与复制开销。

最小中间读出可固定每step开头、block9后、block30前和block39后，及既有4个flow/4个post-step latent；先规划输出量、抽取位置和observer开销。仅选稀疏固定行时称其抽样状态，不声称代表整图；要与已存在Dense参考逐位置比较，必须先确认已有参考确实包含这些边界——现有Dense 10状态是噪声/flow/latent，并不包含逐block hidden，不能从flow倒推或伪造Dense边界。新增Dense边界捕获需要独立预声明与预算，或先仅比较三学生臂相互的实际边界状态。观察位置本身会引入时间和D2H成本；neutral证明observer不改变字节后才能解释。

## 真实执行路径与不能直接复用的部分

[SRC] [native_phase](../../../../tools/e0_raw_bf16_full160_video20_v2.py:565)保存原噪声、初始噪声、4 flow、4后step latent共10状态，输出81帧及全160位置ledger；[LightTap](../../../../tools/e0_raw_bf16_full160_video20_v2.py:498)核查原生计算与162预取。New12 Dense [runner](../../../../tools/e0_s2_new12_dense_baseline.py:352)使用 `mode='dense'`，先过旧源10状态neutral；[V2 wrapper](../../../../tools/e0_s2_new12_dense_baseline_v2.py)调整实际资源授权读取，已完成两路。现有源代码是完整Dense入口，不含逐block混合实现。

[FactorW2Runtime](../../../../tools/raw_bf16_factor_w2_runtime.py:12) 的 `mode` 按整个phase切换；它在W2 `load_state_dict`上决定原权重是否预取，在W2 `apply`上选择原MM或双factor GEMM。Wan的[双缓冲offload](../../../../.runtime/code/LightX2V/lightx2v/models/networks/wan/infer/offload/transformer_infer.py:40)在当前block执行时异步预取下一个block，包括block39循环预取block0。故不能在compute边界简单调用现有 `set_mode`：这会用当前compute模式处理别的block的预取，并可能出现过期buffer。需要独立版本按 **被预取的expert、block和已冻结arm** 决定传输，给每个buffer打带expert/block/原或factor身份的就绪marker，并在apply时核查当前expert、step、block、marker一致；保持stream同步、原b2、全160位置和同h。当前各step的层段固定，预取选择可按expert/block；如将来每step有不同策略，末block预取的block0属于下一计算上下文，必须另设计显式step归属，不能借当前活跃step猜测。

既有 `transfer_geometry` 只接受phase级dense/factor并用它判断是否跳过W2，必须为混合语义独立适配；保持逐次原始copy来源、逻辑字节和实际dispatch事件，不要复用全factor的162跳过断言。已运行pilot的因子属于共享两step/专家的固定value10，需由新协议验证同一factor bank、tensor SHA和step→expert映射，防止不同训练family或新拟合。保留原四个original CUDA W2 buffer的实际驻留；如果事先把80个value10 factor常驻GPU，E/L均会付这笔preload和占用，不能以理论跳过40份factor推断真实显存省额。

对每个视频，原 offload 每个step循环预取40次，两个expert各额外初始化block0一次，总计162次。按固定位置策略，在实际dispatch正确的前提下，E预计原W2装载42次、跳过120次；L预计装载40次、跳过122次；C预计跳过162次。E/L计算次数相同但预取流量不同，E多2×141,568,000=283,136,000 bytes的原W2请求；这些是由源码循环推得的预算假设，需真实日志核验，而不是硬件总线实测字节。流水线重叠、双缓冲占用、预取初始化和kernel效率使等位点数不等于等wall或等实际物理成本。

注册和配对：为复用Dense，验真其原source、image/prompt、seed、initial noise、四step、video及原实际RUN/EXIT；新registered manifest需引用原生成并以诚实的来源标记建立登记/收集适配，不能把旧Dense文件伪装成新模型调用。旧 `collect` 要求job与source内容逐字一致，现有单Dense注册行不可机械套入新多臂manifest；实际初始噪声文件还须按现有规则逐byte比较。每臂视频/状态必须是自己的真实生成，三臂不可共用输出文件或连续推进同一随机generator。CORE和完整多轴视频质量应以真实Dense作为相同source/prompt/seed参考；官方质量与人工视频评审分别登记，不把latent SSE当视频验收。

## 有界工作量和停止条件

以每part六张新图×三臂，另加一条旧曝光源的**全factor neutral**做已有value10十状态逐byte控制为提案：每part实际新增19视频、76 flow、3040 FFN位置，其中2560 factor位置＝5120次factor GEMM，480原W2计算位置；全两part共38视频、152 flow、6080位置、5120 factor位置＝10240 factor GEMM、960原W2位置。每part190状态、133 stage pairs、3078原block预取观察。需先证明旧value10参考的checkpoint、factor各tensor SHA、调度和numeric recipe确与新bank一致；缺任一证据时这条历史neutral不可虚称通过。若扩展EL，新图每part另加6视频、24 flow、960位置（480 factor/960 factor GEMM、480原W2），对应state、decode、传输及输出另增预算。两路每项一GPU，沿当次有效会话/注册上限调度；资源、wall、HF和显存额度必须由新的可测scope冻结，本文不授权运行。

每part单三臂提案的19视频按原每视频10状态payload `60,802,560` bytes计约1.155 GB，19个128 MiB视频输出上限hold约2.550 GB，合计预留约3.705 GB，未包括metadata、日志、模型读、factor preload及中间观察；4 GiB输出上限看似可装但必须按实际累计检查，不能由此保证足够。每个输出writer先predebit并追踪实际字节，原D2H/日志、factor CPU加载/GPU预载、转移、observer、原W2/因子GEMM、每个位置的动作与完整80因子驻留等逐项计数。记录峰值GPU、host、wall及CUDA事件；不同顺序/缓存和独立Dense参考采集时间意味着现成FFN耗时不是严谨的同条件端到端速度比较。各part六图可冻结六种三臂排列各一次，以降低固定顺序偏差，不作为重复计时或消除所有缓存效应。

启动前要求源码闭包、全部12源与Dense实际EXIT、factor完整SHA、同seed实际noise、neutral来源和GPU/CPU/HF预算齐备；先做合成双buffer测试覆盖factor→dense、dense→factor、block39→0和expert切换、marker错配拒绝，再真实neutral。runtime任何错误加载或missing site、混入错误factor、neutral10字节不一致、未重置generator、非有限值、flow/帧数/几何不全、资源超限、未授权和计数predebit不闭合均立即停止并保留已写RUN/外层真实EXIT，不自适应改位点、换源或自动重试。未完成结果只保留有界失败身份，不宣称质量、速度或机制确认。
