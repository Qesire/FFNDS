# 固定完整 A、压缩锚残差：有辨别力的实验设计

作者 /root；2026-11-05 北京时间。状态 DESIGN_DRAFT / 待独立精确版本复审。本文件是研究设计，不是新实验的 ACTIVE 或执行收据。原40步133998/133999已完成退出0，结果与质量另行验收；本方案不修改它们。

## 1. 实验必须作出的决定

研究目标是相对同模型 Dense，在真实视频质量保持的条件下，降低 FFN 实际执行时间、端到端耗时和工作集。首轮保持当前全部 A 矩阵、原生 BF16 数值族和锚点策略，专门改变历史残差表示。缓存最小不是唯一目标；必须允许容量稍大但更快的方案获胜。

首轮回答四个可以改变架构选择的问题：

1. 固定 A 后，锚残差是否存在跨来源可复用的低维通道表示，还是只在各个来源内部可低秩？
2. 残差是否依赖视频时空邻接，因此可先压到小网格再做通道编码？空间和视频时间分别贡献多少？
3. 历史有效来自残差本身小/低秩，还是来自高维残差随去噪过程变化小？
4. 压缩后的误差是否真的小，还是被原预测误差抵消、被后续动力学削弱？哪种格式能把规则 GEMM 的算术收益变成真实时间收益？

不将「学得 U 失败」直接判为「残差不低秩」，不将「视频还能看」直接判为「残差重建精确」。

## 2. 固定对象与数值边界

代码依据：video_template/anchored_hook.py、normal_pipeline.py、cal_observer.py。实际行矩阵计算是 `pred=(u @ A).float()`，输入 u 和 A 都是 BF16；F 是完整原生 W1→tanh-GELU→W2 含偏置的 BF16 输出，位于当前 gate 之前。实际 C=F.float()−pred。数学下文 token 作列，A 相应转置，保持同一个算子。

原40步正常模型，两个独立 expert，CFG=3.5；实际每步只使用一个 expert，40层×2 CFG=80个位点/步，共3200个位点。当前 grid=(21,29,52)，T=31668，d=5120，h=13824。high 步0–14、low 步15–39；实际锚点0/10/15/20/30，窗口末9/14/19/29/39。所有新scope仍从真实dispatch推导窗口，不跨expert外推缓存。保存 actual timestep、solver参数、branch、layer、source、seed、坐标与dtype。

A绑定现 matrix_manifest SHA 41863779c03b85bfdb1f6140c73a78b10f426b7f35d8c0f61ebf6fbafcc81ae2，原权重manifest SHA df3e1fe3654b6d5af48d1193f7d88e99d4224e7dbc7a6ed70b47afa87323abba。每个新的运行范围仍需重新核验真实文件，引用此SHA不代替launch校验。

命中核心是规整的 BF16 GEMM：M=31668、K=N=5120，无动态稀疏索引/逐token路由；50MiB/层的A，当前expert40层约1.953GiB。主GEMM MAC仅占原两层FFN的 d/(2h)=18.5185%。完整路径还包含残差读取、解码、相加、舍入、gate与后续残差。当前hook还保留输入clone/诊断、无用GELU中间计算和整block预取；clean执行不能沿用这些开销后声称已充分优化。

## 3. 误差对象及可直接核验的恒等式

同一 Dense 轨迹上，令 R_t=F(X_t)−A X_t，Delta_t=R_t−C_s，c_s=C_s−Ctilde_s。忽略本段明确分离的最终舍入时：

    e_raw(t)=Delta_t,
    e_codec(t)=Delta_t+c_s,
    ||G_t e_codec||² = ||G_t Delta_t||² + ||G_t c_s||²
                         + 2 <G_t Delta_t, G_t c_s>.

每一项都保存，另保存 native BF16 最终输出误差及「float公式−native结果」的舍入余项。若合计误差降低而 c_s 不小、交叉项显著负，则结论是误差抵消，而非缓存准确。该效应需跨来源/seed检查，不能以一次改善推广。

另将创新分成均值与零均值纹理：当G_t只作用于通道时，||G_t Delta_t||²=T||G_t(mu_t−mu_s)||²+||G_t[(R_t−mu_t1^T)−(C_s−mu_s1^T)]||²。这样可辨别「动态均值已足够」与「高维纹理很重要但缓慢变化」，两者对应完全不同的缓存结构。所有归一化同时保留绝对能量、相对残差能量、相对FFN输出能量和零分母标记。

完整当前 A 不变，mu_s 单独存 FP32，C0_s=C_s−mu_s 1^T。统一结构为

    Ctilde_s=mu_s 1^T + U Z_s W,
    Z_s=E C0_s W^T.

保留 A 不等于改写 native 舍入路径。先池化再乘 A、预计算 EA、最后BF16量化，都须与原 full-M `u@A` 形成的 C 单独比较。统计时先算真实 C，部署时再检验代数重排的额外误差。mu 恢复必须在最后量化后核验；需要时补偿DC。

## 4. 机制预测与最小对照

|假说|能区别的观察/干预|结果如何改变方案|
|---|---|---|
|H1 跨来源共享通道低维|CAL冻结U在DEV保留能量，接近DEV全网格最优共享rank谱尾；同rank随机正交基更差|优先通道codec；随机基同样好时不声称学得独特子空间|
|H2 各来源内部低维，但基随source变化|DEV自身最优谱尾小，CAL-U在DEV误差大；CAL源数增加能改善源外风险|先补来源/研究结构基；不能宣布本质高秩，也不能在线免费拟合每视频基|
|H3 视频时空邻接形成可压缩性|共同置换token→同一空间codec→逆置换后，误差明显高于原坐标codec；通道协方差与谱严格不变|支持固定时空编码；量化/通道对照基本不变是干预合法性检查|
|H4 历史主要靠去噪创新小|C_s中心化谱尾不小，但Delta_t幅值显著小于C_s；no-history和mean-only明显差|保留更多历史方向，以低比特/时空表示为先；创新低维仍属后续待测，不将C本身强行低rank|
|H5 主要是均值/偏置校正|Au+mu与raw G19接近，复杂codec无稳定额外收益|优先极小mu缓存；同时报告均值解释比例和其跨步漂移|
|H6 输出对特定误差方向不敏感/跨步抵消|同state重建误差较大，但完整rollout稳定；同能量方向扰动响应明显不同|转向带下游敏感度的保留规则；不能从门控范数推导视频误差保证|

H1–H6可以同时成立；实验报告贡献量与反例，不强制单一标签。

结构主臂在同一真实 C 上并行统计，不为每个臂重新生成视频：

- R0：完整 FP32 C，现有G19参考。
- R16：完整 BF16 C，最低复杂度的强压缩基线。
- R8：分块INT8 C，块128元素、明确scale/布局/舍入；先只一种冻结格式。
- M：仅mu；Z：零历史，仅Au。二者主要作机制对照。
- S：保留全部通道，只做固定正交时空压缩。
- C：仅通道投影，r∈{128,256,512,1024}，使用同一嵌套CAL基。
- CS：通道+固定时空压缩；先筛查可达误差与字节，不展开所有组合的视频网格。
- CR：固定随机正交通道基，同rank/同mean/dtype，至少3个预注册随机种子，只作局部机制对照。
- SP：共同token置换→S→逆置换，至少3个预注册置换；不改变F、X、网络本身的位置，只改变codec处理残差的邻接。

对SP必须逆置换，否则是把历史注入错误位置。相同置换作用于所有通道；独立通道置换会改变通道关系，属于另一问题。R8的分块分组可能随置换变化，因此「置换不变」只要求无分组量化的R16/纯通道投影等合法检查，不强加给R8。

该负对照还有精确的条件预期：固定空间正交投影rank K并包含常数方向时，对任一已中心化场C0，均匀共同置换Pi满足 E||C0 Pi W^T||²=((K−1)/(T−1))||C0||²。它无需假设token独立；来自置换平均投影在常数空间及其正交补上的对称性。原坐标保留率若远高于此值，才是邻接组织的额外收益。三个置换是诊断重复，不能当三个新source，也不未经交换性论证就报普适p值。

纯空间与纯视频时间压缩采用正交子空间分解：P_tau、P_xy是作用于不同轴的交换投影，分别保留/舍弃形成四个正交频带。保存每带能量，直接识别视频时间、平面空间和混合高频贡献。2×2×2分组在奇数边界用真实块大小正交归一化；不得填零后把padding当真实token。纯空间和纯时间的K不同，不能冒充同字节率比较；另画实际byte–distortion曲线作成本比较。

## 5. 不依赖64行秩上限的全网格谱证据

每个anchor场先显式减自己的全token通道均值，再求全网格数值Gram：

    S_{j,e,l}=平均_{本source的该expert anchors和两CFG}
                  (C0_s C0_s^T / T).

每source、expert、layer一个5120×5120 FP32矩阵，共80个、7.8125GiB。该平均对应「一个expert/layer共用同一个通道基、anchor分别带mu、两CFG共用基但分别编码」的明确部署假设。跨source平均时source等权；另按真实2个high/3个low窗口分别保存计数，不把token当独立source。

source自身最优共享rank误差为 tr(S_j)−sum_{i<=r}lambda_i(S_j)；CAL冻结正交U在DEV的误差为 tr(S_j)−tr(U^T S_j U)。后一个式子只用于未量化、列正交U的欧氏正交重建；保存对应未量化直接投影误差作为Gram核验，BF16 U/E、EA重排、coeff量化及最终写回的native误差另算，不能用该trace式代算。因此可以区分：

1. source自身最优尾大：否定这个rank、共享范围与度量下的线性子空间精确表示；
2. 最优尾小但CAL-U误差大：基的迁移/拟合有问题；
3. 两者都小：支持共享codec进入原生量化与闭环验证。

source自身oracle允许在DEV数据上拟合，不能部署，也不是无泄漏测试成绩。平均各source各自oracle尾≤跨source共用一个U的最优尾。若将来每window/CFG分别换基，此pooled谱尾不能否定那个更自由的类；需另行分组Gram及计入基库。首轮不从80池声称每个单field自身的精确rank。

「全网格」不是无限精度。Gram产品FP32、TF32关闭；先中心化，避免二阶矩减均值的消减。保存FP64 scalar trace参考、mean、对称误差、累加顺序、特征对残差和正交性、负特征值尾。线性代数后处理单独在计算节点运行，逐矩阵读取，禁止forward中做SVD。若Gram数值不确定性与声称的尾同级，不下rank结论。源外在线直接重建误差也用于校验Gram推算。四个落盘完整field只能直接复核这四个field自身的投影/舍入误差，不能认证其余80池或某池全部anchor。任意pooled谱边界结论需要重新捕获其对应完整fields进行直接投影/更高精度复核，另审采集scope；资源不足时保留数值不确定，不把四点验证泛化。

主Gram是未门控指标。跨anchor合池会丢失field与目标gate的配对，不可事后声称从此池精确恢复每个目标step的门控误差；门控和量化后的codec误差在冻结U的DEV现场逐field计算。若要学习gate加权U，必须另行冻结固定度量或保存对应加权统计，不能事后变更估计目标。

非AI数学依据包括数值线性代数的POD/Ky Fan谱尾，以及独立随机误差检验。若需降成本而使用sketch，必须用与拟合独立的Gaussian检验或有界PSD近似证书；不得把sketch的截断谱当原矩阵真谱。首轮优先全网格Gram，sketch作为资源受限时重新审定的替代方案。

## 6. 一次轨迹收集什么，哪些东西刻意不重复运行

每源完整40层、40步、实际两个expert、两CFG。full-field主采集为5个anchor和5个窗口末，共800个位点；在anchor计算全网格Gram和各codec统计，窗口末计算全网格Delta与交叉项。其余2400个位点保留配对样本统计，增加中间尖峰的可见性，但不能排除未采到的局部尖峰。均匀近似下单个异常token被1024/31668抽样发现的概率仅约3.23%，固定样本可能各步持续漏掉它。全网格位点的精确分位/最坏，与其余位点的加权样本分位/样本最坏分开标记；只在发现具体风险时另注册中间step全网格补采，不能宣称窗口末误差单调代表全窗口。

首轮没有另存Delta的通道Gram，只能判断创新幅值、均值/纹理分量和signed交叉，不能称创新低rank已证。为避免膨胀预算，不默默增加另一套7.81GiB Gram；如要判断创新低维，后续另审固定投影误差或创新谱的采集范围。

CAL尚未有冻结U时只做Gram和无需学习的格式；C/CS的完整源外统计在U冻结后的DEV做。现场对每组候选同时评价不意味着事后从Gram能恢复全部量化/时空误差。

所有3200个位点：

- 全网格u的均值、RMS、方差/能量及分组总量；group=(视频tau三段, y四段, x四段)，共48格，保存每格真实count。
- 预注册同坐标1024个分层随机token，在每格内无放回抽样、固定source/expert/layer seed、两CFG同坐标；保存Ng/ng权重。每步观察u与真实full-batch F选中行，计算输入变化和残差创新的样本统计。小M A舍入与full-M不同，在800个full-field位点测二者差异；非full-field只标SAMPLED_NUMERIC_PROXY。
- 在同一轨迹的当前W1输出与GELU输出采集分组pre/post均值、方差、固定histogram和基于原生tanh-GELU连续公式的曲率代理。不能把BF16离散执行称为可微函数定理。
- 按expert/layer/branch/anchor-age、实际timestep距离、u范数分位组、输入增量分位组报告误差，分位边界只在CAL冻结。每格报告输入能量、残差能量、codec误差、gate后误差和signed交叉项；不只存总均值。
- 两个预注册连续3×4×8小块用于时空一/二阶差分，坐标与边界明确；与随机样本分开汇总，不把刻意过采的连续块等权当全局抽样。去噪二阶差商按真实timestep间距计算，跨expert不连接。

分组统计轴明确为：每token先在通道轴求mean、mean-square、variance、norm，再按token组用真实数量或采样权重归约scalar的均值/分位/尾；pre/post直方图在该组采样token×hidden-channel上按冻结bins归约，保存加权count和分母，不存每组每channel直方图。默认禁止48格×全部通道×每step的向量大表：仅pre/post每channel mean/var就会约31.64GiB/source，超过本方案预算。少量全局per-channel摘要若实现需要，必须单独列字节并仍满足20GiB硬界。所有input统计有明确full-grid或sample标志。

1024行与连续块默认只在线累计统计，不每步写全部向量。raw审计保留：每anchor固定128条包含u/pre/post/F的原生行（约3.61GiB/source，不含封装），四个预定完整field的BF16 u/F/实际Au（约3.62GiB/source），加80个Gram约7.81GiB，总基础载荷约15.05GiB/source，统计/metadata后磁盘上限20GiB/source。完整field审计点冻结为(0,0,conditional)、(10,13,unconditional)、(20,26,conditional)、(30,39,unconditional)，只用于数值核验；主统计仍覆盖全部层，不能拿四点代表总体。

全网格C不全落盘：每source仅anchors也需241.61GiB，40步则约1.888TiB。为窗口末比较，在host只保留当前expert的anchor C，共48.3215GiB，换expert释放；输入sample anchor状态另有明确上界。统计未保存的原始信息不能日后宣称可恢复。

## 7. 来源、seed、拟合与检验隔离

所有chip10来源一律视为开发暴露，旧planned HELDOUT名称不赋予盲性。主结构采集全部使用seed43，便于只比较source变化，并与本轮00/05的Dense hash作观测中性核验。

- Codec CAL四源：02室内、04动物/水体、07植物近景、09多人动作；当前A仍是已冻结的旧A，绝不重新拟合。
- DEV六源：00抽象液体、01建筑/海岸、03人物舟船场景、05交通、06静物、08单人场景。source列表是开发设计，实际图像与prompt/文件hash沿现cohort审计。
- 首个有限producer先两CAL源02/09，用于容量/耗时/数值观测校验；通过实际结果复审后再另外两CAL源。四源Gram接受后离线训练U、冻结r嵌套基与全部transform定义，再采六DEV源。不能在DEV上调U后仍称该DEV源外验证。
- 随机seed稳健性是后续独立轴：预定CAL02/09和DEV00/05增加seed44；与43形成同source配对，尤其观察subspace与高频尾。若未做，只报告seed43下source迁移，不外推所有噪声。

如果出现「source oracle好而CAL-U差」，再启动同总行数的1/2/4来源子设计，均衡全部CAL子集而非挑最优源，另保留固定source加行数轴。仅比较一源全网格与四源全网格同时改变source数和行数，不能定因。此扩展需要新增原始行预算/独立scope，首轮不默认采几十GiB行来回答尚未出现的问题。

## 8. 下游敏感性：按疑点触发，不先做巨型雅可比

先看同state误差与真实自由生成是否一致。若低重建误差对应明显质量下降，或高误差却基本无损，才做有限方向的非侵入式可观测性实验。

固定一个anchor window，在未压缩G19轨迹的锚点缓存加入v；v在该窗口所有命中重复使用，而不是每步重新抽噪声。设从该缓存扰动到后续输出的线性化映射H=sum_t L_t G_t，则窗口敏感度是O=H^T H，包含跨时间交叉项，不能替换为sum_t G_t^T G_t。

每个选定window用真实codec舍弃方向v_c与同Q窗口能量、同DC约束的随机方向v_r比较。选取基于CAL规则的高/低各一个window和两DEV源，先小范围完成一组。±epsilon扰动与±epsilon/2检查响应线性范围；如果BF16将扰动舍入掉，报告不可分辨并调整经审范围，不能给零敏感度结论。输出使用后续latent/完整RGB与已冻结视频质量向量；仅latent Gram不当作质量定理。

受限方向Gram由输出中心差分的内积构成；只对实际探测子空间得结论，不声称构建整个全维O。单个配置两方向×两幅度×正负=8次rollout，代价高，故是歧义出现后的专项范围，不加入首轮全部source×window的笛卡尔积。末端真实视频codec验证先于大规模敏感度拟合。

这借用控制与模型降阶的非侵入平衡思想，但Wan不是其原文线性稳定系统，不能照搬全局误差上界。

## 9. 执行鉴别：将GEMM优势与缓存/传输瓶颈分开

结构初筛后只保留2–3个有希望的格式进行原生执行测试：优先简单R16/R8、纯时空S，再与C/CS竞争。比较包括：

执行可行性的第一小段前移至E1a之后：先复用试点实际field，测完整XA+简单缓存格式的anchor/hit路径，不必等10源统计结束。若某格式明显被简单格式支配，停止其昂贵实现。此时S的时空压缩质量尚未知，计时仅证明执行潜力。U冻结后再测C/CS；完整clean视频benchmark仍在质量候选明确后进行。

复用已保存F/C所做的计时只能称codec-only；完整anchor计时必须包含真实Dense W1/GELU/W2及编码，或者逐项标明后按实际关键路径组合，不能把事先计算F当免费。纯S同时区分「先算全C再pool」的参考实现与「先pool X/F再在K位置乘A」的重排生产实现；前者仍付T d²，不可据其慢否定后者。E4a只淘汰被测实现/预算，不把局部kernel成绩当完整block或视频速度。

1. 仅 `XA`（真实M/K/N、BF16、同matrix layout）；
2. 从u可用到FFN gate前输出可用的完整anchor/hit路径，包含读取、编码/解码、mean、quant、临时buffer；
3. 包含gate/残差及offload/prefetch的完整block，使用相同非FFN工作；
4. 完整40步视频wall及冷启动准备成本。

每种格式比较host与GPU驻留的可行配置；R0全GPU放不下时记录infeasible，不假装已测公平GPU基准。Dense允许相同合理的offload/预取优化。两种等价CS执行必须比较：每hit重新算UZ，和anchor算一次D=UZ后保存D。K=4290/r512示例后者历史约3.273GiB，前者约0.327GiB；同时保留D和Z不能只报小者。纯时空S同样约3.273GiB但无需通道基，是很强的对照。

不在线SVD、不Python逐token循环、不把全维C解码后再持久保存。W1/W2预取由anchor/hit在预取前决定；单独计能跳过的FFN权重字节。去除诊断用输入clone/额外GELU/全网格isfinite、逐site同步和hash的性能干扰，但保留另一次instrumented正确性范围。生产质量路径改变后，先对同scope中性模式检查完整40latent和81帧RGB，不能只看首步。

局部计时采用CUDA events且同步包围测量窗口，计启动/同步成本；至少5次warmup、20次重复、交错顺序，报告中位数/p90。端到端clean配对用至少4个DEV source、3个重复，随机/轮换Dense-G19-codec执行顺序；相同种子、参数、硬件、模型驻留约束，报告median与source配对差。重复计时不是新独立source。

本会话在clean计时时只保留一个活跃benchmark任务，避免本会话另一作业争用同节点PCIe/host带宽；记录节点、GPU UUID、可见共租/利用率/时钟条件。不能保证排除集群其他用户的干扰，异常负载需标注并按预注册规则处理。当前两个视频在同一节点并行且带诊断，尤其不能冒充已完成单卡clean benchmark。

原40步理论固定A上限：q=5,N=40，rho>=q/N+(1−q/N)d/(2h)，主FFN最多省71.2963%。四步两个anchor时上限40.7407%。这轮缓存实验若已接近理论界而全系统仍无实质收益，下一结构必须改变当前输入算子/刷新成本；不能继续缩小系数却称会无限提高省算。

## 10. 视频与统计判据：Dense始终是最终基准

本轮原40步R0对Dense视频与全指标先独立闭合。若R0已经失败，仍可完成廉价结构发现，但不继续用有问题的R0证明codec质量，也不忽略Dense。

DEV完整视频只推进一个结构候选，与Dense及R0配对；如果候选不是R16/R8，最便宜且通过局部误差筛查的量化基线进入DEV竞争，避免复杂结构只战胜旧FP32 host实现。最多4个科学臂，未入围的随机/置换/mean-only主留在同state诊断。

预注册局部风险只用于筛查：报告codec新增gate误差相对gated F、相对R0漂移的比值，以及每source的p50/p90/p99与最坏layer/window。不以一个人为MSE阈值宣称视频无损。优先保留在多数source/window同时降低字节、没有系统性尾部放大且在误差–耗时Pareto前沿的格式；任何DEV选出的格式在blind阶段固定，不再按blind结果改rank/频带。

最终确认计划为12个经历史图像hash/近重复/图像簇审计的新source，按场景运动压力4组各3源：主体动作、镜头/全局运动、细纹理/流体形变、低运动/细节。分组以图像prompt与事先冻结的标注为准，不能看候选坏例后改组。每source两个真实seed，Dense、R0、冻结codec共72个完整视频；若DEV量化基线胜出，则它就是冻结codec。该阶段12来源是首个确认规模，不承诺足以证明很小质量差异。

独立统计单位是image-cluster，先聚合两seed，再计算source配对差。81帧、1024tokens、CFG、层与seed重复不能膨胀n。十维质量：subject/background consistency、I2V subject/background、motion smoothness、dynamic degree、aesthetic、imaging/MUSIQ、camera_motion、temporal_flickering；另做连续RAFT完整序列/预定时间窗、逐源完整81帧审阅，明确动作、主体、镜头、拖影、物体增生。人工按盲化随机臂序审阅，保存实际观看范围。

使用配对source区间及全部原值，不只平均排名。非劣效阈值必须在blind前形成单位明确的MARGIN.json：有方向的质量维用实用差异尺度，motion amplitude/camera不能机械按越大越好；同时审动作遵循与相对Dense幅值。当前不已有可辩护的统一阈值，因此设计审阅不假称非劣效合同已冻结。DEV只可帮助估计尺度/方差，不能为让候选通过而放宽阈值。

以预定主要连续指标的配对标准差sigma、非劣效margin delta作样本量规划，单侧alpha=.05/power=.80的正态近似n约6.18 sigma²/delta²，最后使用小样本修正/实际分析方案。示例sigma=4分、delta=2分需约25source，12个不足；这个例子不是本轮实测sigma。确认样本数在看确认结果之前由DEV方差和冻结margin确定；若12不够则先调整到有预算的固定N（例如24/更大另行注册），不在每12源后反复偷看显著性。预算不足时报告不确定，不把未显著差异称为等价。

性能推进建议为相对Dense端到端至少20%缩短且峰值工作集下降，属于工程目标，不是已有结果；相对R0还必须明确压缩带来的时间/容量净收益。达不到工程目标但机制发现成立的结果保留，不称部署成功。

## 11. 预算和执行次序

已测原40步instrumented Dense约653.55s/视频=10.89min，不含整个作业模型加载；neutral约23min，它计算Dense并增加A/cache/诊断，不能作为候选执行时间。后续时间均是规划，不是实测承诺。

现已取得两条原40步candidate完整收据：00为1305.23s、05为1303.42s，约Dense的2倍。这是带shadow、全网格检查、CPU缓存搬运和未跳过FFN权重预取的当前原型实测，不能宣称有真实加速，也不能把全部差额单独归因缓存。新增实验必须把「规则GEMM容易执行」与「当前完整路径是否快」分开。两个neutral全40latent/81帧RGB均与Dense一致；candidate质量仍待十维评测。收据分别见../video_chip10_00_r1/results/EXIT.json与../video_chip10_05_r1/results/EXIT.json。

|阶段|规模|现阶段预算身份|
|---|---|---|
|E0 现有原40视频及质量|2source的Dense/R0/neutral已生成；质量待实际结果闭合|不改原scope|
|E1a producer校验|2 CAL Dense轨迹、full-field+Gram+分组统计|每job1GPU、4CPU、192GiB、<=2h；按实际峰值决定后续|
|E1b 剩余CAL|2 CAL Dense轨迹|相同有限scope逐项复审，不自动重试|
|E2 Gram求基与数值核验|逐矩阵、最多80 pooled训练Gram；source oracle按需要逐一处理|独立compute scope，首批2h上限；不给每forward加eigh|
|E3 DEV结构验证|6 DEV Dense轨迹，冻结U、多臂同场统计|若每源30min则3GPUh；30min仅排期假设，按E1实际修正|
|E4 执行筛选|2–3格式，真实shape、anchor/hit、host/GPU|最多2个1h compute scope，先局部再clean视频|
|E5 DEV自由生成|最多6source×入围臂，Dense合格资产可复用|只运行有决策价值的候选，不对全部rank做视频|
|E6 确认|初拟12source×2seed×3arm=72视频|以Dense当前耗时作同速预算约13.1GPUh，两GPU约6.54h；非下界/上界，候选速度、装载、排队和指标另计|

E1/E3总10个结构Dense轨迹若按30min，约5GPUh、两卡理想2.5h；其FP32 Gram可能明显更慢，E1先测再冻结剩余预算。理论额外fullA(10步)+anchorGram(5步)主FLOP约Dense FFN的6.944%，不代表额外wall6.944%。Gram禁TF32时尤其不能借BF16吞吐。

host粗算：既有CAL peak RSS约98.07GiB，当前expert完整C48.32GiB，80 Gram池7.81GiB，合计约154.2GiB，尚需原生A、sample状态、pinned buffer、写回瞬态和page cache。该不同阶段峰值相加不是安全证明。每次仅一source/job，cgroup/RSS/GPU allocated/reserved全记；内存软阈值176GiB、GPU阈值以实际80GB卡可用空间减预留为准，producer在新site前检查预算并失败保留已完成收据，不能OOM后自动删样本换口径。每source落盘20GiB硬上限，超限失败并重审。

最新三pass实际RSS高水位约144.36GiB已包含host C；与新增7.81GiB Gram池粗加约152.17GiB，比CAL推算更接近现执行，但仍非新方案峰值上界。GPU allocated约15.34GiB不含host C。最终以新producer cgroup/allocated/reserved实测为准。

任何模型、tensor、eigh、质量指标实验均在Slurm计算节点；登录节点只做metadata/文件传输/提交。会话同时最多2个live任务（包括pending）、2GPU；逐scope单GPU，wall上限不得超过既有3h。质量评测与新采集共享这个额度，优先完成现有质量证据。失败保留，重试另版本另审。

## 12. 结果→决策规则

- R16/R8已接近R0且最快：先用简单低比特缓存部署，保留高级结构为科学发现，不为形式复杂而强行使用。
- source oracle与CAL-U均低尾、置换影响小：通道结构更主要，优先C；不能据此说A低秩。
- 通道尾高、S低误差、置换后显著变差：优先纯时空粗网格，保留满通道，甚至不需要学U。
- C与S分别有收益、联合新增尾仍可接受：比较CS每hit解码与decode-once，以实测全窗口wall选型。
- 只有Delta幅值小而C难压缩：保留高维低比特锚点；创新低维尚未测定，另开创新更新/更廉价当前算子的结构研究。
- 同state好而自由生成差：补学生真实状态/误差反馈证据，不能靠在Dense数据上继续减MSE就宣布修复。
- 重建差但视频好：做受限可观测性/误差抵消专项，而不是把重建指标删掉。
- 固定A方案已接近71.3% FFN算术节省但端到端弱：根据profile改变权重预取/缓存布局；若主要时间仍在Au，则下一结构必须改A执行，不在缓存率上继续局部调参。

## 13. 文献使用与科学身份

本设计的主体是本仓真实FFN残差和条件代数，不是将别领域全套算法照搬。数学工具负责形成可证伪预测：

- Tropp等，Streaming Low-Rank Matrix Approximation，SIAM SISC2019，Sec6：独立检验sketch及Frobenius误差估计的条件。https://tropp.caltech.edu/papers/TYUC19-Streaming-Low-Rank.pdf
- MGARD误差控制：多尺度表示和派生量误差之间需明确映射，不能把像素/latent范数直接当全部视频质量。https://www.ornl.gov/publications/multilevel-techniques-for-compression-and-reduction-of-scientific-data-quantitative-control-of-accuracy-in-derived-quantities/
- BlockMGARD2026，SecIV：GPU分块/层级与边界容差传播，启发固定快速空间实现与边界审计；首轮并不宣称已移植其全部codec。https://arxiv.org/html/2609.00205v1
- Padovan等 Non-Intrusive Balancing2026，Sec2.1：从有限方向响应识别受限可观测性，不能直接继承线性稳定系统界。https://arxiv.org/html/2610.15219v1
- Roofline原报告：实际GEMM与搬运关键路径分离。https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf

本轮已核原文相关段落；应用到Wan的贡献仍需原始证据。本方案不声称方法新颖性已经排除所有同领域工作，且本轮优先验证结构和执行价值。

## 14. 作者自审与执行前缺口

AUTHOR_SELF_AUDIT：

- 已固定A、锚策略、dtype/舍入边界、实际3200sites，首轮只改缓存。
- 全层/两expert/两CFG主统计；四个完整field只作raw审计，非选层总体结论。
- 明确source/seed混杂、oracle与部署、谱数值误差、token置换逆操作、误差交叉项。
- 新原生采集与昂贵谱分解拆scope，fullfield不全落盘；提供内存/磁盘/时间软硬边界。
- Dense是质量和端到端比较对象；R0定位历史预测损失，量化基线定位codec价值。
- 完整视频只推进少数候选；新source数量由margin/方差决定，不用token数假造统计辨别力。
- 还需实现producer、数值/观测中性单测、input manifests、完整margin合同、原生kernel与精确代码SHA，再作独立实验复审和ACTIVE冻结。设计通过不等于这些执行条件已满足。

下一步最小具体产物是E1a producer及两个有限scope，完成自审和独立代码/数值/资源recheck后才能提交。本轮文档不预填APPROVE或ACTUAL_RUN。
