# 保存latent的步进signed几何：Wan四步scheduler源码与代数审查

2026-10-27 · SRC/ALG / DISCUSSION_ONLY。**当前资产允许先研究真实状态误差怎样逐步叠加，不需要立即新增32次full-flow。** 本轮蒸馏路径在相邻step间没有图像条件覆盖；原生BF16 post-state升格保存为FP32，下步pre的cast对这些值幂等。因此保存的实际initial state与四份post-state可用于步进增量恒等式。除以实际sigma差得到的是有效状态速度，包含原生舍入、处于各自轨迹的模型响应，不能叫作恢复出的`noise_pred`或同状态算子误差。

本稿只读取源码及小JSON元数据，没有打开实际safetensor数据区、计算真实tensor统计、运行Torch/GPU/模型/新评分，亦未修改运行文件、配置或ACTIVE。后述CPU统计尚待协调者冻结独立范围。

## 1. 精确源码路径：这里不是vanilla Wan2.2 scheduler

所有路径均指当前 `/home/xiang/projects/FFNDS`。下表链接为仓库相对源码，行号来自本次逐行读取。

| 入口与源码行 | 实际作用 | 对保存state解释的约束 |
| --- | --- | --- |
| [原生合同](../../../../.runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/lightx2v_wan_distill_contract.json) L4–6、13–15、29–38 | `model_cls=wan2.2_moe_distill`，`Wan22StepDistillScheduler`，sample_shift5，CFG关闭，四步1000/750/500/250，boundary_step_index2。 | 不采用vanilla `wan2.2` 的条件prefix规则；四步是high/high/low/low。 |
| [Wan22MoeDistillRunner](../../../../.runtime/code/LightX2V/lightx2v/models/runners/wan/wan_distill_runner.py) L120–121、156–189 | 两个`WanDistillModel`和`MultiDistillModelStruct`；`NoCaching`时建立指定step-distill scheduler。L57–79按step索引切expert。 | expert切换改变模型，不在边界修改latent或重新加噪。 |
| [DefaultRunner.run_segment](../../../../.runtime/code/LightX2V/lightx2v/models/runners/default_runner.py) L228–255 | 每步严格`step_pre → model.infer → step_post`，之后可调用progress callback。构造时L90将callback设None，本任务driver没有安装它。 | 保存post之后、下次step_pre之前没有额外latent动作。 |
| [BaseRunner](../../../../.runtime/code/LightX2V/lightx2v/models/runners/base_runner.py) L231–244；[DefaultRunner.init_run](../../../../.runtime/code/LightX2V/lightx2v/models/runners/default_runner.py) L432–442 | 默认只有一个video segment；请求开始时一次prepare；segment初始化只设segment_idx。 | 本合同没有分段prefix拼接、每步reset或第二次prepare。 |
| [WanStepDistillScheduler.prepare](../../../../.runtime/code/LightX2V/lightx2v/models/schedulers/wan/step_distill/scheduler.py) L21–33 | 创建FP32随机初始latent，随后生成FP32 sigma表并选取四索引。 | 生成随机数的dtype与进入首个模型调用时的dtype需要分开。 |
| [BaseScheduler.step_pre](../../../../.runtime/code/LightX2V/lightx2v/models/schedulers/scheduler.py) L22–27；[dtype查询](../../../../.runtime/code/LightX2V/lightx2v/utils/envs.py) L31–43 | `keep_latents_dtype_in_scheduler=False`；若常规/敏感dtype相同，则将latent转成常规dtype。 | 这是本轮必须计入的cast边界；保存FP32不能说明原生一直FP32。 |
| [WanScheduler.step_pre](../../../../.runtime/code/LightX2V/lightx2v/models/schedulers/wan/scheduler.py) L353–357 | 除父类cast，只设置当前timestep_input；只有条件prefix分支才使用mask调整时间输入。 | 当前蒸馏model_cls不进入prefix分支。 |
| [WanPreInfer](../../../../.runtime/code/LightX2V/lightx2v/models/networks/wan/infer/pre_infer.py) L94–126 | 取`x=scheduler.latents`；I2V条件编码作为`y`，`torch.cat([x,y],dim=0)`后patch embedding。 | 图像条件是网络输入的额外通道，不是对scheduler.latents原地覆盖；cat/conv生成网络中间量。 |
| [WanModel](../../../../.runtime/code/LightX2V/lightx2v/models/networks/wan/model.py) L103–122、191–197；[WanPostInfer](../../../../.runtime/code/LightX2V/lightx2v/models/networks/wan/infer/post_infer.py) L17–30 | 模型返回unpatchified的`u.float()`，CFG关闭时直接将该结果赋给`scheduler.noise_pred`。 | `noise_pred`是模型原生输出边界，不是从下一latent计算的量。此路径不修改scheduler.latents。 |
| [Wan22StepDistillScheduler.step_post](../../../../.runtime/code/LightX2V/lightx2v/models/schedulers/wan/step_distill/scheduler.py) L80–89 | FP32 `flow_pred`；先算`latent−sigma·flow`，非末步再加`next_sigma·flow`，最后cast回原latent dtype。 | 真正的update是顺序FP32算术再cast；不是在实数中一次FMA执行`z−h·flow`。 |
| [本轮保存回调](../../../../tools/run_ub1_consumer_disjoint20.py) L449–478 | 首个original_post之前保存真实原dtype latent；original_post之后`.float().cpu().contiguous()`保存四post-state及各sigma。 | initial_noise实际是首个post更新之前、已经经历step_pre的状态；不是必然等于刚由torch.randn产生的FP32字节。 |

此外，[adapter hooks](../../../../.runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/lightx2v_wan_distill.py) L726–773只建立expert/step/block上下文并替换所选MM，未写scheduler.latents；[build pipeline](../../../../.runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/lightx2v_wan_distill.py) L845–854核对scheduler类和四步/boundary。源码全树中`calculate_alpha_beta_high`只有定义，没有本路径调用；不能因该函数存在就向实际update添加额外high-noise变换。

## 2. 图像条件与dtype的两个容易混淆的边界

**条件没有在step之间覆盖latent。** [WanScheduler._uses_conditioned_latent_prefix](../../../../.runtime/code/LightX2V/lightx2v/models/schedulers/wan/scheduler.py) L42–53只接受`wan2.2`/`wan2.2_matrix_game3`。当前model_cls是`wan2.2_moe_distill`，故L94–96的初始prefix替换不触发；当前step_post又是子类L80–89，未调用父类L359–399的UniPC/条件prefix回写。图像mask及VAE条件由[WanRunner](../../../../.runtime/code/LightX2V/lightx2v/models/runners/wan/wan_runner.py) L627–674构建，再由pre_infer拼接到模型输入。把它们叫作“每步固定首latent帧”会错认本模型路径。

**实际原生状态是BF16，保存容器是FP32。** [pipeline](../../../../.runtime/code/LightX2V/lightx2v/pipeline.py) L4–6默认`DTYPE=BF16`、`SENSITIVE_LAYER_DTYPE=None`；GET_SENSITIVE_DTYPE在None时返回GET_DTYPE，BaseScheduler因而cast成BF16。子类step_post计算后cast回同BF16。仅只读本轮四part首个Dense的PHASE小JSON，`initial_noise.dtype`均为`bfloat16`，与此路径相符；没有打开它们的tensor数据。本轮报告字段`trajectory.dtype=float32`对应保存前的`.float()`，不是相反的原生dtype证据。

因此，对非首步而言，上一步post原值`q∈BF16`被精确升格保存成FP32；下一step_pre把原生`q`再cast至BF16是幂等的，值不变。pre_infer只读取它，随后cat/conv不别名写回。首步initial_noise在step0模型已运行、original_post尚未运行时捕获；该模型调用没有latent写回，因而其值仍等于首步post的实际输入。完整路径支持把该文件作为s=0的previous state。

这里的结论依赖当前完整源码链及实际dtype元数据。未来若改SENSITIVE_LAYER_DTYPE、keep_latents、model_cls、进度callback、changing_resolution、CFG或条件注入，不得自动套用“post值等于下一pre”。CPU分析应记录native state dtype与serialized dtype两个字段，原有结果的FP32保存身份保持不变。

## 3. sigma差是确定的尺度，实际记录优先于名义步号

源码生成\(\sigma(q)=5q/(1+4q)\)，在名义索引q=1、3/4、1/2、1/4取四点。实数参考sigma为`1,15/16,5/6,5/8`，末端sigma是0。因此正步长h_s=σ_s−σ_(s+1)的实数参考为：

| denoise step索引s | 名义denoising_step_list | 实数参考h_s | 用于理解的近似 |
| --- | ---: | ---: | ---: |
| 0 | 1000 | 1/16 | 0.0625 |
| 1 | 750 | 5/48 | 0.1041667 |
| 2 | 500 | 5/24 | 0.2083333 |
| 3 | 250 | 5/8 | 0.625 |

这里只对源码常量做代数推导。实际sigma表用FP32算术；本轮小JSON已记录`[1.0,0.9375,0.8333333134651184,0.625]`。未来分析必须从每条轨迹记录读取并核对Dense/candidate相同sigma，以其差计算h_s；末步使用0，不能把没有写入表的第五项当作缺数据。不要直接用“750→500差250/1000”作为步长，也不要把名义step标签当作网络实际timestep值。

**竞争解释**：末步h是首步的10倍、前一步约3倍。若两轨迹的有效速度差范数相近，状态增量差范数便会相应放大，能量则按h²放大；仅见末步latent误差增长，不足以认定末步函数拟合更差。它同样不能排除早步差异被后续响应放大。

## 4. 可从保存状态做的恒等式：真实增量，不命名缺失flow

对每个source/seed，令\(z_{-1}^a\)为各臂实际initial_noise，\(z_s^a\)为该臂s步保存post-state（s=0..3）。全部先从保存值精确升格至FP64再相减，a∈{D,C}：

\[
e_s=z_s^C-z_s^D,\quad
\Delta_s^a=z_s^a-z_{s-1}^a,\quad
d_s=\Delta_s^C-\Delta_s^D.
\]

\[
\boxed{e_s=e_{s-1}+d_s},\qquad
\boxed{\|e_s\|^2-\|e_{s-1}\|^2
=\|d_s\|^2+2\langle e_{s-1},d_s\rangle}.
\]

初始文件逐字节一致才设e_(−1)=0；这使第一步e_0=d_0。上式对任何一对保存状态序列都成立，甚至不需要Euler假设。完成源码审查后的额外价值，是这里的\(\Delta_s^a\)确实跨越一次原native update，未混入未记录的条件overwrite；恒等式本身不能证明这个语义，也不是机制因果证书。

建议记录的三个基本量是`previous_error_energy`、`increment_difference_energy`、`twice_previous_increment_dot`，另存`current_error_energy`、闭合误差、cos（零范数时null）。能量增长可能主要来自正增量能量，也可能来自与旧误差同向叠加；负cross表示该步增量有抵消旧误差的分量，不保证总能量下降。

若P固定为沿latent视频时间轴的一阶差分，输入形状`[16,21,58,104]`，则`Pz=z[:,1:]-z[:,:-1]`，输出时间维20。线性性给出

\[
Pe_s=Pe_{s-1}+Pd_s,\quad
\|Pe_s\|^2-\|Pe_{s-1}\|^2
=\|Pd_s\|^2+2\langle Pe_{s-1},Pd_s\rangle.
\]

P全来源/全step固定，不根据观测运动选ROI或warp。它是latent视频时间方向的数学诊断，既非RAFT、物体速度、动作语义，也不是沿四个denoise-step求差。

## 5. 可以归一化成effective velocity，但不能逆Euler恢复noise_pred

把真实FP32顺序算术及末尾BF16 cast的总差记为\(\rho_s^a\)，在实数记号下可以**定义**：

\[
z_s^a=z_{s-1}^a-h_s v_s^a+\rho_s^a,
\quad v_s^a=\text{该次实际模型的noise_pred}.
\]

这里只有v的边界由源码定位，v和ρ都没有保存。实际update先减σ_s·v、再加σ_(s+1)·v，与在FP32中重算一次`z−h·v`也可能不逐位相同，末尾还有BF16舍入。因此可以从保存状态定义

\[
v_{s,\mathrm{eff}}^a=-\Delta_s^a/h_s
=v_s^a-\rho_s^a/h_s,
\quad
w_{s,\mathrm{eff}}=v_{s,\mathrm{eff}}^C-v_{s,\mathrm{eff}}^D=-d_s/h_s.
\]

这给出纯状态数据的尺度诊断：

\[
\|d_s\|^2=h_s^2\|w_{s,\mathrm{eff}}\|^2,\qquad
\|e_s\|^2-\|e_{s-1}\|^2
=h_s^2\|w_{s,\mathrm{eff}}\|^2
-2h_s\langle e_{s-1},w_{s,\mathrm{eff}}\rangle.
\]

后一个式子只是前一个状态恒等式的单位变换，不新增可识别的因果量。两臂v分别处于各自state，故即使ρ忽略，它们之差仍混合直接算子变化与状态响应。

**为何不能称“恢复原flow”**：BF16 cast是多对一；一个输出舍入区间内，不同FP32 update甚至不同模型预测可以落在相同post-state。已知z_prev与z_post也不能唯一求回v。小h还会把\(\rho/h\)放大，早步可能有较明显的有效速度舍入影响；某坐标`post==prev`也不证明该坐标v=0。FP32保存仅精确保留已有BF16值，无法找回舍去的update位。当前条件没有prefix覆盖，是排除了一个混杂，而不是解决了这个不可逆问题。

## 6. 这项CPU观察能区分哪些竞争解释

| 可证伪的预测 | 建议观察 | 合法解释范围 |
| --- | --- | --- |
| **纯步长尺度是末步大增量的主要解释** | 比较各源raw `||d_s||`与归一化`||d_s||/h_s`；同时看原始/归一化cross。若后者没有末步峰，而前者随h放大，则该解释更有力。 | 归一化量还含舍入及不同state；不是控制住噪声分布后的模型因果实验。 |
| **末步有新的方向性叠加** | 末步`2<e_prev,d>`大且正，或归一化增量差也明显增长；若大增量仍与旧误差相反且距离改善，则“末步同向积累”预测被削弱。 | 只能说当前两条实际路径的增量关系，不分出当前W2误差与Dense对状态的反馈。 |
| **先前误差已有纠偏，但新分量能量抵消其收益** | cross负而`||d||²`超过抵消幅度，出现当前误差更大。 | 支持几何上的部分抵消/新增方向，不自动意味着λ应调整或局部容量不足。 |
| **latent视频时间变化受到不同影响** | 同时报告P=I与固定时间差P的三项signed分解；若P方向与全局结果不同，说明全局SSE掩盖了时间变化方向。 | 不是运动真实性证明；若只P=I变化而P差分稳定，不能说动作改变。 |
| **原生低精度update是需保留的竞争项** | 分析前核实际noise.dtype，并可在未来CPU协议中核post FP32值是否均可由BF16无损表示；可另预先登记每臂`post==prev`坐标比例等描述量。 | 这些不能估计缺失ρ或v；若要隔离舍入，仍需另做同状态/同flow控制的新native调用。 |

如果源级raw/归一化与cross模式已能排除最简单的“末步函数一定更差”说法，先完成这个低成本观察有信息价值。若状态响应与直接算子误差依然是关键分歧，[此前32-full-flow提议](ub1-consumer-cross-mechanism-synthesis-20261027.md)仍是严格不同的下一证据；本CPU恒等式不能取代该2×2同状态交叉。

## 7. 供协调者冻结的最小CPU对象与分析边界

可只读新20源的两臂实际initial_noise与四post-state、对应immutable GENERATION/RUN、canonical图/prompt/seed和source/image cluster身份。每source每step的四张量差分在FP64内分块归约即可；P=I和固定latent-time差P先冻结，不需要模型权重、bank payload、视频解码、FFT、光流或learned metric。结果以20个实际独立来源为单位，四step/21槽嵌套，不将80个step条件当80个独立样本。

分析前需要核完整文件SHA/shape/serialized dtype、Dense/Candidate source/seed/prompt/condition/sigma、初始文件真实字节一致，以及source的两个generation已完整。只按预冻完整性规则纳入，缺失不填补、不重跑，不按误差方向择例；父RUN的资源失败/其它状态须原样保留，CPU代数分析不为原生执行重新授予PASS。此处不据已有文件名推断所有20pair均已验收。

建议字段明确写`actual_state_increment`、`effective_velocity_from_state_difference`、`native_noise_pred_saved=false`、`native_state_dtype=bfloat16`、`serialized_post_dtype=float32`、`new_native_flows=0`。先给逐source四步signed三项及有效尺度，再给来源均权汇总和subject/background分层；所有比值的零分母规则和闭合容差事先冻结。新的CPU协议即使无模型，也应绑定实际source与byte预算；本稿没有执行该分析。

## 8. 本次读取的源版本

以下是只读源码SHA，方便新CPU协议绑定；不是给当前原生protocol追加门槛或修改它：

| 文件（LightX2V相对路径） | SHA256 |
| --- | --- |
| `lightx2v/models/schedulers/scheduler.py` | `0b43c8452ee734fffbd0a753f8176a5b530980723aa54981b34a8070c68db7f7` |
| `lightx2v/utils/envs.py` | `68c2f49686cb58559e8f1beaa0641b1141b5639083fe4ec3e8df771395c3189c` |
| `lightx2v/models/runners/wan/wan_distill_runner.py` | `26a2a503137b0125cf4a2f8b2c9fc5dae716121c6e600b7ce85517af5ee62dc3` |
| `lightx2v/models/networks/wan/model.py` | `f0b0e9904d9708ff67412d8c98cb3f384b86426bb96ebf84d17e439e7a8580ff` |
| `lightx2v/models/networks/wan/infer/pre_infer.py` | `f3f88ca8ccbbd778206f8f04805c124beb84c19dcd22c4c38a6028b42c1502b5` |
| `lightx2v/models/networks/wan/infer/post_infer.py` | `6567c439e2f16f322a9f10cd2ca326d3f924e9da75b3ae2ae442ec1327a327a2` |

其余scheduler/runner/adapter及保存driver本已在40视频冻结协议中绑定；新CPU scope仍应引用那些原始身份。本文只新增源码解释与条件代数，复用既有P28及finite-map讨论，不新增图节点或数值结果。
