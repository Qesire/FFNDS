# 独立机制挑战：局部误差几何、离轨输入与跨步纠错不是同一件事

作者：`independent_e0_audit`；日期：2026-10-26；阶段：`E0_OPEN_DISCOVERY`。

**新认识。** Wan四步scheduler下，若定义 `δ_k=student_latent−dense_latent`，理想的一步重合速度目标确为 `v*(z_k+δ_k)=v(z_k)+δ_k/h_k`，符号为正。这是既有CTEC一般affine采样器目标的特例，不是新的免费在线纠错规则。另一方面，TCEC v1 Eq8的inverse产品在原PDF也如此印刷；它与同页Eq7及初值不一致，不能归咎于HTML转换或只改一个下标就成立。此数学问题不否定TCEC实际校准组件或论文实证。

本记录亲读真实scheduler、模型head/输出路径与两篇原论文；只做源码、数学和已有JSON标量核对。没有新模型、GPU、拟合、张量实验或视频生成。论文PDF只下载至临时文献缓存并渲染页面阅读；仓库仅新增本文件，不修改先前独立审查。

## 1. 实际计算对象与版本

本记录的teacher是**原dense D-Q/HiF4参考函数**，不是论文中的未量化FP16/FP32模型。固定图锚点是 `F09→F12/G20→F13/G21→G07(head flow)→G08(scheduler)→G09(next step)`；上游 `G02` 决定两专家。局部FFN误差、最终flow误差及scheduler输入偏移分别位于这些不同边上，不能互相代替。

| 实际读取源码 | SHA256 |
| --- | --- |
| `LightX2V/lightx2v/models/schedulers/wan/step_distill/scheduler.py` | `2d560449e0559aea44d80afb539f9e97bd544e5e26abcc06236e61918d01e1db` |
| `LightX2V/lightx2v/models/schedulers/wan/scheduler.py` | `c86af28a3b8d709692fb3a88d67ec441a77624929d397c216e0106f1c0142f4a` |
| `LightX2V/lightx2v/models/runners/wan/wan_distill_runner.py` | `26a2a503137b0125cf4a2f8b2c9fc5dae716121c6e600b7ce85517af5ee62dc3` |
| `HiF4_Sp/HiFloat4/ptq/adapters/lightx2v_wan_distill_contract.json` | `f6104c256b9fb881fe33fda2707095cd8591c4010f6fff1be2cffd2b648d56aa` |
| `LightX2V/lightx2v/models/networks/wan/model.py` | `f0b0e9904d9708ff67412d8c98cb3f384b86426bb96ebf84d17e439e7a8580ff` |
| `LightX2V/lightx2v/models/networks/wan/infer/post_infer.py` | `6567c439e2f16f322a9f10cd2ca326d3f924e9da75b3ae2ae442ec1327a327a2` |
| `LightX2V/lightx2v/models/networks/wan/infer/pre_infer.py` | `f3f88ca8ccbbd778206f8f04805c124beb84c19dcd22c4c38a6028b42c1502b5` |

上表路径统一以 `.runtime/code/` 为前缀。它们是本次实际读取版本的绑定，不冒称每个新增阅读文件都已包含在旧视频driver快照中。

合同固定 `denoising_step_list=[1000,750,500,250]`、shift=5、boundary_step_index=2、无CFG、guide scale=1。选中网格σ的理想实数值是 `[1,15/16,5/6,5/8]`；下一终点σ₄=0。故四个h的理想值为 `[1/16,5/48,5/24,5/8]`，均正；实际网格是FP32值，`5/6`等会有舍入。不能把列表中的整数当成模型真正接收的shift后时间，也不能假设四步等间距。

`MultiDistillModelStruct` 按step index选择high/high/low/low。只有0→1、2→3是同专家；1→2换权重。不同专家输出仍在同一latent/flow坐标中，因此跨专家flow历史并非数学上非法，但必须单独校准其关系；不能直接借用同专家FFN历史的函数相同性或长单模型轨迹假设。

## 2. 连续重写的符号正确，原生数值不是合并乘加

`Wan22StepDistillScheduler.step_post` 实际先把noise_pred转FP32，取当前σ为Python标量，再分别执行

`u = latents.float() − flow_pred * sigma`，

非终步再 `u = u + flow_pred * sigma_next`，最后 `latents = u.to(old_latents.dtype)`。

在实数算术中等于 `z_(k+1)=z_k−h_k v_k(z_k)`。在原生FP32中，它不是 `z−(sigma−sigma_next)*v` 的逐位等价替换；中间两个乘法及减/加有自己的舍入。最后一步仅执行减项，不另加零。此前调用路径的norm、BF16/QDQ与head输出数值都已经包含在v里。

固定Wan路径的 `prepare` / `reset` 用FP32初始化latents；`WanPostInfer` 返回 `u.float()`，scheduler再次 `.to(float32)` 不恢复此前网络中丢掉的信息。此路径不是每步把latent强制压回BF16；网络内部BF16 gate/residual与scheduler外层FP32状态必须分清。若另一个调用者改变latent dtype，末尾cast才随之改变，不能把该泛化情况冒充本次固定配置。

将每次原生更新相对理想实数式的舍入余量记为ρ，可写出实际状态差的记账式

`δ_(k+1)=δ_k−h_k[hat_v_k(z_k+δ_k)−v_k(z_k)] + (ρ_student−ρ_dense)`。

因此下面的“精确”都指**给定相同σ、条件和实数更新的代数精确性**，不保证BF16网络可表示该目标，更不保证FP32逐位重合。

## 3. 三种机制的精确分解与不同预测

先定义完整模型在相同外层输入u上的误差

`e_k(u)=hat_v_k(u)−v_k(u)`。

理想更新精确给出

`δ_(k+1)=δ_k−h_k{ e_k(z_k+δ_k) + [v_k(z_k+δ_k)−v_k(z_k)] }`。

第二项是teacher函数对离轨输入的响应；第一项是在该离轨输入上的算子误差。二者都不等于保存的单层teacher-prefix FFN SSE。

| 竞争机制 | 数学对象与信息时点 | 与其他解释不同的预测 |
| --- | --- | --- |
| 局部误差方向/下游几何 | 同一个block输入上的误差e，以及后续blocks/head对它的响应；在FFN后才能直接知道当前完整误差 | 即使外层初始δ₀=0，第一步flow已可大幅改变；同样局部SSE、不同方向可产生不同下游后果 |
| 离轨输入偏移 | `e_k(z+δ)`与`v_k(z+δ)−v_k(z)`；包括同一次模型内早期替换造成的后层输入偏移，以及下一去噪步的latent偏移 | teacher-state校准误差小、student-state误差却更大；teacher-forced与free-running的差距随已发生的前缀改变而出现 |
| 显式跨步纠错 | 使当前速度带有面向下一teacher状态的补偿，而不只模仿teacher当前速度 | 允许当前cross-state flow差更大，却让下一latent差更小；决定量包含有符号内积，不只是误差范数 |

**局部方向反例。** 两个等范数局部误差 `(ε,0)` 和 `(0,ε)` 经同一后续线性映射 `diag(M,1)` 后，分别放大成Mε与ε。这个构造不需要离轨，也不需要错误随时间增长，已经反驳“局部SSE更低就应有更好视频”。实际后续路径还有norm、attention、gate/head，不能把示意线性图当成已测Wan Jacobian。

**两级离轨不可混淆。** 第一去噪步虽然δ₀=0，但同时替换全40层，后面层的输入仍可因更前面的替换而离开保存dense前缀。故“第一步也坏”不能排除**层内**输入偏移；它只排除“所有损坏都必须等下一外层latent偏移才开始”。完整同外层输入的e_k本来就包含这层内传播。

若teacher流场在所考察区域可微，才可条件地写 `v(z+δ)−v(z)≈J_v(z)δ`，从而得到 `(I−hJ_v)δ−h e(z+δ)`。原参考本身含HiF4 QDQ、BF16与分段阈值；常规导数在量化边界附近可能漏掉跳变。不能把某个实现的局部导数为零，解释成有限输入扰动没有作用；本文不采用 `J≈0` 作为Wan事实。

## 4. 一步重合目标与CTEC：符号、γ和离线边界

令 `d_k=hat_v_k(z_k+δ_k)−v_k(z_k)`。要使下一理想状态重合，解 `δ_k−h_k d_k=0` 得

`hat_v_k(z_k+δ_k)=v_k(z_k)+δ_k/h_k`。

正号不是“沿错误继续走”：scheduler前面有负号，所以增加沿δ方向的velocity恰好减去δ。若把目标错写成减δ/h，下一差会变成2δ，而非0。

更一般地，对affine sampler `x_next=a x+b f(x)`，令δ=student−teacher，则目标为 `hat_f=f−(a/b)δ`。这与 [CTEC原AAAI论文PDF第4页 Eq13/14](https://ojs.aaai.org/index.php/AAAI/article/view/34039/36194) 的teacher−student差定义一致；取a=1、b=−h即得到上述Wan式。论文用γ控制纠错目标，并以LoRA蒸馏训练，而非在线访问teacher。这个核心目标的关系不能作为本项目新颖性声明。

对Wan若完美实现缩放目标 `d=γδ/h`，则理想地 `δ_next=(1−γ)δ`；γ=1一步重合，γ=0只保留旧偏移。该条件结论不要求J≈0；真实受限模型、估计误差和原生舍入会留下余项。

另一个直接判别是

`||δ_next||²−||δ||²=h²||d||²−2h〈δ,d〉`。

所以减少偏移的充要条件为 `2〈δ,d〉>h||d||²`（理想实数式）。对同样大小的d，方向反转即可由修复变成加剧；“更像teacher当前输出”与“更好地回到teacher轨迹”不是同一个损失。

**信息边界。** 训练可离线记录teacher z及v(z)，与student实际访问的z+δ组成标签；在线只有student自己的状态、step/expert/σ及可付费缓存，没有免费teacher z或δ。当前视频RUN保存的是latent **hash**，不是这些完整向量，不能由hash计算δ/h或上述内积。CTEC式训练、teacher重算、额外状态保存及新的纠错权重都需要各自明确范围；本记录没有执行或默许这些动作。既有闭式FFN重构也没有自动变成这个trajectory目标。

## 5. 独立核对TCEC Eq7→8：PDF同样印inverse，展开不闭合

亲读 [TCEC v1 HTML §3.1–3.3](https://arxiv.org/html/2508.12094v1) 及 [v1原PDF第5页](https://arxiv.org/pdf/2508.12094v1)，并实际渲染该PDF页检查排版。下载PDF字节SHA256为 `c8f2f1fa65f5fa2a2f74260e8fa395afdf37b4dbd622fc8e3b28bcf19f39ebe0`。同页印出的关系为

`δ_(t−1)=A_t δ_t+B_t ε_t`，`δ_T=0`，

而Eq8印成 `δ_t=Σ_(k=t)^T (Π_(j=t)^(k−1) A_j^(-1)) B_k ε_k`。

**边界反例已经足够。** t=T时该和只剩空积乘 `B_T ε_T`，通常非零，和声明的δ_T=0直接冲突。这不依赖J是否小、矩阵是否可逆，也不依赖数值实验。

再取T=2、δ₂=0、`B₂ε₂=1`、`B₁ε₁=0`、A₁=2：Eq7给δ₁=1、δ₀=2；Eq8在t=1却给1/2。即使把印式左侧重标为δ_(t−1)，t=1的1/2仍不等于2，故只改一个下标不能解决inverse问题。

**按该递推独立展开，正确的是**

`δ_t = Σ_(k=t+1)^T [A_(t+1) A_(t+2) ... A_(k−1)] B_k ε_k`，

或等价地

`δ_(t−1) = Σ_(k=t)^T [A_t A_(t+1) ... A_(k−1)] B_k ε_k`。

空积为I，矩阵乘法顺序不可交换；不需要逆矩阵。若反过来求解有终端约束的逆问题，才可能出现逆，但边界、求和方向和强迫项都须一并重推，不能沿用文中当前初值解释。

额外检查 [v4 HTML的同一节](https://arxiv.org/html/2508.12094v4) 后，同样的Eq7/8写法仍在；这里没有声称读过v4 PDF或验证其所有附录/实现。具体结论是**所见版本的这一个递推展开命题不成立**。不能据此断言论文全部实验无效、补偿启发无用，或其代码必然实现了这个印式；本文没有验证作者实现。

## 6. 保留TCEC的强竞争组件，但不照搬其简化假设

TCEC另用按step/channel拟合的K乘量化模型输出来估计误差，并保留近邻step输出作补偿。其实际组件仍是可比较的低成本假设；[v1 §3.3 Eq14–18](https://arxiv.org/html/2508.12094v1#S3.SS3)给出对应的离线校准/在线使用方式。它与CTEC的“离线训练模型学补偿”不同，也与本项目宽FFN逐神经元affine重构不同。

但全局flow的channel缩放只能表达每个通道沿当前输出空间模式的修正。一个通道若当前输出在两个位置相同，而需要相反符号的修正，则一个共享K不能表达；当前输出为0而所需δ/h非零更直接。因而它可能解释相关的整体gain/幅度偏差，却不自动覆盖任意局部纹理/重影误差。历史输出或其他低成本特征能否补齐所缺方向，保持为开放问题。

对Wan的条件线性化若写A_k=`I−h_k J_k`，J≈0反而给A≈I，不会推出早期误差自然消失。在这个特例，终态误差是 `−Σ h_k e_k`；不同step等范数误差可同向累加或反向抵消。四步、双专家及非等h不支持未经检验的“只需最近两步”普遍定理。允许研究短历史估计器，但应以实际可获得信息及反预测为依据，不能用不闭合的展开替代验证。

## 7. 可改变解释的最低成本判别与开放自由度

已完成whole热态RUN（HF相对 `e0_function_basis_20261026/prototype_whole_attempt1/RUN.json`，SHA `ed53a172f62cc9d3252b2ec8b905ba4c8814cc2ef6fc9618bcffbf0f62f9fc32`）的T约45.73577秒、FFN边界约4.87997秒，说明真实计算减少与视频功能是不同轴。协调者报告抽帧重影是研究线索，本文不把它写成本人完整视频观看或机制归因。

三种解释应有不同的判别：

- **局部几何主导：** 固定同输入/同前缀，输出方向比局部范数更能区分head/flow后果。若尚未外层离轨的第一步就有明显flow差，这条解释无需等待长轨迹。
- **离轨偏移主导：** 比较teacher-state与student-state上的同一算子误差；仅靠保存teacher面板和最终视频无法区分。即使不做训练，完整paired状态/flow的有限保存也是新的数据范围，不能假装现有hash足够。
- **显式纠错有独立价值：** 在相同student输入上，teacher flow模仿与带δ/h目标的下一状态误差出现不同排序；有符号 `〈δ,d〉` 与剩余一步误差应支持修复。它也允许当前flow SSE变差、下一latent反而更好。

可先对已有源码及标量排除符号/数值口径错误；有必要时再另定有限paired trace或受限校正学习，而非直接扩大网络训练。可能的新计算自由度是：在低维latent/head坐标中学习条件修复，或在FFN内选择更符合下游方向的函数单位；也可能发现固定局部重构本身不足、必须改变支持/模型容量。三者都留为OPEN，并计入离线teacher成本、缓存、拟合和在线执行费用。

**实际WARD记录。** 2026-10-26从“CTEC cross-timestep correction / TCEC error propagation / DDIM affine target”检索，亲读AAAI正式PDF第4页Eq10–14及Algorithm1、TCEC v1 HTML §3.1–3.3和PDF第5页，另核对v4 HTML同式。AAAI直接download端点读取失败后，通过正式PDF view链接成功读到全文；未找到并验证两方法原代码，不声称代码复现。TCEC PDF页面实际查看过，Eq8不是仅凭搜索摘要判断。检索改变了研究判断：Wan纠错目标有明确既有近邻；真正待解释的是当前函数误差、访问状态与可支付修复方向的关系，而非把逆矩阵展开当作可直接部署的公式。

**六视频完成后的归因补界。** 还须保留实际部署bank的all-layer native数值偏差：shape/finite/SHA与少数full-width neutral，不证明所有partial/affine层都贴近保存数据连续构造。此前partial检查只在blocks0/30。故上述三种科学解释尚有一个实现/数值前提未闭合，应先做实际bank的同输入native/连续/dense区分，再决定是否需要离轨trace或纠错学习。此处不是把视频损伤默认归因于J、传播或训练不足；实际六RUN审查另追加在 `independent-matched-calibration-and-runtime-audit-20261026.md` 第13–15节。
