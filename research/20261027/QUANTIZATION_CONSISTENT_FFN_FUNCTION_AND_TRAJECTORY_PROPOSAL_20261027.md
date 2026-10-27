# FFNDS · 量化一致的FFN函数表示与跨步响应研究方案

**2026-10-27｜独立方案 v1｜机制与方法并行探索｜研究状态：OPEN（未授权新模型执行）**

## 0. 研究主张与原则

**核心问题：在Wan2.2少步视频生成中，是否能够以更低物理成本实现原FFN的重要函数作用，并在原生HiF4/BF16执行和完整视频轨迹上维持质量？**

现有证据表明：Affine54已将一部分非线性映射折叠成D×D仿射层，在全40层×4step实际运行中具有显著FFN提速；但它的局部拟合优势没能维持完整视频。现存证据还显示旧FFN后验单通道变化可集中，原始合法64组和M16/K64 tile支持却往往几乎全满。由此形成三个相互关联、但可以分别成功的研究对象：**量化一致的函数表示（I/C）、有限状态扰动与视频作用（F）、实际执行与可部署成本（C）**。

**文献与研究原则：直接复用成熟的好方法，先解决真实问题再准确界定贡献。** TARDIS(2025)及同团队EuroSys 2026 LLMFolder已把FFN常量折叠作为成熟系统研究；Model Preserving Compression(2022)覆盖保留真实激活子集并重构消费者。FLAP、BRECQ、2ndMatch、CTEC与RT-Lynx分别提供其它可用的选择、补偿、传播与真实内核路线。成功复现、合理组合、提出精确数学解释、取得实际视频/系统改进都是有效成果，无需为了“必须不同”放弃成熟技术。**所有论文机制、阅读深度与迁移边界统一查[一手文献审查](PARTIAL_LINEAR_FFN_PRIOR_ART_AND_TRANSFER_AUDIT_20261027.md)**；本文专注可执行研究问题，不重复堆叠文献叙述。

研究科学内容可仅形成定理、函数实测或反例；实际部署主张另需质量和物理性能证据。此提案不以UB1、Affine54、GELU配对、G13或指定loss作为不可替换的路线。I/F/C可独立入场。

## 1. 研究对象：完整原生计算图与已证实的起点

**Wan2.2 I2V A14B D-Q (LightX2V)**：high/high/low/low双expert、每步40 block、四步160个FFN位置；代表性M=31,668，D=5,120，H=13,824=216×64。被研究FFN调用路径为

~~~text
post-attention residual → norm2 / AdaLN / e3,e4 modulation
→ Smooth1 → original HiF4 Q1 → BF16 W1+b1
→ tanh-GELU → Smooth2 → original HiF4 Q2 (64→8→4共同scale)
→ BF16 W2+b2 → e5 gate → BF16 residual
→ attention / later blocks / flow → original scheduler → actual frames
~~~

量化基线实际为 **fake-QDQ + BF16 Linear**；真正packed FP4是可以并行研究的其它算子，并非已在原Wan部署。任何函数分析要固定source、seed、expert、step、block、输入边界与caller的gate；Q2选择必须顾及原完整64组scale闭包。

### Affine54已经实现的表示

对真实preactivation \(a=W_1q_1(x)+b_1\) 和保存的原post-Q2响应 \(q(a)\) 进行逐坐标斜率/截距拟合：
\[
q(a)=\kappa\odot a+\beta+r(a),\qquad r(a):=q(a)-\kappa\odot a-\beta.
\]
连续数学上完整余项是恒等分解；丢弃部分余项才是近似。将全部固定仿射基项离线组合成D×D层、保留54个原64组的真正W1/GELU/Q2并对新读出V作闭式ridge校准后，**实际部署执行形式**为
\[
\widehat F_{\mathrm{A54}}(q_1(x))=
A_{\mathrm{bank}}q_1(x)+b_{\mathrm{bank}}
+V_S\,Q_{2,S}\!\left(\phi(W_{1,S}q_1(x)+b_{1,S})/s_{2,S}\right).
\]
其中A_bank、V_S、b_bank是新BF16参数，读取q2的旧fit与缩窄Q2部署的数值路径存在待核对差异。门控残差由旧caller处理，和运行时bank的前馈输出是两个不同边界。

理想两投影MAC密度为 \((40+54)/216=43.5185\%\)，只对原Dense两层FFN GEMM，不把QDQ、融合、门控、bank内存及前置预测器算作免费。\n\n真实结果：两个持出但已开发暴露来源×160位置=320局部例，A54 gated SSE平均8.54%，Whole94为11.67%；A54在305/320例占优。三臂两来源共六条真实81帧视频均已生成；来源12热态FFN 9.707→4.449秒（2.182×），完整生成50.750→45.313秒（1.120×），但画面出现模糊/运动及细节变化，因此**尚未通过完整视频质量验收**。历史offline校准为4来源×每expert两步×每步32行=256行/函数，另有有限full-width neutral接线测试；它并未证明所有partial bank的原生部署数值已完全对齐。

[GELU反向配对]的恒等式 \(\phi(z)-\phi(-z)=z\) 只是上述仿射+余项框架的一个条件特例。原Wan权重的真实配对密度、Q2量化后的误差及物理收益尚未证明，**不单列为新算法**。

来源：[bank构建](../../tools/build_e0_reconstruction_bank.py)、[实际运行hook](../../tools/e0_reconstructed_runtime.py)、[全层匹配对照](../20261026/E0_ALL_BLOCK_RECONSTRUCTION.md)、[六视频原生记录](../20261026/E0_RECONSTRUCTED_VIDEOS.md)、[此前Skill审查](UB1_TO_FFNDS_EVIDENCE_DRIVEN_ROUTE_REASSESSMENT_20261027.md)。

**已完成的相关数值控制**：[W2形状对照](E0_W2_SHAPE_CONTROL_RESULTS.md)在八个原Dense保存条件上，M160有约0.292%差异，恢复原M/row位置则全部输出逐字节复现；[原M D/Q](E0_Q2_DIRECT_COLLATERAL_FULL_SHAPE_RESULTS.md)严格通过Q2和W2基线，保留直接/补集两种作用。该结果支持WP0–1的精确数值边界研究，尚未核对Affine54全部partial bank或解释其视频失败。本方案的H1–H3与A1–A4继续分别保留OPEN。

## 2. 最重要的三个机制解释及其不同可观察结果

| 假说 | 具体机制 | 区分性预测 | 已有证据状态 |
| --- | --- | --- | --- |
| **H1 数值部署失配** | 使用保存Dense post-Q2拟合的bank在原生缩窄W1/Q2、BF16矩阵与写回时产生额外误差 | same-input同gate比较离线拟合输出、native partial FFN2、post-gate/writeback，第一差异在数值路径出现 | SRC可确认拟合/部署边界不同；误差份额OPEN |
| **H2 函数响应/后缀失配** | 当前值SSE较低，但对真实前缀扰动的有符号有限响应与Dense不同，后续block/step传播改变视频行为 | 同值误差级别的候选，finite signed响应或相同前缀下suffix行为出现不同排序；模长敏感性不足以预测方向 | E0有局部-vs-视频矛盾和其它候选的P/N跨seed证据；A54专属归因OPEN |
| **H3 表示容量/校准失配** | 固定仿射+54 whole64无法表达自然视频所需的联合非线性；也可能原256行校准覆盖不足或正则未适配 | 增加独立来源/改变监督目标/改变合法表示，各自对持出函数响应与视频质量的影响不同 | A54强于全仿射且仍失真；各因素贡献OPEN |

三种假说可以同时成立；研究优先以**最少变更区分因果来源**，不以“必须与论文不同”作为成功标准。能直接复用论文方法改善质量或系统效率，是有效的研究推进。

## 3. 成熟先例作为真实的研究组件

本章压缩为**研究与迁移决策**，所有作者原公式/执行细节集中在[共用文献审查](PARTIAL_LINEAR_FFN_PRIOR_ART_AND_TRANSFER_AUDIT_20261027.md)，避免两份方案各自重述并出现版本冲突。

| 成熟组件 | 允许直接使用的作用 | 最关键的Wan条件 |
| --- | --- | --- |
| **TARDIS 2025**（最直接的折叠先例） | 常见激活区间线性化、D×D折叠、2-bit W1代理预测离群、减近似加真实、定制kernel；对Affine54提供固定/动态支持对照 | 预测器成本、M=31668的token粒度、原HiF4 whole64 Q2尺度闭包、动态索引真实内核成本 |
| **LLMFolder EuroSys2026** | 同团队正式发表的常量折叠论文，可作为相关核心文献 | 仅证实出版与作者身份；尚无逐公式核验全文，不能填补未知算法细节 |
| **Model Preserving Compression / FLAP** | 在真实非线性子集上用ID恢复消费者；激活波动度量与bias补偿，均可直接复现 | 原量化全组合法性、校准来源泛化及是否需额外D×D主干 |
| **BRECQ / OBS-Diff** | block/组级重构、二阶感知支持/时间敏感性 | 是否能在原native BF16/QDQ和实际后缀上稳定改善，而非只改局部损失 |
| **2ndMatch / CTEC / TCEC** | 扰动响应保留、跨时间状态蒸馏与低成本补偿 | 原真实scheduler、同源同seed教师/学生配对、离线可见Teacher与在线真实信息的区别 |
| **RT-Lynx ICML2026** | 独立2:4 activation-sparse GEMM+范数恢复+LoRA+融合CUDA，比折叠之外有更强物理执行对照 | 是图像DiT/H20的成果，不是Wan I2V native HiF4验证；实际融合、训练和全生成成本另算 |

Affine54的数学表示是 \(\widehat F=Aq_1(x)+b+V_Sq_S\)。TARDIS最关键的新增对照是条件性**异常修正**，而非“论文首次允许保留真实非线性”；MPC的 \(H\approx H_ST\Rightarrow HW_2\approx H_S(TW_2)\)表明重构消费者亦早有先例。我们已经有\(V_S\) ridge，所以应比较其选择/泛化效果与论文实现，而不是再次命名同一操作。

### 可以借用的响应与轨迹目标

- 原值：\(\mathcal L_{\rm value}=\|F_S(b;c)-F_D(b;c)\|^2\)。
- 经典有限差分：\(\mathcal L_{\rm secant}=\|[F_S(b+\delta;c)-F_S(b;c)]-[F_D(b+\delta;c)-F_D(b;c)]\|^2\)；条件\(c\)含AdaLN/gate，真实前缀变化时不能只改\(b\)且暗中固定\(c\)。
- 2ndMatch的\(J^\top J\)匹配是有价值的既有基线；有限有符号差分可作为另一项实验比较器，**不预先假定更优，也不宣称其数学公式新颖**。
- CTEC理想Euler流场下：若 \(\delta_t=z_t^S-z_t^D\)，则使下一状态吻合Dense所需**整模型**teacher目标为\(v_S^*(z_t^S)=v_D(z_t^D)+\delta_t/h_t\)；需要按原Wan shift5和精度重新推导，单FFN未必能达到。TCEC论文较早版本的具体闭式展开需独立核对正式版本。现有E0有限映射\(P_t,N_t\)交互是严格代数恒等式，不是独立首创机制。

**直接复现优先级**由哪种机制能够解释Affine54真实模糊、运动与细节偏差决定，而不是由论文年份/会议排名决定。

## 4. 函数与实际工作单位：四个并行对照族

| 类别 | 代表计算表达/成熟方法 | 需要区分的现象 | 执行条件 |
| --- | --- | --- | --- |
| **A：固定部分线性** | 现有Affine54（D×D+54 whole64）；same-cost Whole94；MPC式真实激活子集+消费者重构 | 视频退化来自支持/表示能力，还是数值校准失配？ | 固定source/seed/FFN成本和真实HiF4路径 |
| **B：条件线性+修正** | 直接复现TARDIS热区、在线预测及“减近似加真实”，必要时以whole64组或更合适的合法算子为单位 | 校准区间在不同step/来源是否稳定？online outlier支持是否实际集中成省算tile？ | 观察器先于原W1/GELU/Q2有信息、并计2-bit proxy/metadata、group scale和kernel价 |
| **C：函数/训练恢复** | FLAP/ID/OBS/BRECQ支持与消费者恢复、2ndMatch/CTEC/TCEC的局部和轨迹目标，允许直接采用 | 额外训练或不同评价边界能否改善独立视频质量？ | 相同校准来源及时间/参数预算比较，不把模型整体teacher状态当在线标签 |
| **D：不同物理计算表示** | RT-Lynx的N:M activation sparse+LoRA+SpTC，原生packed FP4或其它量化代码域/融合、可选块耦合近似 | 既有D×D折叠与原whole64是否已经受限于布局/数值闭包？ | 实际大M形状、现有设备和native kernel；不能用后验SSE替换部署成本 |

可保留一个开放数学种子：以合法64组内的 \(q_g(z_g)\) 拟合 \(K_gz_g+d_g+r_g\)；A0的逐坐标仿射是对角近邻。这种共享scale块映射可能解释Q2条件关联，**但64×64块需要大量参数**，4来源×每函数256行校准不足以先验保证持出稳定，也不能自动减少GEMM。若证据不支持，直接转向TARDIS、MPC或其它更有效的已知基线。

\(\phi(z)-\phi(-z)=z\)（tanh-GELU连续值）是已知函数恒等式和A类仿射表示的条件特例；原Wan是否存在可利用对称权重、能否保留原Q2/BF16仍OPEN，**不单独升级成新方法**。

## 5. 工作包、竞争预测与资源

**WP0｜真实性与证据登记（现存源码/资料，无新模型执行）**  
完整复核Affine54的bank构造、V_S ridge、Q2/smooth/gate/BF16写回、同预算Whole94与6条81帧原生视频；记录4来源×256行/函数的校准和source0/12已开发暴露的持出身份。明确当前现存哪些full vector可读、哪些仅有聚合RUN/哈希；整理LLMFolder全文未读和需要复核的方程。交付单一source/step/block/boundary矩阵。

**WP1｜数值与表示的第一分歧（静态推导与已有载荷）**  
将(A)完整native执行和(B)校准标签/折叠bank与原Dense在同条件下的偏差分离。来源相同、同Q1输入、同AdaLN/gate时，先检验Q2 scale/round/bias；再比较原Affine54、同成本Whole94、MPC式消费者重构与TARDIS式热区残差，若已有材料足够只需读取/数学检验。成功判据是找到**决定下一构造的数值原因或稳定反例**，不要求某个方法必须新。

**WP2｜条件计算可行性（允许静态成本/数学构造）**  
按真实分布考察逐组热区、当前输入outlier频度和在M/K tiles上的占用；在不执行实际模型前，构造必要W1提前信息的成本下界/方案并比较TARDIS原2-bit predictor。一个有价值的反例是outlier分布跨所有tiles，使索引/补算超过saved GEMM；即使如此也只关闭指定粒度/硬件路径，仍可采用固定方法/其它函数。

**WP3｜已有扰动与轨迹结论的迁移（保存资料只读）**  
在同source/seed/step和同真实prefix比较值误差、有限方向响应、BF16写回与下一正常计算边界。其他E0候选的P/N交叉变号与local-vs-video分离只能生成Affine54待判别假说，**不可认作Affine54自己已经验证**。如果完整张量不存在，保留OPEN且先数学/文献推进。

**WP4｜经用户新授权后的最小native判别**  
在真正有区别力的两个方案上固定同预算、同source/seed、同数值精度和可见信息：例如原Affine54 vs“原生Q2对齐”的Affine54；或Affine54 vs匹配成本的TARDIS动态组恢复；或Affine54 vs直接MPC-ID重构。小试针对机制，不能替代全四步视频验收；如采用CTEC/2ndMatch做少量训练，需另报真实训练成本。

**WP5｜经授权后的终端研究结果**  
完整40block×4step自由生成及帧视频，原Dense resident、UB1、Affine54、同预算Whole94、可运行论文迁移方法为强比较器。质量：per-source Aesthetic/Imaging/Motion及Dynamic Degree、subject/background consistency、flickering、局部细节与真实运动；速度：全160 FFN、DiT与完整save-video wall、GPU显存、驻留/冷启动/在线proxy+correction+QDQ/packing及训练摊销。结果对同一原生设备/模型/seed形成Pareto，而不是跨LLM或图像论文复述比值。

**可修订决策**：数值路径不足则优先A；固定支持足够而条件outlier强时可直接复现TARDIS；V_S/消费者重构不足可直接采用MPC/OBS；局部重构仍不能解释视频则发展CTEC/2ndMatch与真实suffix；若硬件更受约束，直接比较RT-Lynx/packed FP4类系统。以上研究可以并行，没有强制顺序，也不设“必须不同”的算法门槛。

**所有新模型运行、CPU数值模型测量、GPU、视频、训练仍服从[ACTIVE](../ACTIVE_RESEARCH.json)；当前文档修改不是新的计算授权。

## 6. Skill复审：保留开放性与成熟技术可用性

| AutoResearcher Skill | 已落实的检查 | 应继续保留的开放性 |
| --- | --- | --- |
| **mechanism-research** | 源真实Q1/W1/GELU/Q2/W2/门控，分别提出数值、响应、容量竞争解释与最小判别 | 假说可以先于全部素材取得；纯数学结论不必跑视频 |
| **research-expansion** | 表示、数值算子、传播目标、训练/不训练、硬件布局都有不同入口 | A3只是探索候选，不规定为最终方法 |
| **problem-reformulation** | 将删通道问题改为函数实现及条件成本；反向GELU已归入Affine54特殊情况 | 容许完全采用TARDIS/2ndMatch/CTEC/OBS成熟方案 |
| **web-research-discovery** | 官方原论文与真实先例映射到具体计算步骤、目标和适用范围 | 先例应促进合理借鉴，差异发现后再谈新颖性 |
| **FFNDS领域Skill** | I/F/C并行、实测/推导/假说区分、强Dense和原生质量/成本保留 | G13仅为F线专题；未改ACTIVE、CI或一次性交接 |

**交付主线**：复现并成功迁移成熟机制、准确定位数值/函数/传播失配、或获得真正省算的计算表示均为有价值成果。下一轮判断优先来自同Wan强比较器与真实视频，而不是论文名称差异。

## 7. 精简后的唯一证据与论文入口

- [部分线性FFN与条件修正一手文献主审查（本轮新增）](PARTIAL_LINEAR_FFN_PRIOR_ART_AND_TRANSFER_AUDIT_20261027.md)：TARDIS、LLMFolder、MPC、FLAP、BRECQ、CTEC、RT-Lynx与实际Wan迁移边界

- [已审核UB1和科研空间](UB1_TO_FFNDS_EVIDENCE_DRIVEN_ROUTE_REASSESSMENT_20261027.md)
- [E0全40层匹配局部](../20261026/E0_ALL_BLOCK_RECONSTRUCTION.md)、[六视频与系统计时](../20261026/E0_RECONSTRUCTED_VIDEOS.md)
- [E0完整hidden与物理tile](../20261025/E0_INTRINSIC_FULL_HIDDEN.md)、[Q2合法组与物理算子](../20261025/E0_SHARED_TILE_PRE_W1.md)
- [独立G13方法](G13_FUNCTION_LOCALIZATION_METHODOLOGY.md)、[三线目标](../RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md)

**静态文档完成标准**：本文所有已测数字与原RUN/源码一致；内部链接可访问；理论等式标明成立条件；每个解释存在对照及可修订结论；不把复用成熟方法排除在外，也不预先宣称新方法/新执行授权。