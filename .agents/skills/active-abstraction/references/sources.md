# 来源与使用范围

以下原始材料为方法论提供具体实例和进一步理论入口。引用时沿原文核对目标主张；本技能中的一般推导、类比和新预测分别按其条件表达。

## 已核对的一手材料

- **Rotstein et al., Pathways on the Image Manifold: Image Editing via Video Generation, CVPR 2025.** [CVF 正式页](https://openaccess.thecvf.com/content/CVPR2025/html/Rotstein_Pathways_on_the_Image_Manifold_Image_Editing_via_Video_Generation_CVPR_2025_paper.html)；[arXiv v4 正文与附录](https://arxiv.org/html/2411.16819v4)；[作者项目页](https://rotsteinnoam.github.io/Frame2Frame/)。§3 给出时间描述、视频生成与选帧；§4 为几何可视化；附录 B–C 为接口消融。支持任务迁移案例，不建立测地线求解或身份保持定理。
- **Zhu et al., Generative Visual Manipulation on the Natural Image Manifold, ECCV 2016.** [作者项目页](https://www.cs.cmu.edu/~junyanz/projects/gvm/)。支持用生成模型学习自然图像集合，并将编辑写成约束优化的先例。Frame2Frame 的具体贡献应放在视频先验和时间过程接口上比较。
- **Tishby, Pereira, Bialek, The Information Bottleneck Method, 1999 manuscript / 2000 arXiv.** [原文](https://arxiv.org/html/physics/0004057v1)。§1、§3 把相关信息相对于目标变量定义，并研究压缩与预测相关性的权衡。本技能的确定性充分表示关系由定义推导，不与信息瓶颈优化目标混同。
- **Li, Walsh, Littman, Towards a Unified Theory of State Abstraction for MDPs, 2006.** [作者论文](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/camera-ready-9.pdf)。§3 区分不同信息保留与规划、学习保证。支持将可表示性、可规划性、可学习性分开判断。
- **Dembo, Eisenstat, Steihaug, Inexact Newton Methods, SIAM Journal on Numerical Analysis, 1982.** [出版页](https://epubs.siam.org/doi/abs/10.1137/0719025)；[原文镜像](https://wiki.math.ntnu.no/_media/tma4180/2018v/inexactnewton.pdf)。将精确内部求解放宽到残差控制，并分析局部收敛。具体收敛阶结论应沿原文假设引用。
- **Clarke et al., Counterexample-Guided Abstraction Refinement, CAV 2000.** [原始技术报告](https://cs.uwaterloo.ca/~bbonakda/teaching/ECE725/papers/CEGAR.pdf)。支持由伪反例驱动精化的具体形式验证方法。向科学发现迁移的是修正抽象的策略；形式验证的保证不自动迁移。
- **Dao et al., FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, 2022.** [原文](https://arxiv.org/pdf/2205.14135)。§2–3 通过数据移动模型、分块和归约重组说明，计算重构的收益与资源层级有关；讨论精确数学运算和浮点执行时分别说明。

## 知识发现与问题形成

- **Gentner, Rattermann, Forbus, The Roles of Similarity in Transfer: Separating Retrievability From Inferential Soundness, Cognitive Psychology, 1993.** [作者机构原文](https://groups.psych.northwestern.edu/gentner/papers/GentnerRattForbus93.pdf)。四项人类故事材料实验区分记忆检索与给定对应后的推断合理性评价：表面相似与共同关系结构在两类任务中的作用不同，部分关系检索失败不能归因于遗忘。支持为候选发现单独设计动作；不能直接推断搜索引擎或 LLM 具有相同机制。
- **Gentner, Loewenstein, Thompson, Forbus, Reviving Inert Knowledge: Analogical Abstraction Supports Relational Retrieval of Past Events, Cognitive Science, 2009.** [作者机构原文](https://groups.psych.northwestern.edu/gentner/papers/gentnerLoewensteinThompson&Forbus_2009.pdf)。五项实验与模拟研究比较类比案例如何促进关系检索，涵盖自传性与受控记忆材料。支持用比较形成新的检索线索；类比案例由实验提供，不能据此声称解决了第一批候选从哪里来的问题。
- **Dorst, Cross, Creativity in the Design Process: Co-evolution of Problem–Solution, Design Studies, 2001.** [作者机构原文](https://oro.open.ac.uk/3278/1/Creativity_-_coevolution.pdf)。对九位有经验工业设计师的过程记录进行分析；§3 展示部分问题结构与部分方案反复交换信息，方案可以重新界定问题。支持共同演化的流程启发；小规模设计研究不构成 AI 科研技能有效性或必然创新的证明。

## 继续扩充

按研究需要加入新的原领域定义、定理、构造、反例与来源位置。保留每条迁移所使用的关系和缺失条件；案例尚未有 AI 用途时，可以先记录其数学结构与可能后果。新颖性、实测效果与实现状态分别查证，不由知识库覆盖情况推断。
