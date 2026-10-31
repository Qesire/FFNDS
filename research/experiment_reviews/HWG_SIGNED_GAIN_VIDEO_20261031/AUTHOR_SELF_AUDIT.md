# HWG Signed + Gain v2 · 作者复审记录

状态：**AUTHOR_SELF_AUDIT / INDEPENDENT_REVIEW_REQUESTED**。这是同一作者对实验设计的复查，尚无真实不同agent的review、recheck或ACTIVE授权。主设计：[2026-10-31 v2](../../20261031/HWG_SIGNED_GAIN_VIDEO_DESIGN_V2_20261031.md)。

## 已修订的实质问题

1. 原BF16实验误写HiF4 Q1/Q2：现按RAW_BF16无激活Q1/Q2定义h，HiF4单列后续复验。
2. 缓存旧输出必须是pre-gate Y0，当前e5 gate应用于增量后的新Y；b2仅一次。
3. 主投影MAC按不同路径分段：双expert稀疏H/X+Y为68.5185%/77.7778%，仅单expert稀疏、另一expert Dense则84.2593%/88.8889%（都在K=5120条件下），后者不能对Dense侧假付三次W1。
4. Signed Gram是固定算法近似组合求解，不能称全局最优；盒约束α∈[0,2]的ridge gain由凸QP求解，无约束闭式解不能直接clip当最优。
5. BF16 DIAG80属于新静态组基线，不冒充原A5参数producer；与历史CAL1、UB1 consumer及Chipmunk的先例关系已说明。
6. 96 camera prompt被修改过，明确完整prompt/图片SHA/初始noise SHA配对，不能按source ordinal替代conditioning身份。
7. BF16校准操作数先真实核查，train/calval与P0/P1/P2分image簇；24新确认来源不是现成名单或统计功效承诺。
8. P1中GAIN80即BOTH，High/Low单边臂给出四格因子交互；D/G/R均预算同组，G/R全80site mask hash必须一致。
9. P0/P1/P2最多14/72/288是条件容量，不是视频已执行或自动GPU额度；video质量、运动、语义和真实速度是最终判断。
10. 不修改ACTIVE、不写假独立批准记录，实验前分别申请外部Agent重审/修订/recheck。

## 向真实独立评议者提出的问题

- 长度216的组Gram在不同image cluster、gate和学生轨迹下符号是否稳定？给出相反源例。
- 有界gain是否产生放大/闪烁，收益是否仅来自teacher同状态训练；哪些连续视频和student capture能区分？
- 当前packed BF16无observer真实执行中，组scale能否在不改热路径/不付额外W2复制时生效？
- 两次相同expert的缓存旧Y、当前gate、旧bias与BF16数值重排可否在随机真实输入上建立中立门？
- 数据是否覆盖全80更新site，以及新样本量能否支持逐类别结论？
- 对P64需要付费64行W1代理，哪些同K/同系统成本对照才公平？

**独立评议尚未返回**；Reviewer身份、决定、批准SHA和实际RUN/EXIT均不填充。下一scope遵循 [实验前独立复审](../../agent_protocol/EXPERIMENT_PREEXECUTION_INDEPENDENT_REVIEW_20261029.md)。
