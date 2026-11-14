# G20 两条研究方向：执行交接手册

**2026-11-14 · rev5 · 固定规格，待实现与具体预审；本次未运行模型、拟合或采集。**

本手册的目标是让接手者直接完成有限实验。研究问题和默认实现已经选好；执行者只补真实文件、代码和环境绑定，不重新设计问题、扩大矩阵或增加审批层。与旧文档冲突时，当前手册及[机器规格](G20_TWO_CORE_EXECUTION_SPEC_20261114.json)规定本轮范围；已冻结旧结果与真实执行权限保留原身份。

## 1. 给接手 agent 的起始指令

> 按本手册实现 A、BW、BWS 三个固定比较阶段，服务“新算子”和“已有方法差异归因”两个目标。先完成已有 bank 能支持的 A 与 BW 的 Q1/ICC；P64 真正缺少哪64行才补哪64行。BW 后按固定 CAL3/DEV3 复用或补齐 BWS，不用 A 结果或 BW 显著性决定是否继续。沿现有协议一次自审、独立审查、集中答复和定向 recheck，冻结有限 ACTIVE 后单次调度。科学阴性正常完成；没有具体输入、身份、数值或资源错误，不得增加 pilot、参数网格、样本、复审层或用户确认。

| 步骤 | 实际动作 | 产物与退出条件 |
| --- | --- | --- |
| 1 一次绑定 | 读取固定 manifest、路径、PANEL token ID、权重和现成方法代码；逐消费者登记可用或缺失 | `B_BINDINGS.json` 一份；缺 Chip 输入不阻止其他绑定完成 |
| 2 A | 实现下述四臂，复用已发表 Q1 alpha，拟合三个固定残差 | 528 行数学表＋528 行 BF16 表，六站点完整可重载参数；正负均完成 |
| 3 BW | 同一 D4 U、同一 G_ref；两权重各自产生目标，运行三种固定方法 | Q1/ICC 的264个 DEV 单元先完成；加 P64 后396个 |
| 4 缺项生产 | 仅在实际 P64 64行不齐时复用、恢复或补捕获 | `B_CHIP_P64_INPUTS`；最多既有DEV11各一条D4，0新图文 |
| 5 BWS | BW Q1/ICC 完成后复用或补齐指定六源 O40；混合 CAL、固定参数四格 | Q1/ICC 144个 DEV 单元；加 P64 后216个；最多6条O40 |
| 6 收尾 | 输出逐源差分、有限结论、缺失单元和实际消耗；回填现有资产树 | 三阶段完成或明确局部阻塞；不自动启动全层/视频或另一轮搜索 |

A 与 B 可以并行准备，计算遵守实际会话资源限额。BWS 不等待 Chip 缺口全部解决；能完成的 Q1/ICC 四格先完成。步骤4是输入生产，不增加一种科学比较。

### 1.1 固定图文与语义覆盖

使用已绑定的 VBench **CAL_A11 + DEV11**，共22个现有图文条件。下表仅帮助核对角色，图像和 prompt 字节继续取当前 bank/绑定 manifest；不从网页重新下载或改写 prompt。CAL_B11 保留，当前不使用。

| 官方类别 | CAL_A | DEV |
| --- | --- | --- |
| single-human | vbi_1005 | vbi_1013 |
| multiple-human | vbi_1039 | vbi_1040 |
| animal | vbi_0895 | vbi_0886 |
| transportation | vbi_0937 | vbi_0947 |
| food | vbi_1094 | vbi_1109 |
| plant | vbi_1082 | vbi_1084 |
| other | vbi_0966 | vbi_0950 |
| architecture | vbi_0304 | vbi_0176 |
| scenery | vbi_0648 | vbi_0800 |
| indoor | vbi_0456 | vbi_0392 |
| abstract | vbi_0032 | vbi_0056 |

六站点固定 high/low × layer4/19/34（0-based），seed42、同一有序 PANEL512。D4 配对 high 0→1、low 2→3。当前几何832×464×81，token grid21×29×52，M=31668。旧O40的480高度只提供 scheduler/CFG 实现参考，不是匹配当前面板的状态。

bank 路径：`/public/home/scc/pb23061276/projects/wan-quant/work/g20_v2_capture_20261113_rev2/stage/output/bank`；同级 `BANK_MANIFEST.json` SHA256 为 `f6f8e100da6f75b2db612edd8c8c4d21729ed9bf99bfb5092941f2381a4d0bee`。已有V2 verification SHA为 `85c3eb27226190b89bebbf125224bc0ca6d7b2ef09def2ddd7658f07d1a34c30`。当前节点可读性由执行者一次核实；GitHub凭据缺页不等于必须重跑。

A 的 `VB_LOCAL_FITS.json` 取已发表V4输出，SHA256 `55888f223e6962927cedc694d91c8ee247220239857231d668e9da82b12cd38c`。每个 `fits['512|'+site].alpha_Q1` 含全部13824系数。六个实际证书已收敛；硬编码 `CERTIFICATE_UNRESOLVED` 字符串不是失败，不重跑PCG。

## 2. 当前范围与资源默认值

| 任务 | 新完整轨迹上限 | 默认计算范围 |
| --- | --- | --- |
| A | 0 D4 / 0 O40 | 1 GPU、4 CPU、160GiB host、78GiB GPU、3h；输出2GiB以内 |
| BW | 0 D4 / 0 O40 | 同A的本地拟合/回放资源上限；输出8GiB以内，按site流式处理 |
| B_CHIP_STATS | 最多11条D4，只补实际缺失DEV条件 | 每捕获job默认1GPU、4CPU、192GiB host、76GiB GPU、2h、4GiB输出 |
| BWS | 最多6条O40 | 缺失来源逐条捕获；拟合/四格使用本地回放上限；0视频decode/评分 |

设计并发上限2GPU；实际会话/ACTIVE若更严，执行更严者。默认Python为 `/public/home/scc/pb23061276/miniconda3/envs/hif4/bin/python3.11`，优先已有V4环境（Torch2.10.0+cu128）。当前节点、分区/账户和必要依赖只核对一次；这些数字是拟定上限，不代表已分配资源。失败消耗跨版本累积，不能换scope名清零。

三种方法主表报告本轮采样函数回放，数学/数值/视频身份分清即可。不要求先完成整模型中性、全80导出、视频质量、系统速度或功效研究才能运行当前局部比较。必要的目标、恒等、参数重载和中性检查合并到实际消费者。

## 3. A：固定数学与代码规格

### A2. 唯一固定数学规格

四臂名称固定：

    CURRENT_Q1
    Q1_PLUS_LINEAR64_Q32
    Q1_PLUS_DELTA_N32
    Q1_PLUS_ENDPOINT_N32

不包含 STABLE_FREE_LINEAR、L32 或 256 行；没有额外 rank、激活、lambda、eta 分支。

#### A2.1 Q1 祖先：直接重建，不重新拟合

对每个 site：

    alpha = torch.tensor(
        fits["512|" + site]["alpha_Q1"],
        dtype=torch.float32, device="cuda")
    A_q1_f32 = (w1.float().T * alpha) @ w2.float().T
    A_q1 = A_q1_f32.double()

此运算顺序与 q1_fit_core.Problem.matrix 一致：先 FP32 形成完整矩阵再 lift F64，不能换成 W1/W2 全 F64 重建后称为同一 Q1。TF32 关闭。无需实例化 Problem 产生大型 Gram，也不调用 q0_prior/fit_q1/stable_free_linear。

#### A2.2 P、Q、端点中心与尺度

每个 site 的所有参数只用 CAL11×512。源权重 w_s=1/(11*n_s)，n_s=512。

P64：继续用旧 select_pq 的 weighted stacked X 的 F64 thin SVD，取 top64 右奇异向量；P32=P64[:,:32]。不改成随机投影、不用 DEV、不给额外特征基底候选。

本版明确使用门前残差基底 Q32：

    R_s = Y_s.double() - X_s.double() @ A_q1
    R_stack = cat(sqrt(w_s) * R_s)
    Q32 = svd(R_stack, full_matrices=False).Vh[:32].T

这是新 scope 的固定选择：从 Q1 的门前目标残差选择门前写出方向，真正的 gate 权重在原 fit_readout 正规方程中保留。与旧 V4 “对 gated residual 做 SVD 却将 Q 用作门前基底”的方向提议不同；两者都可定义模型，但此次不让坐标含义悬空。不要同时再运行旧 gated-Q 作为第五臂，也不声称此改动单独解释了 V4 的失败。

设 h0=U0.double()@P32，h1=U1.double()@P32：

    b = sum_s w_s * (h0.sum(0) + h1.sum(0)) / 2
    var = sum_s w_s * (((h0-b)**2).sum(0)
                      + ((h1-b)**2).sum(0)) / 2
    sigma = sqrt(var).clamp_min(1e-6)

固定 tanh，不换 GELU/SiLU，不做温度搜索。Delta 分支不减 b，保持奇性。Endpoint 两端使用同一 b/sigma，端点交换仍反号。

#### A2.3 特征与读出

原始特征如下，每个残差臂均有 64 列：

    L: cat-free X.double() @ P64

    D: h = X.double() @ P32
       cat([h, tanh(h / sigma)], dim=1)

    E: h = X.double() @ P32  # 与Delta的前32列使用完全相同的运算
       h0 = U0.double() @ P32
       h1 = U1.double() @ P32
       cat([h,
            tanh((h1-b)/sigma) - tanh((h0-b)/sigma)], dim=1)

为避免 raw 线性特征和 bounded tanh 在 ridge 中因单位不同而出现任意惩罚，统一对三个臂使用同一确定性 CAL RMS 规则；不减特征均值，避免破坏零增量和反对称：

    gamma_m = sqrt(sum_s w_s * Phi_raw_m(s).square().sum(0))
    gamma_m = gamma_m.clamp_min(1e-6)
    Phi_m = Phi_raw_m / gamma_m

每个臂保存自己的 gamma；共享的前32个线性特征自然具有相同 gamma。该规则是本版固定坐标定义，不是一个需要额外消融或预实验选择的超参。

每个臂 C_m 为 F64 64×32；预测：

    pred_math_m = X.double() @ A_q1 + Phi_m @ C_m @ Q32.T

C=0 的基线就是相同 Q1。P/Q、b/sigma、gamma 与三个 C 全部保存真实张量，不能仅保存 norm/hash。

### A3. 原 fit_readout 的求解规则直接保留

只把 a0 参数换为固定 A_q1，feature_map 换为上述固定特征；F64 sum-of-Kronecker 正规方程逐字复用：

    K += kron(w_s * Phi_s.T @ Phi_s,
              Q32.T @ (g_s.square()[:,None] * Q32))
    T += w_s * Phi_s.T @ ((R_s * g_s.square()) @ Q32)

    eps = 1e-6
    rho = eps * trace(K) / 2048
    vec(C) = solve(K + rho*I, vec(T))

源权重只出现一次，不能在两个 Kronecker 因子各乘一遍。rho 就用旧 READOUT_EPS 的相对 Tikhonov 规则，不换成 1e-3，不按 DEV 更改，也不重跑另一档 lambda。

如果 trace(K)=0 且 RHS=0，返回 C=0 并记 NO_FEATURE_SIGNAL；这是合法无信号结果，不应加 rank、换源或调 ridge。非有限数、负的 trace、求解异常属于工程问题。正常分支保留旧正规方程相对残差诊断，默认要求 <=1e-8；不做第二套全量 SVD/eigh 求解来重复确认这个已验证读出求解器。

本版的目标是：

    J(C) = sum_s w_s * ||g_s * (X_s A_q1 + Phi_s C Q32.T - Y_s)||_F²
           + rho * ||C||_F²

### A4. 实际 BF16 回放：随本次实验完成，不单开 floor

直接复用 ridge_fit_core.bf16_replay 的 Q1 cache/gate 语义。它表示 512 行本地 BF16 算子/缓存/门控回放，不含完整残差流 x 的最终相加，也不是整模型视频或原生性能结果。

统一 Qb(x)=x.float().bfloat16()；GEMM 的 BF16 输入与权重保持 BF16 输出；tanh 的输入变换在 FP32，tanh 后转 BF16；gate 乘法为 FP32 相乘后转 BF16。

固定 P32_BF16=Qb(P64[:,:32])、Q32_BF16=Qb(Q32)、b_FP32=b.float()、sigma_FP32=sigma.float()。不把 b/sigma/gamma 偷并进 P32 后再改变投影舍入顺序；gamma 按下文并入各自读出系数。

### Q1 与 L64

Q1：A_Q1_BF16=Qb(A_q1_f32)，直接调用原 bf16_replay。

L64 将所有线性残差离线折叠：

    A_L_math = A_q1 + (P64 / gamma_L[None,:]) @ C_L @ Q32.T
    A_L_BF16 = Qb(A_L_math)
    bf16_replay(..., A_L_BF16)

### Endpoint

先折叠前32个线性项：

    A_E_linear = A_q1 + (P32 / gamma_E[:32][None,:]) @ C_E[:32] @ Q32.T
    A_E_linear_BF16 = Qb(A_E_linear)
    C_E_N_BF16 = Qb(C_E[32:] / gamma_E[32:,None])

    endpoint_B(u):
        h = u.bfloat16() @ P32_BF16
        v = tanh((h.float() - b_FP32) / sigma_FP32).bfloat16()
        n = (v @ C_E_N_BF16) @ Q32_BF16.T
        return Qb((u.bfloat16() @ A_E_linear_BF16).float() + n.float())

    cache = Qb(y0.float() - endpoint_B(u0).float())
    base  = Qb(endpoint_B(u1).float() + cache.float())
    gated = Qb(base.float() * g.float())
    target = Qb(y1.float() * g.float())

端点分支不额外常驻 h0；其锚点贡献已在单残差 cache 中。

### Delta-only

同样折叠前32线性项，C_D_N_BF16=Qb(C_D[32:]/gamma_D[32:,None])。先用 A_D_linear_BF16 的原 bf16_replay 取 base，再按下式加增量：

    h0 = u0.bfloat16() @ P32_BF16    # 在线方法需要保存这个锚点投影
    h1 = u1.bfloat16() @ P32_BF16
    v  = tanh((h1.float()-h0.float()) / sigma_FP32).bfloat16()
    n  = (v @ C_D_N_BF16) @ Q32_BF16.T
    base = Qb(base.float() + n.float())
    gated = Qb(base.float() * g.float())
    target = Qb(y1.float() * g.float())

不能把 Delta-only 写成某个单端点 B(u) 的差，也不能假装它不需要锚点投影缓存。本次无需开发生产融合 kernel，只如实固定并执行这条 BF16 本地路径。

参数导出时保存这些实际参与回放的 BF16 A、P32/Q32/C_N，以及 b/sigma FP32，重载后评分；不依赖运行结束后的临时张量。

### A5. 仅三组必要检查

### 检查一：同一 Q1 与零修正

- 重建 Q1 的 CAL/DEV 数学风险与 V4 公开 512 配置逐源值核对，rtol=1e-5、atol=1e-12；此为祖先身份检查，不是再次拟合 Q1。
- 每个 site 选 manifest 排序第一条 CAL 源，分别将三个 C 置零，用真实预测/回放函数检查与 CURRENT_Q1 相同；数学预测与 BF16 gated prediction 均 torch.equal。
- 同时确认四臂读的是同一 source/token panel，零修正没有更改目标、gate 或基线。
- 身份或零修正检查失败是 ENGINEERING_FAILURE；不进入效果解释，不用“效果小”来放宽数值身份。

### 检查二：数学 CAL 总目标

在该 site 全 CAL11、完全相同权重下计算 J0=J(C=0)、Jfit，以及未加正则的 CAL 总风险。容差固定：

    tol_J = 1e-8 * max(abs(J0), 1e-30)
    require Jfit <= J0 + tol_J

结合正规方程残差报告。保证仅限数学精度 CAL 聚合目标/风险，不要求每个 CAL 来源改善。不能用某一个 CAL 源没有改善，或 DEV 退化，判工程失败。

### 检查三：真实 BF16 回放有结果

四臂在全部 CAL/DEV 的六站点上保存有限的 BF16 gated 风险，使用同一 target=Qb(y1*g)。BF16 不再优于 Q1是科学结果，工程上不要求“BF16 必须保持数学收益”。

同时可保存 BF16 预测与相应未量化数学完整输出的差异作为 implementation_gap；不得把 BF16 对 Dense 的总误差命名为“纯舍入 floor”，也不据此将风险宣布为不可测。

没有第四类“必须先证明视频收益/功效/最小改善幅度”的前置门槛。输入有真实损坏、代码不满足上述三组合同、或者原有资源/权限条件不成立，才是运行阻塞。

### A6. 完整输出与固定读出

每个 site 保存一个可 weights_only=True 重载的参数文件；至少包括：

- A_q1_f32、alpha_Q1、P64_F64、Q32_F64；
- b_F64/sigma_F64、三个 gamma_F64、三个 C_F64；
- CURRENT_Q1 / L64 / Delta-linear / Endpoint-linear 的实际 A_BF16；
- P32_BF16、Q32_BF16、两个非线性 C_N_BF16、b_FP32/sigma_FP32；
- 参数祖先/数值路径使用 JSON metadata，而不是自定义 Python 类。

不需要保存 A_q1 的重复 F64 副本，也不存全 Jacobian/Hessian。完整参数文件预计约2GB以内，使用V4的2GiB输出上限；实际以文件字节计数，不能只报张量估算。拟合完成先保存并从文件重载，再进行最终数学和 BF16 评分，保证下游消费真实产物。

固定输出：

    output/params/{high04,...,low34}.pt
    output/A_Q1_RESIDUAL_PARAMS.json       # path/bytes/SHA/dtype/shape/ancestry
    output/A_Q1_RESIDUAL_READOUT.json      # raw CAL/DEV per source/site/arm
    output/CHECKS.json                    # 上述三组检查及求解诊断
    output/SUMMARY.json
    RUN.json / EXIT.json                  # 继续使用原 guard

读出统一按每个来源：层等权→expert 等权；再对来源等权。数学和 BF16 分表。保存：

- 每个来源、站点、arm 的绝对 gated MSE；
- R_Q1-R_m，对三个残差臂分别报告；
- R_linear-R_endpoint；
- R_delta-R_endpoint；
- 以上差值的来源等权均值、11 个逐源值及 leave-one-source-out 均值；
- 相对 Q1 差值为辅表；只有真实分母为 0 才记 undefined，不调用旧 floor 旗标；
- high/low 分报，不用 token/block 增加样本 n。

四臂、六站点、22 源及所需检查齐全后：STATUS=COMPLETE。科学读出直接给均值的正/负/零方向、数学与 BF16 是否同向、11 个 LOO 值的符号，不发明 SMALL 的数值分界，也不把这些描述写成工程通过条件。保留全部四臂，不按 DEV 另选参数、隐藏失败臂或自动跑全层/视频；本版报告后由研究负责人进行一次部署价值判断。

### A7. 最小源码差异，不重新造框架

在新的 stage 复制旧文件作新版本，旧 reports/source 不修改：

1. v4_repr.py → a_q1_residual_repr.py：保留 source_weight、gated_mse、fit_readout 的求解主体；修改 select_pq 的基底参数与残差坐标，加入上述固定中心/尺度/RMS特征；模型列表只有四臂。
2. v4_runner.py → a_q1_residual_runner.py：保留 load_site/build_pack/权重提取/同源聚合结构；PREFIXES 只含 512；fit_config 改为读取已发布 alpha、构造共享特征、拟合三个 C；不调用 free ridge/Q0/Q1 PCG。
3. 加入本规格的四臂 BF16 回放、真实参数保存/重载、三组检查；删除 cal_numerical_floor、source_density_delta 和“11/11 unmeasurable”路径。
4. ridge_fit_core.py 的 _f64_sse/bf16_replay 继续原样复用。q1_fit_core.py 仅为 Q1 祖先依据，不另改其算法。
5. 原 guarded_runner / dispatch_once / job.sh 只改 scope_id/phase、runner import、脚本名称和新 stage；保留已有一次提交、资源、review、ACTIVE、RUN/EXIT机制。不要新造审查流程或重复一遍完整正式拟合做“预演”。

原 guard CLI 形状可直接沿用：

    python dispatch_once_a.py --stage NEW_STAGE --scope-sha NEW_SCOPE_SHA

只有实际独立预审/recheck、有限 ACTIVE 和 code/input hash 已完成才执行该命令。本规格不是 ACTIVE，也不伪造其授权状态。

资源默认复用已成功 V4 envelope：单 A100 80GB、session_max_gpus=2、4 CPU、160GiB host、78GiB GPU上限、3h wall、单个正式 job；新完整生成/D4/O40/视频均为0。新范围不是要求占满预算，不增 reviewer 全量重跑。具体集群分区/账户和可用设备按当前已核实环境一次绑定，不从文档假定资源可用。


## 4. B：固定输入、实现与四格

### B2. B-W 的输入已经确定

| 字段 | 固定值 |
|---|---|
| 来源与角色 | V2/V4 已有 CAL11、DEV11；以当前 `BANK_MANIFEST.json` 的 `cycle_role` 与 source ID 为准 |
| D4 bank | `/public/home/scc/pb23061276/projects/wan-quant/work/g20_v2_capture_20261113_rev2/stage/output/bank` |
| bank manifest | 上述 `output/BANK_MANIFEST.json`；SHA256 `f6f8e100da6f75b2db612edd8c8c4d21729ed9bf99bfb5092941f2381a4d0bee` |
| 站点 | `high/4, high/19, high/34, low/4, low/19, low/34` |
| 端点 | high：step 0→1；low：step 2→3；不跨 expert |
| 行 | 原 PANEL512 的有序前缀 512，保留原空间 token index；CAL/DEV 不重采样 |
| 共同预处理/grid | 当前 cohort 的宽832×高464×81帧；实际 token grid `21×29×52`、M=31668；O40 也使用此预处理和空间映射 |
| 形状 | `U: [512,5120]`；`W1: [13824,5120]`；`W2: [5120,13824]` |
| 共同门 | 每 source/site 使用 D4 hit 的原 gate，作为 `G_ref`；同一值用于 O/D 权重，以及后续 BWS 的 O/D 两种状态 |
| 图文 | 0 新 source、0 新图片、0 新 prompt；旧曝光状态不再触发扩数据任务 |

真实 bank 读取格式来自 `v4_runner.py`：

```python
path = bank_root / source / f'{source}__seed42__s{step:02d}__b{layer:02d}.pt'
data = torch.load(path, map_location='cpu', weights_only=True)
U = data['U'][:512]
# Y_pre_gate 可作历史捕获参考；本轮共同函数目标必须按本节的共同数值目标小节重算。
G_ref = hit_data['gate'].reshape(-1)
```

#### B2.1 两套权重的实际位置

蒸馏模型 `W_D`：

```text
/public/home/scc/pb23061276/projects/wan-quant/weights/Wan2.2-I2V-A14B-LightX2V-Distill-4step/high_noise_model/wan2.2_i2v_A14b_high_noise_lightx2v_4step.safetensors
SHA256 39d3af74e46e59e989db5cc36f401e666f3ae77c7648eb55d3aff9fee47716fa
/public/home/scc/pb23061276/projects/wan-quant/weights/Wan2.2-I2V-A14B-LightX2V-Distill-4step/low_noise_model/wan2.2_i2v_A14b_low_noise_lightx2v_4step.safetensors
SHA256 a3b8680affe2ab2e2121210a83ffa7d351f77da18191e57bb359d7839beba8a2
```

原模型 `W_O` 根目录：

```text
/public/home/scc/pb23061276/projects/wan-quant/weights/Wan2.2-I2V-A14B-bf16
```

该目录名虽含 bf16，已执行的原始证据明确记录 W1/W2 的 storage dtype 为 F32，native conversion 为 `F32_TO_BF16_RNE`。不要根据目录名假定存储 dtype；只取六站点对应 tensor，转换一次后记录实际 BF16 payload SHA。

| expert | layer | shard 后缀 | 已绑定完整 shard SHA256 |
|---|---:|---|---|
| high | 4 | `high_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors` | `aeea563f9d38ec434b6761c497027ed2843220ee3efaf8c92c815255ade955e3` |
| high | 19 | `high_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors` | `c5d7ddaae0ef24ab452d852a7a53dde7bdad598cd8542e6824e65b0666f39e7f` |
| high | 34 | `high_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors` | `a95d645bfdac3bf13f96d52299fac1a416c7a3bc8741a3ea58e5b7fd0eb3505f` |
| low | 4 | `low_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors` | `1127e3dea8c08cd746e36d1a7047a3197449adf13d90725ae0a276aeccaf8521` |
| low | 19 | `low_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors` | `d0f75d2f41fdab239dbc0624c13ca1c56d79196f2aa0ac57f5f5c68cb53220ea` |
| low | 34 | `low_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors` | `8b024bda8fb709ba69ec91f1efa5edc19e173d40dcb5b0936050c985167a1be9` |

各站点读取 `blocks.{layer}.ffn.0.weight/.bias` 与 `blocks.{layer}.ffn.2.weight/.bias`。W1/W2 的 raw 与 native BF16 tensor SHA 已在旧 `matrix_manifest.json[site].provenance.w1/w2` 中，不需要重建原模型的整个资产树。

原权重 manifest SHA 为 `df3e1fe3654b6d5af48d1193f7d88e99d4224e7dbc7a6ed70b47afa87323abba`。上述路径/sha 的原始证据见文末证据路径。旧 original40 的 `alpha_*` 来自 `chip10_02/chip10_04`，不得当作本轮 CAL11 的 Q1。

### B3. 默认数值目标：两套权重都做相同的局部函数 replay

**每个 W 都在同一 512 行 U 上重新计算自己的 anchor/target。** 不混用旧 full-M 捕获的 `Y_D` 与新 512 行重算的 `Y_O`；GEMM 行数/实现不同可能改变 BF16 结果。整个本轮默认均为相同设备、后端、权重转换、张量 layout 与操作次序。

复用现有 `fit_core_v2.native_ffn` / V4 `ridge_fit_core.native_ffn` 的真实配方：

```python
z = torch.addmm(b1, U, W1.T)       # BF16 输入/权重与输出
h = F.gelu(z, approximate='tanh')  # BF16 输出
y = torch.addmm(b2, h, W2.T)       # BF16 输出
```

对全部 22 source × 6 site × 2 W × 2 endpoint，得到各自 `z0,h0,y0,z1,h1,y1`，所有方法共享这些 own-W 端点。总计 528 次 512 行局部 FFN 计算，匹配身份的输出可直接复用。存源图文/面板 SHA、weight SHA、recipe/runtime SHA、shape/stride/dtype 与端点 payload SHA 即可；没有新完整轨迹。

主结果统一为：各方法**真实 BF16 pre-gate prediction** 与 own-W BF16 `y1` 的误差，升到 F64 后乘同一 `G_ref` 并计算 SSE，除以 `512×5120`。Q1 用 `bf16_replay(...)[0]` 的 pre-gate base；ICC 用现有两残差 BF16 prediction；Chip 用真实5120列indices驱动的既有packed pre-gate prediction。此主风险不混入某一个方法独有的额外 gate 舍入。

统一辅表再计算 `BF16(pred.float()*G_ref)` 对 `BF16(y1.float()*G_ref)` 的 F64 MSE。Q1 的 `y0 + ΔU A` 数学 lift 可以同时保存作诊断，但不能用数学 lift 的 Q1 与 BF16 ICC/Chip 混成一个主排名。无需为 floor 另起任务。

这是**采样局部 FFN 函数 replay**，不是完整M GPU原生视频、自由轨迹性能或系统加速。P64也按本节P64小节明确的局部数值构造消费真实64行；本轮不需要full-M均值、128分组统计或跨两套权重的全M FFN重算。

### B4. 先完整完成 Q1/ICC；配置不再留给执行者选择

### Q1：`Q1R2_CAL11_P512_OWNW`

每 W/site 从自己的 CAL11 × 512 行端点构造 `X=U1-U0`、`Y=y1-y0` 和 `G_ref`，重新生成 Q0 prior，再按 V4 的 `fit_q1` 拟合一次。固定 `ridge=1e-3,maxiter=240,tol=1e-5`，初始化 `alpha_Q0`，沿用已发表 R2 PCG 的 restart 行为。总计 12 次 Q1 拟合；不做 old/new alpha 选择、不按 DEV 优化、不扫超参。

Q0 的逐隐藏维比例沿用已有 `q0_prior`：以各来源等权的 `Δz·Δh` 与 `Δz²` 矩生成 alpha，构造对应 W 的 A0。 **函数内部维持原数值顺序：U/W1/b1升FP32，FP32 addmm及GELU(tanh)，再将dz/dh升F64累积；分母clamp1e-30，alpha返回FP32，A0按原F64权重与比例构造。不要把主表缓存的BF16 z/h替代这段FP32 prior计算。** 主拟合目标仍是前述own-W BF16端点差分。Q1 求解器仍优化既有门控数学目标；主表实际读出使用固定 BF16 residual cache 路径。方法规则在 O/D 间相同，不声称 BF16 风险也有优化器训练单调性保证。

旧 W_D alpha 只有在 CAL/source/ordered row/U、重新计算的 own-W 目标、G_ref、Q0 prior、数值 recipe/求解规则全部一致时才可复用。**默认重新拟合这 6 个 W_D 站点**，因为旧 bank 目标源自 full-M 捕获，不能盲用旧 alpha。已完成的新 512 行原生目标只生产一次并供所有方法复用。

### ICC：`ICC_SVD_R512_TWO_RESIDUAL`

固定普通 SVD，W1、W2 各 rank 512；不改成 CD-SVD，不加第 4 臂，不扫 rank。旧 W_D factors 是 weight-only 资产，若 native BF16 权重/算法身份匹配可复用，不受 Q1 的新 CAL 目标影响；W_O 对自己的 W1/W2 按同一固定方案各分解一次。缺少一个 site 的 factor 只补这个 site。

旧远端 factor 索引：

```text
/public/home/scc/pb23061276/projects/wan-quant/work/g20_r1_screen_20261111/icc_svd_manifest.json
SHA256 6d0a576481f8d36c367171bd47f2b81029d5470bff06a3584740f550047c876e
```

沿 `manifest[site]['path']` 加载，核对 `expert,layer,method='svd',rank=512`；四因子形状固定为 `(13824,512),(512,5120),(5120,512),(512,13824)`。按照 `fit_core_v2.icc_two_residual_decomposed` 原顺序：

```python
first(x) = linear(linear(x, fc1_right), fc1_left)
second(h) = linear(linear(h, fc2_right), fc2_left)
q1 = z0 - first(U0)
q2 = y0 - second(h0)
pred = second(gelu(first(U1)+q1, approximate='tanh')) + q2
```

所有 `z0,h0,y0` 均来自该 W；不得沿用另一个 W 的 cache。这是现有两残差移植的结论边界，不是完整作者 ICC 系统已复现的证明。 原 `icc_two_residual_decomposed` 内部会重算native端点且只返回诊断字典；新wrapper直接提取上面相同的first/second/q1/q2/pred算术，接收已经计算好的own-W z0/h0/y0并返回pred，再进入共用评分。不要为了取pred重复调用整套native回放或误把诊断字典当张量。缺失因子沿已有producer的精度与奇异值拆分规则生成，函数入口在一次绑定时从factor provenance定位，不另选SVD算法。

### 成本冻结

`IDENTICAL_WITHIN_METHOD_ACROSS_WEIGHTS`；`CROSS_METHOD_COST_REPORTED_NOT_MATCHED`。CAL 数据、rank/selector 参数和方法数值路径在 O/D 间相同；跨方法的真实成本分列，不先扫等预算配置。

| 方法 | 固定 hit 主算术 | 必需 cache 与选择成本 |
|---|---|---|
| Q1 | 每 token `5120²` MAC；A 可预构造 | 一个 5120 维 BF16 residual cache |
| ICC r512 | 每 token `2×512×(5120+13824)` MAC，另有 GELU/加法 | 13824 维首残差 + 5120 维次残差 |
| Chip GLOBAL_P64 | 固定K=5120；每被更新token的W1/W2合计 `2×5120×5120` MAC；另有每site的64行W1/GELU/RMS与排序 | hidden/output cache、64行input/hidden读取、indices、gather/packing/cache update都计入；没有block mean cache |

本轮不用采样回放 wall 冒充 full-M latency；可报告本轮真实计算消耗和上述解析成本，不额外开 full-M 性能网格。

### B5. Chip唯一主配置：已有 `CHIPMUNK_GLOBAL_P64`

本轮解释用户已经实现的ICC与Chipmunk，唯一Chip主比较器恢复为**真实已测的FFNDS `CHIPMUNK_GLOBAL_P64`**。它以64个采样token估计隐藏变化，选一个作用于全部token的全局列mask；不是global-mean selector，也不是128-token分块selector。

`AUTHOR_MLP_TK030` 仅在方法区别说明中保留：它是另一套作者selector。它不进入当前queue、不作fallback、不要求新CUDA扩展、分块wrapper或padding实验。恢复实际P64也使此前提出的full-M块统计资产不再适用于当前B；改为 `B_CHIP_P64_INPUTS`，任务名 `B_CHIP_STATS` 可保留。

#### B5.1 原来真正执行的P64规则

权威源为 `tools/chipmunk_fullm_proxy_math.py::rms_score/selection`，以及 `tools/chipmunk_wan_global_p64_tap_v2.py` 中的实际candidate调用；配置 `e0_chipmunk_global_p64_video_v2_20261030_part0_pending_review_v4.json` 明确 `K=5120,p64_seed=42,M=31668`。

固定64行是CPU `torch.Generator().manual_seed(42)` 下 `torch.randperm(31668)[:64]` 的原始顺序。它们来自完整真实token域，**不是当前PANEL512的前64行**：

```text
29814,13390,19510,21217,3134,5134,10930,19066,11058,5825,13134,25134,
17942,4262,30241,19747,10647,22026,26685,20204,16905,28112,9885,4294,
16583,19825,3843,22491,25475,27699,25116,21,17163,28327,15803,13552,
4607,15752,11320,7509,12590,12136,30301,19937,6303,11549,1469,15805,
9981,16693,24094,24909,8422,9622,2872,16788,14615,25043,12326,30753,
18494,21183,21370,21523
```

这些字面index亦保存在已执行packed配置的 `sample_prefix64` 中。当前VBench panel使用独立namespace `G20_VBENCH_RESTART_V4_20261113_ROWS_V1`，不能因为生成seed也为42就断言覆盖这64行；只需一次读取实际panel row IDs并求集合差。

历史已执行selector顺序是：anchor全M原生调用保存 `h0_fullM,y0_fullM`；hit从当前U取这64行，做BF16 `F.linear` 和 `GELU(tanh)`，与**先前全M缓存取出的h0[64]**做FP32差，再沿64行计算RMS。每个隐藏列一个score：

```python
h1_proxy = F.gelu(F.linear(U1_64, W1, b1), approximate='tanh')
scores = (h1_proxy.float() - h0_cache_64.float()).square().mean(dim=0).sqrt()
ranked = torch.argsort(scores, descending=True, stable=True)
indices = torch.sort(ranked[:5120]).values
```

`stable=True` 使分数相同时保留原隐藏列顺序；原列顺序是递增的0..13823，因此tie优先小列号。选完再将5120个indices升序排列，固定selected GEMM累加顺序。K恰为5120，不是近似30%，没有向256取整，也没有随机加列。一个mask同时作用于全部真实token；本轮只对共同PANEL512评分。

#### B5.2 本轮局部数值身份，避免追加全M重算

主方法ID仍为 `CHIPMUNK_GLOBAL_P64`，数值适配ID固定 **`OWNW_LOCAL512_P64ANCHOR64_V1`**。两权重都采用同一构造：

1. 主评估anchor/target与所有其他方法一致，按本节的共同数值目标小节对PANEL512做own-W `native_ffn`，得到 `h0_eval512,y0_eval512,y1_eval512`。
2. 按上述真实64个空间index读取 `U0_64,U1_64`。无论其中多少行已包含在PANEL512，都将完整64行按固定原顺序一次性构造。anchor以64行 `F.linear→GELU(tanh)` 求own-W `h0_selector64`；hit以64行 `F.linear` 求 `z1_proxy64`。
3. 原样调用 `scores=rms_score(z1_proxy64,h0_selector64)` 和 `indices=selection(scores,5120)`。`rms_score` 内部已执行hit的GELU，外面不要再做一次。selector学习规则、K、行indices、tie与列顺序均不变。
4. 以得到的同一个mask对PANEL512执行下段的既有packed action；action anchor cache使用 `h0_eval512,y0_eval512`，不是另一权重或selector64张量。

历史full-M缓存抽样的 `h0_cache_64` 与本轮64行重算的 `h0_selector64` 可能因GEMM形状/数值次序不同而不逐bit相同。因此本轮明确报告局部函数replay的数值适配身份，不声称原样重现旧full-M视频mask。**不能为追求旧视频的bitwise mask而新增两套权重full-M F1或全M状态资产。** 这种统一适配在O/D之间固定；BWS继续使用相同适配。

真实旧方法没有读取当前完整hidden来选mask；本轮也不能把已算的 `h1_eval512` 或完整target hidden当selector输入。固定64行只用于在线因果proxy，没有DEV阈值拟合。

#### B5.3 直接复用已执行的packed pre-gate action

唯一默认action入口为 **`tools/chipmunk_wan_packed_pregate.py::packed_pregate`**，与V2实际视频调用相同；不引入作者新kernel。该函数已存在，不需要逐group wrapper。计算顺序固定：

```python
packed_w1 = W1.index_select(0, indices)
packed_b1 = b1.index_select(0, indices)
packed_w2 = W2.index_select(1, indices).contiguous()
hc, yc = h0_eval512.clone(), y0_eval512.clone()
old_selected = hc.index_select(1, indices)
new = gelu(linear(U1_eval512, packed_w1, packed_b1), approximate='tanh')
delta = (new.float() - old_selected.float()).bfloat16()
hc.index_copy_(1, indices, new)
fp32 = torch.mm(delta, packed_w2.T, out_dtype=torch.float32)
pred = (yc.float() + fp32).bfloat16()
```

W1线性输出先舍入BF16，再GELU输出BF16；delta舍入BF16；W2乘积用FP32输出，先加旧y的FP32值，再舍入BF16。b2已在own-W原生y0中，不再次加入。mask必须按已有selection输出的升序；记录indices SHA和固定 `selected_k=5120`。上述操作、cache clone、gather和selection都算成本，不能只报被选矩阵乘。

其他已有证据包括 `tools/chipmunk_a100_selected_delta_kernels.py`、`chipmunk_p4_sparse_action.py` 与真M same-mask packed/Triton比较；它们证明历史算术移植可用，不要求执行者重新做一个作者系统复现。若实际环境缺旧packed所需的Torch接口，只报告这一具体运行环境缺项并保留已完成Q1/ICC；不启动新作者extension框架。

#### B5.4 最少输入缺口与有限补捕获

每个DEV source/site只需要：真实固定64行的 `U0_64,U1_64`、上述row IDs及来源/endpoint身份。h0/h1、score、mask均由own-W局部计算产生；**无需full-M global mean、128块mean、full-M h/y、tail有效行统计或padding。** `M=31668` 仅确定空间index域与既有固定64行身份。

优先级唯一：当前PANEL512中已含行 → 其他同身份已存真实U或可恢复原生状态 → 对缺失DEV source各补一次D4，仅捕获这64行及本轮六site的必要端点。一次source捕获同时供O/D权重的局部重算，不因weight/site拆成多条轨迹。额外保存PANEL512身份校验点以确认同一状态即可，不扩大采集范围。

`B_CHIP_P64_INPUTS` 的缺项清单列 `source,site,anchor_or_hit,missing_row_ids,reusable_path`；一次合并到每source。最多11条D4，0新图文、0CAL轨迹、0decode、0视频评分；现存U足够则0新增D4。每个source每端点每site最多64×5120个BF16值，远小于此前错误提出的full-M分组资产。

新捕获若与既有条件的共同校验行不一致，保存差异并只阻塞该source的Chip单元，不将不同状态悄悄拼入原B-W主表，不重跑全部CAL11/DEV11。

#### B5.5 已执行证据

`research/20261030/CHIPMUNK_GLOBAL_P64_V2_ALL160_VIDEO_RESULTS_20261030.json` 的状态为 `TWO_SOURCE_ACTUAL_GPU_AND_VERIFY_EXIT0`：来源898/96，seed42，各自四个video phase含global_p64；学生自己的step0/2 h和pre-gate y缓存，step1/3自己的P64+packed更新；每source 80个P64 hit，完整160站点路径。报告中的pre-gate控制在两source的high/low检查点均为relative L2=0、absmax=0。

这证明实际P64实现已执行；它不替代当前VBench CAL11/DEV11的配对函数表，也不把历史固定顺序video wall作为本轮公平系统速度证据。

### B6. B-W 表格、比较与完成标准

`BW_CELL.csv` 每行一个 DEV source/site/W/method，主表应先有 `11×6×2×2=264` 行；Chip 补齐后为 `396` 行。字段至少：

```text
source_id, role, expert, layer, state_mu, endpoint_steps, weight_id,
method_id, config_sha, fit_sha, U0_sha, U1_sha, row_indices_sha,
G_ref_sha, own_anchor_sha, own_target_sha, numerical_recipe_sha,
n_rows, output_dim, pred_sha, gated_sse_f64, gated_mse_f64,
bf16_gate_mse, chip_indices_sha, chip_selected_k, chip_proxy_rows_sha, cell_status
```

非Chip的indices/K/proxy rows字段填null。另出 `BW_COST.csv`：各方法参数/因子字节、cache字节、解析MAC、P64每site额外64行proxy与排序成本、此次真实计算消耗；不伪装成等成本排名。

先在每 source 内对三层等权，再对 high/low 等权得到 `R_source,m,W`，再对 11 个来源等权。high/low 分报，六 site 原始表保留。统计单位为 source，512 行/六 site 不变成独立 N。

固定符号：`d_m(W)=R_m(W)-R_Q1(W)`，正数表示 comparator 比 Q1 风险高；`theta_m=d_m(W_D)-d_m(W_O)`。同时给全部绝对风险和 11 个逐源 theta。汇总采用逐源差、来源等权均值/范围，以及 leave-one-source-out 的11个均值与符号；不新增 bootstrap、CI 或 p 值门槛。

theta>0 表示该固定 comparator 相对 Q1 的差距在蒸馏权重下更大；theta<0 表示差距缩小。若 LOO 均值变号，写明“均值方向对单一来源敏感”；这不是显著性检验，不决定是否执行 BWS，也不等于排除状态/间隔交互。

### B7. BWS 是既定后续，限定原生复合状态问题

**触发规则固定：** `BW_Q1ICC_COMPLETE` 且没有同源/同方法/同拟合身份的匹配四格实测，进入本节；完成一次独立预审后执行。仅真实的 weight/interface/合法 endpoint 缺失阻塞相应单元，noise/gap 无重叠不阻塞复合四格。

固定 CAL3：`vbi_1005,vbi_0895,vbi_0648`；DEV3：`vbi_1013,vbi_0886,vbi_0800`。图文、seed=42、geometry 与 D4 manifest 相同；先复用真正匹配的 O40 状态，否则每条件一条，最多 6 条 O40。旧 chip10 原生轨迹只提供 recipe 参考，不自动当这六源的状态。

### O40 默认 recipe 与 endpoint

使用已执行 original40 producer 的原生 scheduler/CFG recipe：`wan2.2_moe,i2v,infer_steps=40,fps=16,sample_shift=5,guide_scale=[3.5,3.5],enable_cfg=true,boundary=0.9`；Torch SDPA、Torch norm modulation、double precision RoPE、block offload 等继承旧 `required_runtime_config`。**共同图像预处理与几何覆盖为当前 V2 cohort 的宽832×高464×81帧，实际 token grid 必须为21×29×52；相同空间 panel index 才有确定对应。** 旧 scope 的480×832只提供原生 scheduler/CFG 与运行入口证据，不能直接当本轮匹配状态，更不能跨 shape 复用其面板。这里不把 D4 scheduler 移植给原模型。

**唯一评分分支为 conditional。** CFG 的 unconditional 调用仍计入 O40 轨迹成本；它影响生成状态，但不新增第二个评分臂。每 expert 取实际调度的**首对相邻且同 expert 的 scheduler step**，读取这两步 conditional pre-W1 U；记录 timestep/noise/gap，不靠数字相近配对，不跨 expert。D4 仍用高 0→1、低 2→3。

本合同不再等待选择长短间隔网格。有限四格估计对象是“包含各自 scheduler/CFG/interval 的 native conditional 输入状态分布”的函数风险交互；单独的纯间隔/CFG/纯状态成分没有匹配控制时记 `UNIDENTIFIED`。

### 混合 CAL 与四格

两状态均沿相同空间 panel index 读取 512 行，先按本节的共同数值目标小节统一 512 行 native_ffn recipe 重算每个 W 的 anchor/target。每 CAL 来源用 D 前256 + O前256，来源等权且状态各半；每 W/site 学一套 Q1 参数。ICC同一own-W r512 factors；Chip同一GLOBAL_P64规则与 `OWNW_LOCAL512_P64ANCHOR64_V1` 数值适配。不要分别按测试状态重新拟合。

每条选定端点对都从该 W 的 fresh anchor cache 开始，只执行一次 hit；Chip 的局部逻辑坐标为 anchor→hit，不把 O40 的绝对 step 奇偶当成新增 full-step schedule。O40 全轨迹仅生成原生输入状态，本轮没有在其中部署 Chip/ICC/Q1 学生。

`G_ref` 始终是同源 D4 hit gate；混合 CAL 的 O/D 两半和 DEV 四格都使用它，不随 W 或 μ 改。DEV 每 source/state 均使用 512 行，报告 3 源 × 6 site × 2W × 2μ × 2methods =144 行；加 Chip 后 216 行。

`BWS_CELL.csv` 与 B-W 同字段，另加 `cal_mixture_sha,native_recipe_sha,branch,actual_t0,actual_t1,noise0,noise1,gap`。`BWS_SOURCE.csv` 给每来源的四格风险、d、同权重状态差，以及：

```text
I_m = [d_m(W_D,mu_D)-d_m(W_O,mu_D)]
      -[d_m(W_D,mu_O)-d_m(W_O,mu_O)]
```

BWS 的μ_D P64单元复用 `B_CHIP_P64_INPUTS` 中DEV3的D4固定64行；μ_O的固定64行由本次DEV3 O40同时捕获。任一状态缺这些行只阻塞该状态的Chip单元，不阻塞Q1/ICC。

BWS 的 Q1 拟合身份是 CAL3 混合，不得与 B-W 的 CAL11 D-only 表拼接交互。n=3 是机制开发的小样本，只报告逐源值与均值/范围，不靠 token 数制造精确显著性。四格对角线也不代表两模型分别做了 native 最优校准后的部署成绩。

### O40 的真实 producer 与可继承资源上限

可直接定位的旧 producer 目录：

```text
/public/home/scc/pb23061276/projects/wan-quant/work/g19_anchor5_gpu_20261105/anchor5_gpu_chip10_00_r8
runner_video.py / normal_pipeline.py / scope.json
Python: /public/home/scc/pb23061276/miniconda3/envs/hif4/bin/python3.11
LightX2V: /public/home/scc/pb23061276/projects/wan-quant/code/LightX2V
```

继承原生 pipeline 构建和 scheduler/CFG recipe，并明确覆盖为上述共同图像预处理/464×832几何；不原封不动运行旧三 pass `dense/gpu_bf16_anchor5/gpu_fp32_anchor5`。本轮只需要一条 dense 原生轨迹的被动六站点 endpoint/统计捕获。旧运行记录的单 job 上限为 1 GPU、4 CPU、192 GiB host、76 GiB GPU、7200s wall、4 GiB output；这些是参考上限，不是本轮耗时测量，也不继承旧作业授权。六条件串行；本轮正式 scope 把实际新增量冻结为最多 6×40 scheduler steps、最多 6×80 native model flows（CFG 两次）、0 视频评分、0 student 部署。

在这最多六条轨迹内，DEV3同时保存Chip所需O状态的真实固定P64 anchor/hit输入64行与空间index映射；只能六site，不扩40层，不采full-M块统计。P64不拟合CAL，CAL3只需Q1混合拟合的既定panel输入。own-W selector隐藏值在本地重算；不暗增第七条原生轨迹。


## 5. 最小实现差异与文件边界

A 的具体函数改动已列于上文。B 只新增两个小模块：`b_common_replay.py` 封装共用loader、own-W目标、现有Q1/ICC/P64调用及F64风险；`b_weight_state_runner.py` 根据固定BW/BWS模式组织数据、拟合和结果表。P64沿用 `rms_score/selection/packed_pregate`；Q1沿用V4的`q0_prior/fit_q1`；ICC沿用已有两残差函数。缺输入时才编写 `b_capture_inputs.py`，从既有V2/O40 producer借用被动捕获入口。新增模块不是本手册已经实现好的代码。

O40只跑一个dense原生轨迹，删除旧runner的另外两个candidate video pass，不增加代理方法，不改native scheduler。捕获hook只复制规定的pre-W1输入，返回原值；按已记录同身份资产验证所需数值。原guard/dispatch机制继续复用，只修改本scope的消费者、输出和身份清单；不得照抄旧scope的批准、输出数量或成功状态。

统一运行目录：`/public/home/scc/pb23061276/projects/wan-quant/work/g20_two_track_core_20261114/{task_id}/{scope_version}/stage`。归档根：`reports/20261114/g20_two_track_core/{task_id}/{scope_version}/`。每个版本仅保存一份source snapshot、scope、实际RUN/EXIT、必要日志、参数和既定结果。上游bank、cohort角色、旧source/scope/RUN/EXIT只读。

| 阶段 | 必须保存的科学产物 |
| --- | --- |
| A | `params/{site}.pt` 六份，`A_Q1_RESIDUAL_PARAMS.json`，`A_Q1_RESIDUAL_READOUT.json`，`CHECKS.json`，`SUMMARY.json` |
| BW | `BW_CELL.csv`，`BW_SOURCE.csv`，`BW_COST.csv`，可重载Q1/ICC参数与P64 indices，`PARAMS_MANIFEST.json`，`CHECKS.json`，`SUMMARY.json` |
| B_CHIP_STATS | 固定64行U与token IDs、同状态校验、来源/endpoint/recipe身份；输入manifest及必要缺项清单，不保存full-M块均值 |
| BWS | 所需O40端点输入及DEV3的P64行，混合CAL参数，`BWS_CELL.csv`，`BWS_SOURCE.csv`，`PARAMS_MANIFEST.json`，`CHECKS.json`，`SUMMARY.json` |

所有结果目录有实际 `RUN.json/EXIT.json` 和一个 `MANIFEST.json`（路径/字节/SHA/祖先）。沿用launcher需要的治理文件名即可，不另造同义报告或批准文件。张量按site处理与保存，不为“以后可能有用”归档全M hidden、全部current target hidden、Jacobian或Hessian；主表记录必要payload hash和可重建输入。

## 6. 什么阻塞，什么照常完成

阻塞必须记录四项：`code / evidence / affected_unit / unblock_action`。只能暂停依赖这个事实的单元，不把一个Chip输入缺口传播成全项目停止。

| 观察 | 处理 |
| --- | --- |
| 实际输入不可读或哈希不符；CAL/DEV角色混用；非法expert端点/token对应 | 停受影响消费者，定位那个具体输入或实现错误 |
| 非有限值、零C恒等失败、固定正规方程/目标检查失败 | 保留失败现场，修源码；按原协议用新版本定向复核，不自动重试 |
| 当前scope尚未获真实预审/ACTIVE，或实际资源不足 | 不启动该计算；完成其他静态准备或已具备条件的范围 |
| P64的64行缺失 | 只标对应Chip单元缺失，按固定输入补缺规则处理；A/Q1/ICC继续 |
| 历史曝光、DEV已见、仅六site、没有第二架构、没有完整视频 | 继续当前开发比较，限定对应结论 |
| A无收益、收益很小、BF16吃掉收益、BW差分不稳定 | 有效科学结果，当前阶段正常完成；不改参数补救 |
| O/D noise/gap不重叠 | 合法复合四格继续；纯成分归因标UNIDENTIFIED。即使有重叠，也仍需匹配控制才能拆分 |
| 旧receipt未公开但执行节点实际输入可核 | 记证据公开缺口后继续，不重复G0/V0/V2 |

### 常规审查只走一轮闭环

作者一次自审 → 真正不同agent一次独立审查 → 集中答复 → 同reviewer一次定向recheck → 有限ACTIVE → 调度前核对冻结集合 → 单次dispatch → 实际RUN/EXIT与结果闭合。没有新阻塞或身份变化，就不重开完整审查，不要求独立reviewer完整重跑科学实验，不增加第三方投票。非阻塞建议在原记录答复，不派生新pilot。

有真实阻塞不能强行批准。工程失败先保存现场与成功单元，再从日志定位具体根因；修复以同目标新版本复核，失败消耗继续计账。同类根因修复后再次失败，向研究负责人报告该具体错误，其他独立任务继续。没有科学阳性不属于工程失败。

### 当前明确不做的工作

不新增图文/seed/CAL_B，不扫rank/activation/lambda/eta，不重复R1b/V1/V4补档，不单开floor/接口/导出/中性pilot，不复现作者另一套Chipmunk框架，不为追求历史mask逐bit一致补两套权重full-M F1，不将全80或视频作为这三个阶段的前置，不扩Attention/跨架构/终端Hessian/JVP。接手者可以记录一个以后值得考虑的具体问题，不能自行把它加入当前queue。

## 7. 完成后如何写回

执行成功与科学结果分开。完整结果矩阵、合法身份、参数/数据manifest和实际EXIT齐全即阶段完成；收益可以为负。缺一个单元就精确标PARTIAL和缺项，不静默删来源重归一化，也不把局部负结果升级为关闭整个A/B方向。

每次完成只更新现有[资产与队列](../20261113/G20_VBENCH_ASSETS_AND_QUEUE_V4_20261113.json)、[判断台账](../20261113/G20_VBENCH_DECISION_LEDGER_20261113.json)及ACTIVE对应scope；不要再建第二份总资产树。结果摘要固定回答：完成了哪些单元、主差分是什么、哪些结论仍未识别、实际消耗多少、既定下一步是什么。A结束后不自动跑部署；BW之后BWS照既定顺序。三个固定比较结束后收尾汇报，不再提交一份要求用户重新批准这两个方向的泛化计划。

## 8. 依据与本次状态

- [原两条P0合同](../20261111/G20_TWO_TRACK_OPERATOR_AND_DISTILLATION_ATTRIBUTION_20261111.md)：本轮目标是新算子及**用户实际实现**的方法差异。
- [V4冻结scope](../../reports/20261114/g20_v4_repr_density/evidence/scope.json)、[V4数值实现](../../reports/20261114/g20_v4_repr_density/source/v4_repr.py)、[runner](../../reports/20261114/g20_v4_repr_density/source/v4_runner.py)、[公开完整fits](../../reports/20261114/g20_v4_repr_density/output/VB_LOCAL_FITS.json)：A祖先、固定求解器和已完成负结果。
- [ICC两残差函数](../../reports/20261112/g20_r1b_recheck/source/fit_core_v2.py)与[R1 scope](../../reports/20261112/g20_r1_screen/evidence/scope.json)：当前实际ICC身份。
- [P64 proxy](../../tools/chipmunk_fullm_proxy_math.py)、[packed action](../../tools/chipmunk_wan_packed_pregate.py)、[实际tap](../../tools/chipmunk_wan_global_p64_tap_v2.py)、[已执行P64结果](../20261030/CHIPMUNK_GLOBAL_P64_V2_ALL160_VIDEO_RESULTS_20261030.json)：64行、K5120、真实数值顺序与历史边界。
- [原权重manifest](../../reports/20261105/g19_evidence/browse/work/g19_original40_anchor10_20261104/original40_matrix_build_r1/results/matrix_manifest.json)与[O40旧producer scope](../../reports/20261105/g19_evidence/browse/work/g19_anchor5_gpu_20261105/anchor5_gpu_chip10_00_r8/scope.json)：权重映射、原生scheduler/CFG与资源参考；旧480高度不直接复用为当前464状态。
- [现行实验前协议](../agent_protocol/EXPERIMENT_PREEXECUTION_INDEPENDENT_REVIEW_20261029.md)：保留真实独立复审与有限执行权限；本手册不增加审批层。

本次只发布执行方案，真实设计复核记录见[机器规格的design_review](G20_TWO_CORE_EXECUTION_SPEC_20261114.json)。所有执行scope、job ID和RUN/EXIT仍应由后续真实运行产生，不能从这份设计文档生成虚假批准或结果。
