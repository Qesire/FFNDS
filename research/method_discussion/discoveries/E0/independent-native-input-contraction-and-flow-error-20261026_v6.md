# Student 局部误差降低与四步轨迹恶化可以同时成立：两来源 native 配对记录的独立复核

日期：2026-10-26。作者：独立讨论 Agent `independent_e0_audit`。这是独立发现与挑战记录，不是正式方法 endorsement、质量验收或新执行授权。保留此前 v2–v5 文件及其 SHA，不改冻结源码、原始回执或共享状态。

**新认识：** source0/seed42 与 source12/seed54 各自 159/160 个实际 site 的 student-input native 同输入 operator SSE 小于 teacher-input SSE；第一步第一层相等。与此同时，实际 post-Q1 输入方向已明显不同，后续三个 flow 差都与已有 latent 差反向，实际更新因此继续扩大轨迹距离。这个组合不矛盾，也不足以单独识别“数值误差”“信息损失”或“dense 动力学放大”哪一项是主因。

## 1. 原始身份与亲自完成的核对

读取 HF derived 下相对根 `e0_paired_full_depth_20261026/v3_attempt1/` 的两个 `dc_source00/`、`dc_source12/` 目录；每个目录读取 `TRACE.json`、`RUN.json`、`NATIVE_RUN.json`、`BUNDLE.json`、`STRICT_BUNDLE.json`。仅进行 JSON 标量算术、文件 SHA 和源码阅读；没有运行模型、张量计算、拟合、测试、GPU、SSH 或 Git，没有观看本轮完整视频。

| 文件 | source0 SHA256 | source12 SHA256 |
|---|---|---|
| `TRACE.json` | `26789cad90a54019df901838d6354fad40b229d12b652b9afc5ec4bd180c22c8` | `fb486649c7d51c65a5abc73f9e39d05e502d1e79272d305057cb3e1f844723db` |
| `NATIVE_RUN.json` | `14201944fb5b627e9c31957eb91df8fcdcf06243dae1e3f697695e765a5f4c2a` | `5974b4749eeb42a39bbbb006507dbf0fa1f614bc3c52d60b41956a7e47af4ae2` |
| `RUN.json` | `4b51d835b2f493a06dbac1d1b100c66d380b3ea128fa645591cec6531c49fc98` | `b4cb875f08064deaf24dc2ef67e23cb74ea96b9a0d862da84a925b1c56d8fe1e` |
| `BUNDLE.json` | `d669318069ad8b54ed5a61177de11cfa3874bcf43483dacd8cef9ed2f8a717c8` | `769e77942a38bdb401d06ad96b91bdde31c632154b8ffbcb1d7ec5a4796ec47f` |
| `STRICT_BUNDLE.json` | `55a6fa6e25275f9b92701f008863b87aeff49c970e6a7733d5dd96657056c746` | `92757288bbeefdf2958452b9187a3d940451ee38c9bb22168eb7d93ee4e45533` |

两个 TRACE 均恰好含 `{0,1,2,3} × {0,…,39}` 共 160 个唯一 site，主路径种类逐 site 实际记录为 teacher=`dense`、student=`candidate`。每 site 的 FFN/input/block 标量覆盖 `FULL_M`、矩阵 `[31668,5120]`；完整 latent 形状为 `[16,21,58,104]`。这是两条配对路径的 320 个相依 site，不是 320 个独立样本或跨来源普适性证明。两者 `quality_accepted=false`。

独立重新计算的文件绑定一致：RUN/NATIVE_RUN 的 TRACE SHA、RUN 的 NATIVE_RUN SHA、strict manifest 的 core bundle/native receipt SHA，以及 bundle 中四个实际视频和四个 metric 文件 SHA 均与文件匹配。这里没有重新运行 gate 或重新评估视频；哈希和 PASS receipt 也不构成视频质量验收。

共同冻结协议为 [config/e0_paired_full_depth_20261026_v3.json](/home/xiang/projects/FFNDS/config/e0_paired_full_depth_20261026_v3.json)，SHA `3027bc17588efdcdab32aacc5084cca6705b58d76f75529be43edeeda1b7bcc5`。两个 TRACE 的共同 provenance 为 native source aggregate `32bf1c4ff135e7f0a284d34181a4898b7b89fc6a78db63e8bd5a8d9454f36594`、model digest `decf99fc1ba6763726256d04a263754bd390a9d58f404a484c5cabc74bf05639`、bank RUN `fd38f0c148a9284da26adf35bc43c06e720ecb89b00c35500e5b4d375c32dc81`。

亲读的决定性源码与本地 SHA 均匹配该协议：

| 源码 | SHA256 |
|---|---|
| [tools/e0_full_depth_trace_contract.py](/home/xiang/projects/FFNDS/tools/e0_full_depth_trace_contract.py) | `28154f7d340cf164f07bbb908e31b37f04414b3fb7ef4f9f03e45fbc377ba98e` |
| [tools/e0_paired_ffn_observer.py](/home/xiang/projects/FFNDS/tools/e0_paired_ffn_observer.py) | `691a2e66f3326ce1f6b65536d576f8b89c9861db217fdee4885463a54d196476` |
| [tools/e0_paired_wan_controller.py](/home/xiang/projects/FFNDS/tools/e0_paired_wan_controller.py) | `bd99bdba0651dc59fcc5a6cdf72922906a5c26eb08dfb4d4650f1d6cefa560bd` |
| [tools/run_e0_paired_full_depth.py](/home/xiang/projects/FFNDS/tools/run_e0_paired_full_depth.py) | `c4d362c5b7dca25edc5eb63e6dd769bffde236517ac147d9445be36799d7d0ca` |
| [Wan transformer_infer.py](/home/xiang/projects/FFNDS/.runtime/code/LightX2V/lightx2v/models/networks/wan/infer/transformer_infer.py) | `5fa53d102aed4e2793b4d639da65a71e1e63061d9b2c35574db1b4ac2256c225` |
| [Wan step_distill/scheduler.py](/home/xiang/projects/FFNDS/.runtime/code/LightX2V/lightx2v/models/schedulers/wan/step_distill/scheduler.py) | `2d560449e0559aea44d80afb539f9e97bd544e5e26abcc06236e61918d01e1db` |

## 2. 同分母比较是真的；由此推“student 变准确”仍然越界

对一个固定 step/block，记实际 post-Q1 输入为 qT、qS，native dense/candidate FFN 为 D、S。四臂记录直接测量

\[
e_T=S(q_T)-D(q_T),\qquad e_S=S(q_S)-D(q_S),
\]
\[
p=D(q_S)-D(q_T),\qquad t=S(q_S)-D(q_T)=e_S+p.
\]

源码中两种局部 SSE 的公共参考能量均是 `reference_energy=||D(qT)||²`。两来源各有 159 小、1 等、0 大，而且比较原始 SSE 也成立，故不是“两个不同分母”造成的数值假象。唯一相等者都是 `(step=0,block=0)`，对应未受先前替换影响的共同初始输入。

然而，`||eS||² < ||eT||²` 是在不同输入上评价同一对算子，并没有要求 qS 接近 qT，更没有要求 S(qS) 接近 D(qT)。这里只能说 candidate 在**自己已经访问的输入**上与 dense 的绝对局部差更小。将它读成整网功能更好、扰动越来越小或学生更接近 teacher 都不成立。

`pair_geometry` 还保存了 ET、差能量和 teacher–difference 点积；所以无需新张量或拟合，就能精确重建

\[
\|q_S\|^2=E_T+\|q_S-q_T\|^2+2\langle q_T,q_S-q_T\rangle,
\]
\[
\cos(q_T,q_S)=\frac{E_T+\langle q_T,q_S-q_T\rangle}{\sqrt{E_T\|q_S\|^2}}.
\]

下表都是逐 step 的 40 层中位数；不是把所有层先合成一个向量。

| source | step | `||eS||²/||eT||²` | `||qS||²/||qT||²` | `cos(qT,qS)` | 输入 relative SSE | 局部 SSE 再按各自 q 能量归一后的比值 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.46880 | 0.65111 | 0.48418 | 0.82553 | 0.74055 |
| 0 | 1 | 0.44111 | 0.63781 | 0.43672 | 0.90948 | 0.66433 |
| 0 | 2 | 0.36428 | 0.66356 | 0.38038 | 1.03045 | 0.55691 |
| 0 | 3 | 0.38687 | 0.68119 | 0.35816 | 1.07216 | 0.54168 |
| 12 | 0 | 0.53119 | 0.65486 | 0.51007 | 0.80044 | 0.80269 |
| 12 | 1 | 0.49774 | 0.68071 | 0.60670 | 0.67508 | 0.71436 |
| 12 | 2 | 0.42608 | 0.65713 | 0.49954 | 0.83932 | 0.62857 |
| 12 | 3 | 0.42613 | 0.66830 | 0.50260 | 0.82933 | 0.65937 |

最后一列定义为 `(||eS||²/||qS||²)/(||eT||²/||qT||²)`，在 source0 的 158/160、source12 的 159/160 个 site 小于 1。它只是描述性幅值校正，**不是** FFN 输出相对误差或反事实“把输入放大”的实验。q 是 norm/modulation、Smooth1 和真实 HiF4 QDQ **之后**的坐标；范数缩小不自动等于 latent、norm2 前激活或语义信息整体衰减，也不能假设 QDQ 满足连续齐次缩放。

因此，“post-Q1 幅度降低”是实际存在的解释成分，但简单的纯缩幅说解释不了显著方向变化及上述幅值校正后仍较小的多数误差。更强的竞争解释是学生前缀将输入移到固定 bank 较容易表示的方向，可能同时丢失任务相关创新。当前没有直接测量这种任务信息损失。

## 3. “输出塌缩”要分清 FFN 输出与残差块输出

TRACE 的 full-M 四臂账本没有单独记录 `||D(qS)||²` 或 `⟨D(qT),D(qS)−D(qT)⟩`，因而**不能**从现有 full-M FFN 标量精确恢复 student-input dense FFN 的输出范数、各自输出归一误差或输出塌缩。8 行向量也不能替代完整 M。

但实际 `block_output` 是 pair_geometry，可以用同一恒等式恢复实际 student 残差块输出能量：

| source | step | block student/teacher 能量比中位数 | 最小–最大 | block39 比值 | block 方向 cosine 中位数 |
|---|---:|---:|---:|---:|---:|
| 0 | 0 | 0.89655 | 0.62711–1.34133 | 0.96260 | 0.77339 |
| 0 | 1 | 0.93752 | 0.63139–1.11029 | 0.96761 | 0.76537 |
| 0 | 2 | 0.95793 | 0.61866–1.10050 | 1.01097 | 0.83351 |
| 0 | 3 | 0.95928 | 0.61361–1.09639 | 1.02821 | 0.82434 |
| 12 | 0 | 0.94030 | 0.69081–1.07644 | 0.98767 | 0.83541 |
| 12 | 1 | 0.92383 | 0.57337–1.05944 | 0.96768 | 0.84195 |
| 12 | 2 | 0.94443 | 0.55654–1.05721 | 0.98623 | 0.86446 |
| 12 | 3 | 0.96403 | 0.56287–1.07183 | 0.99236 | 0.86597 |

这不支持“所有 block 输出整体范数随深度塌缩”的强叙述。common residual 可以保留大部分总能量，任务相关的小方向仍可能损失；接近 1 的总范数不能排除这种失真。

## 4. signed 重组成立，但项的大小不是起因份额

逐 site 独立核算 `||t||²=||eS||²+||p||²+2⟨eS,p⟩`。两来源 FFN 与真实 native writeback 的 `recomposition_SSE` 全部为 0，标量恒等式 gap 相对 total SSE 均小于 `1.3e−15`（最大绝对 gap `1.79e−7`）。这是四臂向量减法与 FP64 归约的闭合性，不能单独证明被测算子与某个理想连续函数等价。

FFN 中 `prefix_response_SSE > student_operator_SSE` 的计数 source0 为 157/160、source12 为 153/160；真实 writeback 分别 158/160、158/160。两来源的 FFN operator–prefix 点积分别在 157/160、154/160 个 site 为负。因此只有相加 SSE 而省略 signed cross 会高估总差。

这仍然不是“prefix 是主要起因、operator 无关”的识别：prefix 状态本来就由更早的 operator 替换与网络计算共同生成；eS 和 p 还共享 D(qS) 这个中间参考，负点积本身不证明一种独立的纠错机制。跨 site 的 SSE 不能直接累加成最终 latent SSE。

源码额外限定：当前普通 I2V 配置中，controller 每 step 显式校验两轨 `embed0` 相等；同 block 的 `c_gate_msa` 来自共同 modulation 与 embed0 的 `chunk(6)`。所以这个成功运行中的同 step/block gate 相同。通用 TRACE warning 中“可能含 gate changes”不是本次已观测到 gate 改变。真实 writeback prefix 仍包含不同 residual base、dense FFN 输入响应及 BF16 运算，而非纯 FFN 输入偏移。

32 行 `numerical_check` 比较的是 native candidate 与**同一套已部署 BF16 系数升为 FP32、去掉 Q2/中间 BF16 cast**的连续表达式；它不是原始未 cast 拟合系数。其覆盖和参考函数与 full-M D/S SSE 不同，且没有同 32 行 D/S SSE 及所需 signed cross；因此不能把两者直接比较后宣称 H-NUM 占主导，也不能把 Q2 的设计差异自动叫实现 bug。

## 5. 四步 flow 的符号与 BF16 边界

原 scheduler 实际先做 FP32 的 `z−sigma*v`，前三步再加 `sigma_next*v`，最后 cast 回输入 latent dtype。记录证实两轨每步 step_pre 后及 step_post 后 latent 都是 **BF16**，flow 为 FP32；此前独立旧文“外层一直 FP32”的表述在 v2 勘误后，本次得到实际 dtype 证据的进一步否定。

定义 step_pre 后 `δ=zS−zT`，实际整网 flow 差 `d=vS(zS)−vD(zT)`，`h=sigma−sigma_next>0`。理想实数关系与真实有限算术残差为

\[
\delta_{next}=\delta-hd+\eta,
\]
\[
\|\delta_{next}\|^2=\|\delta\|^2+h^2\|d\|^2-2h\langle\delta,d\rangle
+2\langle\delta-hd,\eta\rangle+\|\eta\|^2.
\]

| source | step | h | `cos(δ,d)` | `h²||d||²` | `−2h⟨δ,d⟩` | 实际 next latent SSE | `||η||²` | 实际 SSE − ideal SSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.06250000 | 未定义，δ=0 | 1721.89 | 0 | 1731.58 | 9.895 | 9.698 |
| 0 | 1 | 0.10416669 | −0.870257 | 4517.26 | 4867.85 | 11125.33 | 7.747 | 8.633 |
| 0 | 2 | 0.20833331 | −0.905023 | 18968.87 | 26294.60 | 56392.66 | 5.221 | 3.862 |
| 0 | 3 | 0.62500000 | −0.941810 | 198723.13 | 199401.58 | 454530.65 | 7.059 | 13.275 |
| 12 | 0 | 0.06250000 | 未定义，δ=0 | 464.57 | 0 | 474.42 | 9.834 | 9.851 |
| 12 | 1 | 0.10416669 | −0.790130 | 1477.73 | 1323.15 | 3283.15 | 7.707 | 7.852 |
| 12 | 2 | 0.20833331 | −0.800810 | 6558.88 | 7432.23 | 17278.61 | 5.088 | 4.349 |
| 12 | 3 | 0.62500000 | −0.884869 | 62163.48 | 58000.43 | 137447.95 | 6.331 | 5.435 |

理想 signed 恒等式按 JSON 原值重算最大绝对差 `5.83e−11`。末步从 sigma=.625 直接走到 0，h=.625；没有未来额外一步替它消化残差。后 3 步的 `⟨δ,d⟩<0` 意味着实际增量 `−hd` 与已有偏差 δ 同向，扩大距离，不是 toward-teacher 纠错。末步两个增长项大小接近，不能仅用“大 h”忽略符号方向，也不能假设 J≈0 来解释它。

两来源初始两轨 FP32→BF16 cast 各有约 5.585/5.589 的单轨 SSE，但两轨 cast 相同，差分 cast SSE 为 0；后三次 step_pre 本来已是 BF16，实际 cast 差均为 0。`η` 包括本次 step_post 的运算顺序与输出 cast，`||η||²` 不是实际能量改变量，后一项还含与 ideal 差的 cross。表中最终更新的直接 defect 很小，相对于本步几十万/十几万的轨迹差不能解释其主要即时增长；这不排除之前网络内部 BF16/QDQ 误差的积累，也不是所有数值机制已排除。

理想要求下一状态严格贴回 teacher 的目标仍为 `vS(zS)=vD(zT)+δ/h`，符号是正的。teacher 状态及其 flow 不是学生部署时免费的在线纠错信息，而且在 native BF16 下该式只是理想目标，不保证 bitwise 下一状态相等。

## 6. 一个无不稳定 teacher 也能产生该现象的反例

取标量 flow `vD(z)=c>0`、`vS(z)=c+bz`，b>0，两个轨迹从同一正 z0 开始，以实际正步长 h 更新。选择 `0<h_k b<1`，并令 z0 足够大，使有限步内两轨仍为正。则第一步后 `0<zS<zT`，每一步的同输入近似误差分别为 `b²zS²<b²zT²`；然而 `δ=zS−zT<0`，`d=bzS>0`，所以 `δd<0` 且 `δnext=δ−h b zS` 继续远离零。teacher flow 导数为 0、teacher update 导数为 1；student update 导数在 (0,1)，无需不稳定或膨胀的 teacher Jacobian，现象已经成立。

这是解析反例，没有运行模拟。它证明“更小的 student-input 局部 SSE + outward flow”本身不能识别 dense suffix/flow 的不稳定传播。它不是 Wan 的拟合模型，也不声称实际收缩只有幅度原因。对于方向变化，一个各向异性误差算子 B 也可以让输入移向其近零方向、局部 SSE 下降，同时丢失任务相关方向；保留大的 common residual 又能维持接近原值的 block 总能量。现有数据与“学生更易表示但功能更差”机制相容；功能/语义损失仍是待检验解释。

相竞争的机制是：student 输入仍有足够功能信息，但 dense 模型对其状态偏移的有限响应已经沿现有 δ 的 outward 方向。两者可共存；当前每 FFN 的四臂把**局部函数**分开了，却没有把**整个 flow 网络**的模型改变和 latent 输入改变分开。

## 7. 最小下一判别：在实际 student latent 上求整个 dense flow

未来若另行授权，在一个预先冻结且 δ≠0 的 step（例如 step1），保持相同条件、time/expert 和实际配对前缀，额外得到整个原始 dense 网络的 `vD(zS)`。现有 `vS(zS)` 与 `vD(zT)` 于是可拆成

\[
d=\underbrace{vS(zS)-vD(zS)}_{c:\text{相同 student latent 上的模型差}}
+\underbrace{vD(zS)-vD(zT)}_{s:\text{dense 网络对 latent 输入偏移的有限响应}}.
\]

保存两个 `⟨δ,c⟩`、`⟨δ,s⟩`、两范数和 `⟨c,s⟩`，并比较理想 `||δ−hs||²` 与 `||δ−h(c+s)||²`。这只需要一项额外整网 dense flow，不需要新 VAE 解码或完整视频；当前逐 FFN shadow 每次都被丢弃，既没有把 student 前缀改回 dense，也不等于这一 whole-model 反事实。当前 TRACE 标量无法事后生成该缺失输出。

不同预测：若 `⟨δ,c⟩` 主要为负，而 `⟨δ,s⟩` 近零或为正，则即时 outward 方向更多来自在同一 student latent 上的整网替换差；若反过来，dense 对状态偏移的有限响应才是该 step 的主要 outward 项；若两者都大且为负，则需要保留交互解释。比较不能省 cross，理想更新比较也不能冒充未执行的 native 混合干预。即使补齐这项，仍只识别一个 step 的有限机制，不会自动识别最初哪层产生了偏差或最终视频损失的主因。

Graph 锚点为实际输入 `F03`，局部算子 `F04/F06/F08/F09`，native gate/residual `F12`，后续块 `F13/G21`，以及 head/flow→scheduler→下一 latent 的 `G0` 路径；`P23/P24` 的这两来源全深度 signed 标量现已有观测，但“哪项导致质量下降”的边保持 OPEN。`D09/P25` 的完整视频与 metric 账本不把以上机制推论自动提升为质量接受。它们都是付过两主路径、同输入 shadow 和归约成本的后验诊断，不能作为免费的 pre-W1 观察或候选真实部署时延。

本次检索范围按委派限制为冻结协议、原始 JSON 与上述原始实现；没有新增文献检索，也不把先前 CTEC/TCEC 讨论当作当前 Wan 机制的实证。新增的区分来自亲自重建的四臂/调度公式、两来源标量恒等式和上述解析反例。没有预选方法、训练或后续执行承诺。
