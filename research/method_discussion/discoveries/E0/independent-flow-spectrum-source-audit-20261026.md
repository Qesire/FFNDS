# DISCOVERY · 原生整步差的有符号 Fourier 子空间审阅

**研究对象是同一组原生整步状态差在固定正交时空子空间中的能量和相消，而不只是全局 SSE。** 若少量低频实坐标承载大部分差异，原先全局缩放/常数偏置所遗漏的结构便有具体描述；竞争解释是大部分差异仍分散于许多坐标，或低频集中只是周期延拓/平滑场的常见性质。最便宜的判别是读取已有八份状态包，在完全相同子空间里分别报告 dense 响应、候选注入及有符号交叉项。

本审阅只读最终 analyzer、三项 CPU fixture、协议和原生结果 JSON；未运行 fixtures、保存张量分析、模型或 GPU。执行及实际结果由 root 另行冻结/登记/记录，本文件是源与数学审阅。

## EVIDENCE：原生对象、FFT 公式与维数

1. 接线使用真实保存的 `z_dense_after`、`z_student_after`、`z_dense_at_student_after`，在 CPU FP64 作差：`P=dc-zd`、`N=zs-dc`。因此 `P+N=zs-zd` 是原生整步后状态差；本程序名含 flow，但实际没有把 flow 速度当 latent 差。原始数值为 BF16 状态，提升 FP64 后计算这些有限差。
2. `torch.fft.fftn(...,dim=(-3,-2,-1),norm='ortho')` 在21×58×104 latent 网格上使用单位正交 Fourier 变换，16个 channel 各自变换。设系数为 `A,B`，则每个频点使用 `|A|²,|B|²,2 Re(conj(A)B)`，故每个相同 mask 下 `E_next=E_P+E_N+cross`；cross 的符号得到保留。
3. `abs(fftfreq)` 构造的三轴矩形 mask 对共轭频点对称，保留实数场的正交子空间。代码使用完整 complex FFT，未套用 rFFT 的倍2规则。每一对不同共轭频点对应两个实维，DC/Nyquist自共轭点各一维，所以 `mask.sum()*16` 正好是实子空间维数。
4. 当前固定 cuts 在真实 shape 上给出 `d=16(2k_t+1)(2k_s+1)²`。例如DC空间维数16；最大 `k_t=10,k_s=8` 为97,104维，占2,026,752总实维约4.79%。时间cut10覆盖全部21个 temporal modes，空间cut8没有覆盖全部空间频率。其它 shape 的 Nyquist 情形由实际 mask计数处理，而非套用此简式。
5. 全谱分别核验 P、N 的 Parseval 及 signed cross 恒等式；容差是 `1e-10*max(|direct quantity|,1)`。每个面板保存独立的 signed recomposition error；完整 `P+N` SSE另与既有原生结果以绝对 `1e-8`核对。源代码不会把这些误差校验当视频质量测量。

## 最终接线与已闭合项

- `spectral_panels`现在要求两个输入均为 CPU、等 shape、rank4、实数且有限；meta-device与complex拒绝fixture对应此前遗漏的第二输入检查。当前 main 用 `load_file(device='cpu')`，没有任何 GPU tensor 构造。
- 协议必须同时绑定 analyzer、tests 和实际 native-result JSON SHA；原 native-result 必须是已完成状态。来源/seed严格为 `(0,42),(12,54)`，每个步骤严格为 `[0,1,2,3]`，结束时强制八份输出；各状态文件另按原 `tensor_sha256`核验。
- 数值执行仍要求精确协议 SHA 在 ACTIVE 中唯一登记且 `AUTHORIZED_NOT_STARTED`。固定输出目录 `exist_ok=False`，异常结果保存为 `FAILED_PRESERVED_NO_RETRY`，输出上限262,144字节。源文件没有新增模型调用、拟合或可部署校正 bank。
- 阅读的三项 fixture 覆盖：非CPU/complex拒绝；全谱负cross与Parseval/24维计数；DC与单一空间余弦的分离。这里记录其覆盖意图，不冒称本审阅独立运行或验证了实际八份数据。

## DISCOVERY 边界与竞争预测

固定20个面板是**嵌套的累计矩形子空间**，不同面板之间重叠，不能相加。每个面板内部的 Fourier 投影正交；这并不使不同累计面板彼此正交。需要互斥band时应明确取差集，当前协议没有输出此类新选择。

频率单位是整个latent轴上的cycles，不是每视频帧频率、径向空间频率或语义运动。21个latent-time样本不是81帧视频；58×104两个空间轴的同一cut也对应不同轴长。周期延拓连接时间及空间的两端，端点不连续会产生频谱能量；该表示的seam影响应随结果一起解释。

**可区分的预测：** 若低维mask中的能量占比显著超过其维数占比，可描述本组有限状态差的该基集中；若P/N在同一mask内有不同signed cross，说明全局交叉项掩盖了频带差别。它们都没有证明在线能获得修正系数、该基可压缩真实FFN、可改善视频，或局部高频等于坏细节。输入各向异性/平滑先验也可产生集中，维数占比只是一个参考，不能作为完整统计零假设。

对 FFN compute 的意义仍是 conditional：该描述有助于询问哪些输出方向需要保留，当前代码本身仅增加离线计算。已完成P23/P24原生整步证据是本任务的输入；`F09→F12/F13→P24`为posthoc描述，通向新FFN执行和完整视频收益的边保持OPEN。源与公式审阅在以下字节边界内未发现阻断数学缺陷。

## 源绑定

| 实际读取对象 | SHA256 |
| --- | --- |
| `tools/analyze_e0_flow_spectrum.py` | `0ea6ad4ec586275913c5951d6866444d5ef6bd4e7fe8e94180f70713282d9171` |
| `tests/test_e0_flow_spectrum.py` | `984ca6020a2d058b4cb0dc837d44b9ae2f9bc21def4272266e994eb5a548f0c1` |
| `config/e0_saved_flow_spectrum_20261026.json` | `48503cd33237136d4db6358b8e93fd6841a631d67a25e4f6283a8550ecfbf109` |
| `research/20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.json` | `d4c1db3a644e7972bdc43e551da2ea59e28508b994730f760396f480a6c99f3a` |

此为本地源/标准正交变换代数审阅，没有新增论文阅读或实证结果；没有改动 analyzer、protocol、ACTIVE、graph、atlas 或已冻结报告。
