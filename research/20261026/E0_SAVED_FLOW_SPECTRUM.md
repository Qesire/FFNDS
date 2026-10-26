# 原生整步状态差的有符号频谱：继承、增益与频带相消

**新结构是：小部分空间低频坐标承载大部分累计状态差，但当前候选注入更分散；同一交叉项在低频带与全域甚至可反号。** 末步97,104维、占全部2,026,752实维4.79%的子空间，保留来源0/12状态差的69.99%/53.47%，dense响应的69.66%/58.23%，候选注入仅27.17%/20.59%。竞争解释是先前偏移已集中于这些坐标，而非dense普遍优先放大低频。核对原生跨步连续性后，可用现存标量比较输入/输出频带能量；该比较显示明显的来源差异。

## EVIDENCE：同一个原生有限状态对的正交分解

令 `P_t=TD_t(zS_t)−TD_t(zD_t)`、`N_t=TS_t(zS_t)−TD_t(zS_t)`，均取保存的真实 native step_post状态，于是 `delta_(t+1)=P_t+N_t`。对固定正交Fourier投影 `Π_B`，

\[
E_B(delta_{t+1})=E_B(P_t)+E_B(N_t)+2\langle\Pi_BP_t,\Pi_BN_t\rangle.
\]

程序对16通道、21×58×104 latent网格作单位正交3D FFT，固定20个累计矩形mask。下表只显示预先固定的最大mask `|k_t|≤10,|k_y|,|k_x|≤8`，含全部时间频率及部分空间频率。每列百分比都除以**该列对应量的全域能量**，不是误差因果百分比。

| 来源/seed | step | 下一状态差在B中 | P在B中 | N在B中 | B内signed cross | 全域signed cross |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0/42 | 0 | 77.37% | 未定义 | 77.37% | 0 | 0 |
| 0/42 | 1 | 77.00% | 74.77% | 53.28% | +300.688 | −220.532 |
| 0/42 | 2 | 74.82% | 73.49% | 37.77% | +386.091 | −1914.644 |
| 0/42 | 3 | 69.99% | 69.66% | 27.17% | +9579.900 | −227.407 |
| 12/54 | 0 | 57.66% | 未定义 | 57.66% | 0 | 0 |
| 12/54 | 1 | 57.49% | 70.99% | 39.47% | −111.416 | +119.694 |
| 12/54 | 2 | 57.10% | 66.44% | 28.00% | +1187.698 | +2629.576 |
| 12/54 | 3 | 53.47% | 58.23% | 20.59% | +2911.843 | +4938.870 |

来源0后三步在全域有相消，在B内却均增强；补空间的负交叉项抵消了B的正交叉项。来源12 step1反向：B内相消，全域增强。因而“全局signed抵消”不能替代各功能子空间的方向描述。`N`在B中的份额逐步降低，也不能单凭其全能量判断它对集中累计偏移的作用。

## DISCOVERY：低频继承与有限频带增益可以区分

本报告读取**实际原生RUN/TRACE**，核其SHA与已验证[whole-flow结果](E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md)一致。两来源、teacher/student两分支、step1–3的12条边界链全部满足：

`previous.after_step_post == current.before_step_pre == current.after_step_pre == counterfactual.input_hash`。

每个后续入口的teacher/student cast SSE均为0，BF16 dtype/shape/bytes参与哈希。原controller在step_pre前后克隆并记录状态；whole-flow保存入口取实际step_pre后状态，原CPU分析器已将状态包中的before/after身份与TRACE边界逐一核对。当前controller、whole-flow及trace consumer源码SHA也匹配原native冻结协议。本报告没有重复读取张量作数值复核。

因此在本范围，`delta_t`的B能量就是上一step的`next_B`能量，可以派生

\[
g_B(t)=\frac{E_B(P_t)}{E_B(delta_t)},\qquad
g_{B^\perp}(t)=\frac{E(P_t)-E_B(P_t)}{E(delta_t)-E_B(delta_t)}.
\]

| 来源 | step | B内能量比 g_B | 补空间能量比 | 全域能量比 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1 | 5.767 | 6.654 | 5.968 |
| 0 | 2 | 4.752 | 5.738 | 4.978 |
| 0 | 3 | 7.187 | 9.304 | 7.720 |
| 12 | 1 | 6.178 | 3.438 | 5.018 |
| 12 | 2 | 4.190 | 2.862 | 3.626 |
| 12 | 3 | 6.790 | 6.482 | 6.658 |

这组固定有限偏移中，来源0的B份额从输入到dense响应逐步降低，来源12则提高；大低频能量并不支持一个跨来源通用的“dense优先放大低频”解释。两来源step0已经有77.37%/57.66%的初始注入落在B中，继承本身是不可忽略的竞争解释。

这些比值是完整输入偏移下的**输出/输入频带能量比**。非线性网络可跨频带搬运能量，当前计算没有单独注入`Π_B delta`；因此它们不是B上的算子范数、Jacobian、因果隔离增益或能相乘预测新轨迹的传递系数。验证低频输入本身是否导致优先响应，需要另设计原生隔离干预；当前记录已经足以否定把“频谱集中”直接等同于该机制。

## EXECUTION与表示边界

root登记的唯一CPU执行PASS：八份保存状态、160个面板，循环阶段2.658355秒，4个CPU线程；0新模型/GPU/拟合，`GPU_initialized=false`，`quality_accepted=false`。全谱P/N/cross最大绝对Parseval误差`1.16e-10`，各mask重组误差最大`4.91e-11`。公开JSON为原RUN的71,694字节精确复制；本轮只读JSON/源、派生上述标量比与检查隐私，没有FFT重跑。

20个mask是**嵌套累计面板，不可相加**；本表的B与其补空间才构成一个正交分割。频率单位是整条latent轴的cycles，矩形各轴cut不是径向频率；21个latent-time位置不是81帧视频。周期延拓的时间/空间端点seam属于该表示，频谱不能直接标成语义运动、背景漂移或细节质量。各向异性和平滑场也能产生低频集中，4.79%维数占比仅是参考，并非充分统计零假设。

最便宜的后续研究可复用既有20个mask标量，检查上述来源差异/符号反转在固定cuts间是否稳定，先明确功能子空间再选择新表示。在线获得teacher目标、校正系数、真正减少W1/W2以及完整视频收益均未测。对FFN的启发是同时描述继承偏移、dense有限响应和当前注入的必要输出方向，而非自动选择低通修复。图锚点`F09→F12/F13→P24`为posthoc已存状态证据，通向新算子和质量的边保持conditional/OPEN。

## 字节绑定与独立审阅范围

| 对象 | SHA256 |
| --- | --- |
| 原RUN / [公开精确JSON](E0_SAVED_FLOW_SPECTRUM.json) | `e7ff5a48a1084b1588ea593476c8d88b2dc1e058ac9a984c943894f605414120` |
| `config/e0_saved_flow_spectrum_20261026.json` | `48503cd33237136d4db6358b8e93fd6841a631d67a25e4f6283a8550ecfbf109` |
| `tools/analyze_e0_flow_spectrum.py` | `0ea6ad4ec586275913c5951d6866444d5ef6bd4e7fe8e94180f70713282d9171` |
| 原whole-flow结果JSON | `d4c1db3a644e7972bdc43e551da2ea59e28508b994730f760396f480a6c99f3a` |
| 来源0 native RUN / TRACE | `3e915a01255ddd9a1fc5188a98cc9d1be20b4465da5fb5acb4ce9edb68fb4ae5` / `791f08648063e91392487f7b77a9f83590bddd80061be4a9aba20282352b0dc8` |
| 来源12 native RUN / TRACE | `dd2e346a9ba1684fd74f7946bf94595a2cb830f0554f277e1dd480f688f2830d` / `448b26276bf6cec6c49aa18968503900cfae46910d7d438ac9a4881be6b417e6` |

连续性源：`e0_paired_wan_controller.py`=`bd99bdba0651dc59fcc5a6cdf72922906a5c26eb08dfb4d4650f1d6cefa560bd`；`e0_whole_flow_counterfactual.py`=`0592cfe2a3b92b64b36ecc6e12b99ca653d5939f20305ae02b5ce35d7115306b`；`e0_full_depth_trace_contract.py`=`28154f7d340cf164f07bbb908e31b37f04414b3fb7ef4f9f03e45fbc377ba98e`。源码数学审阅另存[独立记录](../method_discussion/discoveries/E0/independent-flow-spectrum-source-audit-20261026.md)。发布前对新MD/JSON只作私有标识文本扫描；没有修改原RUN、冻结协议、ACTIVE、atlas或共享graph。
