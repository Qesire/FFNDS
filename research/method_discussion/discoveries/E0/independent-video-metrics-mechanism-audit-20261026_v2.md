# 独立审查：全帧指标能描述失败，但不能把模糊、运动缺失与传播机制自动分开

作者：独立讨论 Agent `/root/independent_e0_audit`。日期：2026-10-26。状态：源码、原始文献与旧保存标量的独立审查；不是方法 endorsement、质量验收或新增实验授权。保留所有先前独立记录。

## 1. 先明确被解释的离线算子

这里的对象是离线固定校准的 Wan `affine54`，不是实时/流式算法。源码 `tools/e0_reconstructed_runtime.py` 在原 FFN 调用点保留调用者的 norm/modulation、tanh-GELU、gate 和 residual 逻辑，替换 `.ffn.0/.ffn.2` 的矩阵求值。令 `q` 为原 Smooth1/Q1 后的 BF16 输入，保留 54×64 个 hidden 坐标，再以新读出和一个输入仿射支路重建输出，概念式为

`fhat(q) = retained_native_Q2(tanh_GELU(q W1_J + b1_J)) V + q A + b`。

实际 Q2 之前有保存的 Smooth2 缩放，矩阵/偏置是新 BF16 参数，支路相加也有有限精度；上式只是显示计算组成，不能把它当连续精确恒等式或原 HiF4 权重不变。`whole94` 保留 94×64 hidden 坐标，用拟合读出重建。原 dense 和候选自由生成已经覆盖所有 40 层×4 步，但全 160 位替换不是全 160 位同输入 native 数值验证。

因此视频结果必须分开四类未知：(a) 部署 bank/partial-native 在同输入上的数值或打包误差；(b) student 前缀改变 block 输入；(c) signed 误差在后续 block/gate/residual/head 的传递；(d) 四步 scheduler 的 flow-state 偏离。终端像素/运动指标是这些因素及解码/编码共同作用后的消费者，单独不能识别哪一种主导。已有 all40 保存面板的局部误差不是 student 轨迹的替代。

图锚点为 `F12→F13/G21→P22→D09`，本次补评属于 `P25` 的 **posthoc** 消费者观察；`P23/P24` 全深度 paired-native/有符号传递仍是 **OPEN/conditional**，不能因指标报告完整就补画成已观测因果边。它不提供 W1 前信息，不减少原 W1/W2 计算，也不授予新模型调用。

## 2. 审查身份与范围

实际阅读了根/讨论 AGENTS、ACTIVE 当前 E0 阶段、Graph↔Agent 协议、registry、最新六视频报告及本 Agent 先前轨迹审查。使用 pinned AutoResearcher/WARD（manifest commit `8b88a52d894568dc303db727decfbf13b5fabf3b`）。本次没有运行 evaluator、视频解码、光流、张量计算、科学拟合、模型/GPU/SSH/Git；只做只读文本/JSON 检索、普通 Python 标量汇总及文献/官方源码阅读。

实际读到的版本：

| 文件 | SHA256 |
| --- | --- |
| `tools/ffnds_video_metrics.py`，uint8-flow + finite-motion-diagnostics 版 | `f1dbfa33ce9f0329c7a09c2bd2617f241febf98d91d896ae17364c457ebddc71` |
| `tools/ffnds_video_metric_gate.py` | `3f1b70fa25b7c961f8141ba396ee3b9507e028d60fd168e642e0548de99a2d07` |
| `research/agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md` | `b629b79970685f7d53395770f879ebc59e974a4145e1b0eb00c28ab106bd1ab8` |
| `research/agent_protocol/GRAPH_AGENT_BINDINGS.json` | `53fba11b861d110344d78e79b49f8950cc9ebb843e2b72f12b9907df383f0ac5` |
| 旧 `research/20261026/E0_RECONSTRUCTED_VIDEOS.json` | `f01554b974741191b39faffd31fc23badc7a395de87a621a95d98674a621e7c8` |

协调者在本次审查期间修正工具并准备独立后验补评。下面区分亲见的早期归一灰度调用、已修正的源码与尚未收到的新结果，不把将来的数据写成已经审阅。

## 3. 一个决定性的源码问题：Farneback 不是对任意强度尺度不变的实现

初读版本把 OpenCV gray float32 `[0,1]` 直接交给 Farneback。官方 API 指定 8-bit 单通道；C++ CPU 路径虽然接受 float，并不会自动恢复其强度尺度。官方实现中的多项式系数随输入线性缩放，normal-matrix 和右端为二次，更新使用固定 `det(G)+1e-3`。依据这些操作，我独立推得：在固定 warp/采样位置处，若图像强度乘 `c`，某次更新由 `N/(D+eps)` 变成 `c^4 N/(c^4 D+eps)`。输入除255时，固定正则相对放大 `255^4`；这足以打破精确算术理想公式中的强度尺度抵消。完整多尺度迭代还会改变采样位置，故不是所有像素都有同一个已知衰减倍率。[OpenCV 官方 CPU 实现](https://raw.githubusercontent.com/opencv/opencv/4.13.0/modules/video/src/optflowgf.cpp)

这不是凭参数文档猜测“float 不合法”：代码能执行与物理量正确是两回事。初版接近零的 flow 不能被当作候选或 dense 冻结证据。已向协调者报告；当前 SHA 对应的 `downsampled_farneback` 已把 gray×255 四舍五入为 uint8 再 resize，修正了这一具体尺度问题。[OpenCV Farneback 参数定义](https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html#gae91f303e2f96e5e4f7b3ce28500c82e5)

原论文从局部二次多项式平移关系得到 displacement，再对邻域约束作最小二乘，见 Eq.7–13；存在唯一解需要可辨识的局部结构。论文的 confidence/残差讨论和 aperture 条件并不允许将“估计为零”等同真实零运动。这也是为何尺度修复后还需要保留纹理退化这一竞争解释。[Farnebäck 2003 原论文，§3.1–3.2](https://www.diva-portal.org/smash/get/diva2:273847/FULLTEXT01.pdf)

已知纹理的固定平移软件测试可以检验单位、方向、量级与边界处理；它检验的是 evaluator 实现，不是 Wan 运动质量。应记录实际通过版本，不能仅凭上述静态推导宣布测试通过。本 Agent 未执行此测试。

## 4. 修正后 motion diagnostics 的几何正确性与强反例

读到的 `motion_diagnostics` 采用 forward `u: previous→current`，在 `x+u(x)` 采样 backward `v` 和 current luma；因此 `u+v(x+u)` 的一致性方向及 `current(x+u)-previous(x)` 的重建方向正确。端点越界被排除，一致性阈值为 `0.5+0.01||u||`，单位是最长边320的下采样像素。这是本仓库定义的工程阈值，不是已经验证的真值 flow 精度。

然而，“一致”不是“正确”。在无纹理或严重模糊区，两个方向同时给零，forward/backward 检查可以完全通过；恒色区域的补偿残差也为零，仍无法知道它属于移动物体还是静态背景。water/highlight、遮挡、重复纹理及解码噪声也会违背或削弱对应关系。有效面积和各类失败/无有效像素状态应与残差共同报告，不能只比较条件均值。

**最强与 affine54 当前外观直接相关的反例：** 设 dense 的图像只是平移，`R_t(x)=R_0(x-u t)`；候选为同一个运动的空间模糊 `C_t=B*R_t`。因卷积和平移交换，候选的真实位移仍然精确是 `u`。但边缘能量、相邻帧差可以下降，光流估计也会因纹理/梯度损失更保守。故“低 flow + 低 edge + 低帧差”既兼容真实运动受抑，也兼容原运动未变而可观测纹理退化。它们不是三个独立的冻结证据。

另一个反例是冻结主体叠加移动纹理：纹理提供可追踪平移，flow 可以大、几何一致且补偿残差小，主体动作仍然缺失。反过来，静态画面加随帧曝光偏移可以使原始帧差与补偿残差升高，但不包含几何运动。没有语义主体对应或实看片段，不能把升高的 flow/帧差统称“恢复动作”，也不能把它们统称纹理噪声。

当前 paired flow L2 是同一 raster 位置两套独立估计的差；它没有建立 candidate/dense 物体间的对应。两套各自 motion-compensated 残差之差同样以共同 raster 起点比较，而不是先跨视频对齐语义区域。候选改变构图/物体位置后，比较含义会改变；这不使指标无用，但限制了“相同背景/物体的闪烁”这一解释。

`static_region` 的原指标仍是 **uncompensated low-reference-flow luma-delta error**：dense `||u||<=0.25` 的 mask，未在该标量内应用相机运动补偿、遮挡排除或语义背景分割。新版另存 mask 面积及与一致性 mask 的交集面积，是有用信息，但没有将原低运动区指标转成语义背景指标。blur 影响 dense/candidate flow 的程度也不相同。`0.25` 下采样像素约相当于本轮832宽图的0.65原图像素，不是所有分辨率通用的物理阈值。

## 5. 颜色、锐度与时间差分别测什么

ITU-R BT.709-6 的 nonlinear primary signal 权重是 `.2126/.7152/.0722`，并另定义 OETF。当前工具对已解码 gamma-encoded RGB 加权，能得到明确可重复的 **BT.709-weighted encoded-luma proxy**。它未逆 OETF、未校验实际编码 primaries/range/transfer metadata，不是场景线性光度，也不保证等于原压缩码流的 Y 平面。把其 signed mean 解释为“编码亮度偏移”正确，解释为物理曝光或原模型 illumination 通道的直接变化则越界。[ITU-R BT.709-6，Table1 item1.2、Table3 item3.2](https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.709-6-201506-I!!PDF-E.pdf)

工具的 flow/Laplacian gray 来自 OpenCV `RGB2GRAY`，权重是 `.299/.587/.114`，与上面的 luma proxy 不同。两个定义并存可以，但不能把 gray flow 称为 BT.709 luma flow。[OpenCV 原始颜色转换定义](https://docs.opencv.org/4.13.0/de/d25/imgproc_color_conversions.html)

`mean(abs(Laplacian(gray)))` 是 L1 二阶边缘能量，不是 variance-of-Laplacian，更不是感知锐度。增益/对比度、噪声、重影的额外边界、振铃均可改变它。原始 focus 文献在受控 focus-stack/局部窗口里评价 Laplacian 等算子，特别比较噪声、对比度与窗口的影响，不能直接把整帧 mean 能量比当成重建细节保真度。[Pertuz/Puig/Garcia 2013，§2.1、§4.2.2](https://isp-utb.github.io/seminario/papers/Pattern_Recognition_Pertuz_2013.pdf)

我给出的明确数学反例是内部像素的棋盘噪声 `n(i,j)=a(-1)^(i+j)`：5点离散 Laplacian 的响应为 `-8n`。在灰平面上添加这种不真实的高频可显著“提高边缘能量”，却不恢复任何真实细节。固定噪声还可以完全不增加帧差；逐帧噪声会增加帧差。空间和时间两个指标必须联合解释。[OpenCV Laplacian 官方定义/降噪前处理示例](https://docs.opencv.org/4.13.0/d5/db5/tutorial_laplace_operator.html)

令配对像素误差 `E_t=C_t-R_t`，当前 paired temporal RMSE 测 `||E_t-E_{t-1}||`。恒定颜色/结构偏差 `E_t=B` 对这一指标不可见；低 temporal error 不意味着 low spatial error。完全冻结的 `C_t=R_0` 会让自身 adjacent difference 为0，同时 paired temporal error 保留 dense 的变化。帧差无法区分该变化来自物体动作、相机、曝光还是纹理。

这也说明不能把该指标改名官方 VBench temporal flickering。VBench 原仓库的 flickering 函数本身使用相邻帧 MAE 并明确要求静态视频；原论文/补充将其放在特定静态场景筛选与独立动态维度的框架中。它不是“任何视频帧差越低质量越好”的普适结论，复制其简单公式也不等于完成官方 I2V 评价。[VBench 官方实现](https://raw.githubusercontent.com/Vchitect/VBench/master/vbench/temporal_flickering.py)、[CVPR2024 原论文补充的静态筛选与公式](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Huang_VBench_Comprehensive_Benchmark_CVPR_2024_supplemental.pdf)

## 6. 亲自重算的旧六视频标量：实际已有相反运动表征

只从冻结旧 JSON 的逐帧标量计算，每候选81帧、80相邻差；source0/12 的真实 seeds 分别42/54。下表 RMSE 是逐帧/逐差先开方再均值，不能与全像素 pooled RMSE 互换。

| source | 候选 | mean frame RGB RMSE | mean paired temporal RMSE | 自身 adjacent RGB MAE | dense adjacent RGB MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | whole94 | 0.1578974 | 0.0575771 | 0.0146360 | 0.0235361 |
| 0 | affine54 | 0.1541602 | 0.0564177 | 0.0086696 | 0.0235361 |
| 12 | whole94 | 0.1616454 | 0.0507843 | 0.0185399 | 0.0201946 |
| 12 | affine54 | 0.1305286 | 0.0651147 | 0.0295449 | 0.0201946 |

**可支持：** affine54 在 source12 的 mean spatial RMSE 比 whole94 低，而 temporal discrepancy 更高；affine54 自身帧差 source0 下降、source12 升高。这是一个真实的 source×operator 交互和指标排序反转，不能用“所有候选都冻结”或“affine54 各维度都更好”概括。

编码 luma signed mean：source0 whole/affine 为 `−0.01158393/+0.00667025`；source12 whole/affine 为 `+0.00027715/+0.01519584`。source12 affine 的81帧均为正，范围 `+0.01162134..+0.03788992`。这支持该 decoded-luma 定义下持续变亮，不支持自动称为背景闪烁、线性曝光或某层偏置错误。正均值可来自占据面积/构图变化以及其他局部变化。

旧结果缺真正 flow/warp/新一致性指标，仍保持 legacy partial-quality 身份。这里不借新工具的存在为旧收据补 PASS。先前实际查看过同 SHA 的两个五帧 contact sheets，仅支持“所抽帧 whole94 重影/纹理损坏，affine54 相对连贯但模糊”的有限观察；没有看完整视频，不把接触图或本次标量汇总记作 full-video human review。

## 7. Gate 真正闭合的证据与没有闭合的证据

已读 SHA 版本的 gate 确实逐条核实际 candidate MP4、metric JSON、RUN 文件 SHA；paired report 还核 dense MP4/RUN 的 SHA，检查 source/seed/prompt 匹配、预期帧数、连续索引及部分逐帧字段。输出默认外部质量 NOT_RUN、`quality_accepted=false`，这是正确的证据/验收分离。

但这些边界需要保留，不能把 gate PASS 解释成“政策所有维度真实完成”：

1. `validate` 对多数 metric summary/逐帧数值主要检查 key 存在，未递归检查有限值/数值类型/summary n，也未从逐帧值重建 summary。新增 `motion_diagnostics`、FB/补偿残差/有效面积尚未成为 mandatory family/逐帧字段。合法 evaluator 自身的 `summary` 会拒绝非有限值，不代表对外部传入 JSON 的 audit 同样严格。
2. gate 核 RUN 的字节身份，但不解析其执行内容。因此 source/seed/model/variant/cost 与 RUN 的语义关系仍依赖生成 report/bundle 的代码或人工复核；hash 证明绑定文件，不能证明文件内部宣称来自这个视频。`expected_video_count` 也是 bundle 自报，漏登记一个实际生成 clip 不会自动被发现。
3. `external_quality` 为 PASS 时未要求官方结果/人工记录的文件、版本、评审者或 SHA；`--require-acceptance` 目前证明状态字段满足规则，不能取代真实 evidence attachment。不可仅改 PASS 布尔值完成质量验收。
4. 政策还要求结构/感知维度、连续窗口与背景/运动解释。当前核心实现提供边缘 proxy、全帧/全差分布与几何 flow 诊断，未自动提供语义背景、感知距离、官方 VBench、人类完整观看、连续窗口摘要或 causally controlled 160-site trace。原始 per-frame 数据允许后续预先定义窗口汇总，但尚未计算不能写成已有结果。

这些是具体实现与证据边界，不是反对先归档有 NOT_RUN 的客观诊断包。强制 future generation driver 调用 evaluator/gate 比仅附 MP4 更可审计；若要宣称强 fail-closed，仍应补上述语义/数值/外部凭据检查。本 Agent 未修改根工具，也未构造或执行新测试。

## 8. 最小反预测与下一项有意义的观察

在不新增模型调用的层面，协调者正补评同六个原 MP4。应保留真实工具 SHA、每条 MP4/RUN 身份和有效面积，分别报告 source0 与source12；任何新 flow 结论只来自已验证单位的版本。

竞争解释有不同的预测，但不是单指标的充分判据：

| 解释 | 区别性预期 | 最强混杂/不能独得的结论 |
| --- | --- | --- |
| 主体动作抑制 | 在可追踪主体区域，较少位移/姿态变化；低 texture-loss 区仍减少运动 | 整帧平均 flow 被背景、camera 或 blur 支配；须实际区域对应/观看 |
| 模糊使运动不可观测 | edge 与 flow 一同下降；低梯度区 flow 置信诊断退化，保留纹理边界可能仍遵循原位移 | 零/零 FB 可假一致；高一致面积不能直接否定 |
| 多余纹理/亮度变化 | raw temporal error 升高而可解释几何运动未同比增加，残差/亮度序列异常 | 真实水面、曝光或不同合法运动也会有同一表现 |
| 仅跨视频运动相位/构图变化 | paired pixel/flow L2 可大，自身几何一致/补偿残差仍小 | 自洽视频也可能语义错误或冻结主体；不是质量通过 |

**下一项决定性机制问题仍在算子内，而非继续堆终端指标：** 先在所部署 bank 的 all40×4 同输入 native 对照中区分连续近似和部署误差，再在实际 student prefix 量化 `e_op(xS)` 与 `fD(xS)-fD(xD)`，最后保留 signed downstream/latent 方向。若部署误差在若干层已经显著，不能把自由视频失败唯一归因跨层传播或 trajectory；若部署误差小、student 输入上的 operator error 也稳定，而终端变化大，则 signed transport/flow-state 成为更强竞争解释。P23/P24 提供判别设计，当前没有执行授权或新增结果。

## 9. 实际 WARD 检索及其改变的解释

2026-10-26 使用 web 检索与 direct-open 的实际路径：Farnebäck 2003 原论文→OpenCV4.13 文档→官方 `optflowgf.cpp` 的 PolyExp/UpdateMatrices/Blur/convertTo；BT.709 官方 PDF（web 文本部分失败后，以只读内存下载和 `pdftotext` 读取 item1.2/3.2）；focus-measure noise/contrast 查询→Pertuz2013 原文 §2/§4.2.2 及 OpenCV Laplacian/颜色定义；VBench static flicker/dynamic-degree 查询→CVPR原文/补充的可访问检索正文及官方 temporal_flickering.py。CVPR main-PDF direct-open 403，未声称完整读完该 PDF；官方补充的静态筛选和公式可读，代码可完整读。

搜索把初始“补 motion 指标即可区分冻结/噪声”的解释改为两个层次：先验证求解器强度单位，随后承认 **texture-dependent observability** 是与真实运动变化竞争的机制。新诊断是有价值的 posthoc 测量，而不是 ground-truth motion、背景语义或传播因果的替代。closest known ingredients 是局部多项式光流、双向一致性、运动补偿残差和二阶边缘 proxy；这里没有把这些既有原理命名为新视频质量方法。

## 10. 审查期间工具修订的准确限定

随后实际重读 `tools/ffnds_video_metrics.py` SHA `0c6ab066674adb2a04097978ced52abcd7cbf1125918b74d59cf976a56ffec66`：新增递归 `finite_tree` 拒绝嵌套 NaN/Inf，并在特定 `implementation` 字符串下要求三项逐帧 FB/补偿诊断字段。因此第7节第1项针对 `f1dbfa33…` 的这两处缺项已被**部分修复**，不应转述为新版本仍完全没有检查。summary n/与原 series 重算的一致性、值类型、逐项运动汇总内容及外部凭据/实际 RUN 语义等边界仍需分别理解。

还亲读了 `tests/test_ffnds_video_metrics.py` 的已知纹理平移测试（192×192，预设位移 `(2,-1)`、去边界后检验估计中位数，误差阈值0.15）与显式 forward/backward warp 测试（横移2，检查补偿差与越界有效面积），以及 metric gate 的 hash-only fixture。它们检验的对象与上述单位/符号问题相符；没有运行它们，不把读到测试定义写成亲自验证 PASS，更不当成真实视频语义运动准确率。

本次重新阅读的 `tools/e0_reconstructed_runtime.py` SHA 为 `957588f5dc3b32334982b6247064ddece1d21ab627267eb717bc72fb31003872`。该绑定补充第1节的当前源身份；不改先前原生结果的冻结 source receipts。

## 11. 新六视频 CPU 补评：实际原始身份与独立复核

后续收到并实际读取 HF 相对目录 `e0_function_basis_20261026/six_video_quality_backfill_v2_attempt1/` 的 `RUN.json`、`BUNDLE.json`、六份逐帧报告。原始补评 RUN SHA 为 `3057f2facba0748f36a59cdab41b3ef7219d5c47f0e908c6c554995a2279a2d9`；BUNDLE SHA 为 `c433a0c9c0a34ef36ada4c40af455d5d0559e3f85722fbaf59a7f33ffc388ff1`。冻结协议 `config/e0_existing_video_quality_backfill_20261026.json` SHA 为 `6978cd9c41101f725fe384e07956c2293adc31f041c10fb0264343aca9088144`；driver `tools/backfill_e0_video_quality.py` SHA 为 `1d3efdc44e0a1f385c1b0a35c8e80ae032173d4329aab16532c1da2c01744307`，绑定第10节修正后的 metrics SHA 与原 gate SHA。

| metric report（上述目录内） | SHA256 |
| --- | --- |
| `dense_source00_metrics.json` | `07165673afe108c6ece5e6d7adf532f03c2103e95a40ed5eb83f77fdb14f7350` |
| `dense_source12_metrics.json` | `7dec7eb334126e57d71bbc90440db6fcaac6626807cd1fc790edc76b5539c685` |
| `whole_source00_metrics.json` | `2825ab753dd917c46101f58fea836fc6a31f8f613d32287f9eaa58794509cb2b` |
| `whole_source12_metrics.json` | `5f1eba0d61f4e116fbf44f58ed867fa8c7566a751292858ebde7c0cd0b1c441c` |
| `affine_residual_source00_metrics.json` | `2955ba313bd329494f3b85318e76f303ab2ec29bc73c6ad13fa56f8144ac4b6e` |
| `affine_residual_source12_metrics.json` | `168ac8ff55d22e3a72113c52c218cfe00da8c2503bd8576575a19d68bd38c106` |

本 Agent 没有重新运行解码/flow。通过普通 CPU JSON 算术和文件哈希，独立完成：六实际 MP4 SHA、三原 generation RUN SHA、六 metric SHA 与 bundle/RUN 绑定一致；report identity、source0/12 对应 seed42/54、81连续帧、80转换、16FPS、`[464,832,3]` 一致；wall/FFN/setup/采样显存字段逐项吻合原运行收据；四候选的 reference MP4/RUN/source/seed/prompt 对齐。实际读取原收据内容，不仅接受 gate 自报 PASS。

从 per-frame JSON 重建了88组已有标量序列的 `mean/p50/p95/max/n`，包括 flow、edge、颜色、FB、warp、paired空间/时间及low-reference-motion量；对已存 summary 的最大差为 **0**。这是数据/汇总一致性核对，不是独立 flow 重算或算法准确率验证。所有报告仍 `quality_accepted=false`，三类 external status 均 NOT_RUN；新 generation/GPU/official quality 调用均为0。旧视频与旧报告 SHA 保留。

**一个局部元数据问题：** driver 的 `reference_identity={**candidate_identity, run_sha256:dense_sha}` 会将候选 `model_digest` 也复制到 reference identity。因此四候选报告中的这个额外 model 字段不是 dense 模型 digest；source/seed/prompt/dense RUN/video SHA 绑定正确，标量计算与像素配对未因此变化。冻结文件应保留；引用时须以实际 dense RUN/报告身份为准，不把复制的额外字段当作 dense 模型 provenance。已向协调者报告此勘误。

## 12. 新测量反而排除了“仅作普通低通模糊”的单一解释

下表 flow 为下采样像素/相邻帧，edge 为 mean absolute Laplacian，FB 是几何一致面积；均为各自已存序列的 mean。warp 为在几何一致有效区测得的自身 encoded-luma 补偿 MAE。

| source | 模式 | flow | edge | mean paired edge ratio | FB面积 | 自身warp MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | dense | 0.84600524 | 0.05084669 | — | 0.98162987 | 0.01351107 |
| 0 | whole94 | 0.24702117 | 0.03098603 | 0.61613480 | 0.99093991 | 0.01230297 |
| 0 | affine54 | 0.07693327 | 0.05356027 | **1.06513680** | 0.99280789 | 0.00559069 |
| 12 | dense | 0.12543616 | 0.14086858 | — | 0.99248859 | 0.01344117 |
| 12 | whole94 | 0.09613984 | 0.09879469 | 0.70150564 | 0.99202159 | 0.01332814 |
| 12 | affine54 | 0.10618077 | 0.16302160 | **1.15696596** | 0.99223315 | 0.01766572 |

第4节的 blur 平移反例是一个辨识性反例，**不是当前 affine54 的已证机制**。现有数据比“flow下降、edge下降”更有张力：affine54 source0 的 edge ratio 在53/81帧超过1，source12 在81/81帧超过1。ratio-of-mean edge 分别为 `1.05336786/1.15726020`，也都超过1，不能把 mean-of-ratios 的聚合差异当成该现象的唯一来源。

如果候选真的仅为同一 dense 帧经过非负、归一化空间卷积 `B`，在可交换/忽略边界或周期域条件下有 `||L(BR)||1=||B(LR)||1<=||LR||1`。所以该非常具体的 **同内容、同对比度、只加正核低通、没有新纹理/噪声/编码变化** 模型不足以解释升高的 edge。它不排除真实模糊与伪高频、振铃、对比度/构图变化并存。主观“模糊”与本 L1 高频能量上升并不矛盾：能量可能来自错误的结构而非真实细节。这一新反证使简单 blur-only 机制降级，同时不把“高频=噪声”当作已经测出的事实。

source0 affine 的 mean flow 约为 dense 的9.09%，mean per-transition flow ratio 却是10.63%；二者不同，不应互换。其低 flow 与抽帧主体动作变化减弱相容，不能唯一证明冻结。另一方面，source12 affine 的 flow 平均只略低于 dense，自身 raw 帧差明显更高，且自身补偿 luma 残差 `0.01766572` 也高于 dense `0.01344117`；它至少不符合“整段只更静止、更稳定”的统一解释。

从已有 per-transition 标量另作固定16转换窗口汇总（终帧索引1–16、17–32、33–48、49–64、65–80；普通算术，无新视频计算）：

| source | 窗口 | affine flow | dense flow | affine raw adjacent MAE | dense raw adjacent MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | 1–16 | 0.098321 | 0.545409 | 0.010135 | 0.019831 |
| 0 | 17–32 | 0.077082 | 0.738228 | 0.008742 | 0.022639 |
| 0 | 33–48 | 0.069508 | 0.940183 | 0.008197 | 0.024548 |
| 0 | 49–64 | 0.068490 | 0.985877 | 0.008034 | 0.024029 |
| 0 | 65–80 | 0.071265 | 1.020329 | 0.008240 | 0.026633 |
| 12 | 1–16 | 0.114571 | 0.124561 | 0.032149 | 0.020841 |
| 12 | 17–32 | 0.103374 | 0.150672 | 0.030087 | 0.022751 |
| 12 | 33–48 | 0.100499 | 0.124603 | 0.027764 | 0.019599 |
| 12 | 49–64 | 0.102621 | 0.126181 | 0.027794 | 0.019423 |
| 12 | 65–80 | **0.109839** | **0.101163** | 0.029931 | 0.018360 |

source0 五个窗口均低 flow/低帧差；source12 五个窗口均高帧差，最后窗口 flow 甚至高于 dense。这支持“来源/内容相关的不同失败表现”，不支持所有位置或主体同一种失败因果。这些是视频帧索引，**不是去噪 step**。

低参考运动 mask 平均面积 source0 为10.53%，source12 为84.95%。它们的 raw low-motion luma error 在 affine 为 `0.02014986/0.02420659`；paired compensated luma residual difference 为 `0.01557388/0.02323373`。两个来源的 mask 覆盖与运动基线相差很大，这些值不能直接成为跨来源背景 flicker 排名，也不能把补偿前后差额全归功于去除真实 camera motion。FB面积均约98–99%表明该几何阈值接受很多像素，仍不是 subject/background 语义正确性的证书。

因此新补评确实增加了**可复核的反例和故障表征**，没有恢复质量验收。也没有关闭第1节四类数值/前缀/传递/轨迹混杂。

## 13. 对本人先前“外层一直FP32”结论的独立纠正

另一位源审查 Agent 通过协调者指出继承链前置 cast 后，我重新亲读 `BaseScheduler.step_pre`、`WanScheduler.step_pre`、`WanStepDistillScheduler/Wan22StepDistillScheduler`、runner 的每步 pre→infer→post 调用及 `envs.GET_*DTYPE`，并 `rg` 了实际 runtime 树内 `keep_latents_dtype_in_scheduler` 的赋值。

**我先前漏看了继承链的 step_pre cast。** 旧独立轨迹记录第2节“此路径不是每步把latent强制压回BF16；scheduler外层FP32状态”的断言过强，在当前配置/静态调用链下应撤回。旧文件 `independent-trajectory-error-and-correction-20261026.md` SHA `da7f943ab8966b9757ab518d4356aa621d860a9543593ce57a413043ef35316c` 原字节保留，本节是明确勘误，不悄悄改历史。

实际代码给出：

- BaseScheduler 初始化 `keep_latents_dtype_in_scheduler=False`；在 `GET_DTYPE()==GET_SENSITIVE_DTYPE()` 且 keep 为 false 时，`step_pre` 将 `self.latents` cast 至 GET_DTYPE。
- 本地配置 `config/worker.env` 指定 `DTYPE=BF16`、`SENSITIVE_LAYER_DTYPE=None`；后者在 envs 中返回 GET_DTYPE，因此条件相等。WanScheduler 的 `step_pre` 调 `super()`，Wan22 未覆写以跳过它；检索未见 Wan22 将 keep 设为 true。
- prepare/reset 的 FP32 初始化不阻止首个 step_pre cast；post 内两次 FP32乘减/加只说明**更新算术**的 dtype；最后 `.to(self.latents.dtype)` 在该调用条件下会回到 BF16。不能再据 prepare 的 FP32 和 `noise_pred.float()` 声称状态跨步保持 FP32。
- 原视频 driver 在 post 之后执行 `scheduler.latents.detach().float().cpu()` 再保存 shape/hash。该哈希的载体为 FP32，不能证明取出之前 scheduler state 的 dtype 为 FP32。旧 RUN 没有相应逐边界 dtype 记录。本结论是条件源码预测，未通过新模型运行追溯测量历史进程。

本次源身份：

| 源文件（LightX2V相对路径） | SHA256 |
| --- | --- |
| `lightx2v/models/schedulers/scheduler.py` | `0b43c8452ee734fffbd0a753f8176a5b530980723aa54981b34a8070c68db7f7` |
| `lightx2v/models/schedulers/wan/scheduler.py` | `c86af28a3b8d709692fb3a88d67ec441a77624929d397c216e0106f1c0142f4a` |
| `lightx2v/models/schedulers/wan/step_distill/scheduler.py` | `2d560449e0559aea44d80afb539f9e97bd544e5e26abcc06236e61918d01e1db` |
| `lightx2v/utils/envs.py` | `68c2f49686cb58559e8f1beaa0641b1141b5639083fe4ec3e8df771395c3189c` |

旧视频 driver SHA 仍为 `7df86bdc4ebb5a8c218883e93358f5f690e536f55bdf970949c449766ed0d005`。这里不暴露节点、硬件UUID或私有资产路径。

### 对轨迹公式的影响

正步长 `h=sigma_k-sigma_(k+1)>0` 和理想纠偏符号 **不变**：对实数更新，`delta_next=delta-h[hatv(z+delta)-v(z)]`，使下一状态匹配 teacher 的理想 target 仍为 `hatv(z+delta)=v(z)+delta/h`。这是在给定相同条件和实数状态更新下的代数结果，不是实际 BF16 可表示/可访问的训练 target 证书。

但有限更新应把 BF16 状态 cast 与 FP32 两乘减/加共同纳入 rho。在每步 pre 之后定义实际状态、`delta=zS-zD` 和 `d=hatv(zS)-v(zD)`，可精确定义

`delta_next = delta - h*d + eta,  eta=rhoS-rhoD`，

其中 rho 是实际 finite update 相对理想 `z-h*v` 的差；其含 post 的 BF16 cast，必要时还要分列 pre cast。相应范数变化多出 `2<delta-h*d,eta>+||eta||^2`。只计算原 `h^2||d||^2-2h<delta,d>` 不能宣称给出原生 latent 能量变化的精确值。

因此 `delta/h` 保留正确的连续目标方向，却不能保证在两次独立 FP32乘法和 BF16回写后逐位匹配；quantization cell/舍入可能使小修正不可见，也可能使相近更新落到不同状态格点。未来如另行获得授权，记录应逐 pre/post 保存真正 dtype 与有限向量差，而不是从 FP32哈希缓冲区反推。当前不启动它，也不据此把六视频的失败归因于 BF16量化。
