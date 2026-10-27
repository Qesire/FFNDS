# G11 / L74 SharQ：FP4符号幅值、稀疏主路径与稠密残差的原文/源码双证据

2026-10-27 | 直接强竞争（原文+官方Python/CUDA源码），**LIT / PRIMARY METHOD VERIFIED + SOURCE INSPECTED**，不构成FFNDS本地GPU或原生Wan执行。和 [五篇G11原始论文审查](G11_FIVE_PRIMARY_PAPERS_EQUATIONS_CODE_AND_COUNTERFACTUALS.md)、[量化可用信息主问题](G11_QUANTIZED_SIGN_MAGNITUDE_INFORMATION_AND_FFN_FUNCTION.md)联读。

## 1. 为什么这篇是最直接的新增优先级

[arXiv:2606.26587v1完整原始论文](https://arxiv.org/html/2606.26587) §3 Eq(6)–(13), §4.3 Table3, §4.4 Tables4–5/Fig4；[作者官方GitHub](https://github.com/actypedef/SharQ)。SharQ把**已知可读的activation magnitude**用于input-adaptive、硬件合法的N:M分解，并可用于NVFP4/HiF4/MXFP4，作者在**Wan2.2-T2V-A14B、4张RTX5090**实测全片延迟。它是当前G11最强方法级+数值格式+近缘视频工程竞品之一；其基础模型**T2V不等于FFNDS的Wan2.2 I2V双expert四步D-Q**。它主要解决FP4量化误差与保留值尺度，而**并未证明冻结原FFN可免去dense残差路径**。

## 2. 严格算法式与真正总成本

输入X与weight W，在线每local unit取硬件合法mask M(X)（NVFP4: 4:8按pair保留；HiF4: 2:4），定义

\[
X_{\mathrm{sp}}=M(X)\odot X,\quad
\widetilde X_{\mathrm{sp}}=\mathcal{D}_{\mathrm{sp}}(\mathcal{Q}_{\mathrm{sp}}(X_{\mathrm{sp}})),\quad
R=X-\widetilde X_{\mathrm{sp}}.
\]

关键等式是 \(R=(X-X_{\mathrm{sp}})+(X_{\mathrm{sp}}-\widetilde X_{\mathrm{sp}})\)：R补**掩码丢失的信息**与**稀疏分支的量化误差**。Eq(9)的两路都是实际运算：
\[
\widehat Y
=\operatorname{SparseFP4GEMM}(Q_{\mathrm{sp}}(X_{\mathrm{sp}}), Q_{W,\mathrm{sp}}(W))
+\operatorname{DenseFP4GEMM}(Q_{\mathrm{dn}}(R), Q_{W,\mathrm{dn}}(W)).
\]
数学上 \(X=\widetilde X_{\mathrm{sp}}+R\) 完全成立；第二路的R再量化后仍有误差，Eq(12)–(13)给出后者在固定W下的上界。两个路径共享低位weight payload，但**有不同的weight scale view**；不应该把“共享weight数据”理解为“两个GEMM变成一个GEMM”。当前W1/GELU/W2如果采用SharQ，每个linear可能各有这两路核，原本真实FFN MAC不必然下降；真实延迟比较相对FP16/FP8或NVFP4基线才有意义。

## 3. 官方代码的确切执行与限制

源码逐行读取：
- [model/qLinearLayer.py](https://github.com/actypedef/SharQ/blob/main/model/qLinearLayer.py) blob SHA 1db675d181663d4bd76868f40279e40db8428c31。QLinearLayer(quant_type=SHARQ)量化weight调用quantize_weight_shared_nvfp4取得**weight_q、scale_w_sparse、scale_w_dense**；prepare_input用quantize_activation_sparse_residual_nvfp4取得**a_comp、e、sfa_sparse、qx_res、scale_x_res、scale_x**；apply_prepared严格**先**sparse_matmul_tensor_scale，再matmul_accum_tensor_scale把dense residual累加到y_sparse，若disable fusion则另算dense matmul后加和。公开真实kernel主路径是**NVFP4/Blackwell**；HIF4_SIM、SHARQ_HIF4_SIM和SHARQ_SIM是PyTorch仿真分支，不等于官方已发布同等HiF4 native kernel。
- [kernels/src/fused_sparse_prepare.cu](https://github.com/actypedef/SharQ/blob/main/kernels/src/fused_sparse_prepare.cu) blob SHA 687470cc965cfd524157d7ea16afedba35b84a76。源码每32输入划分4个8元素单元，每个8元素内比较4对相邻pair的max abs，保留分数最大的**两对**（总4/8），得到稀疏FP4码，并直接将残差改为 \(x-\mathrm{dequant}(q_{\mathrm{sp}})\)；随即在两个16元素子组计算residual_fp4动态scale/码并写pack；包括sparse mask metadata e。故可核实**动态幅值决策确实在线**，同时它访问真实当前activation，不能称依照checkpoint weight FP4 sign/scale已获得该mask。
- [kernels/README.md](https://github.com/actypedef/SharQ/blob/main/kernels/README.md) blob SHA 8a09910a72256969ed94ca3d482fc8407e131c2b。CUDA/CUTLASS主扩展默认sm120a RTX50，sm100a B200需另配置；standalone benchmark分quantize/GEMM/whole linear，weight quant离线计账。
- [README](https://github.com/actypedef/SharQ/blob/main/README.md) blob SHA b4d2b0d3c0e1b28c32f78489499a40d9dc816316。SHARQ、SHARQ_SIM、NVFP4；模型评测默认可禁RMSNorm融合，benchmark/demo可选启用RMSNorm及epilogue fusion。**不能把该公开实现直接描述为对FFNDS原HiF4 QDQ的原样drop-in native sparse kernel**。

## 4. 物理数据要严格拆分

原论文Table3: Wan2.2 **T2V** A14B，**4 RTX5090**，全片生成时间秒：
- 480p: FP16 223.41；SharQ 182.21（1.23×）；SharQ+SageAttention 141.63（1.58×）。
- 720p: FP16 717.88；SharQ 648.04（1.11×）；SharQ+SageAttention 457.88（1.57×）。
本文只有作者报告的generation latency，没有当前I2V 4step、高/低专家、完整VBench或视觉一致性对照的等价结果；**1.58×不是单独SharQ**。

**尤其关键的强反例**：原论文Fig4(a)的Llama3.1-8B **prefill latency** 相对于dense NVFP4 baseline，SharQ为111.2%，即**比dense NVFP4慢11.2%**。虽然融合把quantize+RMSNorm两阶段开销34.1%减到26.7%，两路实际GEMM的占比由30.1%升到48.4%。Fig4(b)仅比较**孤立sparse FP4 GEMM**和dense NVFP4 GEMM，sparse kernels是0.66/0.72/0.71×，不等于叠加dense残差后的整体linear就同倍提速。说明对Wan大M/小chunk必须比较**完整两分支/完整模型**与dense packed FP4，而不是只报半稀疏GEMM speedup。

Table5仅用sparse路不补残差，Llama3.1-8B WikiText2 PPL达到82.95、MMLU25.78；SharQ的sparse+dense恢复到PPL6.73、MMLU63.76。对FFNDS而言，**残差有真实功能**，研究若要让残差也变少，必须提出比局部元素小幅值更强的有符号/下游证据。Table4：HiF4+RTN Qwen3-30B-A3B WikiText2 PPL9.08→SharQ+HiF4 8.92（官方仿真或方法适配成绩；真实发布内核仍是NVFP4），与FFNDS originalHiF4三层数值尺度不应直接等同。

## 5. 最新竞争解释与可证伪机制

**H1“既有编码包含足够的便宜信息以证明可省W1/W2”**与**H2“运行时幅值分解+低位补偿使精度/硬件吞吐更好，但没有跳整个FFN函数”**必须区分。SharQ的直接证据强力支持H2，尚不支持H1。真正新的科学问题：在input和weight的编码/scale早已可用且pairwise读价低于稠密TensorCore的条件下，能否预测**net gate×W2作用小的联合函数组/可跳tile**，从而让原来的R也无需dense GEMM，而不是仅仅压缩activation分布？

三个可比较而未实施的路线：① certified signed margin预判负侧GELU函数范数；② 在已存scale/sign编码中预示下一FFN的有符号W2 value子空间，按合法HiF4 group/tile决策；③ 重新表示编码计算（CodeGEMM查表、training-structured Spark）替代主投影。比较所有路径必须计入Q1在线量化、selector/scale、第二路是否dense、两套权重/缓存、整段video质量及SLO，原实验暂停不变。
