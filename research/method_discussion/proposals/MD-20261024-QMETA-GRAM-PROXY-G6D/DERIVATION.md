# Quantizer metadata as low-cost Proxy — causal and format-aware derivation

**Agent:** `gpt6-ffnds-method-agent-20261024-a` · **2026-10-24** · AutoResearcher @ `5bd7ec5`
**Status:** Independent method discussion, source/math only; no new model inference, GPU/CPU tensor collection, benchmark or quality result.

## 1. Reframed objective: use the representation's existing side information

We should not require full FP4 code equality, code-orbit memoization or quantization-cell certificates before opening a compute-saving avenue. The motivating question is:

> Can a low-dimensional statistic already produced by an activated quantizer predict enough **relative benefit of actually executing FFN work** to make a cheap packed decision?

The proxy is a decision instrument, not a substitute for the FFN activation value. Q1 scale metadata is a potential **pre-W1** instrument; Q2 metadata is available only after W1 and can influence W2 or a later decision. This respects Q1 information, Q2 economic decision and Q3 genuinely eliminated GEMM.

### Source-exact native reference

- `tools/packed_ffn_reference.py`: `v1=qdq((input/s1).BF16)`; `z=F.linear(v1,w1,b1)`; `h=GELU_tanh(z)`; `v2=qdq((h/s2).BF16)`; `v2=BF16(v2*alpha)`; `y=F.linear(v2,w2,b2)`. `alpha` is applied **after Q2, before W2**.
- `upstream_audit/.../ptq/adapters/wan_i2v.py`: optional Smooth division before the activation QDQ, then ordinary `F.linear(work,weight,bias)` with a dequantized tensor. The actual current path is **QDQ + BF16 GEMM**, not native packed FP4 multiply.
- `upstream_audit/.../hifxg_quant_cuda.cu`: `hifx_quant_cuda_inner` computes within-64 maxima, global `lv0`, nested 8/4 scales and rounded outputs. Its `lv0,lv1,lv2` intermediates are CUDA local variables; this source path does not expose a persistent metadata tensor to `F.linear`. Prior R12 group separability is limited to complete ordered64 groups.
- OCP MX v1.0 Table 1: **MXFP4** has 32 input elements per 8-bit E8M0 scale and E2M1 FP4 codes.
- NVIDIA Transformer Engine NVFP4: **NVFP4** has per-16 E4M3 local scales plus an FP32 global factor, with rowwise activation and optionally 2D weight scaling. Its tensor-global scale may be computed by a training recipe, frozen by inference calibration, or represented differently in an actual engine. Do not assume it changes per-token without checking the selected inference adapter.

Native packed representations store scale metadata that the MMA consumer needs; whether the routing system can access the same metadata **without new synchronization, copying or swizzle** is a separate measurable question. A QDQ baseline does not get its discarded scale metadata for free.

## 2. A direction-aware scale-gram proxy

Partition the already quantized Q1 input along the input feature/K dimension into `B=D/k` blocks. Denote real decoded input `x_{t,b}=s_{t,b} c_{t,b}` and hidden W1 group `g` with (A_{g,b}\in\mathbb{R}^{h_g\times k}).

\[
z_{t,g}-b_g=\sum_b s_{t,b} A_{g,b} c_{t,b}.
\]

Scale-only cannot predict the vector output. However, under the explicit conditional null model

\[
\mathbb E[c_b\mid s]=0,\quad
\mathbb E[c_bc_{b'}^\top\mid s]=
\begin{cases}
\nu_b I,&b=b',\\0,&b\ne b',
\end{cases}
\]

the expected squared preactivation norm equals

\[
\boxed{P_{t,g}^{\mathrm{scale}}=\mathbb E[\|z_{t,g}-b_g\|_2^2\mid s]
 =\sum_{b=1}^B \nu_b \|A_{g,b}\|_F^2\,s_{t,b}^2}.
\]

A fixed precomputed `K[g,b]=ν_b||A[g,b]||²_F` turns the quantizer's native `s_b` into a **group-conditional** proxy. This is more informative in principle than the scalar input norm `Σ_b s_b²` because it weights input blocks by each output group's weight geometry. It remains statistical and can be wrong with directional correlations, W1 bias, signed GELU effects, per-channel Smooth, gate and downstream transport.

For actual learned code statistics one can add a mean/covariance correction:
`E[z_g-b_g | s]=Σ s_b A_gb μ_b(s)`;
`E||z_g-b_g||² = ||Σ s_b A_gb μ_b||² + Σ_bb' s_b s_b' tr(A_gb Σ_bb'(s) A_gb'^T)`.
A full covariance model would defeat the inexpensive-proxy goal. The minimal useful question is how much conditional orientation is recoverable from **one or a few cheap code statistics** without reconstructing W1.

### Less information versus more information

`S0`: one scalar total input RMS or RMS inferred from scales

`S1`: the full original Q1 scale vector + precomputed group weight profiles

`S2`: S1 plus minimal FP4 packed-code statistic (signed subblock sum, code moment, saturated/zero fraction)

`S3`: tiny low-dimensional signed code-weight projection

`S4`: full W1 group preactivations (oracle information only; its MACs have already been paid)

Each increase must demonstrate **incremental action value after its own read/computation cost**, not merely improve score–label correlation. Static rank and same-budget native FP4 dense baselines compete with S0–S3.

## 3. The minimal counterexample the metadata score must survive

Choose one 2-coordinate part of an MXFP4 32-coordinate block, all other coordinates zero. Use **the same** FP4 scale `s=1/4` and E2M1 codes `c_A=(4,4)`, `c_B=(4,-4)` (both 4 values legal in E2M1). Thus effective decoded inputs are `x_A=(1,1)`, `x_B=(1,-1)`.

Their block max, scale, code squared norm and per-coordinate magnitudes are identical. For W1 row `w=(1,1)`:

\[
w^\top x_A=2,\qquad w^\top x_B=0.
\]

`GELU_tanh(2)` is nonzero while `GELU_tanh(0)=0`. No decision rule using only their **shared unsigned scale + energy** can rank these W1 responses differently. A signed code statistic or another upstream state may distinguish them; that is a constructive way to extend a proxy, not a reason to demand full equality.

Also, all-zero except one scale leader and a block with many equal-magnitude leaders share the same amax/scale but have different sum of squares. For a quantizer scaling by amax, this is an independent insufficiency of scales alone even before sign matters.

## 4. Native compute accounting

For Wan `D=5120,H=13824`, one W1 has `D*H = 70,778,880` nominal scalar MAC per token.

Use hidden **decision groups** of 64 (`G=216`) for an initial Q2-aligned candidate, but respect the chosen format's actual grouping and weight layout. The Q1 input microscale blocks are *not the same axis* as the hidden decision groups.

| Format / Q1 scale block | B input scales | G×B scalar proxy contributions per token | 4-bit input codes | scale metadata raw byte count per token |
| --- | ---: | ---: | ---: | ---: |
| MXFP4 / 32 | 160 | 34,560 | 2,560 B | 160 B (E8M0) |
| NVFP4 / 16 | 320 | 69,120 | 2,560 B | 320 B (E4M3 local) + global state |

These are representational arithmetic/byte counts, **not latency or measured data movement**. NVFP4 scale tensors may be padded/swizzled; global state may be tensor-level rather than per-token. Native kernel compatibility/launch/packing costs dominate the actual decision.

A static low-rank factorization `K≈U_r V_rᵀ` changes `GB` scalar proxy contributions to `r(B+G)`; for `r=8` this is 3,008 (MX) or 4,288 (NV) nominal operations per token plus loading, decision and dispatch. It is only useful if rankings/quality constraints survive factorization and routing integrates cheaply enough into the real producer/consumer pipeline.

For within-token homogeneous score `P_g(s)=Σ K_gb s_b²`, if NV local+global `s_b=G·a_b` and `G>0` is shared across b, then `P_g=G² Σ K_gb a_b²`; `G²` cancels in **within-token group ranking** only. Cross-token thresholds, biased scores and nonlinear decision utilities can depend on G.

## 5. Most discriminating saved-asset workflow, not a claim of results

The previous `OFFLINE_SAVED_ASSET_QUANT_STRUCTURE_R1.json` recomputed **true archived scalar native state drift / nextnorm labels**, and exactly enumerated FP4 code orbits. It did **not** have decoded/native Q1 codes or Q1 scale vectors; these archived scalar aggregates do not answer the present Proxy question.

When authorized native operand payloads are accessible, use one real, source-indexed subset and freeze identical token/step/expert coordinates:

1. At the **Q1 producer boundary** capture actual scales and the minimal packed-code counters **without recomputing the full W1**. For the current QDQ implementation record whether a capture kernel has to compute/export discarded scales; account for this as a separate control.
2. Offline pair those features with actual W1 group activation/GELU/Q2-group values at the same scope, using historical available tensors only if fields are validated. Measure source-held-out group relevance ranking and output margin, not just oracle exact duplicate percentages.
3. Factor `K` with `r\in\{1,4,8,16\}`; compare S0, S1, S1-low-rank, S2 and cheap learned router against matched static group selection. Take cross-source and cross-step generalization seriously; 31,668 rows in one scene are not independent scenes.
4. Only after the lightweight information test: discuss a true packed group-elision path with physical W1/W2 work counted. Q2 info may give an independent W2-only upper bound but cannot retroactively improve W1.

**Do not call a proxy successful simply because scales correlate with max activation; compare independent improvement over original RMS / proxy+HWG as well as fixed group selection.** The most useful result could be that S1 is insufficient but one very cheap S2 code-sign summary resolves its errors.

## 6. Scientific relation to prior ideas

- [OCP MX specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) gives the exact representational metadata contract.
- [NVIDIA TE NVFP4 documentation](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html) specifies the two-level scale decomposition and packed tensor layout.
- [Agile-Quant](https://ojs.aaai.org/index.php/AAAI/article/view/29860) and activation-aware quantization show existing uses of activation statistics to drive model processing; this idea is not automatically novel.
- Existing FP4 code-orbit reuse (G6C) and exact Q2 cell certificates (G6B) remain **independent competitors**, not required gateways for this method. The current objective is inexpensive *ranking information* rather than exact quantized operand equality.

**Proven here:** conditional second-moment formula, causal timing, representation byte counts, deterministic sign counterexample and nominal cost formula.
**Not proven:** conditional code isotropy in Wan, source-held-out proxy utility, availability of Q1 scale without overhead in a particular kernel, or native speed/video quality.
