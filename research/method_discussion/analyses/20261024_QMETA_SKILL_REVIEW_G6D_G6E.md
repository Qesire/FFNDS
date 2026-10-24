# Skill-driven independent self-audit and exploration — G6D v2 / G6E v1

**Date:** 2026-10-24 · **Agent:** `gpt6-ffnds-method-agent-20261024-a` · pinned AutoResearcher `5bd7ec537ae63f2b6a6dd579f0e886aff58a177c`

**Status:** Method discussion only. This is the originator's critical self-audit, **not an independent peer review or SUPPORT vote**. Existing G6D v1 and all independent review files remain immutable.

## Strongest new result: conditional W1-energy prediction need not provide dynamic group information

G6D v1 presented `P_t=K q_t` with `q_t=(s_{t,b}^{eff})²` and `K_{g,b}=ν_b||W1[g,b]||²_F`, grounded in conditional isotropic/zero-mean quantized codes. The conditional second-moment derivation is mathematically valid **under those assumptions** but ranks `E||z_g-b_g||²`, not actual GELU, signed W2, the gated writeback, nextnorm or video.

### Exact double-heterogeneity deductions

Let `K∈R_+^{G×B}` and `q_t∈R_+^B`. Score at a fixed token: `p_t=Kq_t`.

1. **No input scale-shape variation → no dynamic rank.** If `q_t=λ_t q0, λ_t>0`, then `p_t=λ_t Kq0`. All top-k sets are independent of token, regardless of `K`.
2. **Separable weight profiles → no dynamic rank.** If `K=a b^T+1_G v^T`, where `b≥0`, then `p_t=a(b·q_t)+1_G(v·q_t)`: for `b·q_t>0`, ranking is the fixed ranking of `a`, even with wildly different scale shapes.
3. **Positive-cone partial ordering.** If `K_{g,b}≥K_{h,b}` for every input microscale block `b`, then `p_g(q)≥p_h(q)` for every possible nonnegative `q`. A ranking reversal requires the realized `q` to cross `(K_g−K_h)·q=0`; mixed-sign differences alone do not guarantee such a crossing in the *feasible Wan distribution*.
4. **Useful dynamic coordinate.** `q_t=a_tu_t` with `a_t=Σq_{t,b}>0`, `u_t=q_t/a_t`. At fixed per-token top-k budget, ranking depends only on `Ku_t`. Consequently an NVFP4 scalar tensor-global scale **cannot by itself** create dynamic group ranking, though it can matter for absolute thresholds, bias/GELU and quality.

The double condition is falsifiable: **both input scale-shape diversity and nonseparable output-group weight profiles must interact**, and the resulting order changes must align with a real action reward. This is a more valuable first check than a large correlation coefficient for `E||z||²`.

### Where a more constructive Proxy arises

The natural next correction to G6D is not an arbitrary extra score. Move the target from W1 energy to **output-space contribution and signed downstream effect**. For a fixed decoded Q2 operand `h`, original W2 output columns `B_g`, actual AdaLN gate `γ_t`, omitted panels S,

`δ_t(S)=diag(γ_t) Σ_{g∈S}B_g h_{t,g}`.

The exact local loss contains **cross-group signs** `2<δ_g,δ_h>` and cannot be recovered from independent W1 activity scores alone. Conditional group output-energy may be modeled by
`tr(M_g Σ_{h_g|q})+μ_{h_g|q}^T M_g μ_{h_g|q}`, `M_g=B_g^T diag(γ)^2 B_g`. This is a *conditional-moment identity*, not a predictive model until its moments are estimated from real data; joint group error needs cross-covariances. A cheap proxy could condition these estimates on quantizer scale SHAPE plus a tiny signed nibble sketch and gate/upstream state, then be compared with exact static and old linear selectors.

The **two distinct causal timings** should not be merged: Q1 scale/codes can guide omission of original W1 hidden groups if early extraction/dispatch is cheap; Q2 native scale is available AFTER W1 but BEFORE W2, and directly fits a W2 K-panel skip hypothesis. [G6E v1](../proposals/MD-20261024-Q2-NATIVE-SCALE-PANEL-G6E/v1.json) develops the latter using a conditional exact real-arithmetic error bound.

## Source audit: native FP4 versus historical HiF4-QDQ

- `tools/packed_ffn_reference.py`: `Q1→F.linear W1→GELU_tanh→Q2→alpha cast→F.linear W2`. The optional `alpha` is **after Q2 and before W2**. Q1 pre-W1 and Q2 pre-W2 timing must be respected.
- HiF4 archived `upstream_audit/source_deltas/HiF4_Sp/files/HiFloat4/hif4_gpu/quant_cy/base/cusrc/hifxg_quant_cuda.cu`: inside `hifx_quant_cuda_inner`, at lines about 229–282, 64 input entries are reduced into 4/8/64 maxima and `lv0/lv1/lv2` *CUDA local temporaries*, and the kernel returns quantize-dequantize values. `wan_i2v.py` then sends dequantized BF16 to `F.linear`. Historical scale metadata **is not an exposed persistent GPU tensor**; a QDQ proxy must charge instrumenting or recomputing it.
- OCP MXFP4: E2M1 code, one E8M0 exponent scale per **32** K elements. NVIDIA Transformer Engine NVFP4: E2M1 codes, local E4M3 per **16**, and an FP32 per-tensor factor, with swizzled/padded scales. This is a **representation fact**, not proof of free scale reads before a routing barrier.
- Selected GPU execution environment matters: the historical A100 HiF4 QDQ→BF16 GEMM **is not** a direct benchmark for Blackwell native FP4 Tensor Core code. No speed ratio may silently transfer across paths.

The [PyTorch low-precision FlashAttention 4 discussion](https://pytorch.org/blog/low-precision-flash-attention-4-end-to-end-block-scaled-attention-for-blackwell/) is informative execution prior: producer+consumer fusion, scale-factor memory layout and overhead overlap matter even when the hardware implements scaled native MMA.

## Preserved saved evidence: what was actually learnt before this proposal?

The original `reports/worker_20261020/HP1_proxy_RUN.json` says **COMPLETED_HP1_PROXY_GATE_REJECTED**; it is a source-held-out `C`-score support proxy, **not a sparse kernel/video acceptance**. Across its two validation folds:

| Historical HP1 case | Static retained C | Old linear Proxy | Post-hoc oracle | Linear's captured oracle-static headroom |
|---|---:|---:|---:|---:|
| Step2, Block20, 108 of 216 groups | 0.534831 | 0.560428 | 0.635718 | 25.37% |
| Step3, Block20, 162 of 216 groups | 0.774155 | 0.789953 | 0.836206 | 25.46% |

`(linear−static)/(oracle−static)` gives the displayed percentage. The old C-score proxy improved a **local ranking metric** but failed its own gate. `HP2_proxy_RUN.json` nonlinear head was worse than the old linear control, and `NP1_proxy_RUN.json` nonlinear512 did not beat its matched linear512. These failures only concern the archived features/targets/source split and costs; they do **not** eliminate a quantization-specific early side-channel that those heads did not use.

The newer `research/20261024/OFFLINE_SAVED_ASSET_QUANT_STRUCTURE_R1.json` measured step-specific state differences from old native scalar summaries and enumerated E2M1 code-orbit cardinalities. **It did not observe real Q1 FP4 scales/codes or actual quantizer-proxy ranking labels.** Do not relabel it a QMeta success.

## Distinguishing research moves after this audit

| Mechanism | New observable/prediction | Most efficient source/analysis discriminator |
|---|---|---|
| G6D v2: Q1 scale-SHAPE × weight-profile interaction | Actual **change in the output-group ordering** above static RMS and old HP1/NP1 controls | Offline row-dominance and pairwise ranking-crossing on verified W1 weights, then source-held-out actual Q1 scale vectors |
| G6D v2 + cheap signed sketch | Equal scales/energies but opposite code orientation change W1 and gate/W2 value | Signed-code counterexample plus small actual-payload sample; *compare incremental value after code read cost* |
| G6E v1: Q2 native scale → W2 K-panel omission | Exact local output-error upper bound makes some whole native W2 panels cheap | Native Q2 scale/code visibility and entire physical token-tile K-panel occupancy; compare static skip and dense block-scaled GEMM |
| Competing explanation: generic RMS/weight static dominates | Rank reversals are absent/too close or physically not useful | Test ranking margins and matched action value, not activation-energy regression MSE |

This is constructive rather than a general moratorium: if G6D's matrix is near rank-one, the result identifies a distinct **Q2 decision-time** action; if there is strong input×weight heterogeneity, it identifies the exact information needed to exploit it. No preset dense completion, rank256, HiF4 format monopoly or requirement to produce an experimental result before recording a theory.

**Current evidence:** two conditional exact algebraic theorems (rank collapse and W2 bound), precise native source timing, real prior HP1/HP2/NP1 numerical controls, and published FP4 format/physical execution constraints. **No new CPU/GPU/model experiments** and no new native FP4 data.
