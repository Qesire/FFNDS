# Paired-value transport: mechanism and source screen

Date: 2026-11-27. Scope: analytical candidate, source audit, and static accounting; no runtime modification or GPU experiment.

## Decision

Proceed with one candidate: estimate a full per-head channel map from a small set of same-position anchor/current value pairs, then apply that map to the already cached omitted-attention response. Use full 128 × 128 maps for the first mechanism test. Low-rank maps are a later execution option whose approximation error must be measured.

This changes the information used for prediction: current value rows supervise the transport of a historical attention field before P64 attention-output labels correct the remaining innovation. It adds a new identifiable response family beyond the current diagonal transport; it does not claim a new least-squares algorithm or a new generic forecast-and-correct framework.

## Typed construction and exactness

Suppress batch, layer, head labels; each map is fitted separately per layer/head. Let query/key positions number N, and value channels d = 128. Let W_U ∈ R^(N×N) be the full-softmax weights on omitted key positions, with zeros elsewhere; its row sums are ρ. S ∈ R^(N×d) is selected-only attention, normalized over selected keys. Let μ be the mean value over key positions, broadcast over query positions, and D the diagonal value standard-deviation matrix.

At the anchor a:

X_a = (V_a − 1 μ_aᵀ) D_a⁻¹,
Z_a = W_U,a X_a.

The existing method caches Z_a and uses Z_a D_t. Add only J paired anchor rows U = X_a[J,:]. Current V_t already exists as an attention input. Fit

M̂ = argmin_M (1/|J|) ||U M − (V_t[J,:] − 1 μ_tᵀ)||_F² + λ ||M − D_t||_F².

Use the base prediction

b_t = (1 − ρ_a) ⊙ S_t + ρ_a μ_tᵀ + Z_a M̂,

then apply the existing P64 innovation recovery with unchanged probes, dictionary and ridge settings.

**Conditional exactness.** Suppose selected support and full-softmax weights remain fixed, V_t − 1 μ_tᵀ = X_a M_* for every key, and the sampled design identifies all channel directions present in X_a. With λ = 0, exact arithmetic and exact state storage, fitting yields X_a M̂ = X_a M_*; hence

W_U,a X_a M̂ = Z_a M̂ = W_U,a (V_t − 1 μ_tᵀ),

and b_t = A_t before P64. J ≥ d with full column rank U is a simple sufficient identifiability condition. More generally it suffices that ker(U) ⊆ ker(X_a); full rank is stronger than necessary. For V_t = V_a T + 1 cᵀ, the correct map is M_* = D_a T.

The mathematical fact that makes the state useful is associativity of two operations on different axes:

W_U,a (X_a M_*) = (W_U,a X_a) M_*.

It permits transporting the whole query field without storing the omitted N × N weights or all anchor value rows. This is a conditional sufficient-state statement for an explicitly restricted value-evolution family.

D_t is the existing diagonal special case. The fitted family permits channel mixing. Scalar gains, rewriting the baseline as SparsePR plus projected history, and renaming the objective do not create this extra information.

**Numerical boundary.** λ > 0 shrinks toward D_t and does not preserve the general exactness theorem. BF16 anchor samples and cached Z introduce state error. Use QR or a regularized stable solve; inspect singular values/conditioning rather than infer stability from J ≥ d alone.

## What is left to predict

Write V_t^c = V_t − 1 μ_tᵀ. For arbitrary routing changes, the pre-probe error is exactly

A_t − b_t =
(ρ_a − ρ_t) ⊙ (S_t − 1 μ_tᵀ)
+ (W_U,t − W_U,a) V_t^c
+ W_U,a (V_t^c − X_a M̂).

These are respectively mass drift, routing drift, and value-transport mismatch. The new map directly addresses the third term. P64 still observes and corrects their combined remaining effect.

A low held-out value-pair error supports the proposed channel relation. It does not by itself guarantee smaller attention or video error: omitted weights may emphasize a different part of the key field, routing can drift, and downstream responses can magnify particular directions. Use independent value-pair validation, withheld attention rows, and downstream block/final-output checks as separate evidence.

## Difference from SparsePR v2

SparsePR v2 fits a current-step affine output correction from current sparse outputs and current full-attention output probes. This candidate fits an anchor-to-current value relation from key-position pairs, then acts on a historical query-position field. The input space, regression targets and action on the computation graph differ.

A structural discriminator exists: selected outputs may be identical for different query rows while omitted responses differ. A shared affine function of selected output must make identical predictions on such rows (except rows overwritten by probes). Historical Z_a can retain their distinction and Z_a M can update its channel content. The method is therefore not generally an algebraic rewriting of current-output affine regression. If Z_a already lies in the current affine feature span and the estimators have matching constraints, overlap is possible on that restricted instance.

The current low baseline already uses Z_a, so an improvement over SparsePR alone would not isolate the new map. Compare diagonal history transport and fitted full-map transport using exactly the same P64 correction. Add shuffled anchor/current value pairing as a mechanism control; it preserves marginal value statistics while destroying the proposed correspondence.

## Directly relevant sources and scope

| Source | What was checked | Mechanism relation |
| --- | --- | --- |
| SparsePR v2 | [Primary-source and equivalence audit](../../sparsepr_overlap/primary/SPARSEPR_PRIMARY_AUDIT.md) | Current sparse-output/full-output probe regression is shared recovery territory. Paired values identify a different map with different supervision. |
| RECAP, arXiv:2610.26244 | Official repository README; operator.py and kernels/detail_cache.py at commit 139956c51514567baba62f41f03485402fe85123 | Direct prior for fresh coarse tail statistics plus cached omitted-interaction refinement. The inspected core code stores tail-output and tail-log-mass discrepancies and regulates their use by current log-mass drift. It does not estimate the proposed same-position V_a → V_t channel map. |
| Koopman Observers for Diffusion Acceleration, arXiv:2610.10366 | Official arXiv abstract retrieved; full HTML/PDF fetch failed, so no full-paper exclusion claim | Calibrated time-dependent joint shallow/deep feature-increment dynamics, current shallow innovation correction, and periodic full refresh make observer language prior territory. The abstract's operator is identified from calibration trajectories; this candidate identifies an online value map within the current layer/head and uses attention's right-linearity. |
| Re-ttention, NeurIPS 2025 | Official project and proceedings abstract | Normalization-ratio correction and cached masked-token residual already exist; avoid claiming tail caching or denominator correction in general. |
| PISA, arXiv:2602.01077v5 | Primary HTML §3 | Current coarse/Taylor tail numerator-and-denominator approximation is prior territory; not the proposed value-pair history transport. |

### RECAP source details

Official repository: https://github.com/lyraxtg/RECAP

Fixed sources:

- https://github.com/lyraxtg/RECAP/blob/139956c51514567baba62f41f03485402fe85123/recap/operator.py
- https://github.com/lyraxtg/RECAP/blob/139956c51514567baba62f41f03485402fe85123/recap/kernels/detail_cache.py
- Paper: https://arxiv.org/abs/2610.26244

At refresh, let O_tail,a be the exact conditional omitted output and g_a the coarse conditional output. RECAP stores their difference, as well as the difference between exact tail log-mass and coarse log-mass. At a sparse call it recomputes g_t and its coarse log-mass ℓ_t. Its admission factor is

α_i = exp(−|ℓ_t,i − ℓ_a,i − mean_i(ℓ_t,i − ℓ_a,i)|).

It mixes the coarse tail with the corrected coarse-plus-cached-detail tail in mass space, then merges that tail with exact selected attention. This is strong evidence to keep the novelty statement on the new paired-value identification and its exact operator action, rather than on broad combinations of current coarse estimates and cached residuals.

### Other primary links

- Koopman: https://arxiv.org/abs/2610.10366
- Re-ttention: https://kgmills.github.io/projects/rettention/
- PISA: https://arxiv.org/html/2602.01077v5

## Static resource accounting

For N = 31668, H = 40, d = 128, J = 256, and 40 layers in the active expert:

| Item | Static amount |
| --- | ---: |
| Added paired anchor samples, BF16 | 100 MiB = 0.09765625 GiB over 40 layers |
| Full M application per layer/hit | 20,753,940,480 MACs = 41.50788096 GFLOPs |
| Full M application across 40-layer low hit | 1.6603152384 TFLOPs |
| UᵀY per layer/hit | 167,772,160 MACs = 0.33554432 GFLOPs |
| Rank-8 factored application per layer/hit | 2,594,242,560 MACs = 5.18848512 GFLOPs |

The paired-row count is not an attention-teacher budget: V_t rows are already computed input activations. The solve, anchor factorization/storage, data movement and kernel launches still have costs. Full M increases arithmetic and can increase memory traffic; these numbers are not latency measurements. The four-step setting gives only one low-stage hit after its anchor, so anchor setup has little within-video amortization.

If rank-r compression is tried later, exactness requires rank(M_* − D_t) ≤ r for the diagonal-plus-low-rank parameterization. A low-rank T − I does not by itself imply this condition. Optimal reduced-rank least squares is obtained by truncating in whitened prediction coordinates, not by indiscriminately truncating the raw OLS coefficient.

## Recommended claim and next discriminator

A defensible candidate claim is:

“Estimate current value-channel transport from a small paired sample, and push a cached omitted-attention response through that transport; use sparse current full-attention probes for the remaining innovation.”

The derivation gives a real expansion of the recoverable family and concrete conditions for success. Whether it improves Wan wide8192 accuracy or wall-clock performance remains a measured question. First compare diagonal and full-map transport with fixed P64 and fixed sparse support, and separately report pair generalization, withheld attention error and downstream block/final-output error. This single comparison isolates the new mechanism more directly than expanding unrelated features or changing probe budgets.

