# G20 V4 — Representation / Row-Density Comparison — FORMAL RESULTS (EXIT 0)

**Status: FORMAL job 163279 EXIT 0; scientific interpretation revised (doc rev2).**
Author: `qoder_root_g20_20261111`. Date: 2026-11-14. Route: `Qesire/FFNDS` main
`4107f75f15ca9982aab2f3d80401260dd213ec5d`, `research/20261113/G20_VBENCH_RESTART_V4_20261113.md`
§4.3 + §6; queue V4.

> Formal job **163279 EXIT 0** (`V4_REPR_DENSITY_COMPLETE`, elapsed 1122.25s, anode19 A100-80GB,
> peak RSS 1.51 GiB). Formal `VB_SOURCE_DENSITY.json` is **byte-identical**
> (SHA256 `a6409eec607a0c193cc213e4900722da9a28cffc85345f4f3b2afff19aff86f4`) to the independent
> reviewer's full run on the same rev2 code (job 163050).
>
> **Doc rev2 (this revision):** incorporates the independent scientific review of the published
> results — adds the same-method **absolute** density analysis (§4.2), **clarifies the floor
> semantics and retracts the "11/11 unmeasurable" claim** (§5.1), explains the Q1 `certificate`
> placeholder (§5.2), **tightens the mechanism claims** (§4.3), records the review **direction
> erratum** (§5.3), and registers the **asset status** accurately (§5.4). The numerical results
> and the code/math/binding APPROVE verdict are unchanged; the corrections are to interpretation
> and prose. Execution source is now published under `source/`.

## 1. Scope identity

| field | value |
|---|---|
| scope_id | `G20_V4_REPR_DENSITY_20261114` |
| phase | `G20_V4_REPR_DENSITY` |
| scope.json SHA256 | `4a1508712e66e356c26fd29189a0e182c13557d2d1ea9ff4794a52f361e571f7` |
| ROOT_SELF_AUDIT.txt SHA256 | `2a842a0efb381e216673e172870c0fb0455a79930ca6e401651bf5d4377500fe` |
| INDEPENDENT_REVIEW.json SHA256 | `a949cc7b979dfb163daa4c2eaf7dfad0dcd37af05b5039f9609dd166b403b3e8` |
| ACTIVE_RESEARCH.json SHA256 | `b8a770ef7ddee47e5ca9394460a9f307ba604be1cd2fb5505a3a4355d16ff9be` |
| formal job | **163279 EXIT 0** (anode19, elapsed 1122.25s, peak RSS 1.51 GiB) |
| consumes | `VB_DENSE22` (V2 rev2 job 161964 EXIT 0; bank verification PASS 31/31; manifest `f6f8e100`) |
| produces | `VB_LOCAL_FITS`, `VB_SOURCE_DENSITY` |
| new D4 full executions | **0** (pure local fitting + FFN-weight read + scoring) |
| new videos | **0** |

## 2. What V4 tests

Six models × {256, 512} row prefixes × six primary sites (high/low expert × layer 4/19/34),
fitted on **CAL_A11** (pooled, source-equal), scored on **DEV11** (full 512 panel).

- **CURRENT_Q1** — full 5120×5120 `A(α)=W1ᵀ diag(α) W2ᵀ`, R2 Problem PCG (verbatim), fresh fit.
- **STABLE_FREE_LINEAR** — free D×D correction `D` on the Q1 residual, R1b `stable_ridge_fit`
  verbatim (F64 weighted thin-SVD direct solve + eigh cross-check), `A_free=A_Q1+D`. **Base = A_Q1.**
- **L32** — linear-only `Φ=X@P32` (readout 32×32). **Base = A0 (Q0).**
- **DELTA_N32** — `Φ=[X@P32, tanh(X@P32)]`, `X=u_t−u_s`. **Base = A0.**
- **ENDPOINT_N32** — `Φ=[u_t@P32−u_s@P32, tanh(u_t@P32)−tanh(u_s@P32)]`. **Base = A0.**
- **L64_Q32** — linear-only `Φ=X@P64` (readout 64×32=2048, capacity control matching the N methods). **Base = A0.**

`A0` prior is **fresh per-config Q0** (user adjudicated 2026-11-14): `α_Q0_j=E[dZ_j dH_j]/E[dZ_j²]`
on the CAL training subset; common base shared across representations within a config; no
old-CAL4 reuse, no zero prior. `P/Q` fixed by CAL data SVD. Anti-leakage: 256 and 512 are
**separate fits**; DEV never enters any fit.

**Primary estimand** (§6): `M(c,e,m)` = equal-mean over layers {4,19,34} of `gatedSSE/(rows·5120)`;
`r=(M_m−M_CURRENT_Q1)/M_CURRENT_Q1`; expert-equal within source then source-equal. Scoring is the
**F64 gated pre-gate increment math lift** (`SSE=Σ((pred−Y)⊙g)²`, `Y=y_hit−y_anchor`); BF16
cache/gate replay is a **separate V4D** check. Six-site primary; all-80 is a pre-registered
separate later table.

## 3. Governance chain (complete)

1. **Author self-audit** (`ROOT_SELF_AUDIT.txt`, rev2 `2a842a0e`).
2. **Two material code bugs found by independent quantitative review of rev1, fixed in rev2:**
   - **F1 — readout source-weight scaling.** `fit_readout` put `w_c=1/(S·n_c)` on **both**
     Kronecker factors → normal matrix carried `w_c²` → `C` inflated by `1/w_c=S·n=2816`.
     **Fix:** weight once (on `GPhi_c`); `GQ_c=Qᵀ diag(g²) Q` unweighted. `v4_repr.py` → `8ead06e5`.
   - **F2 — floor BF16 dtype crash.** `cal_numerical_floor` called `bf16_replay` with fp32 `u0` ×
     bf16 `a` → RuntimeError. **Fix:** cast `u0/u1/y0/y1` to bf16, gate to fp32. `v4_runner.py` →
     `9c6d8940`. (Note: this fixed the *dtype crash*; the floor *semantics* issue is separate — §5.1.)
3. **Author smoke 162978 PASS 14/14** via the formal runner path.
4. **Genuinely independent review** — reviewer `qoder_independent_reviewer_g20_v4_rev2final_20261114`
   (non-author), **APPROVE / recheck APPROVE**, bound to scope `4a150871` + audit `2a842a0e`; full
   reviewer run job 163050 EXIT 0; own third re-aggregation matched all tables to max err 0.0.
5. **Author responses**, **mechanical ACTIVE freeze** (`AUTHORIZED_FOR_ONE_JOB`, session_max_gpus 2).
6. **Dispatch** via `dispatch_once_v4.py` (all guards passed) → **formal job 163279 EXIT 0**.

## 4. Formal results (job 163279 EXIT 0; byte-identical to the independent reviewer full run)

### 4.1 Primary estimand — relative risk vs Q1

`source_equal_relative` `r=(M_m−M_Q1)/M_Q1` (expert-equal then source-equal; **0 = equal to Q1,
positive = WORSE than Q1**):

| model | r @256 | r @512 | Δr (512−256) |
|---|---|---|---|
| CURRENT_Q1 | 0.0 | 0.0 | 0.0 |
| STABLE_FREE_LINEAR | 1.59851 | 2.20291 | +0.60440 |
| L32 | 0.04991 | 0.05517 | +0.00526 |
| DELTA_N32 | 0.04872 | 0.05369 | +0.00498 |
| ENDPOINT_N32 | 0.04607 | 0.05055 | +0.00448 |
| L64_Q32 | 0.03883 | 0.04278 | +0.00396 |

**Ranking (correct direction):** Q1 is best (r=0). The four constrained readouts are **WORSE than
Q1 by 3.88%–5.52%** (r=+0.0388…+0.0552). Among them, closest to Q1 (= best) is **L64_Q32**, then
**ENDPOINT_N32 < DELTA_N32 < L32** at both prefixes. `ratio_of_mean_risks` (256/512): L32
0.06980/0.07519; DELTA 0.06712/0.07307; ENDPOINT 0.06404/0.06877; L64 0.05354/0.05678;
STABLE_FREE_LINEAR 1.56798/2.12469.

### 4.2 Same-method ABSOLUTE density effect (added in doc rev2)

The registered `source_density_delta` above is **Δr** — how the *gap to Q1* changes. It does **not**
answer whether more calibration rows help a method in absolute terms. The same-method absolute DEV
MSE change (ratio of source-equal mean risks, common DEV512 panel; per-source improve counts) is:

| model | abs MSE @256 | abs MSE @512 | absolute change | DEV sources improved |
|---|---|---|---|---|
| CURRENT_Q1 | 1.5032e-2 | 1.4899e-2 | **−0.884%** | 11/11 |
| STABLE_FREE_LINEAR | 3.8603e-2 | 4.6556e-2 | **+20.603%** | 0/11 |
| L32 | 1.6082e-2 | 1.6020e-2 | **−0.385%** | 9/11 |
| DELTA_N32 | 1.6041e-2 | 1.5988e-2 | **−0.332%** | 9/11 |
| ENDPOINT_N32 | 1.5995e-2 | 1.5924e-2 | **−0.443%** | 10/11 |
| L64_Q32 | 1.5837e-2 | 1.5746e-2 | **−0.579%** | 10/11 |

**Correct interpretation:** increasing CAL rows 256→512 **improves Q1 and all four constrained
models in absolute DEV MSE**; **Q1 improves the most**. STABLE_FREE_LINEAR **worsens** (overfit).
The positive Δr for the constrained models in §4.1 means *Q1 improved more, so the gap to Q1
widened* — **not** that the constrained models got worse absolutely. Calibration density is
effective for Q1 and the constrained readouts; it is harmful for the free linear correction.

### 4.3 Mechanism claims — tightened (doc rev2)

The ranking is robust, but two confounds limit causal interpretation:
- **Base confound.** STABLE_FREE_LINEAR corrects the **Q1** base (`A_Q1+D`); the constrained
  readouts correct the **Q0** base (`A0`). "Free overfits while constrained generalizes" therefore
  cannot be attributed to degrees-of-freedom alone — the base differs. The free correction's poor
  DEV risk is consistent with overfitting (D has 5120² parameters vs ≤5632 CAL rows; certificates
  prove the solve is correct), but the base difference is a co-existing explanation.
- **Capacity confound.** **L64_Q32 (linear, matched readout capacity 2048) beats ENDPOINT_N32 on
  11/11 DEV sources.** A matched-capacity *linear* readout outperforms both nonlinear (tanh)
  variants, so the small ENDPOINT-vs-DELTA difference cannot be cleanly attributed to "endpoint
  state supplying information delta lacks." The ordering is consistent with linear
  structure/capacity, not with a nonlinearity benefit.

**Bounded conclusion:** no causal claim that endpoint state adds information, nor that
degrees-of-freedom alone cause overfitting, is supported by this design. These are registered
observations for the route's J_REPRESENTATION question, to be discriminated by later scopes.

### 4.4 Numerical health (all 12 configs, formal `VB_LOCAL_FITS.json`)

Q1 CONVERGED (max `true_relative_residual` 9.82e-6 < 1e-5 tol); free-ridge certificates pass
(max `cross_check_f64_eigh_rel_l2` 1.88e-11, `D_out_of_rowspace_rel_fro` 4.43e-12; D numerical rank
2816@256 / 5052–5112@512 = CAL rowspace); readout normal relative residuals ~1e-16; Q1
self-relative = 0 in both tables; independent aggregation max err 0.0; α_Q0/α_Q1 each 13824 finite
values per config; P/Q/α distinct across prefixes at all 6 sites. F1 real-data probe:
`C_code/C_correct=1.0` (dev 2e-12), grad_rel 2.1e-14, DEV ratio vs Q1 L32 1.1790 / DELTA 1.1678 /
ENDPOINT 1.1565 / L64 1.1294, all `rel_diff_vs_rev1_correct=0.0`.

## 5. Clarifications, errata, and asset status (doc rev2)

### 5.1 Floor semantics — material clarification; "unmeasurable" claim RETRACTED
The published `floors` are computed by `cal_numerical_floor` (v4_runner.py:121) via `bf16_replay`
(ridge_fit_core.py:48). In `bf16_replay`, `target = (y1·g).bfloat16()` is the **gated DENSE target**
and `gated` is the BF16-replay Q1 prediction. Hence
`floor = mean_CAL ‖BF16_replay_prediction − gated_dense_target‖²/(n·D)`
= the **total replay-vs-dense error** (Q1 model approximation error **plus** BF16 rounding), **not**
the pure BF16 cache/gate **rounding** noise the route §6 intended.
- **Evidence:** `floor / R_Q1,CAL ∈ [0.99936, 1.00043]` — the floor ≈ Q1's total CAL math risk
  (rounding alone would be ≪ total risk).
- **Consequence:** the registered small-denominator flag `M(c,e,Q1) < floor` fires for **11/11**
  sources because `floor ≈ Q1 risk` by construction, **not** because of a genuine
  small-denominator / noise-resolution problem. The earlier "11/11 sources unmeasurable"
  interpretation is **retracted as stated**.
- **Correct floor (follow-up):** the intended numerical noise floor isolates rounding,
  `‖BF16_replay_prediction − FP32_math_prediction‖²` (same Q1 model, differing only in BF16
  rounding). Recomputing it requires `A_Q1` (from α_Q1 + W1/W2) and the bank `u0/u1/y0/y1` → a
  follow-up reviewed scope (V4D or a correction scope). **Until then no "unmeasurable" claim is
  made.** The primary estimand is the F64 math lift and is unaffected by this issue.

### 5.2 Q1 `certificate: CERTIFICATE_UNRESOLVED` — vestigial placeholder, not a failure
In the verbatim R2 primitive `q1_fit_core.py:65`, the returned `certificate` field is the hardcoded
literal `'CERTIFICATE_UNRESOLVED'`, emitted **regardless of convergence**. It is a placeholder that
was never wired to a separate certificate computation in R2; it does **not** indicate solve failure.
The operative convergence certificate is `status: CONVERGED` together with `true_relative_residual`
(max 9.82e-6 < 1e-5). All 12 Q1 fits converged. (Field retained verbatim for R2 hash fidelity;
clarified here.)

### 5.3 Independent-review direction erratum
`INDEPENDENT_REVIEW.json` OBS-1 states the constrained readouts "**IMPROVE** DEV relative risk by
3.88%–5.52% vs Q1". **Direction error:** `r=+0.0388…+0.0552` means the readouts are **WORSE** than
Q1 by 3.88%–5.52%. Corrected wording: "the four constrained readouts are worse than Q1 by
3.88%–5.52%; ordering L64_Q32 < ENDPOINT_N32 < DELTA_N32 < L32 (closest to Q1 = best among
constrained)." OBS-3 "the readouts' relative **gain** over Q1 shrinks" should read "relative **loss**
vs Q1 grows slightly at 512." This is a prose direction error in non-blocking observations; the
review's APPROVE verdict on code/math/binding/anti-leakage is unaffected. Erratum recorded here
(the review JSON is hash-bound evidence and is preserved unchanged).

### 5.4 Asset status — fits published; full candidate export + reload PENDING
`VB_LOCAL_FITS.json` (9.55 MB) publishes, per config: **full α_Q0/α_Q1 arrays** (13824 finite each);
**P/Q SHA256 only** (not the matrices); readout **C norms/dimensions/solve diagnostics only**; free
**D norms/rank/numerical certificates only**. Therefore the asset tree status is:

> **Fits recorded and published; full candidate-predictor export (P/Q/C/D matrices or locatable
> parameter files) and close/reload/consumer acceptance are PENDING (V4D).**

These candidates **cannot be reloaded from this JSON alone**; reuse requires exporting the actual
P/Q/C/D (or α-derived A_Q1/A_free plus P/Q/C) and a reload acceptance check.

## 6. Boundaries / limitations
- `quality_metrics`: **NOT_MEASURED**. `performance_comparison`: **NOT_BENCHMARKED**.
- V4 fitting wall time (18.7 min) is **not** production timing; no observer-free production claim.
- **No deployment / noninferiority claim.** DEV11 supports only the next finite experiment decision.
- Six-site primary table; **all-80 is a separate pre-registered V4D table**. Full-matrix export and
  BF16 cache/gate replay acceptance are **V4D**, not this scope.
- No "unmeasurable" claim is made pending the correct rounding-floor recompute (§5.1).

## 7. Next steps
1. **Correct-floor recompute + full candidate export (V4D or correction scope):** export P/Q/C/D
   (and A_Q1/A_free), recompute the pure BF16 rounding floor (§5.1), and run close/reload/consumer
   acceptance. Clarify V4D's purpose — Q1 full-80 deployment chain vs advancing a new candidate (if
   the latter, justify the candidate and the question it answers).
2. **V3** — conditional CAL22 expansion **only if** source breadth changes the judgment. Current
   evidence does **not** warrant CAL22 expansion or new image-text pairs.
3. B-lane weight/state attribution (B-W / B-WS), independent of this A-lane operator result.

## 8. Evidence and source inventory
- `evidence/` — governance chain (final): `scope.json`, `ROOT_SELF_AUDIT.txt`,
  `INDEPENDENT_REVIEW.json` (preserved unchanged; erratum in §5.3), `ACTIVE_RESEARCH.json`,
  `AUTHOR_REVIEW_RESPONSES.json`; formal receipts `DISPATCH_INTENT.json`, `SUBMITTED.json`,
  `RUN.json`, **`EXIT.json`** (exit_code 0), `slurm-163279.out`/`.err`.
- `evidence/reviewer/` — independent reviewer full-run evidence on rev2 code: `REV2_FULL_RUN.json`,
  `REV2_PROBE.json`, `REV2_TINY_CORRECTED.json`, `SUMMARY.json`, `VB_SOURCE_DENSITY.json`.
- `output/` — **formal job 163279 outputs**: `SUMMARY.json`, `VB_SOURCE_DENSITY.json`
  (SHA `a6409eec…`), `VB_LOCAL_FITS.json` (9.55 MB raw fits + certificates).
- `source/` — **execution source** (hash-bound to scope.code_sha256): `v4_repr.py`, `v4_runner.py`,
  `q1_fit_core.py`, `ridge_fit_core.py`, `runtime_guard.py`, `guarded_runner_v4.py`,
  `dispatch_once_v4.py`, `job.sh`.

**Cluster paths:** formal stage
`/public/home/scc/pb23061276/projects/wan-quant/work/g20_v4_repr_20261114/stage`;
reviewer evidence
`/public/home/scc/pb23061276/projects/wan-quant/work/g20_v4_review_scratch_20261114/rev2`.