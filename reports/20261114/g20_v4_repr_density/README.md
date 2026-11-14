# G20 V4 — Representation / Row-Density Comparison — FORMAL RESULTS (EXIT 0)

**Status: FORMAL job 163279 EXIT 0.** Author: `qoder_root_g20_20261111`.
Date: 2026-11-14. Route: `Qesire/FFNDS` main `4107f75f15ca9982aab2f3d80401260dd213ec5d`,
`research/20261113/G20_VBENCH_RESTART_V4_20261113.md` §4.3 + §6; queue V4.

> The formal dispatched job **163279** completed **EXIT 0** (status `V4_REPR_DENSITY_COMPLETE`,
> elapsed 1122.25s, anode19, A100-80GB, peak RSS 1.51 GiB). Its `VB_SOURCE_DENSITY.json` is
> **byte-identical** (SHA256 `a6409eec607a0c193cc213e4900722da9a28cffc85345f4f3b2afff19aff86f4`)
> to the independent reviewer's full V4 run on the same rev2 code (job 163050) — the formal run
> exactly reproduces the independently verified results. This publication includes the formal
> `EXIT.json`, `VB_LOCAL_FITS.json`, `VB_SOURCE_DENSITY.json`, and `SUMMARY.json`.

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
  verbatim (F64 weighted thin-SVD direct solve + eigh cross-check), `A_free=A_Q1+D`.
- **L32** — linear-only `Φ=X@P32` (readout 32×32).
- **DELTA_N32** — `Φ=[X@P32, tanh(X@P32)]`, `X=u_t−u_s`.
- **ENDPOINT_N32** — `Φ=[u_t@P32−u_s@P32, tanh(u_t@P32)−tanh(u_s@P32)]` (state through per-endpoint tanh).
- **L64_Q32** — linear-only `Φ=X@P64` (readout 64×32=2048, capacity control matching the N methods).

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
2. **Two material bugs found by independent quantitative review of rev1, fixed in rev2:**
   - **F1 — readout source-weight scaling.** `fit_readout` put `w_c=1/(S·n_c)` on **both**
     Kronecker factors → normal matrix carried `w_c²` → `C` inflated by `1/w_c=S·n=2816`
     (reviewer measured `C_code/C_correct=2816`, DEV MSE ~19308 vs ~0.012, grad_rel 73 vs 1.6e-14).
     **Fix:** weight appears **once** (on `GPhi_c` only); `GQ_c=Qᵀ diag(g²) Q` unweighted.
     `v4_repr.py` → `8ead06e5`.
   - **F2 — floor BF16 dtype crash.** `cal_numerical_floor` called `bf16_replay` with fp32 `u0` ×
     bf16 `a` → RuntimeError (rev1 smoke never exercised the floor path). **Fix:** cast
     `u0/u1/y0/y1` to bf16, gate to fp32. `v4_runner.py` → `9c6d8940`. Added `cal_floor_finite`
     and `readout_mse_sane_vs_Q1` smoke checks.
3. **Author smoke 162978 PASS 14/14** via the formal runner path; readout DEV ratios match the
   reviewer's independent correct values exactly.
4. **Genuinely independent review** — reviewer `qoder_independent_reviewer_g20_v4_rev2final_20261114`
   (non-author), **disposition APPROVE / recheck_disposition APPROVE**, bound to scope `4a150871`
   + audit `2a842a0e`. The reviewer ran a **full V4 run on a compute node** (job 163050, anode19,
   elapsed 1146.6s, JOB_RC=0), re-hashed the bundle local+remote, proved the executed copy is
   byte-identical to rev2, and did its **own third Node re-aggregation from raw DEV scores matching
   all published tables to max err 0.0**.
5. **Author responses** (`AUTHOR_REVIEW_RESPONSES.json`), **mechanical ACTIVE freeze**
   (`ACTIVE_RESEARCH.json`, `AUTHORIZED_FOR_ONE_JOB`, session_max_gpus 2).
6. **Dispatch** via `dispatch_once_v4.py` (all guards passed) → **formal job 163279 EXIT 0**.

### Reviewer adjudication of three flagged driver checks (all reviewer-test artifacts, not code bugs)
- `build_24cal_12dev`: runner does 12 CAL + 12 DEV builds = 24 total (`build_calls=24`); the test
  wrongly expected 24 CAL builds. Correct is 12 CAL (one per config).
- `cal_prefixes_correct`: same miscount — actual CAL prefixes `[256]*6+[512]*6` (12 entries), test
  expected 24. Anti-leakage independently proven (120/120 fit calls CAL-only, `leaks=[]`, DEV scored
  512-only, P/Q/α_Q0 distinct across prefixes at all 6 sites).
- `readout_ratios_sensible`: test applied a ratio threshold `(1,2)` to the **relative risk** `r`
  (~0.039–0.055); the ratio is `1+r`≈1.039–1.055 ⊂ (1,2). Readouts are sensible.

## 4. Formal results (job 163279 EXIT 0; byte-identical to the independent reviewer full run)

**Primary estimand — `source_equal_relative`** `r=(M_m−M_Q1)/M_Q1` (expert-equal then source-equal;
0 = equal to Q1, positive = worse than Q1):

| model | r @256 | r @512 | density Δ (512−256) |
|---|---|---|---|
| CURRENT_Q1 | 0.0 | 0.0 | 0.0 |
| STABLE_FREE_LINEAR | 1.59851 | 2.20291 | +0.60440 |
| L32 | 0.04991 | 0.05517 | +0.00526 |
| DELTA_N32 | 0.04872 | 0.05369 | +0.00498 |
| ENDPOINT_N32 | 0.04607 | 0.05055 | +0.00448 |
| L64_Q32 | 0.03883 | 0.04278 | +0.00396 |

**Secondary — `ratio_of_mean_risks`** (256 / 512): L32 0.06980/0.07519; DELTA_N32 0.06712/0.07307;
ENDPOINT_N32 0.06404/0.06877; L64_Q32 0.05354/0.05678; STABLE_FREE_LINEAR 1.56798/2.12469.

**Numerical health (all 12 configs, formal `VB_LOCAL_FITS.json`):** Q1 CONVERGED (true relative
residual <1e-5); free-ridge certificates pass (max `cross_check_f64_eigh_rel_l2` 1.88e-11,
`D_out_of_rowspace_rel_fro` 4.43e-12; D numerical rank 2816@256 / 5052–5112@512 = CAL rowspace);
readout normal relative residuals ~1e-16; Q1 self-relative = 0 in both tables; independent
aggregation max err 0.0; floors finite positive (high04_256 floor 0.00948068833486425). F1 probe on
real data: `C_code/C_correct=1.0` (dev 2e-12), grad_rel 2.1e-14, DEV ratio vs Q1 L32 1.1790 / DELTA
1.1678 / ENDPOINT 1.1565 / L64 1.1294, all `rel_diff_vs_rev1_correct=0.0`.

### Scientific signal (bounded; DEV11 supports only the next finite decision)
- The constrained representations are **~4–5.5% worse than full Q1** on DEV (aggregated). Among them
  **L64_Q32 is best** (lowest r), and **ENDPOINT_N32 is slightly better than DELTA_N32** at matched
  capacity — consistent with the route expectation that endpoint state carries information delta-only
  lacks, and that the linear capacity control (L64) is the strongest readout variant.
- **STABLE_FREE_LINEAR overfits CAL** (r=1.60@256 → 2.20@512; the free D×D correction has far more
  parameters than CAL rows and the relative ridge `ρ=λ/‖A_Q1‖²` is weak). The certificates prove the
  solve is correct; the poor DEV risk is a **legitimate finding**, not a defect. More CAL rows increase
  the overfit (density Δ +0.604).
- **11/11 DEV sources flagged `q1_denominator_below_floor`** (registered small-denominator flag, §6);
  carry to V4D. Relative quantities for those sources are flagged unmeasurable against the BF16 floor.

## 5. Boundaries / limitations
- `quality_metrics`: **NOT_MEASURED**. `performance_comparison`: **NOT_BENCHMARKED**.
- V4 fitting wall time is **not** production timing; no observer-free production claim.
- **No deployment / noninferiority claim.** DEV11 supports only the next finite experiment decision.
- Six-site primary table; **all-80 is a separate pre-registered V4D table**. Full-matrix export
  (`A_Q1`/`D`) and BF16 cache/gate replay acceptance are **V4D**, not this scope.

## 6. Next steps
1. **V4D** — full-80 export + close/reload/consumer + BF16 replay acceptance (separate scope).
2. **V3** — conditional CAL22 expansion only if source breadth changes the judgment.
3. B-lane weight/state attribution (B-W / B-WS) independent of this A-lane operator result.

## 7. Evidence inventory
- `evidence/` — governance chain (final): `scope.json`, `ROOT_SELF_AUDIT.txt`,
  `INDEPENDENT_REVIEW.json`, `ACTIVE_RESEARCH.json`, `AUTHOR_REVIEW_RESPONSES.json`;
  formal dispatch/run receipts `DISPATCH_INTENT.json`, `SUBMITTED.json`, `RUN.json`, **`EXIT.json`**
  (exit_code 0), `slurm-163279.out`/`.err`.
- `evidence/reviewer/` — independent reviewer full-run evidence on rev2 code: `REV2_FULL_RUN.json`,
  `REV2_PROBE.json` (F1 real-data probe), `REV2_TINY_CORRECTED.json` (PASS), `SUMMARY.json`,
  `VB_SOURCE_DENSITY.json`.
- `output/` — **formal job 163279 outputs**: `SUMMARY.json`, `VB_SOURCE_DENSITY.json`
  (SHA `a6409eec…`, byte-identical to reviewer), `VB_LOCAL_FITS.json` (9.5MB raw fits + certificates).

**Cluster paths:** formal stage
`/public/home/scc/pb23061276/projects/wan-quant/work/g20_v4_repr_20261114/stage`;
reviewer evidence
`/public/home/scc/pb23061276/projects/wan-quant/work/g20_v4_review_scratch_20261114/rev2`.