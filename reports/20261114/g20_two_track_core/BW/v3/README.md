# G20 BW — Weight-State Effect on Fixed Method Gaps (BW)

**Status:** FORMAL_EXIT_0 · `BW_Q1ICC_COMPLETE` · job 164312 · scope `G20_BW_20261114_V3`
**Direction B (rev5 TWO_CORE):** explain Q1r2 / actual ICC / actual GLOBAL_P64 differences under two weight sets and composite native states.
**Boundary:** sampled local FFN function replay on shared D4 PANEL512 states (mu_D). **0 new D4, 0 new O40, 0 new videos, 0 reference generation/benchmarks.** quality `NOT_MEASURED`, performance `NOT_BENCHMARKED`. **theta sign does NOT gate BWS.**

## 1. Fixed design

- **Two weights:** `W_O` (original FFN, F32 storage → BF16 round-to-nearest-even once) and `W_D` (distilled BF16). Each weight recomputes its **own** anchor/target on the identical 512-row `native_ffn` panel — no mixing of a full-M D target with a sampled O target.
- **Shared state:** same D4 PANEL512 U and same `G_ref` (same-source/site D4 hit gate) for both weights and all methods.
- **Methods:** Q1 (`Q1R2_CAL11_P512_OWNW`, refit per W/site, **12 fits all CONVERGED**, true relative residual ≤1e-5) and ICC (`ICC_SVD_R512_TWO_RESIDUAL`, ordinary SVD rank 512, gesvd, **no CD scales**; W_D factors reused only on checkpoint/method/rank/hash match, W_O produced fresh by the same scheme). **Chip (`CHIPMUNK_GLOBAL_P64`) DEV cells are BLOCKED_INPUT_GAP** this scope (see §4).
- **Primary risk** = F64 gated pre-gate SSE/(512·5120); **auxiliary** = BF16 gate MSE, kept separate (no mixed ranking).
- **Contrast:** `d_m(W) = R_m(W) − R_Q1(W)` (positive = comparator worse than Q1); `theta_m = d_m(W_D) − d_m(W_O)`. Aggregation layer-equal → expert-equal → source-equal; 11 paired sources + leave-one-source-out; high/low separate.
- **Cells:** 11 DEV × 6 sites × 2 weights × 2 methods = **264 valid Q1/ICC cells**; 132 Chip cells blocked (396 total expected).

## 2. Primary result (DEV source-equal mean gated MSE)

| weight\method | Q1 | ICC |
|---|---|---|
| W_O | 0.0149025 | 0.0214279 |
| W_D | 0.0148998 | 0.0214386 |

| contrast | value |
|---|---|
| d_ICC(W_O) = R_ICC − R_Q1 | **+6.525e-3** |
| d_ICC(W_D) = R_ICC − R_Q1 | **+6.539e-3** |
| **theta_ICC = d(W_D) − d(W_O)** | **+1.329e-5** |

- **ICC is worse than Q1 for both weights** (d_ICC ≈ +0.0065, i.e. ICC risk ~0.0214 vs Q1 ~0.0149, ~44% higher). Q1 beats ICC on this sampled replay.
- **theta_ICC ≈ 0** (+1.33e-5, negligible vs the ~6.5e-3 method gap). The distilled-vs-original weight change has **essentially no effect on the ICC-vs-Q1 method gap**.
- **Robustness:** all 11 per-source theta_ICC positive (5.7e-6 … 2.2e-5); all 11 leave-one-source-out theta_ICC positive (1.24e-5 … 1.41e-5) — consistent tiny-positive sign, **no sign flip**. Q1 baseline risk is nearly identical across weights (0.0149025 vs 0.0148998).

## 3. Checks (all pass)

- 12/12 Q1 fits CONVERGED (≤1e-5); ICC W_D factor checkpoint/method/rank/hash identity + W_O fresh ordinary-SVD shape asserts; shared G_ref; all 264 primary/auxiliary risks finite.
- **Reloadable Q1/ICC parameters** (handoff §5/A6): real tensors saved per site|weight (`alpha_Q1`+`A_q1_f32` FP32; four ICC BF16 factors), `weights_only=True` reload verified, **scoring uses the reloaded artifact**; 12 files ~1.72 GB total recorded in `output/PARAMS_MANIFEST.json` (path/bytes/sha256/shapes). Large tensors remain on cluster; hashes bind them here.
- `BW_CELL.csv` carries full payload identity (U0/U1/G_ref/own_anchor/own_target/pred BF16 hashes, `row_indices_sha` from the frozen panel token-id hash, per-cell `fit_sha` from the reloaded alpha). `BW_COST.csv` reports all five cost fields (cost reported, not matched across methods).

## 4. What remains unidentified / blocked

- **Chip GLOBAL_P64: 132 DEV cells BLOCKED_INPUT_GAP.** The fixed64 selector rows (CPU seed42 `randperm(31668)[:64]`) are absent from PANEL512 for all 11 DEV sources (696 missing rows). Unblocked by **B_CHIP_STATS** (separate concrete scope + ACTIVE, max 11 D4, fill-only). B remains PARTIAL until Chip is filled.
- Whether the Q1>ICC gap and weight-independence transfer to full-M native video / free-trajectory / system speed — not measured here.
- **BWS** (composite native-state four-cell follow-up) is the fixed next step after BW_Q1ICC_COMPLETE, independent of theta sign; separate scope + ACTIVE, max 6 O40.

## 5. Version history (fix-before-rerun; failed consumption accounted)

- **v1** job 164189 — failed: reloadable params not persisted (review F1). Fixed.
- **v2** job 164225 — failed: `sha_tensor` `.numpy()` on BFloat16, then `aggregate_bw` method-key mismatch. Both fixed.
- **v3** job 164312 — **EXIT 0**. Smoke hardened to exercise the full aggregation+serialization integration path (`AGGREGATE_OK` on a complete grid) before dispatch. v1/v2 artifacts preserved read-only on the cluster.

## 6. Reproduction / governance

- `source/` — execution code (verified vs `scope.code_sha256`); `evidence/` — scope, RUN/EXIT, independent review (APPROVE/APPROVE), ACTIVE, author response, self-audit, dispatch records, slurm logs; `output/` — SUMMARY/CHECKS/BW_AGGREGATE/PARAMS_MANIFEST/BW_CELL/BW_SOURCE/BW_COST.
- Independent reviewer `qoder_independent_reviewer_g20_bw_v3recheck_20261114` (not the author). Formal job 164312 EXIT 0, elapsed 460.9 s, peak RSS 1.86 GiB, one A100-80GB.
- See `PUBLICATION_MANIFEST.json` for the full file/byte/SHA binding.