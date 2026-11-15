# G20 BW_CHIP_SCORE — Chip (GLOBAL_P64) Risk Fill and Weight-State Effect

**Status:** FORMAL_EXIT_0 · `BW_CHIP_SCORE_COMPLETE` · job 166479 · scope `G20_BW_CHIP_SCORE_20261115_V2`
**Direction B (rev5 TWO_CORE):** completes the BW Chip column — scores the 132 previously blocked `CHIPMUNK_GLOBAL_P64` DEV cells and diffs them against the completed Q1/ICC table.
**Boundary:** sampled local FFN function replay on the shared D4 PANEL512 states (mu_D) with the produced fixed64 P64 selector rows. **0 new D4, 0 new O40, 0 new videos, 0 video decode/scoring, 0 reference generation/benchmarks, 0 Q1/ICC refit or replay.** quality `NOT_MEASURED`, performance `NOT_BENCHMARKED`. n=11 mechanism development; no deployment claim.

## 1. What this stage does (fill-only)

B_CHIP_STATS produced the fixed64 selector inputs but never computed the Chip selector/prediction/risk; the BW runner wrote all 132 Chip DEV cells as `BLOCKED_INPUT_GAP`. This stage connects the produced inputs to the **actual** `CHIPMUNK_GLOBAL_P64` path and fills exactly those 132 cells:

- **132 Chip cells** = 11 DEV × 6 sites (high04/high19/high34/low04/low19/low34) × 2 weights (W_O, W_D).
- **Reuses verbatim** the proven `chip_predict` / `primary_risk` / `auxiliary_risk` / `build_pack` / `own_w_targets` / `chip_fixed64` from the byte-identical `b_common_replay` core (the same functions BW job 164312 used), the same hash-gated W_O/W_D checkpoints, and the same-source/site D4 hit gate `G_ref`.
- **Reuses the 264 completed Q1/ICC risks** read from the frozen hash-gated `BW_CELL.csv` (`gated_mse_f64`); **no Q1/ICC refit, no parameter reload, no replay**.
- **fixed64 inputs** read from `B_CHIP_P64_INPUTS.json` (sha `7eeb32ab…`, capture_scope `13df6528…`); each of the 264 anchor/hit payloads verified by path/bytes/file_sha256 + (64,5120) BF16 finite before use. fixed64 token IDs regenerated (CPU seed42 `randperm(31668)[:64]`) and asserted equal to the binding.
- **Primary risk** = F64 gated pre-gate SSE/(512·5120); **auxiliary** = BF16 gate MSE, kept separate.
- **Contrast:** `d_Chip(W) = R_Chip(W) − R_Q1(W)` (negative = Chip better than Q1); `theta_Chip = d_Chip(W_D) − d_Chip(W_O)`. Aggregation layer-equal (4,19,34) → expert-equal (high,low) → source-equal over 11 DEV; range + leave-one-source-out.

## 2. Primary result (DEV source-equal mean gated MSE)

| weight | R_Chip | R_Q1 (reused) | d_Chip = R_Chip − R_Q1 |
|---|---|---|---|
| W_O | 0.0115157 | 0.0149025 | **−0.0033868** |
| W_D | 0.0114975 | 0.0148998 | **−0.0034023** |

| contrast | value |
|---|---|
| **theta_Chip = d_Chip(W_D) − d_Chip(W_O)** | **−1.553e-5** |
| d_ICC(W_O) reused (cross-check vs BW) | +0.0065254 |
| d_ICC(W_D) reused (cross-check vs BW) | +0.0065387 |

- **Chip BEATS Q1 for both weights** (d_Chip ≈ −0.0034; Chip risk ~0.0115 vs Q1 ~0.0149, ~23% lower). This is the **opposite sign** to ICC, which is worse than Q1 (d_ICC ≈ +0.0065). On this sampled replay the method ranking is **Chip < Q1 < ICC** (Chip best, ICC worst).
- **All 11 per-source d_Chip are negative for both weights** (W_O range −0.005313 … −0.001536; W_D −0.005289 … −0.001591) — Chip-better-than-Q1 is consistent across every DEV source, not driven by one.
- **theta_Chip ≈ 0** (−1.55e-5, negligible vs the ~3.4e-3 Chip-vs-Q1 gap). The distilled-vs-original weight change has **essentially no effect on the Chip-vs-Q1 method gap**, mirroring theta_ICC ≈ 0 (+1.33e-5) from BW.
- **Robustness:** per-source theta_Chip is 9/11 negative (2 small positive: vbi_0056 +2.47e-5, vbi_1040 +2.86e-6); **all 11 leave-one-source-out theta_Chip are negative** (−1.954e-5 … −1.160e-5) — a robust tiny-negative aggregate sign with no meaningful weight effect.
- **Reuse cross-check:** the reused d_ICC (+0.0065254 / +0.0065387) reproduces the BW publication (+6.525e-3 / +6.539e-3) exactly, confirming the Q1/ICC table was consumed unchanged.

## 3. Checks (all pass)

- 132/132 Chip cells VALID, distinct (source,weight,expert,layer) keys; 264/264 fixed64 payloads verified; all primary/auxiliary risks finite; `d_chip_minus_q1`, `d_icc_minus_q1`, and `gated_sse = prim·n·D` identities hold per cell.
- Independent post-run cross-check (`evidence/verify_post_bw_chip.log`): aggregate per-source risk/d_chip/d_icc/theta/mean/range/LOO reproduce from `BW_CHIP_CELL.csv` to 1e-12; `VERIFY_POST_BW_CHIP_PASS`.
- Zero consumption confirmed in SUMMARY/CHECKS: `new_D4/O40/videos/decode/scoring/source_pairs = 0`, `q1icc_refit_or_replay = false`, `reused_bw_q1icc_cells = 264`, `new_trajectories = 0`.
- Pre-dispatch author verify (`evidence/verify_bw_chip_v2.log`): `PY_COMPILE_OK 13`, `BOUND_HASHES_OK 14`, `CLUSTER_INPUT_HASHES_OK`, `FIXED64_BINDING_OK`, **`CELL_ROW_BF16_OK`** (formal `cell_row` exercised with real BF16 tensors + JSON/CSV serialization), `AGGREGATE_CHIP_DRYRUN_OK`.

## 4. Version history (fix-before-rerun; failed consumption accounted)

- **V1** job 166419 (scope `3cb0b3c6…`) — **failed EXIT 1**: the author re-implemented `sha_tensor` with `.numpy()` on BFloat16 (`TypeError: Got unsupported ScalarType BFloat16`) at `cell_row` — the exact bug BW v3 had already fixed. The V1 verify only dry-ran `aggregate_chip` on synthetic float dicts and never exercised the real `cell_row → sha_tensor` BF16 path; the independent review also missed it. **0 trajectories consumed** (pure scoring; `output/` created empty before the crash). V1 artifacts preserved read-only on the cluster (`…/BW_CHIP_SCORE/v1/stage`).
- **V2** job 166479 (scope `202bbbe3…`) — **EXIT 0**. `sha_tensor` replaced with the proven verbatim BW v3 uint8-view body (`…reshape(-1).view(torch.uint8)`); only `bw_chip_score_runner.py` re-bound (`72b15745…`→`7d2b0fb8…`), the 9 shared-core files + guards + B_BINDINGS byte-identical. Added the `CELL_ROW_BF16_OK` real-path regression so the crashed path is covered before dispatch. Independent recheck APPROVE/APPROVE.
- **Lesson recorded:** "reuse proven core" exempts only byte-identical *imported* functions from real-path verification — never new author-written code in the output path. A cheap scoring job still needs a tight CPU regression that calls each new formal function with real dtypes; the formal job is not free verification for new code.

## 5. What remains / boundary

- This completes the BW **Chip** column (132 cells). Direction B's weight track now has all three methods (Q1, ICC, Chip) on mu_D: **Chip < Q1 < ICC**, both method gaps essentially weight-independent (theta_ICC ≈ +1.3e-5, theta_Chip ≈ −1.6e-5).
- Whether these sampled-replay rankings transfer to full-M native video / free-trajectory / system speed — **not measured here**.
- **BWS** (composite native-state four-cell follow-up, mu_D vs mu_O) is the separate state track; its Q1/ICC/Chip comparison does not depend on this fill. BWS formal job 166225 completed EXIT 0 (publication/closure handled separately).

## 6. Reproduction / governance

- `source/` — execution code (13 files verified vs `scope.code_sha256`) + `B_BINDINGS.json` (frozen input, verified vs `scope.frozen_inputs_sha256`); `evidence/` — scope, RUN/EXIT, independent review (APPROVE/APPROVE), ACTIVE, author response, self-audit, dispatch records, slurm logs, pre/post verify logs; `output/` — SUMMARY/CHECKS/BW_CHIP_AGGREGATE/BW_CHIP_CELL/BW_CHIP_SOURCE.
- Independent reviewer `qoder_independent_reviewer_g20_bw_chip_20261115` (not the author). Formal job 166479 EXIT 0, elapsed 345.8 s, peak RSS 1.86 GiB, one A100-80GB (anode16).
- See `PUBLICATION_MANIFEST.json` for the full file/byte/SHA binding.