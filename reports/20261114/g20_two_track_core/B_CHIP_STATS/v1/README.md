# G20 B_CHIP_STATS — Fill the Chip GLOBAL_P64 Fixed64 Input Gap

**Status:** FORMAL_EXIT_0 · `B_CHIP_STATS_COMPLETE` · job 164928 · scope `G20_B_CHIP_STATS_20261114_V1`
**Direction B (rev5 TWO_CORE):** produce the real fixed64 P64 inputs that BW's 132 blocked Chip cells need, so the Chip (`CHIPMUNK_GLOBAL_P64`) method can be scored on the same DEV states as Q1/ICC.
**Boundary:** 11 existing DEV11 D4 captures, fill-only. **0 new image-text pairs, 0 new O40, 0 new video decode/scoring, 0 reference generation/benchmarks, 0 CAL_B, 0 param grids, 0 full-layer video.** quality `NOT_MEASURED`, performance `NOT_BENCHMARKED`. The 11 trajectory byproduct videos are **NOT decoded**.

## 1. The gap this fills

- BW v3 left **132 DEV Chip cells `BLOCKED_INPUT_GAP`**: the fixed64 selector rows are absent from the PANEL512 bank. The confirmed gap is **11×64 − 8 = 696 (DEV source × fixed64 token-ID) pairs** (8 reusable overlaps), i.e. 696 missing source/token identities — **not** 696 total tensor rows across sites.
- **Each required ID needs real U0/U1 input values at SIX sites × TWO endpoints.** This scope captures, per DEV source, 6 sites (high04/high19/high34/low04/low19/low34) × 2 endpoints (anchor step + hit step) = **12 captures**, each a `U_fixed64` of shape **(64, 5120) BF16**.
- Total: **11 sources × 64 tokens × 12 (site,endpoint) = 8448 fixed64 U rows**, persisted as **132 `.pt` payloads (781,265,364 bytes ≈ 745 MiB)**. Each payload also carries `U_panel_check` (512, 5120) + gate + token indices for the same-state gate.

## 2. Fixed64 binding (verbatim from CHIPMUNK_GLOBAL_P64)

- `fixed64_token_ids` = CPU `torch.Generator(seed=42).randperm(31668)[:64]` — **NOT** the panel first64 (`rows_are_panel_first64=false`).
- `fixed64_token_id_sha256_le_int64 = d53fcf1b0930d6b1ea78b3f20b11dccb7b1c9d27abcad922866b2f26c78bf895` (matches `B_BINDINGS.json`).
- Numeric adapter `OWNW_LOCAL512_P64ANCHOR64_V1`; `selected_k=5120`; grid `[21,29,52]`; `flattened_tokens=31668`; parent recipe `ee780352…` (VB_RECIPE). One global hidden-column mask.

## 3. Same-state gate (all 11 sources PASS)

- For every site and **both** endpoints, the captured `U_panel_check` (PANEL512 U) **and** gate **bit-match the landed V2 bank** (`anchor_panel_u_bit_match`, `hit_panel_u_bit_match`, `anchor_gate_bit_match`, `hit_gate_bit_match` all true). This proves the captured fixed64 rows come from the **same native state** as the bank Q1/ICC already use — no re-simulation drift.
- `same_state_pass_sources=11`, `same_state_fail_sources=[]`. A mismatch would block only the affected source's Chip cells (none occurred).
- Bank manifest `f6f8e100da6f75b2db612edd8c8c4d21729ed9bf99bfb5092941f2381a4d0bee`; inputs manifest `7eeb32ab400a1090b85d54bf13395012175d5c27683a1d754444328c4c87999a`.

## 4. Result and what it unblocks

- `bw_chip_cells_unblocked = 66 = expected` (11 sources × 6 sites). These feed **both** weights → **132 BW Chip cells** (66 × 2), the exact count BW v3 blocked.
- `new_D4_full_executions=11`, `sources=11`, `sites=6`, `target_captures_per_source=12`, `total_fixed64_U_rows_captured=8448`.
- **Reuse, not rerun:** the V2 `NativeTap` hook and pipeline binding are reused verbatim; completed Q1/ICC fits and results are reused directly. This scope only adds the missing fixed64 inputs.
- **Next:** BWS (composite native-state four-cell follow-up, ≤6 O40) and the direction-B unified weight+state explanation consume these inputs. The P64 gap blocks only the corresponding Chip cells, **not** BWS Q1/ICC.

## 5. Checks (all pass)

- **Smoke** job 164771 `SMOKE_CHIP_PASS` on the real formal path (1 source, 62 missing): `CAPTURE_OK sites=12`, `PAYLOAD_OK U_fixed64=(64,5120) bf16 finite; token order matches frozen fixed64`, `SAME_STATE_OK all 12 endpoints PANEL512 U + gate bit-match landed bank`, `AGGREGATE_OK cells_unblocked=6 serialize=OK`.
- **Prelaunch** 16/16 read-only gate checks `PRELAUNCH_PASS`.
- **Post-run** `VERIFY_POST_CHIP_PASS` (compute-node srun, read-only): executed copy re-hashes to scope (7 code + 3 frozen), all **132 `.pt` sha256 match the manifest** and byte sizes match, 8448 rows, 11 sidecars `capture_sites=12`, torch spot-check `U_fixed64 (64,5120) bf16 finite` + `U_panel_check (512,5120)`. See `evidence/verify_post_chip.log` + `evidence/verify_post_chip.sh`.

## 6. Reproduction / governance

- `source/` — execution code (verified vs `scope.code_sha256`); `evidence/` — scope, RUN/EXIT/SHELL_EXIT, independent review (APPROVE/APPROVE), ACTIVE, author response, self-audit, dispatch records, slurm logs, smoke/cluster evidence, post-run verification; `output/` — SUMMARY, B_CHIP_P64_INPUTS manifest, 11 CHIP_CAPTURE sidecars, EFFECTIVE_CONFIG.
- **Large tensors stay on the cluster, hash-bound.** `output/B_CHIP_P64_INPUTS.json` records each payload's cluster `path` / `bytes` / `file_sha256` / `U_fixed64_shape` / dtype; consumers reload from those paths. The 11 byproduct videos also stay on cluster (not decoded).
- Independent reviewer `qoder_independent_reviewer_g20_b_chip_stats_20261114` (not the author). Formal job 164928 EXIT 0, elapsed 1298.2 s, peak RSS 98.07 GiB, one A100-80GB (anode19).
- See `PUBLICATION_MANIFEST.json` for the full file/byte/SHA binding.