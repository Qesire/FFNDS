# G20 A — Q1 Residual Increment Operator (A_Q1_RESIDUAL)

**Status:** FORMAL_EXIT_0 · `A_Q1_RESIDUAL_COMPLETE` · job 164072 · scope `G20_A_Q1_RESIDUAL_20261114_V1`
**Direction A (rev5 TWO_CORE):** is there a useful FFN increment operator around the published Q1 baseline?
**Boundary:** sampled local FFN function replay on the landed VB_DENSE22 bank. **0 new D4, 0 new O40, 0 new videos, 0 reference generation/benchmarks.** quality `NOT_MEASURED`, performance `NOT_BENCHMARKED`. No deployment claim; the research lead makes a deployment-value judgment after this report.

## 1. Fixed recipe (four arms, no others)

| arm | meaning |
|---|---|
| `CURRENT_Q1` | published V4 Q1 baseline (C=0 increment) |
| `Q1_PLUS_LINEAR64_Q32` (L) | Q1 + linear readout on a 64-dim P64 basis + 32-dim **pre-gate residual** Q32 basis |
| `Q1_PLUS_DELTA_N32` (D) | Q1 + 32-dim delta-nonlinear residual arm |
| `Q1_PLUS_ENDPOINT_N32` (E) | Q1 + 32-dim endpoint-nonlinear residual arm |

- **Q1 is rebuilt from the published V4 alpha** (`A_q1 = (W1ᵀ·alpha)·W2ᵀ`, FP32→F64); **no PCG refit**. Ancestry verified against the published V4 gated MSE: **132/132 cells match exactly** (6 sites × 22 sources).
- **Q32 is the PRE-GATE residual basis** (`Y − X·A_q1`), not gated (the key rev5 change). P64 from the weighted stacked X.
- Fixed center/scale `b/sigma`, tanh nonlinearity, per-arm CAL RMS gamma. Readout via the Kronecker normal-equation solver (source weight applied once, `rho = 1e-6·trace(K)/2048`); CAL normal-equation relative residual ~1.8e-15.
- **Math table = F64 gated pre-gate increment lift; BF16 table = separate local cache/operator/gate replay.** The two are never mixed into one ranking. 528 rows per table (4 arms × 6 sites × 22 sources).

## 2. Primary result (DEV source-equal mean gated MSE; layer-equal → expert-equal → source-equal)

| arm | math | bf16 | R_Q1 − R_arm (math) |
|---|---|---|---|
| CURRENT_Q1 | 0.0148995 | 0.0148987 | — |
| Q1_PLUS_LINEAR64_Q32 (L) | 0.0142926 | 0.0142924 | **+6.069e-4** |
| Q1_PLUS_DELTA_N32 (D) | 0.0143781 | 0.0143777 | +5.214e-4 |
| Q1_PLUS_ENDPOINT_N32 (E) | 0.0143526 | 0.0143532 | +5.469e-4 |

- **All three residual arms improve over CURRENT_Q1** (positive `R_Q1 − R_arm`). **L is best** (~4.1% relative lift), then E, then D.
- Arm ordering: `R_L − R_E = −6.0e-5` (L better than E), `R_D − R_E = +2.5e-5` (D worse than E).
- **Math and BF16 agree in sign and magnitude** (BF16 replay is faithful to the F64 math lift).
- Direction is reported as positive/negative/zero with the 11 leave-one-source-out signs; no fabricated SMALL threshold, no engineering pass/fail gate. A DEV regression or a single non-improving CAL source is **not** an engineering failure.

## 3. Checks (all pass)

- ancestry 132/132 match published V4; C=0 increment exactly zero in **both** math and BF16 (`torch.equal`).
- CAL objective `J0 → Jfit` decrease per arm; readout normal relative residual ~1.8e-15.
- `bf16_finite=true`; BF16 export consistency (A_Q1/A_L) + save→reload round-trip verified; final scoring recomputes from the F64 fit + reloaded params.
- Reloadable per-site parameter files (6 sites, ~3.17 GB total) recorded in `output/A_Q1_RESIDUAL_PARAMS.json` (path/bytes/sha256/tensor shapes). Large tensors remain on the cluster; hashes bind them here.

## 4. What remains unidentified

- Whether the ~4% sampled-replay lift transfers to full-M native video / free-trajectory / system speed — **not measured here** (no new D4/O40/video). Deployment value is a separate research-lead judgment.
- All-80-site export and whole-model neutrality are out of this scope.

## 5. Reproduction / governance

- `source/` — execution code (verified vs `scope.code_sha256`); `evidence/` — scope, RUN/EXIT, independent review (APPROVE/APPROVE), ACTIVE, author response, self-audit, dispatch records, slurm logs; `output/` — SUMMARY/CHECKS/PARAMS/READOUT.
- Independent reviewer `qoder_independent_reviewer_g20_a_q1_residual_20261114` (not the author). Formal job 164072 EXIT 0, elapsed 332.7 s, peak RSS 1.46 GiB, one A100-80GB.
- See `PUBLICATION_MANIFEST.json` for the full file/byte/SHA binding and `../` navigation for the BW/BWS companions.