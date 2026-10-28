# BH–A0 matched-head software independent review · 2026-10-28

**SUPPORT** for the exact software versions below. This review establishes a runnable, bounded function-error comparison with meaningful CPU controls. The 96 real full-M neutrals, 192 A0 calls, quality results, and execution performance still require their registered native run.

Reviewer: `/root/compact6016_six_video_build`. The reviewer authored the earlier B-H student core, but did not author or edit this new comparison driver, its tests, or its configuration. The independent comparison implementation was authored by `/root/completion_lineage_review`.

## Exact reviewed versions

| File | Bytes | SHA256 |
|---|---:|---|
| `tools/e0_bh_a0_matched_head.py` | 35398 | `c5432ab9827ff5296301b06adedfa44cef1c07f438021c45d57539d2de75c032` |
| `tests/test_e0_bh_a0_matched_head.py` | 15330 | `62224ef12cb95212afb99900ee8446d24b6ee17bc213c0820e1ed41ec06911f3` |
| `config/e0_bh_a0_matched_head_20261028_draft.json` | 22500 | `6f7ca0b741de0ca5785545dfc1f9977bbe7d51a1a46ae51267ab6d7eedc40b7b` |

All 67 source bindings in the draft were independently checked against actual repository-file SHA256 and length. No frozen producer/runtime source was edited by this reviewer.

## Mathematical and native-source checks

- The A0 operator consumes original saved BF16 **post-Q1 V1**, then performs BF16 W1/b1, tanh GELU, FP32 division by the student's selected Smooth2, the original native group64 Q2 with `force_fp32=True`, and BF16 W2/b2. This matches the original `compact6016_adapter.compact_forward` diagnostic computation; Q1 is not repeated.
- Original `WanTransformerInfer.post_process` uses `x.add_(y * gate)` when both inference and sensitive dtypes are BF16. The new operator separately rounds the BF16 gate product and residual addition in that same order, with the genuine saved pre-write residual.
- Original `infer_non_blocks`, `LNWeight`, and `MMWeight` identify unaffined row-local LayerNorm with epsilon `1e-6`, BF16 `mul_(1 + scale).add_(shift)`, and `torch.addmm(..., out=...)`. The new head follows these operators with M31668, D5120, the original column-major KN head stride, actual modulation strides, and original selected row positions.
- The other full-M rows are explicitly synthetic zero fill. The experiment is **full-M-geometry selected-row replay**, not a recovered complete state. Every one of the 96 real original-u neutrals must first reproduce its captured selected head bytes. Missing, repeated, or failed neutrals block all A0 work.
- Each of the 192 A0 M256 Y outputs must match its original completed RESULT source/step final-diagnostic raw SHA256 before the corresponding head result can be accepted. A mismatch retains expected/actual hashes, case, initialization, and the current arithmetic boundary. The original A0 BF16 reduction flag is correctly identified as not recorded; there is no flag search, teacher recomputation, SHA replacement, or retry.
- The B-H arm reads the 96 saved final BF16 head predictions from their completed immutable prediction packs. Its target is the same actual captured head_rows as the A0 arms. The comparison preserves expert and step, averages the two nested steps within each source, and reports equal-source summaries separately for train16 and exposed heldout8. Rows are not treated as independent N.

## Exact cost and entry gates

The registered numerical work is 96 neutral heads plus 192 A0 heads: 288 original-shape LN/head calls; 192 calls each for W1, GELU, Smooth2, native Q2, W2, BF16 gate multiplication, and BF16 residual addition. Saved B-H introduces no new student forward. Training, decomposition, Q1, teacher-Y numerical reads, full-u reads, full model generation, and video generation are all zero.

Independent enumeration gives **1272 H2D tensor copies / 1,264,047,360 bytes**: four head tensors, five tensors per neutral, five tensors per each of four terminal students, and eight tensors per paired comparison case. The **388 original-layout restoration copies** are separately charged. Label loads/checks are 192/1344, metrics are 288 with 576 FP64 SSE/energy reductions, and 192 original A0 output SHA checks are paid before acceptance.

Import and CPU byte-preflight paths require no Torch/NumPy import or visible GPU. Execution requires a unique exact ACTIVE scope, the exact exited prior independent CPU report and source digest, one assigned A100, dynamic repository GPU policy, and the explicit current session cap2. It rejects forced TF32. Append-only operation records reserve and mark invocation before each actual call; exceptions preserve charged work and no-retry status. This scope measures matched function errors; it does not measure a deployment speed advantage or free-video quality.

## Independent controls actually run

```
.venv/bin/python -m unittest discover -s tests -p test_e0_bh_a0_matched_head.py -v
```

Actual exit 0: **16 tests passed in 1.574 seconds**. The controls cover byte-only import, exact budgets/identities, draft rejection, CPU receipt failures, session/resource gates, wrong layout/affine norm, missing/duplicate/failed neutrals, retained failed debits, exact A0 arithmetic against the original compact forward, wrong Q2 shape, original head replay/strides, invalid rows, output-SHA failure evidence, and nested source aggregation. Numerical fixtures use D4/M9/rows2 synthetic CPU tensors. `py_compile` and `inspect` also exited 0.

This review performed **zero real saved-tensor numerical operations, native CUDA calls, GPU queries, training, decomposition, or model/video execution**. Real asset preflight and the subsequent finite native result remain the coordinator's separately frozen and registered work.
