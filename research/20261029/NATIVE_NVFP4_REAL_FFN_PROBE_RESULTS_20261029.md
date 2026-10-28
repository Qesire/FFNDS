# Native NVFP4 real-operand FFN probe

**NATIVE:** The independent CPU asset producer and the fixed SM120 GPU probe both exited 0. The GPU RUN SHA-256 is `7c90d29928faeecadc7238305898c3a79be59a574bf634e590c0c858fdb2a3b8`, the actual exit receipt SHA-256 is `780aee0948875c158c70e24b10070f644506774c76a134b4b709e8e7a1b2ca84`, and the outer duration was 90.859 seconds. All registered predebits, invocations and returns agree, including 120 dynamic NVFP4 quantizations, 120 packed GEMMs, 62 decoded BF16 GEMMs and 120 timed samples. The two M256 capsule files were independently read and their complete byte counts and SHA identities verified.

CUDA event timings below are mean ± population standard deviation over five samples after two warmups. The ratio compares the complete dynamic-Q1/W1/GELU/Q2/W2 NVFP4 FFN against the decoded BF16 FFN from the **same public NVFP4 checkpoint**.

| Expert | M | Native NVFP4 FFN, ms | Same-checkpoint decoded BF16 FFN, ms | BF16 / NVFP4 |
|---|---:|---:|---:|---:|
| High | 256 | 0.20417 ± 0.00361 | 0.38626 ± 0.00284 | 1.892× |
| Low | 256 | 0.19941 ± 0.00281 | 0.38335 ± 0.00129 | 1.922× |
| High | 31,668 | 11.04864 ± 0.39842 | 34.94652 ± 0.90587 | 3.163× |
| Low | 31,668 | 11.15593 ± 0.36098 | 35.19022 ± 0.31186 | 3.154× |

The BF16 comparator was run with `allow_bf16_reduced_precision_reduction=False` and TF32 disabled. The ratios therefore apply to this precision mode and measured operator setup; they do not establish the fastest default BF16 implementation or end-to-end video speed. M31,668 repeats the same 256 real source-894 teacher V1 rows to construct the shape and measures shape cost; it is not the full native state.

At M256, native versus same-checkpoint decoded BF16 FFN relative SSE was 0.00104670 for high and 0.00114523 for low. The four individual packed linears versus their same-packed FP32 reference cast to BF16 had relative SSEs of `4.69e-14`, `1.11e-10` (high W1/W2) and `4.43e-11`, `2.59e-10` (low W1/W2). These references and the FFN comparator have different purposes: the former checks the packed kernel on identical operands, while the latter includes the dynamic activation-quantization pathway. Q2 is produced from the current call's native W1 and GELU output.

The [machine result](NATIVE_NVFP4_REAL_FFN_PROBE_RESULTS_20261029.json) records every per-cell timing sample and numerical metric, source provenance, operation counts and capsule bindings. This is an operator probe, with no full-depth rollout, original-BF16-checkpoint comparison, generated video, quality acceptance or end-to-end system measurement.
