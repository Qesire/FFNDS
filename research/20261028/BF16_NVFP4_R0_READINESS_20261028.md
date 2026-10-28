# BF16 / native NVFP4 R0 readiness · 2026-10-28

The original two LightX2V distilled checkpoints are present: each has 1,095 BF16 tensors in its safetensors header, including block 0 W1 `[13824, 5120]`. Each file is 28,577,915,880 bytes. Their HF snapshot locators, header hashes and previously recorded full-file identities are in the [machine-readable report](BF16_NVFP4_R0_READINESS_20261028.json). This check read headers and file sizes; it did not recompute the 57 GB of full-file hashes or validate tensor payloads.

The earlier HiF4 `weight_qdq` is an offline quantized and decoded BF16 weight, distinct from original BF16 and packed native NVFP4. A matching complete NVFP4 checkpoint has not been verified in this bounded investigation; its availability remains UNKNOWN.

The 2026-10-26 SM120 primitive actually ran on **RTX PRO 6000 Blackwell**, not an RTX 5090. Two packed CUTLASS examples matched CPU reference; that establishes a small numerical primitive, not a Wan FFN result. Accessible workers observed in the current investigation use A100. The prior SM120 target is not accessible through the investigated approved entry points; failed connection does not establish that a machine is absent. An accessible RTX 5090 remains UNKNOWN. Availability is time-dependent.

Next, identify an authorized accessible SM120 target and bind a matching NVFP4 checkpoint, or define its producer; freeze and register the BF16/NVFP4 comparison before any execution. No new model or GPU work is claimed here. Historical primitive identities and byte bindings are recorded in the JSON.

**本会话用户指令：**没有 RTX 5090，使用 A100 继续原始 BF16；NVFP4 原生执行暂缓，直到具备合适设备和已登记范围。
