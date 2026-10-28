# SM120 isolated kernel build · actual CPU result

The pinned LightX2V kernel source closure compiled, linked and installed successfully for SM120a. All three commands and the durable worker exited 0; elapsed outer time was 1,057.28 seconds with two CPU threads and no GPU or model execution.

The isolated `common_ops.abi3.so` is 1,312,256 bytes with SHA256 `474675037c50a97589731cae5225ddf23f9264803205cd538369a75233a3bb88`. Source, compiler, CUTLASS, Torch, command logs and actual exit are bound in the [machine receipt](SM120_ISOLATED_KERNEL_BUILD_RESULTS_20261029.json). The maximum sampled local output was 10,731,200 bytes.

This supplies a loadable-package candidate for the native NVFP4 lane. Numerical packed W4A4 correctness, dynamic Q1/Q2 costs, complete FFN latency and same-recipe full-video fidelity require their own actual runs. The next fixed operator experiment uses original Wan dimensions and both verified high/low NVFP4 checkpoints; synthetic full-M tiling will be identified as shape-cost evidence.
