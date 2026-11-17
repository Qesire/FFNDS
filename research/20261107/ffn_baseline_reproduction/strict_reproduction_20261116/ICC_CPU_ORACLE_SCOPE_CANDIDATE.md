# ICC tiny author oracle — candidate execution scope

Status: REVIEW_REQUESTED. This is not an execution record or frozen ACTIVE scope.

- Fixed authorship: `ccccczzy/icc@dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4`; script verifies Git blob identity for the four upstream files before extracting their actual Python function bodies.
- CPU only, one invocation, no retry; no model, checkpoint, dataset, GPU, production artifact or video. Record actual Python/Torch version, command, stdout/stderr, process exit and input/code identities. Failure stays failure; revise and re-review before another invocation.
- `stat_upd`: FP32 CA/CD, two source batches `[1,2,3]` with signed values and nonzero aligned minima; compare per-source normalized aggregate bitwise, sqrt and ratio 1000 clamp with author; separate constant CA and zero-delta CD.
- `wsvd_merge_lowrank`: seeded FP32 random matrices `[5,3]` and `[3,5]`; rank 2 and full rank, positive left/right scale clones, compare factor action against existing `icc_decompose.factorize` at rtol=2e-5, atol=2e-5; full-rank action matrix versus weight at rtol=atol=1e-5; serialization exact. The author mutates scales, so independent clones per invocation are essential.
- Real `SpMlp.forward_upd/forward_inc` method bodies with identity dropout/norm and tanh GELU: FP32 `[1,4,3]`, hidden width 5, rank 2, nonzero linear biases; two fresh anchors with two distinct hit inputs each (A/H/H/A/H/H stress), and separately the actual author p2 scheduler clr and A/H/A/H gate sequence. Compare output bitwise and verify reset failure. Each hit is compared to the same retained anchor.
- This checks the small FP32 CPU callable and is not original-shape/backend, Wan adapter, BF16, calibration, or scientific-effect acceptance. Host native anchor may only be substituted after validating that its activation is tanh GELU and captured z/h/y correspond to the same native computation; dtype/device and request/expert/layer identity must be bound by caller.

- Four-boundary collector wiring and assignment of scales to W1(u/z), W2(h/y) are deferred; a generic single-boundary statistics fixture does not test this integration.

