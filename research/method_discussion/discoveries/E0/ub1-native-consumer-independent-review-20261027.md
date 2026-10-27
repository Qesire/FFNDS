# UB1 native consumer: independent source review (2026-10-27)

This review covers the unfrozen runtime SHA `5436149336f6dad3d6b76ecb97ead64460e92278cc59bc709589aa887026b2a8` and video driver SHA `b9391256d79a6f5391631b61eb2ac8bbefbc0761f6d6b8657aee3a62f761317d`. It is source review, not an executed generation or a quality or latency finding.

The native arithmetic follows the historical step-budget route: float32 Smooth division, BF16 native QDQ, original alpha on the down path, two BF16 residual linears, and BF16 addition before the native caller gate. The runtime addresses packed objects by step and block; its FFN coverage distinguishes both slots. Candidate generation is guarded on exact equality of all four neutral latent hashes and video bytes against the recorded original UB1 run.

**Must fix before freezing:** the video driver's generation ledger charges its main, residual, and non-FFN linear and QDQ calls only after a complete successful generation. If generation fails partway, the main RUN counts omit the calls already made; the failure phase's observed counter deltas do not repair the budget ledger. Similarly, flow charging occurs after the post-step hook, so a failed in-flight flow is omitted. Precharge the authorized phase or charge individual calls before execution, preserving failed consumption.

The historical `formal20_input_recovery.json` has the exact status and canonical-SHA fields used by preflight; its canonical SHA is `16545be8ec2d951347363211801a375cb0b50d5ac204d8115c5410b35d2e1daa`. The original UB1 policy has the accessed `groups_by_step`, `identity_control`, and cache SHA fields with expected values. The final protocol and its individual image and historical receipt bindings were not available for this review; they require verification after the author finishes the draft. No GPU, model, or new experiment was run.

## Follow-up on revised draft

The revised runtime, video driver, and tests have SHA-256 respectively `bd47565a8e1014f7893d4f15f4b72302e8fc8bbe137ff691102d7c654daa6383`, `05b2f1042b3d4b4126590158da2d374102c7d8addeb3b936e2e1722a9706fa73`, and `a8ee5441bf17beeaccf58e99c74582151e7c641df67d681ba2ef8ea84b2bb410`. The driver now reserves all per-generation flow, FFN and non-FFN linear/QDQ, and residual allowances before dispatch, and preserves this charge on failure. The above accounting finding is resolved in this revision.

A separate CPU byte/SHA check of the draft verified the canonical 20 input manifest and recovery report, original UB1 policy and cache manifest, both actual image files, and all four historical input receipts. For each case the source group, seed, prompt and image digest, and historical four latent and video digests match their referenced records. The draft's bank manifest digest is still pending, so the complete bank chain and worker-specific preflight remain to be checked after it is bound. No model or GPU was run.
