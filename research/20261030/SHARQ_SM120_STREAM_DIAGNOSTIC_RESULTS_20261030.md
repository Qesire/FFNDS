# SharQ SM120 stream diagnostic, 2026-10-30

**OBS / NATIVE:** A bounded matrix diagnostic found differing packed representations under an unbridged non-default stream for both low-expert matrices. The corresponding event-bridged and default-stream cases matched for all four matrices. Original BF16 weights, inputs and reference outputs remained byte-identical and finite. This establishes an integration sensitivity in these observed cases; it does not establish the cause of the earlier V6 full-model nonfinite flow, nor does one observation establish a failure probability.

Four real matrices were tested: high and low expert, each at linear 0 and linear 2. SAFE ran the author's default-stream reference and an event-bridged non-default path for each matrix. The packed digest covers Q payload, logically consumed sparse/dense scale coordinates and global scale, excluding padding. All four bridged digests matched their default references. For each, dense32 and SharQ outputs were finite and had relative L2 **0** against the corresponding default output. The pre/post and post-author BF16 controls were byte-identical and finite.

UNSAFE was a separate process with one unbridged non-default-stream observation per matrix:

| Expert | Linear | Packed digest matches SAFE default | BF16 control unchanged and finite |
|---|---:|---|---|
| high | 0 | yes | yes |
| high | 2 | yes | yes |
| low | 0 | no | yes |
| low | 2 | no | yes |

The [machine-readable report](SHARQ_SM120_STREAM_DIAGNOSTIC_RESULTS_20261030.json) retains each digest and control. A digest mismatch identifies changed bytes within the combined packed digest scope; it does not identify which component changed. The author's default-stream implementation was unchanged. SharQ here uses structured 4:8 sparsity plus a full-K dense residual. This is neither a new video result nor a speed measurement.

Both actual launch receipts report exit 0 with no timeout or automatic retry; their RUNs were written as `COMPLETE_STREAM_PROBE_EXIT_PENDING` before the launcher recorded the exit. SAFE had 8 cases, 24 BF16 linear calls, 8 weight quantizations, 8 dense32 and 8 SharQ calls. UNSAFE had 4 cases, 8 BF16 linear calls and 4 weight quantizations; it did not execute the author output comparisons. Predebit, invoked and returned counts match, with no inflight calls. These count Python entry and return; case CUDA completion follows explicit device synchronization, while intermediate asynchronous completion is not separately observed. Both runs report zero model and generation calls and no monitor violations.

SAFE recorded 63.10 s internal / 66.04 s outer, peak physical GPU use 2,061 MiB and peak host RSS 1,924,583,424 bytes. UNSAFE recorded 19.37 s internal / 22.08 s outer, 2,221 MiB and 1,866,604,544 bytes respectively. These diagnostic observation costs are not system-speed estimates.

Provenance: SAFE protocol `f155c04dbde4ba91b9e840d018a0267be651f726e6e2b28d1a16c566aae8b92e`, RUN `c61c18cbea24c0bc43516da264a1cb76ffb5b6b945f2289bafae02cadf84c8b3`, EXIT `ab07fb775a7be2bb2a1f5a3f9dc09d3f4b7c6b077c440a691d695a7a007f46d6`; UNSAFE protocol `5d46ed00ac0d37efeec253c5e09d39478aeb3c2f4198447cdd56f79c2679ce8f`, RUN `39d720739e253317bcf7d0c2e92773580b864ee10e2794d7ff00509fa600953c`, EXIT `d5131c9b5a9563c6e21e6d5aa80d71e2717fa1327b5f2543ca033f9adc975973`. HF-relative receipt paths, full counters and source hashes are in the JSON. No bundles or tensor states were reread for this report.
