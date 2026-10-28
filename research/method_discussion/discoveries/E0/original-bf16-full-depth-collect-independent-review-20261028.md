# Independent review · original BF16 full-depth collector

**SUPPORT_EXACT_SRC_AND_TINY_CPU_INTERFACE; NATIVE_RESULTS_PENDING.** This review covers the specific code and draft identities below. It supports registration and the prescribed CPU preflight, followed by the separately authorized native runs. It does not attest an actual model run, full-cohort completion, quality, system speed, or generalization.

## Reviewed identities

- `tools/e0_original_bf16_full_depth_collect.py`: SHA256 `3079c7ce5cc40747dbedcb64419026d708973b1e723e558bf292ebaa7d8e4dcf`.
- `tools/raw_bf16_full_depth_sources.py`: SHA256 `27befc974bded5ae19a2f9aee11d14ecafaa9ecf414da925770704842c598ebe`.
- `tests/test_e0_original_bf16_full_depth_collect.py`: SHA256 `df498bbf7295d97ed8598ed87a57f1d1c3a63c74357cb83895b7e08887e8c0a2`.
- `tests/test_raw_bf16_full_depth_sources.py`: SHA256 `1bd1855624f2c92dec3ce53aefaac6d32a5e0294307e021407e8a9f00eb57361`.
- Part 0 draft SHA256 `956b321e0d0ef23e63bca846062d641155f90484850264b0e7c782ebb4e20c60`; part 1 draft SHA256 `c0ed577f72ef82bcc183cf70fd7fa92c8bff808f4a9d49c948fde15c9c7704e5`. The two drafts' full repository-source hashes were independently checked against local bytes. Freezing and registration are separate steps.

## Mechanism and native path

SRC: `CollectorTap` extends the original `NativeTap` full-depth execution. Its `ffn.0` input is the modulated native W1 input V1; the `ffn.2` input is the actual full-width tanh-GELU result h, and its output is Y. The original equal-dtype FFN first adds cross-attention to x, then the block passes the same x and Y to the in-place gated `post_process`. The collector matches the active expert/block, x and Y identity, captures r before the write and u afterward, and flushes each site's selected rows. The gate, shift and scale are recorded from the real FFN call. Native grid shape and the real drawn row coordinates are checked. No Q1/Q2 or synthetic hidden state is introduced.

SRC: the development replay uses the original input encoder, scheduler preparation and `DefaultRunner.run_segment` four-step loop; this loop performs scheduler pre, model inference and scheduler post without VAE decoding. For each part, its exposed development source must match all four original FP32 flow and BF16 post tensor hashes, as well as the original raw and initial noise, before the three new calibration generations. Each calibration source uses original generation and decode and has its own CORE report. Original 0/12 consumer positions are locators for old controls; sampling uses the source bank's native ordinal (0 → 894 and 12 → 256), not consumer position. The six calibration images are historically exposed development images; they are not an independent confirmation cohort.

## Fixed budgets and checks

Each part has one development replay and three calibration sources: 16 native flows, 640 FFN sites, 1,280 FFN linears, 648 block-buffer loads, 640 packet writes and validations, three decoded videos and three CORE reports. Two experts each reset their offload buffer once, then each performs two consecutive 40-block flows: 81 loads per expert per source, 162 per source, 648 per part. The original model path and parent raw checkpoint identity remain BF16; no new training or official quality metric is authorized by this design.

Each packet contains 256 nested rows of V1/h/Y/r/u, gate/shift/scale and the actual row indices. The design flushes a site before proceeding to the next and reserves disk capacity before each packet and state write. CPU and GPU packet copies and serialization buffers can coexist; their total peak is an actual-run observation. Each independent part has its own source identity, scope, no-retry failure record and actual-exit requirement; neither part alone establishes completed six-source calibration.

## Tiny controls and boundaries

Independent execution of `python3 -m unittest discover -s tests -p test_e0_original_bf16_full_depth_collect.py` passed five portable controls. They cover no Torch/NumPy import and unchanged original globals; part budgets and native ordinal/consumer index distinction; packet shape, dtype, hash syntax and unique coordinates; predebit on failure; and duplicate or wrong-full-M operand capture. These tests do not inspect real capture tensors, execute a model, use GPU, or prove native output correctness. Parent exit/asset identity, immutable scope authorization, byte preflight, actual neutral controls, CORE and both exit receipts must still pass in their respective real stages.
