# Original BF16 paired two-video producer

Software prepared on 2026-10-28; native generation and fresh full-model hashing have **not** been run by the author of this implementation.

`E0_ORIGINAL_BF16_CONSUMER00_12_VIDEOS_20261028` generates the two previously exposed consumer source00/source12 clusters with their original image, prompt, conditioning and seed42. It retains the historical four-step LightX2V timetable, sigma shift5, high/high/low/low experts, torch SDPA, original 40-block offload and complete 81-frame trajectories. The underlying DiT matrices come from the original public BF16 checkpoints, independently of the historical HiF4 QDQ path. This is a native BF16 baseline with matched inputs, not a new held-out cohort or a single-factor quantization attribution.

The producer imports LightX2V directly with a lossless config merge. `dit_quantized=false`, `dit_quant_scheme=Default`, unquantized T5/VAE, no adapter, caching or training are fixed. Every DiT Default MM dispatch checks the original class, BF16 CUDA input/weight/bias, transposed weight stride and complete step/expert/block/module order. It delegates to the original `torch.mm`/`torch.addmm(out=...)` implementation once. HiF4 imports, quantized loader and quantized MM construction fail before dispatch. Original CPU checkpoint tensors are checked against the full 1095-tensor BF16 headers. Offload preserves the actual source dictionary and bound RMS scalar auxiliary objects.

The two videos require 8 flows, 320 FFNs, 640 FFN linears, 2560 other block linears, 40 pre linears and 8 head linears. Setup loads 812 DiT source Default MMs and 40 buffer MMs. The original T5 CPU-offload path is accounted separately: 168 source and 7 buffer MM loads; two prompts execute 336 Default MMs and 48 block transfers. DiT block transfers are 324 with four expert-buffer resets. All registered calls debit the budget before invocation and preserve a failed attempt without retry.

Each video saves its raw FP32 draw before conditioning, actual BF16 inference initial state after step_pre0, four native FP32 flows and four actual BF16 post-states. The two noise files must each be byte-identical to the completed historical Dense input control. New BF16 post-states follow their own free trajectory. Two intrinsic CPU CORE reports accompany the videos; official quality scoring remains separate. Timing includes observers and state serialization; the producer grants no quality or speed acceptance.

Root must freeze the draft, register the new scope, then run a CUDA-hidden independent CPU proof before native launch. That proof freshly reads and SHA-checks all 69,025,362,058 model payload bytes with at most two hash streams/CPU4, and separately checks 21,454,628 tokenizer/config bytes. Its immutable RUN and actual exit0 receipt must be bound to the exact frozen protocol. The native gate checks those identities and current payload stat fingerprints before Torch import. Node-local paths are declared explicitly; missing assets stop the attempt. One A100/CPU4, host available at least128 GiB, GPU reserved at most64 GiB, compute1200s/outer1500s and new HF at most1 GiB are fixed. The session two-card constraint remains separate from the repository dynamic policy.

Version under independent review:

- `tools/e0_original_bf16_videos.py`: `5e14e3bc1d0ed84ea51cff31cb77551c55455ff4b5226a5d2d1ec1d5227a24d0`.
- `tests/test_e0_original_bf16_videos.py`: `070560eed2eecea2a495dcf60c0b822ca204e55c3a80ea1c336412927e8b5e3c`.
- `config/e0_original_bf16_videos_20261028_draft.json`: `780ac79f76f5b2d1632a9537b2a894cb2e4a7a8f8e5b843e9fc4349cf856acb7`.

Author verification: 18 tiny CPU software controls exited0 in1.164s, compilation and inspect exited0. A stdlib-only check matched all900 source bindings and the actual historical parent RUN/exit/generation/noise chain. No Torch/NumPy/model imports, native tensor operations, GPU work or fresh69GB model hashing were performed. The controls execute the real Default MM method AST against metadata-only tensor doubles, including two complete DiT/T5 dispatch traces and wrong-phase/order/expert/dtype/stride, source, budget and CPU-receipt failures.
