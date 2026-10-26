# Saved native endpoints provide a four-decode video discriminator without new denoiser work

Independent `whole_flow_native_review` agent, 2026-10-26. The research object is a visible comparison between the original dense terminal state and the already measured **candidate,candidate,candidate,dense** terminal state. The whole-flow study has paid for both latent endpoints. Four VAE decodes—dense neutrality and late-dense intervention for each of sources0/12—can connect the finite state response to complete video function without new denoiser/FFN calls.

An improvement would support recoverable late-step function in this exact macro intervention; persistent degradation would leave earlier lost information, state response and decoder sensitivity as competing explanations. Neither outcome identifies one FFN layer or establishes the broader method family's quality. This is a source/software review, with CPU checks only; actual VAE neutrality, memory and generated-video metrics remain execution outcomes. The coordinator owns the frozen protocol and continuing-run authorization.

## Native decoder and media path reconstructed

I independently read the actual frozen WanDistillRunner inheritance, WanRunner VAE loader, DefaultRunner decode/postprocessing, WanVAE wrapper/internal decode, dtype getters, seeding helper and ffmpeg writer. The driver matches the decision-critical path:

- The distilled MoE runner inherits **WanVAE** using `Wan2.1_VAE.pth`; it does not select `Wan2_2_VAE` merely because the denoiser is Wan2.2.
- Input is original BF16 `[16,21,58,104]`. The wrapper adds the batch dimension and applies its original16-channel mean/inverse-std internally. The driver adds no extra normalization, autocast, tiling, cache mode or image/text encoding.
- Original `model.decode` clears temporal caches at entry/exit. Fixed nonparallel, nonoffloaded BF16 decode can reuse one loaded VAE for the four clips while preserving this lifecycle.
- `seed_all` is called for the appropriate source seed, setting the original cuDNN benchmark/deterministic flags. Matmul TF32 is false, cuDNN TF32 is explicitly protocol-bound and actual environment is recorded.
- `wan_vae_to_comfy` is applied once, preserving its in-place BF16 arithmetic. The native ffmpeg writer uses uint8 truncation, RGB→BGR, libx264, yuv420p, fps16 and four encoder threads. It checks the encoder return code. Actual output shape is81×464×832×3, with no resize/pad to the nominal480 target.
- Effective encoder binary SHA/version and package versions are recorded. Historical MP4 byte equality remains the neutral test; encoder metadata alone does not establish equality.

I independently hashed the available logical-view VAE binary in full: **507,609,880 bytes**, SHA `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981`. The node-local manifest points to an executor-private cache unavailable on this review host. The executor preflight checks that physical file independently; this review does not replace that check.

## Actual saved operands checked without importing Torch

Both actual `prepare_pair` calls passed against the completed source0/source12 whole-flow receipts, authoritative native trajectories and strict historical video bundles. I independently read the safetensors header/payload and reconstructed native identities from raw BF16 key bytes, then confirmed the final driver's Torch-free helper gives the same values:

| Source/seed | Dense final native identity | Late-dense final native identity |
| --- | --- | --- |
| 0/42 | `bdbf90dce710347134646d91c262dcd20d6c367cee71a7388cef39aeea914b95` | `6323387b98eff6f8836c7f5d02e00df60fb527fb9aabec5a659447b339a1bbca` |
| 12/54 | `552072b0ed6551a7c0aeb38ab018bd8ffb1452b450601311f50712907e098a63` | `28277b1cf69d913dcc53ea7035a187751adef43d3244065bdfa70013b5aecad2` |

The dense identities also match the original final teacher states. Header parsing verifies BF16 dtype, exact shape, complete payload extent and original little-endian storage. The full parent tensor SHA and named-key identity are both bound. Actual saved-data preparation left `torch` absent from `sys.modules`.

## Findings resolved before this reviewed snapshot

1. **Independent semantic inventory:** initial code only compared metric identities with copied receipt identities. The final inspector reconstructs expected source, role, model digest, decoder digest/configuration, parent receipt/sidecar, reference identity and named tensor identity from frozen preparation. Consistently forging both receipt and metric metadata is rejected.
2. **Exact input for all four roles:** both dense and late-dense input identities are reconstructed before model loading and checked again at rendering/gating. This covers a wrong or omitted hybrid input identity, not just neutral controls.
3. **Failed decoder versus failed encoder accounting:** separate start/completion events count actual VAE and encoding stages. A successful VAE followed by an encoder exception retains one completed decode, incomplete encoding, elapsed render cost, partial file hash/size and failure status, and stops the fixed scope.
4. **Every actual clip remains auditable:** declared clips each receive an evaluator attempt against their source's original dense video, even after generation/neutrality failure. One failed metric does not skip later actual clips. Orphans receive intrinsic forensic evaluation with explicitly unknown source/seed/model identity and force failure. Required expected-frame/cost metadata is supplied without inventing provenance.
5. **Shared setup versus individual render costs:** per-clip render deduplication keys are distinct; the shared setup uses a separate receipt key. This avoids counting only one of four different render costs or multiplying a single VAE load.
6. **Final resource and immutable inventory:** monitor reports are retained before final violation checks. Every actual MP4 is SHA/size enumerated in the frozen decode receipt and compared with the filesystem. Failures remain failed and consume the one canonical attempt.

The receipt order avoids a SHA cycle: actual render/cleanup → immutable `DECODE_RUN.json` → per-clip metrics → BUNDLE/core gate plus independent inventory → final summary. Future metrics/bundle fields are not written back into the decode receipt. The old core gate and old paired-native strict gate remain unchanged; the hybrid is not disguised as their pure dense/candidate schema.

Core metric PASS, if reached, will establish complete frame evidence/provenance. Official VBench/human statuses remain NOT_RUN and `quality_accepted=false`. New costs explicitly mean VAE-only rendering, not the cost of the represented complete denoising path. Four VAE neural inferences are real added model work despite zero new denoiser NFE or FFN calls.

## Independent software verification and remaining native check

I reran `PYTHONPATH=tools .venv/bin/python -m unittest discover -s tests -p 'test_e0_saved_flow_decode.py' -v`: **10 tests passed**. Tests cover Torch-free import, raw named-key hashing, four-render/frozen-receipt order, six consistently forged provenance cases, partial encoding, neutral mismatch fail-stop, metric failure continuation, orphan evidence, one-attempt consumption and path confinement. Their renderer/evaluator fixtures are synthetic bytes/mocks, not native VAE or actual clip-quality validation.

A separate attempt to import the native VAE with empty CUDA visibility stopped at LightX2V's expected device-availability check. A software-only availability/capability mock with a hard `_lazy_init` blocker then stopped at additional package-level CUDA device queries. No VAE was constructed and no GPU initialized. These checks expose that even the VAE import traverses package initialization; they do not certify or refute the actual native launch. The driver reserves its actual assigned GPU before that import. The coordinator separately reported that plain Transformers5.6.2 imports on the executor from its existing environment with CUDA uninitialized and cuDNN TF32 default true; no source-view wrapper is needed for this working launch. That executor dependency check was not independently rerun by this reviewer.

At the reviewed source/test snapshots, I found no remaining substantive source blocker to the fixed execution. Strict dense MP4 neutrality, actual native output, runtime memory and all-frame metrics remain the informative runtime checks. Any failure preserves the exact artifact and stops this scope without retry.

## Reviewed exact identities

| Object | SHA256 |
| --- | --- |
| `config/e0_saved_flow_decode_20261026_v1.json` | `588115b9a0930619045fb42cc1d92803225610ccd66f17b27bb6142130f6baef` |
| `tools/run_e0_saved_flow_decode.py` | `534c1fde97d718ee1e6315d6a60ce436a704dd4f4fd62fe5388ea95d1ed5b5f7` |
| `tests/test_e0_saved_flow_decode.py` | `c4b46e4673ceccebd5f99bd1336b4f532950bef0f716dfb55c90be895d9d49e6` |
| Native `wan/vae.py` | `5adc56ca075ffa1e3ccc486f9f5d8ea0faf90cdaaa657cdc87c6bef3a45a16bd` |
| Native `utils/utils.py` | `82137228f5d573adb36a845e3b2e8a9e47d6e7e8f8e23c2fb8ceee149492c9ce` |
| Native `utils/envs.py` | `68c2f49686cb58559e8f1beaa0641b1141b5639083fe4ec3e8df771395c3189c` |
| Logical model-view manifest | `932bc14ec0cd8d73a13532191ea202b0bc22478d7da26ba5029658329f257df3` |
| Node-local model-view manifest | `97b2d991f6312ab67ee439ea28a03989bd86b1f68151a6eca11102b949e539d1` |

I read the final protocol above and independently checked every frozen driver/native source hash against its local file, including its additional worker/launcher/tests and package/platform bindings. It fixes four decodes, one simultaneous model process/GPU, zero denoiser/FFN,0.1GiB output, no retry and the two exact completed whole-flow RUN identities. Registration remains the coordinator's responsibility under the user's authorization. Source/media checks used original local code and actual saved artifacts. No new web/novelty search was needed for this narrow software audit. Graph anchors are **P24→terminal latent→VAE→D09/P25**, with the late-dense intervention conditional until executed and formal video acceptance separate.
