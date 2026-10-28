# Compact6016 six native videos · 2026-10-28

**Six complete native videos were generated and the worker exited 0.** Dense, energy94 and random94 each ran on two historically exposed development images at seed 42. Every video contains 81 frames at 16 fps and 832×464; each used four native steps and 160 FFN sites. Both Dense videos reproduced their four historical post-state hashes and video bytes. The actual FP32 and BF16 initial-noise files were byte-matched within each source across all three arms. The operation ledger closed; the two candidate banks loaded 80 functions each, and each candidate video executed 320 narrow BF16 GEMMs.

| Source | Arm | FFN boundary seconds | Generation wall seconds | Mode setup seconds¹ | Peak CUDA reserved GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | Dense | 9.757 | 89.628 | 4.337 | 78.26 |
| 0 | energy94 | 4.870 | 75.119 | 67.207 | 61.86 |
| 0 | random94 | 4.865 | 53.333 | 95.009 | 61.13 |
| 12 | Dense | 9.746 | 64.689 | 4.337 | 78.35 |
| 12 | energy94 | 4.898 | 57.860 | 67.207 | 61.86 |
| 12 | random94 | 4.891 | 57.273 | 95.009 | 61.13 |

¹ Mode setup is paid once for its two sources; shared pipeline/cache/bias setup was another 264.946 seconds. Dense/candidate FFN boundary ratios were 2.004/2.005 for source 0 and 1.990/1.993 for source 12 (energy/random); generation-wall ratios were 1.193/1.681 and 1.118/1.129. These timings include instrumentation, observers, encoder, decoder and saving. Sequential mode/source order and warmup differ, and original checkpoint full buffers and prefetch remain paid. They are not an uninstrumented, balanced deployment benchmark or system-speed acceptance.

**Core video diagnostics reveal source-dependent motion changes.** Downsampled Farneback mean flow for Dense/energy/random was 0.3441/0.1775/0.1966 px for source 0 and 0.0582/0.1438/0.1659 px for source 12. The paired candidate-to-Dense Laplacian energy ratios were 0.662/0.632 and 0.662/0.618, respectively. These cheap encoded-video diagnostics describe this two-source sample; they are not official VBench motion or perceptual scores. Official quality and human full-play review remain pending. Ordered inspection of all81 thumbnail frames per clip also shows persistent blur/noise in both candidate arms: the cow outline remains, while cow/background detail and the street signs lose definition. This is AI still-frame observation and does not replace human full-play review.

[Machine-readable results](COMPACT6016_SIX_VIDEO_NATIVE_RESULTS_20261028.json) contain six video and metric identities, per-video costs and ratios. The v2 actual RUN SHA is `23a34c360ec07df9644a7a8491edc77f4b8164f837aa71e8af78a06a74517a3a`; actual exit receipt SHA is `524a2b2a6ee1f7bad44f7918c6ec6fd3966c633371c1112c2eacb904c6f9add9`. Prior v1's failure before the first block remains a separate historical result.
