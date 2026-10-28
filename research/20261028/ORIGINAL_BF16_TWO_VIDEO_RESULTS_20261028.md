# Original BF16 two-video baseline · 2026-10-28

Two previously exposed sources each produced one 81-frame, 16 fps original-BF16 Dense video. Original checkpoint weights were used without activation QDQ or weight quantization; this is a separate numerical path from decoded HiF4. Matching raw FP32 and initial BF16 noise bytes were verified against prior QDQ input controls.

| Source | Generation incl. observers / encoders / decoder (s) | FFN CUDA boundary (s) | Peak reserved (bytes) | Core Farneback mean |
|---|---:|---:|---:|---:|
| 0 | 55.128 | 6.322 | 21908946944 | 0.524858 |
| 12 | 46.661 | 6.401 | 21911044096 | 0.060728 |

Each core report verified 81 decoded frames and records the full inexpensive diagnostics in the JSON; these are not official quality or human acceptance. The 20 saved noise, FP32 flow and native BF16 post files passed actual file/header/raw-payload identity and shape checks. All 903 executed-source snapshot bindings match the frozen protocol; both actual default-MM traces record BF16 operand shapes and strides, including T5 separately.

Actual worker exit 0, timeout false; RUN SHA256 `59e99d08d206aac4ffabb11453e4b6b27449d046f07d14027d5e65e9644064c9`, EXIT SHA256 `c718ec457194d2e48deed47c9ccb18502ddb44a37355a6f4874867bd4c493b04`, protocol SHA256 `2f5352e91a877e85c0c0e88d8c216f74f5124693161ad412b7226aff73c69122`. The prior CPU proof exited 0 and read/hashed 69,025,362,058 bytes; no large checkpoint rehash was performed for this report. Ledger closes for 2 videos, 8 flows, 320 native FFNs and CUDA event pairs, 324 DiT checkpoint-buffer loads, 2 expert loads and zero QDQ. T5 is separately accounted: 48 block-buffer loads and 336 default-MM calls. Shared raw model setup 83.774 s; outer elapsed 295.412 s.

Times include instrumentation and sequential setup and do not establish a fair system speed comparison. Official quality and human full-play remain pending.

[Machine-readable results](ORIGINAL_BF16_TWO_VIDEO_RESULTS_20261028.json).
