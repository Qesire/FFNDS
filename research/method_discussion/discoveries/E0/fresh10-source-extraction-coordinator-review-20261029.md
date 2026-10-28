# Fresh10 original-image CPU extraction: independent coordinator review

Reviewer: /root. Author: /root/raw_norm_rows_terra. **SUPPORT** the exact bounded CPU-only version below; this is source acquisition, not independent video confirmation.

- `tools/raw_bf16_fresh_source_extract.py`: 13913 bytes, SHA256 `c3d8cdbf97bd0fbce758766860c1f969fe8ff036a6ebe0ac760a34f4077618e9`.
- `tests/test_raw_bf16_fresh_source_extract.py`: 1804 bytes, SHA256 `cf85484f7bed513da14f0cbd13912c4da8052c1de6f40e13bb8cd335415169f9`.
- `config/e0_raw_bf16_fresh10_source_archive_cpu_20261029_draft.json`: 14972 bytes, SHA256 `e87a077578dc8d687e2badec5fa0426e8702c8bd685295689b7547e10cf636e2`.

I read the complete extraction path and ran all three synthetic tests (actual exit0). They exercise exact regular-member selection, duplicate/link/traversal/size rejection, missing-member failure and full compressed-byte accounting. The fixed ten metadata identities, original and camera-augmented prompts, seven historical banks plus task metadata, pinned archive/metadata provenance and 3GiB read/64MiB output/300s CPU2 budget were checked. Archive payload has not been read by this review.

Only exact selected archive members are copied, without new crop. Complete compressed archive SHA and stable inode/size/mtime/ctime are checked after the single stream; each output image is bound. Known historical SHA/name/ordinal aliases or duplicate images cause scope failure and no automatic retry. dHash is a screening aid; output remains provisional pending visual inspection and additional historical coverage, and source count does not grow through a new camera prompt. The source bank and model cohort require separate derivation, review and registration after actual acquisition.

The durable job uses the coordinator-selected private worker profile, CUDA hidden and CPU affinity2. This version does not claim a bound native model or GPU environment; no such work is in scope.
