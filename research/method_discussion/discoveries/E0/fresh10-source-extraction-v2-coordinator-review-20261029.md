# Fresh10 V2: independent coordinator review

Reviewer /root; author /root/raw_norm_rows_terra. **SUPPORT** this exact new CPU scope, with original V1 preserved.

- `tools/raw_bf16_fresh_source_extract_v2.py`: 16326 bytes; SHA256 `060e52630983833e5ce40534634e06dc62e9314cd2da332249ee8980bae84085`.
- `tests/test_raw_bf16_fresh_source_extract_v2.py`: 2450 bytes; SHA256 `a251577109a8bbd5d40af48f0cc4f332b878469f7a0631ba37025ff7e2ebbbfe`.
- `config/e0_raw_bf16_fresh10_source_archive_cpu_20261029_v2_draft.json`: 15639 bytes; SHA256 `26fecb6f7763bbbc93b7014d97ec7ca70d39ec13a45411cb982c235e1220baba`.

I read the full V1→V2 source diff and reran four synthetic tests (actual exit0), including a real temporary snapshot symlink/canonical binding regression. Actual metadata-only preflight checks every ROOT/HF metadata binding and the failed V1 protocol/RUN/exit. The V1 exit is genuinely1, before any archive read or extracted image. V2 fixes only metadata canonical binding, adds exact failed-parent validation and an explicit preflight, and changes scope/output identities; the archive members, source/prompt identities, history checks, tar guards and bounded CPU budgets stay fixed. No archive payload/model/GPU was read or invoked during review.

The ten recovered sources remain provisional until visual and further history checks; dHash alone is not identity or independence proof. This version authorizes one new bounded CPU archive stream after ACTIVE registration, not rerunning V1 and not any native model job.
