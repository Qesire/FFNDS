# Exact recheck: target-worker monitor binding revision

Reviewer `/root/asset_readiness_terra`; owner author `/root/sm120_full_quality_build_terra`. **APPROVE for registration only** for the two V3 protocols and six revision-V3 private job identities in the accompanying four machine receipts.

Compared each V3 protocol against its V2 counterpart: the sole changed field is `execution_monitor_binary`, now bound to the target worker's `/usr/bin/nvidia-smi` (1,243,808 bytes, SHA-256 `384896f294f80e8e5b59c11dbb04d3fdc028cf9bb01d7839154b96f94e966f4e`). Each job changed only its protocol path/SHA and the corresponding private job-spec argument; every other job field and argument is unchanged. Both protocol fingerprints and six job hashes were independently recalculated. Source and scientific recipe remain byte-identical to the previously reviewed V2. The target-worker binary identity is based on the actual remote metadata reported by root and author; runtime will recheck the binary before GPU use.

V2 was metadata-registered without CPU, GPU or model launch. Its original receipts remain preserved. No numerical test, model or GPU work was run for this metadata-only recheck; the 38 CPU software tests from the preceding unchanged-code review remain the evidence. Registration and actual CPU preflight, producer and independent VERIFY exit receipts remain separate requirements.
