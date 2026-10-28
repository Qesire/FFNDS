# Independent review · original BF16 full-depth collector v3

**SUPPORT_EXACT_V3_SRC_AND_SYNTHETIC_CPU_INTERFACE; REAL COMPLETION PENDING.** This review supports registration and independent CPU preflight of the exact frozen scope below. It does not claim GPU execution, completion of the original failed v2 scopes, a complete cohort, or quality/speed acceptance.

## Exact identities

- `tools/e0_original_bf16_full_depth_collect_v3.py`: SHA256 `13bc1ccdfb5b746e802f6324dac211f54e034e5abfc7acd70cbd883b15bb5cec`.
- `tests/test_e0_original_bf16_full_depth_collect_v3.py`: SHA256 `a1fde3bf66a67b15ec18728d1df4e4aba70ea38a1be7c7c0fb5a0399093ee29b`.
- Part 0 draft SHA256 `205df42ea248b86b53264d5168f421a18f88d87ecc215ca69e02a32bf1b6c7b4`; part 1 draft SHA256 `94b83b510a1b6913729f9c415820cc6a6ce433013999d119b2c6db1c2a3d34be`. All 919 repository-source hashes in each draft were independently verified against local bytes.
- The numeric, observer, I/O and accounting implementation is the exact v2 source SHA `e9bc695f5db69f900869c0efaaa47a815f8adab4ef247147080a820a8abbe075`, privately loaded with part-specific limits. Public v1/v2 module globals remain unchanged in the synthetic control.

## Asymmetric completion and costs

Part 0 selects exact v2 parent cases dev ordinal 894 and calibration ordinals 896 and 240: three sources, 12 full flows, 480 FFN sites, 960 FFN linears, 486 original block-buffer loads, two new videos and two CORE reports. Part 1 selects dev ordinal 256 and calibration ordinal 304: two sources, eight flows, 320 sites, 640 FFN linears, 324 block-buffer loads, one video and one CORE report. Each part first replays its original development source and verifies its ten noise/flow/post byte controls. Its setup still loads both full original expert checkpoints and the same T5/buffers. Each part has 1,800 seconds wall and 2,100 seconds outer budget, with four CPU threads, one A100 and 32 GiB reserved GPU maximum; its new-output 12 GiB limit is separate from retained failed v1/v2 output.

The drafts bind actual failed v2 RUN and nonzero exit identities, plus two completed v2 generation receipts each: development/first calibration ordinals 894/897 and 256/899, respectively. v3 requires exact parent sources/rows and rejects a selected calibration already carrying a v2 completion receipt. It preserves the historical v2 failure and does not mark a partial run complete.

## Source union and evidence boundaries

The intended six distinct calibration ordinals are 897, 896, 240, 899, 288 and 304, with exposed development ordinals 894 and 256. Proposed provenance is v2 completed prefixes 897 and 899; new v3 outputs 896, 240 and 304; and separately reviewed v2 partial capture 288. The two v3 development runs repeat neutral checks and do not create additional independent sources. Source 288 has no completed v2 generation receipt: its raw capture, completed video and associated metrics require the independently registered CPU salvage/audit and cannot be inferred from this protocol. The complete eight-source union requires that audit and both v3 actual exit-zero receipts. Retained failed-run and repeated-development work must be included when discussing actual cost.

## Synthetic checks

Independent `python3 -m unittest discover -s tests -p test_e0_original_bf16_full_depth_collect_v3.py` returned exit 0 for three stdlib-only controls. They verify asymmetric per-part compute and byte counts, exact source-index mapping and fixed resources, public namespace isolation/no Torch or NumPy import, and synthetic rejection of incomplete prefix metadata, successful-exit masquerading as a failed parent, and duplicate recapture. Tests did not run a model, use GPU, inspect actual tensor payloads, or establish the source-288 rescue. Actual per-part authorization, source bytes, current execution-worker stat proof, neutral checks and terminal receipts remain runtime requirements.
