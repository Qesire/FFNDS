# Independent review · original BF16 full-depth collector v2

**SUPPORT_EXACT_V2_SRC_AND_SYNTHETIC_CPU_INTERFACE; NATIVE_RESULTS_PENDING.** This review applies to the exact identities below and supports registration and independent CPU preflight. Real GPU execution, complete capture, video quality and system speed remain determined by their actual runs and receipts.

## Bound versions

- `tools/e0_original_bf16_full_depth_collect_v2.py`: SHA256 `e9bc695f5db69f900869c0efaaa47a815f8adab4ef247147080a820a8abbe075`.
- `tests/test_e0_original_bf16_full_depth_collect_v2.py`: SHA256 `b1ad40ab61a957fe1a65373c0022438ba27a668a264e64c2319329c696bafdd3`.
- Part 0 draft SHA256 `f7e440877a1eca4d5835b6aaa20caa5def58e8f78c3c8b02415963b8702f48d6`; part 1 draft SHA256 `fc7954ad2d6d9f225d49e84ae52f52a9b32a766d6af10e1dfe560805e2c92ef6`. Each draft binds 916 repository sources; independently hashing the local source closure found zero mismatches.
- Frozen original collector remains `3079c7ce5cc40747dbedcb64419026d708973b1e723e558bf292ebaa7d8e4dcf`; the v2 code requires that exact SHA. Both drafts bind new-worker mirror proof RUN SHA `aec88a49ae8760f01a9ac82f571301b6339837c3f00d6e0c046bd7a545da7778` and exit SHA `38907a87e555a4c7b05882a2c6b01b79fc6f99195582284e489794194b02beb0`.

## Reviewed execution and byte accounting

SRC: v2 loads private copies of the frozen v1 collector and original native producer. It retains original cases, ordinal-based rows, native 4×40 FFN operations and BF16 boundaries; v1's failed attempt and its worker/master evidence are not rewritten. Each part still requires one original development neutral with four flow and four post byte matches before three full calibration videos and three CORE reports. The exact registered budget remains 16 flows, 640 FFN sites, 1,280 FFN linears and 648 block-buffer loads per part. Current examples distinguish consumer position and native ordinal: development 0/12 map to 894/256. These source images were historically exposed and are not a frozen independent confirmation set.

SRC: `WrittenBytes` accounts owned output writes incrementally rather than recursively scanning the output tree per packet. It includes snapshot copies, append-only journal bytes, completed and partial named outputs, video holds, and actual JSON writes; atomic JSON replacement first reserves the new temporary bytes while retaining the old target. Payload reservations call the existing background GPU monitor's `check` directly; that method only checks already observed violations and does not issue an additional GPU query. Four CPU threads per part and the registered resources remain checked at execution; disjoint CPU affinity between simultaneous parts is a launcher responsibility.

The three opaque ffmpeg outputs each have a 128 MiB reservation. `RLIMIT_FSIZE` bounds individual output files, including inherited ffmpeg writes, and is restored afterwards; the completed or partial named video is accounted from actual bytes. The per-file restriction is part of this exact v2 scope. Partial/oversized external-file accounting records the observed bytes before raising, and the no-retry failure path preserves its receipt subject to the ordinary ability to write the receipt. Remaining filesystem capacity, time and GPU monitoring are separately enforced. Output accounting and timings are instrumented execution costs, not native throughput evidence.

## New worker master evidence

SRC: the execution master proof is a separate immutable `{protocol, RUN, launch_exit}` binding. Consumption verifies the proof protocol, actual successful exit identity, original parent protocol and model identities, fresh 69,025,362,058-byte read totals, high/low native headers, current private worker configuration SHA and all four current stat records. It uses the recorded fresh SHA proof and only rechecks current stat; no second 69 GB payload hash is needed for each part. The historical parent worker/master proof remains intact. The two draft bindings identify the same independent proof; this review did not rerun the proof or read model payloads.

## Synthetic controls and claim boundary

Independent `python3 -m unittest discover -s tests -p test_e0_original_bf16_full_depth_collect_v2.py` returned exit 0 for ten portable controls. A separate temporary-small-file control checked atomic replacement peak accounting, payload-reservation monitor invocation and hold release, journal byte counts, and absence of recursive output traversal. These controls used no model, GPU or real large tensor. They establish the tested interfaces and failure behavior, not actual GPU isolation, completed video outputs, per-part CPU preflight or full-cohort results; those require the registered runs and actual exit receipts.
