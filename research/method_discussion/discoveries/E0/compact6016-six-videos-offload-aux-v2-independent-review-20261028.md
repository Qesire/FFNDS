# Compact6016 six-video offload auxiliary v2: independent review

**SUPPORT — SRC and tiny CPU software controls for the exact versions below.** The reviewer ran no real model, GPU, real tensor analysis, or bank-payload audit. Native v2 success and video quality remain matters for the registered execution and its actual exit receipt.

| Object | SHA-256 |
| --- | --- |
| `tools/e0_compact6016_six_videos.py` | `b7764eecd05552f5dc4fb33175055fd4d4c054cd27a6b574e14c28c79e2f1347` |
| `tests/test_e0_compact6016_six_videos.py` | `68d3b92e7870517e6e4be0e5b41a84bf3294b54257260641a04f20e33e619545` |
| `config/e0_compact6016_six_videos_20261028_v2_draft.json` | `02af8441007df10c65495aafc0772d8b3be984d95d8af8b08455266a10cade81` |

The actual source explains the v1 interface failure. `RMSWeightTemplate._get_lora_attr_mapping` constructs a zero scalar on `AI_DEVICE`; `utils.state_dict` includes existing auxiliary attributes alongside pinned CPU checkpoint weights. Wan explicitly passes `lora_prefix=block_prefix`, producing `blocks.<b>.<attention>.norm_<q/k>.diff`. Treating every source tensor as a CPU checkpoint tensor rejected this legal native state.

V2 binds the four actual source RMS owners, their base/auxiliary mappings, explicit prefix/key and tensor identity. Its only accepted non-CPU values are those original CUDA BF16 scalar objects. It passes the original dictionary and values to the unchanged native copy. The observer reports bytes by source device and does not reinterpret CUDA auxiliary bytes as checkpoint H2D traffic. Zero initialization is SRC evidence; the observer adds no numeric GPU value read, tensor move or synchronization.

Independent execution of `unittest discover -s tests -p test_e0_compact6016_six_videos.py -v` passed **30 tests, exit 0, 8.163 seconds**. New controls execute the original state-dictionary/copy AST with device-metadata fixtures, reproduce the mixed CPU/CUDA counterexample, and verify unchanged dictionary/value identities. Wrong CUDA matrices, scalar shapes/dtypes, replacement objects and incorrect default prefixes fail. An invalid source retains its predebit and never invokes the original copy. Compilation, draft inspection and whitespace checks passed. All 79 software bindings matched their actual sizes and SHAs in a separate stdlib-only check.

Comparison with v1 source `526f0e929c56057c3245fd877194a534e09f9c31a90c6b8b1aea546dd60dd3a4` in commit `31d7b58` confirms unchanged `LIMITS`, `VIDEO`, `BANK_SHA`, modes and sources. The new scope/run IDs and output path preserve v1's failed identity. Existing hook-zero/fallback, complete 80-function coverage, own-Q2, paired real noise, free trajectory, disjoint bank residency and instrumented timing gates remain. Per the coordinator's explicit session context, job1/session2 and the repository's dynamic policy with null global cap are distinct checks.

The separate BH–A0 draft freeze seam was asset-list duplication: both direct native libraries and two build copies were inherited, giving four `.so` entries where the driver requires two. No required source was missing. The coordinator's corrected unexecuted v1 protocol `766e83105775bfccc2862a92f7b24b55b28e4a2ebcf93c92c2070ef739994714` passed the actual frozen structure validator with the two direct libraries. BH–A0 driver/tests remained unchanged; actual byte preflight and execution are separate gates.
