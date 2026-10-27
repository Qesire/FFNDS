# UB1 fixed-RR consumer: 20-source native execution

**NATIVE / VIDEO execution.** The primary composite contains 20 development image-prompt sources, one seed per source, and 40 genuine videos: one fresh Dense and one candidate for each source. The 20 pairs have byte-identical saved initial-noise files. Each completed generation records four native flow steps, all 160 distinct step/block FFN positions, four saved FP32 poststates and an 81-frame, 832×464, 16 fps video. The audit checked the file SHA and bytes of every noise, poststate, video, generation receipt and registered run; frame counts are generation-receipt metadata, not a new video decode.

The primary cohort combines the independently authorized recovery producer for sources 0–4 with the original producer's completed parts 1–3 for sources 5–19. All four RUN and launcher receipts report success and zero GPU-monitor violations. Each part completed five pairs, ten generations and 40 flow steps; its detailed predebited work, anonymous HF-relative artifact paths, SHA-256 identities and per-video bindings are in the accompanying [machine-readable report](UB1_CONSUMER_DISJOINT20_NATIVE_RESULTS_20261027.json).

| Primary part | Sources | RUN SHA-256 | Launcher exit SHA-256 |
| --- | --- | --- | --- |
| Recovery 0 | 0–4 | `828c13921abee168fd556457078d6f0629d73ef8e88a4b178308bf7fd05a86fb` | `e6e6a7be7c64119b8c764f5589d9c8d2dc12356e7d42625fc8847a51d261826b` |
| Original 1 | 5–9 | `45b4b7df02785dec719b6779398bcee4f33d671bf52e3c00cae6877e16fe7154` | `855462d9bb658ee620c4fd51d3a23ef358dad90ca0bb899af47913d57da4e7ed` |
| Original 2 | 10–14 | `6bcbdab41716ac07432fce38b6cba6d57e256c0676b563726f164a3c1289a3a0` | `ae5a40139af388c670acd742d3936b2cd7578d16e2778b7f95d695702490b5c4` |
| Original 3 | 15–19 | `6915260bc873f8d614265ffeef895ed4e2b830bdc2543b4c53653b5d1465e553` | `2cafc207cac9fc08ac59519dd40233d097857dfa34d452efc6951f2216ff7a4c` |

The original part 0 remains quarantined after the NVML `nvidia-smi` monitor timeout and nonzero launcher exit. Its three completed generations remain charged to that original attempt and are excluded from the primary 40. Including the 33 completed generations in the initial attempt and 10 in recovery gives **43 completed generations**, not 43 independent observations. For source 0, both modes' old and recovery initial-noise bytes, four poststate file hashes and video hashes agree; this is a technical repeat of the same source and seed, not a new source.

The sources have development exposure. This execution audit does not establish official quality, human acceptance, or held-out population performance. Generation wall measurements combine differing cold/warm setup and observation costs, including saving and hashing, so they do not establish end-to-end speed. The same-source generated videos enable separately registered quality and dynamics assessments.
