# Original BF16 saved temporal row coordinates

**OBS:** All 1,280 existing source×step×block packets contain an actual 256-element I64 `rows` tensor. A read-only CPU replay checked each row tensor's raw SHA against the independently audited manifest, its packet metadata, the actual recorded `[21,29,52]` grid and the frozen source/step/block draw. It read the manifest and metadata plus exactly 2,048 requested row bytes per packet, using unbuffered packet reads for the confirming replay; no h/X/Y or other operand tensor was requested. The original Wan pre-inference `flatten(2)` establishes `row = τ·(29·52) + y·52 + x`. The [machine result](RAW_BF16_TEMPORAL_ROW_REPLAY_20261029.json) records all individual source/step/block statistics.

| Actual statistic per 256-row packet | Mean | Range | Total across 1,280 packets |
|---|---:|---:|---:|
| Distinct spatial `(y,x)` coordinates | 236.356 | 222–248 | 302,536 |
| Same `(y,x)`, different latent-τ unordered pairs | 20.616 | 8–39 | 26,389 |
| Adjacent latent-τ pairs at same `(y,x)` | 1.962 | 0–6 | 2,511 |
| Complete all-21-τ spatial trajectories | 0 | 0 | 0 |
| Fully covered aligned M16 or M128 native row tiles | 0 | 0 | 0 |

The adjacent count is zero in 178 packets. These counts are local to one saved source, denoising step and physical block: they do not establish a single continuous full-video trajectory or add independent sources. There are usable sparse same-coordinate comparisons for an exploratory S2 analysis, including nonadjacent temporal pairs, but these packets cannot furnish a complete 21-position trajectory or a fully captured native M16/M128 tile. Full temporal or physical-tile studies therefore need a suitably matched capture; one should not reconstruct missing positions from different teacher states.

The saved locators are not shared across positions: among 24,960 same-source/same-step block pairs, 1,920 same-source/same-block step pairs, and 4,480 different-source/same-step/same-block pairs, zero pairs have identical 256-row sets. Their mean exact-row intersections are respectively 2.061, 2.086 and 2.084. The recorded sampler uses a distinct deterministic seed from source ordinal, step and block; equality or overlap across these axes must be checked rather than assumed.

The reproducible stdlib script is [raw_bf16_temporal_row_replay.py](../../tools/raw_bf16_temporal_row_replay.py). An initial buffered pass gave the same result but could have caused file read-ahead; the second unbuffered pass returned byte-identical results and is the direct-read evidence reported here. OS/storage caching behavior is not measured. These are row-coverage observations, not measured FFN function, temporal quality or speed.
