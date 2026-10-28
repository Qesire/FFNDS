# B-F frozen 6016 terminal results · 2026-10-28

Four fixed-basis students completed on two terminal expert functions. Each result is the mean of two steps within each source, then equal-weighted across 16 train or 8 historically exposed held sources. The reference is native-captured teacher Y; FFN evaluation uses M256 and the head uses original full-M geometry with synthetic zero-filled other rows.

| Expert / basis | Held initial head | Held final head | Held oracle head | A0 energy head | Held initial Y | Held final Y | A0 energy Y | Final head wins / 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| high_noise_model / PCA512 | 0.00059891765 | 0.00058653085 | 1.3055483e-05 | 0.00060878882 | 0.0074124552 | 0.0068946322 | 0.008392523 | 7 |
| high_noise_model / HEAD128_PCA384 | 0.00060514368 | 0.00058980605 | 1.0268295e-05 | 0.00060878882 | 0.0074992523 | 0.0070094619 | 0.008392523 | 7 |
| low_noise_model / PCA512 | 0.00065145554 | 0.000638192 | 1.1292529e-05 | 0.00066096615 | 0.0091029552 | 0.0077319171 | 0.009983403 | 8 |
| low_noise_model / HEAD128_PCA384 | 0.00065800211 | 0.00063285716 | 6.438837e-06 | 0.00066096615 | 0.0092433229 | 0.0079233841 | 0.009983403 | 8 |

Oracle is projection of teacher Y into the frozen output basis; it is a representation reference, not a trained student. The accompanying JSON contains train16 and held8 initial/final/oracle means, individual source values, and the continuous direct, radial and signed-cross terms plus the actual BF16 remainder and signed remainder cross. Those components close the head SSE; the remainder is not asserted to be an irreducible floor.

Actual worker exit 0, no timeout; RUN SHA256 `12bc19935b69ab0e5207350ab9241e95fc2718b4268ae7a1a5b3bf2e584d0dc6`, exit receipt SHA256 `97b6967e3ffca55a101c4305ccb66d776d20ff4575b329b4e7de439b3168408a`, frozen protocol SHA256 `8a74628b8160d237b42ed4badc20b8aa7602d0d2431e6b36a7c6cc4e270d06c3`. All predebited, invoked and completed counts equal frozen limits: 1,024 Adam updates, 192 oracle projections, 96 neutral bitwise checks, 96 A0 Q2 and Y SHA matches, 672 full-M head calls, and four student exports with reload. No full-model or video generation was performed.

The static deployed per-row MAC estimate is 36,503,552 versus A0 61,603,840 (0.59255×), with three BF16 GEMMs and dynamic Q1 and own Q2. This is a local arithmetic comparison, not measured system speed. A0 recovery and support costs are inherited; four students add 1,024 updates. Full-video quality, system speed and broader 80-function generalization remain unmeasured.

[Machine-readable results](BF_FROZEN6016_TERMINAL_RESULTS_20261028.json).
