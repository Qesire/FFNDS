# P64 same-stream completion: actual closed results

Two exposed sources, one seed each. Warmups excluded from two measured repetitions per arm. A/B differences are conditional instrumented execution observations; cold NONE is not a repeated SYS control. Source 979 quality failure is not evidence of recovery. Physical bus traffic UNKNOWN.

| Source | A FFN s | B FFN s | B / A | A phase wall s | B phase wall s |
|---|---:|---:|---:|---:|---:|
| 898 | 7.3159 | 6.7927 | 0.9285 | 47.5525 | 47.1724 |
| 979 | 7.5904 | 7.0458 | 0.9283 | 48.5674 | 48.2992 |

Two source-equal means and the mean of within-source B/A ratios are in the JSON; these are two sources with two nested measurements per arm, not four independent sources. Two measurements per arm exclude warmups; raw phase events, four-step infer CUDA and wall sums, per-source differences, receipts, gates, ledger and resources are in the JSON.
