# Input-state native video: actual closed results

Four sources, seed 42, five phases each. This is native execution evidence; quality assessment is separate.

| Source | NONE FFN s | H candidate FFN s | X candidate FFN s | H live allocated GiB | X live allocated GiB |
|---|---:|---:|---:|---:|---:|
| 898 | 6.2966 | 7.2553 | 7.4910 | 52.897 | 32.426 |
| 96 | 6.6093 | 7.7308 | 8.0209 | 52.897 | 32.426 |
| 1040 | 6.3616 | 7.2142 | 7.5052 | 52.897 | 32.426 |
| 979 | 6.3730 | 6.9674 | 7.2809 | 52.897 | 32.426 |

Source-equal FFN seconds: none 6.4101, h_cache_dense 7.0135, x_cache_dense 7.0959, high_h_readonly_p64 7.2919, high_x_recompute_p64 7.5745.

Each phase covers 160 FFN sites; candidate phases use 40 sparse sites and X additionally recomputes old W1 at 40 sites. Dense-cache byte controls, per-arm mask identities, transfer requests, actual CPU/GPU/VERIFY receipts and detailed timings are in the JSON.

One seed per source; no quality acceptance is implied.
Cache-boundary live maxima are discrete observations, not continuous arm peaks; CUDA allocator peaks are cumulative and never reset.
Requested transfer bytes are not measured physical bus bytes.
Prior captures and other-scope timings may use different hardware or ordering and are not a fair system comparison.
