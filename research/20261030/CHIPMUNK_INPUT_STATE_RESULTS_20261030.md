# Input-state recomputation: four full-M local cases

Four source/expert cases passed actual CPU preflight, GPU execution and independently supervised typed verification. Sources 898 and 96 are exposed development images, one seed each; high and low block 20 are paired expert cases, not four independent sources. [Data and receipt identities](CHIPMUNK_INPUT_STATE_RESULTS_20261030.json).

All arms used the same P64 mask. A retained BF16 h; B retained FP8 h and a BF16 sample shadow; C retained old input and recomputed selected old W1/GELU; D recomputed full old W1/GELU before selection. C/D selected-old relative norm and max difference, and gated-output relative norm and max difference, were all numerically zero in these cases. They were not byte-compared, so this does not establish bitwise equality. C/D differed from saved selected h; the exact cause remains open.

| Source | Expert | C vs A gated relative L2 | C/D vs saved selected h relative L2 | Max abs vs saved h | A cache+action ms | B ms | C ms | D ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 898 | high | 0.000202 | 0.0000940 | 0.03125 | 24.557 | 35.708 | 30.478 | 43.010 |
| 898 | low | 0.000184 | 0.0000973 | 0.03125 | 24.670 | 35.667 | 30.625 | 43.708 |
| 96 | high | 0.000233 | 0.0000962 | 0.03125 | 24.606 | 35.824 | 30.485 | 43.451 |
| 96 | low | 0.000202 | 0.0000953 | 0.02344 | 24.674 | 35.539 | 30.544 | 43.315 |

The timing columns add each arm’s own cold cache construction CUDA event to the mean of three action CUDA events after one warmup; a fifth action provides comparison output. They are arithmetic sums, not measured end-to-end latency. Mask-hash checks, gates, observers, durable bookkeeping and full-model creation of the old inputs, h and Y are not included in those sums. The JSON preserves each interval and all five actions, five same-device Dense calls and per-expert source-equal averages. No hypothetical repeated reuse is assumed.

Each case showed a cumulative reserved peak of 13,782,482,944 bytes (12.84 GiB) while original h/x, retained outputs and selected-old tensors and a current-arm cache coexisted. This is a diagnostic peak, not a deployment single-arm peak. Actual Dense versus saved teacher is recorded in the JSON and is not presumed byte-identical. Physical disk and uninstrumented metadata reads remain unknown.

This local test does not establish full-video quality or system speed; no video generation or quality scoring occurred in this scope.
