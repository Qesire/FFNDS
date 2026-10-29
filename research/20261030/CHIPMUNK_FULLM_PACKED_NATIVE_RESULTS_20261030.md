# Chipmunk full-M packed BF16 native control

Two previously exposed sources, each with two expert pairs, use true 31,668-row inputs and one K5120 mask. The mask is required to match the previously closed causal-proxy run. Each selected action pays for selection, mask identity verification, gathers, full cache copies/update, native BF16 W1 and W2 and gate.

| Source | Steps | Arm | Packed gated relative L2 (3-repeat mean; oracle once) | Same-mask Triton vs Dense | Packed vs Triton relative L2 | Packed vs Triton max abs |
|---:|---|---|---:|---:|---:|---:|
| 898 | 0→1 | mean | 0.469899 | 0.469899 | 0.000000 | 0.000000 |
| 898 | 0→1 | p4 | 0.462221 | 0.462221 | 0.000000 | 0.000000 |
| 898 | 0→1 | p64 | 0.426487 | 0.426487 | 0.000000 | 0.000000 |
| 898 | 0→1 | random | 0.547651 | 0.547651 | 0.000000 | 0.000000 |
| 898 | 0→1 | oracle | 0.395269 | 0.395269 | 0.000000 | 0.000000 |
| 898 | 2→3 | mean | 0.302981 | 0.302981 | 0.000000 | 0.000000 |
| 898 | 2→3 | p4 | 0.277083 | 0.277083 | 0.000000 | 0.000000 |
| 898 | 2→3 | p64 | 0.261199 | 0.261199 | 0.000000 | 0.000000 |
| 898 | 2→3 | random | 0.305264 | 0.305264 | 0.000000 | 0.000000 |
| 898 | 2→3 | oracle | 0.251191 | 0.251191 | 0.000000 | 0.000000 |
| 96 | 0→1 | mean | 0.432408 | 0.432408 | 0.000000 | 0.000000 |
| 96 | 0→1 | p4 | 0.408992 | 0.408992 | 0.000000 | 0.000000 |
| 96 | 0→1 | p64 | 0.374251 | 0.374251 | 0.000000 | 0.000000 |
| 96 | 0→1 | random | 0.470129 | 0.470129 | 0.000000 | 0.000000 |
| 96 | 0→1 | oracle | 0.351995 | 0.351995 | 0.000000 | 0.000000 |
| 96 | 2→3 | mean | 0.299165 | 0.299165 | 0.000000 | 0.000000 |
| 96 | 2→3 | p4 | 0.273750 | 0.273750 | 0.000000 | 0.000000 |
| 96 | 2→3 | p64 | 0.256121 | 0.256121 | 0.000000 | 0.000000 |
| 96 | 2→3 | random | 0.295667 | 0.295667 | 0.000000 | 0.000000 |
| 96 | 2→3 | oracle | 0.246337 | 0.246337 | 0.000000 | 0.000000 |

| Source | Steps | Dense 3-event mean ms | Mean ms | P4 ms | P64 ms | Random ms |
|---:|---|---:|---:|---:|---:|---:|
| 898 | 0→1 | 35.524 | 23.699 | 23.666 | 23.604 | 23.192 |
| 898 | 2→3 | 38.662 | 24.092 | 23.965 | 23.853 | 23.450 |
| 96 | 0→1 | 37.110 | 24.003 | 23.760 | 23.794 | 23.667 |
| 96 | 2→3 | 37.660 | 23.783 | 23.781 | 23.730 | 23.266 |

The source-balanced gated errors are mean 0.376113, p4 0.355511, p64 0.329515, random 0.404678, oracle 0.311198. Each expert is averaged within its source before averaging sources. The oracle uses complete current activation after the causal arms and is a paid hindsight reference.

Mask SHA includes GPU-to-CPU synchronization and host SHA; it is included in online events and host intervals and separately reported; it is an extra diagnostic check, not intrinsic selector cost.

Each online and Dense arm has one explicit cold plus three measured invocations in fixed order. All selection, gather, packing, cache cloning/update, both GEMMs and current gate are inside the online CUDA/host interval; ledger and disk persistence are outside. Old-cache build and full-current-h oracle selection/action are separately charged. Five additional Triton same-mask actions are separately charged. These local measurements do not establish system speed or video quality.

Per-repeat CUDA/host intervals, cold and old-cache costs, oracle and Triton checks, actual counts and receipt identities are preserved in the [whitelist JSON](CHIPMUNK_FULLM_PACKED_NATIVE_RESULTS_20261030.json). Saved-h/Y replay and numerical differences reflect their respective BF16 paths. Results are not full-system latency or video quality evidence.
