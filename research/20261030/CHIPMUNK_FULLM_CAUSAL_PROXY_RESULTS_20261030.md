# Chipmunk full-M causal proxy: four source/expert pairs

Two exposed sources and two nested expert pairs per source use actual 31,668 raster rows and one shared K5120 mask. Results are local teacher-state diagnostics; they are not free-running video or population quality evidence.

| Source | Steps | Mean | P4 | P64 | Random | RMS oracle |
|---:|---|---:|---:|---:|---:|---:|
| 898 | 0→1 | 0.469899 | 0.462221 | 0.426487 | 0.547651 | 0.395269 |
| 898 | 2→3 | 0.302981 | 0.277083 | 0.261199 | 0.305264 | 0.251191 |
| 96 | 0→1 | 0.432408 | 0.408992 | 0.374251 | 0.470129 | 0.351995 |
| 96 | 2→3 | 0.299165 | 0.273750 | 0.256121 | 0.295667 | 0.246337 |

| Source | Steps | Dense CUDA ms | Mean CUDA ms | P4 CUDA ms | P64 CUDA ms | Random CUDA ms |
|---:|---|---:|---:|---:|---:|---:|
| 898 | 0→1 | 34.782 | 201.574 | 252.713 | 186.536 | 185.358 |
| 898 | 2→3 | 34.985 | 192.468 | 247.918 | 179.246 | 179.714 |
| 96 | 0→1 | 34.633 | 200.017 | 254.950 | 187.608 | 184.931 |
| 96 | 2→3 | 35.287 | 194.316 | 248.634 | 179.836 | 177.467 |

Source-balanced means (equal experts within each of two sources, then equal sources): mean 0.376113, p4 0.355511, p64 0.329515, random 0.404678, oracle 0.311198.

P4 and P64 use the first 4 and 64 rows of the same pre-frozen random permutation. The oracle uses full current h after the online arms and is an extra paid hindsight diagnostic. Sample-to-full RMS estimates, captured energy, saved-h/Y replay and all per-pair CUDA/host timings are retained in the [whitelist JSON](CHIPMUNK_FULLM_CAUSAL_PROXY_RESULTS_20261030.json).

Each online CUDA event covers the selector, input gather, cache copies, selected W1/GELU and W2/cache add, and current gate. Ledger and disk persistence are outside the event. Old-cache construction, cold kernel setup, same-M dense reference and oracle costs are separate; one-shot local events are not end-to-end system or video latency.

Each part has a real exit 0, closed RUN/EXIT SHA and a matching full operation ledger. Inputs are previously exposed development sources; no new videos or weights were generated.
