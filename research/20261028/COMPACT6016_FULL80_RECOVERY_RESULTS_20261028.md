# Compact6016 full80 local recovery · 2026-10-28

**NATIVE local recovery:** the 160-cell metadata bank assembles 40 original cells, 54 cells retained from the failed completion scope, and 66 newly completed cells. The assembler exited 0; its output retains `METADATA_COMPLETE_TENSOR_LOAD_PENDING`. Original failed scopes and the completion parents' `UNRECOVERED` return-code identity remain unchanged.

For the actual BF16-gated relative SSE, averaging the two expert steps within source, sources equally, then the 80 expert/block functions equally:

| Source split | Energy94 initial → final | Random94 initial → final | Functions where energy final is lower |
| --- | ---: | ---: | ---: |
| Train16 | 0.314963 → 0.129801 | 0.375757 → 0.136573 | 65/80 |
| Heldout8, historically exposed | 0.314906 → 0.129766 | 0.377537 → 0.138127 | 73/80 |

On exposed heldout8, final energy/random means are 0.140715/0.151933 for high expert and 0.118816/0.124320 for low expert. The largest heldout final means are low block 5 energy (0.257252) and high block 29 random (0.259851). These 80 functions are nested model positions, not 80 independent sources; the independent units are the historical 16 training and 8 exposed diagnostic sources. This is local teacher-state recovery, not a free-video quality or latency result.

[Machine-readable result](COMPACT6016_FULL80_RECOVERY_RESULTS_20261028.json) includes four terminal block-39 result/export bindings, group summaries, and lineage. All RESULT metadata and source-step diagnostics were read and checked; deployment tensor payload loading/byte validation remains a separate gate.
