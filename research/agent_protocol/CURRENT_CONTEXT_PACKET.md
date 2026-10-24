# FFNDS · current research evidence packet

**Only current research entry:** [ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json), source pinned `FFNDS@63f7272916baf997937691ccbc7086a61f55861e` and `AutoResearcher@5bd7ec537ae63f2b6a6dd579f0e886aff58a177c`. Older dated "ACTIVE" handoffs remain past observations, not task selectors.

## North star and actual open scientific fork

Lower full FFN plus whole-generation cost under competitive native four-step I2V video-quality constraints. Current questions: **Q1** What useful information exists before a calculation? **Q2** Which information improves cost-adjusted choices? **Q3** Which work is truly disposable after propagation?

**Immediate research choice:** Distinguish predicting the deterministic *local* FFN output from predicting the *downstream value* of a low-cost action under pre-normalization residual/attention state. Recover the true decision-boundary variables before inventing a selector or using older dense-restoration labels.

## Verified source object

Original LightX2V Wan2.2 I2V 4-step distilled D-Q BF16/HiF4, 5120→13824→5120 FFN, native Q1/GELU/Q2, norm2/AdaLN, attention update, residual/gate. Upstream source-file hashes are bound in [C0 data provenance](../20261023/C0_coupling_offline_results.json). The [FFNDS original operator reference](../../tools/packed_ffn_reference.py) documents one *particular* packed 108/216 group baseline, not the full operator search space. Model time values and sigma schedule require original [runtime audit](../20261022/R12_E0_runtime_reconstruction.json).

## Evidence used to decide the next source read

- **Math:** With exact fixed weights/quantization and full post-Q1 `q`, local FFN activation is deterministic: upstream signal cannot add conditional information about that local target. It may affect downstream state through residual bypass or help when online selection only sees a cheap sketch of `q`.
- **Synthetic constructed counterexample:** [C0 finite scale rank reversal](../20261023/C0_coupling_offline_results.json) has equal normalized BF16 inputs at residual scale 1 and 4 but different best next-norm finite-action choices. It establishes logical possibility, not its frequency in Wan.
- **Scoped native past:** R7 two-action finite downstream scalar interactions and [C0 old R17 mask/exposure](../20261023/TACR_sampling_evidence_v2.json) indicate what *was recorded*, not which new action works. R17 independent full-video official quality remains incomplete.
- **Unavailable in old saved OP2 headers:** pre-norm residual, already-computed attention increment, matched native *new low-cost action* finite continuation. Do not infer them from V1/dense/UB1 labels.

## Next constructive discriminator (proposal, no experiment started)

Specify at least two different downstream action-value predictions, then audit actual `infer_ffn` data dependencies. First check whether existing norm output/statistics plus attention update can be exposed with cheap readout and where native QDQ makes them available. For a future *authorized* test, compare metadata-only → norm statistic → attention innovation → q sketch, with actions and physical budget held fixed; test whether cost-adjusted action preference actually changes. A signal improving hidden R² without reducing decision regret answers Q1 but not Q2.

The separate physical-operator search starts when the information/decision mechanism gives a reason to select its granularity. In the meantime mathematical explorations may freely propose competing mechanisms. Dense completion and R256 are optional charged candidates, not prerequisites.

**Permissions:** source/math/literature and administrative writes only. [Explicit experiment pause](../20261024/EXPERIMENT_PAUSE.json) forbids CPU/GPU/model experiments, queued retries and automatic experimental followups; eligible model methods empty.
