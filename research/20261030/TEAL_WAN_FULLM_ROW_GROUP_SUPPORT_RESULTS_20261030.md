# TEAL Wan full-M row-group support, 2026-10-30

Natural contiguous rows have more shared post-GELU support than the same rows under a fixed shuffle. For the fixed q50 mask, grouping 16 natural rows retains **97.289%** of dense W2 logical MACs under an ideal per-group column-union implementation; the shuffled comparison retains **98.582%**. With 32 rows the corresponding figures are **98.049%** and **99.295%**. These are equal-weight means over two development sources after averaging four sites within each source. The largest saving among these source-equal aggregate conditions is therefore about 2.7%, before any cost of identifying or executing the groups.

| Fixed threshold | Rows per group | Natural retained MACs | Shuffled retained MACs |
|---|---:|---:|---:|
| q25 | 16 | 99.126% | 99.647% |
| q25 | 32 | 99.335% | 99.814% |
| q50 | 16 | 97.289% | 98.582% |
| q50 | 32 | 98.049% | 99.295% |

**Every full group had zero entirely empty K16 and K64 tiles**, for both thresholds, both group sizes, both row orders and all eight observed tensors. The short final groups also had zero entirely empty K16/K64 tiles. M16 has 1,979 full groups and a four-row tail; M32 has 989 full groups and a 20-row tail. MAC accounting weights each group by its actual row count and does not treat padding as observed zero rows.

The source-level effect is not uniform across sites. The following q50 values are retained logical MAC percentages; the [structured results](TEAL_WAN_FULLM_ROW_GROUP_SUPPORT_RESULTS_20261030.json) retain all q25/q50 case summaries, group-distribution statistics, tails, original input bindings and measured CPU timing components.

| Source | Step | Block | Natural M16 | Shuffle M16 | Natural M32 | Shuffle M32 |
|---|---:|---:|---:|---:|---:|---:|
| 1040 | 0 | 0 | 94.992 | 97.928 | 96.108 | 99.204 |
| 1040 | 0 | 20 | 99.688 | 99.927 | 99.906 | 99.993 |
| 1040 | 2 | 0 | 94.463 | 96.399 | 96.080 | 97.931 |
| 1040 | 2 | 20 | 99.792 | 99.950 | 99.960 | 99.993 |
| 979 | 0 | 0 | 95.056 | 97.855 | 96.158 | 99.164 |
| 979 | 0 | 20 | 99.835 | 99.965 | 99.959 | 99.998 |
| 979 | 2 | 0 | 94.587 | 96.649 | 96.252 | 98.083 |
| 979 | 2 | 20 | 99.897 | 99.980 | 99.971 | 99.997 |

The input comprised eight real `[31668,13824]` BF16 post-GELU tensors: source 1040 and 979, each at four fixed teacher positions. The q25/q50 thresholds were frozen from the six-source TEAL calibration. Each mask was retained elementwise; the union only described which columns a hypothetical group executor could omit. One PCG64 seed-42 permutation was generated and reused for both sizes, thresholds and all cases; gather and scan CPU times were recorded separately. The actual paid ledger completed 16 masks, one permutation and 95,040 row-group unions, with one full file SHA, one raw H tensor SHA and one H analysis scan per input. The process completed in 57.464 seconds on four CPUs, with actual EXIT 0. The detailed RUN includes complete per-group vectors and is bound by SHA in the structured report.

This result supports a local row-correlation observation and limits this particular fixed-mask, contiguous K16/K64 action at the sampled positions. The column-union MAC figure assumes an ideal implementation and excludes selection, gathering, sparse metadata and execution overhead; it is not measured GPU speed. The two sources have prior development exposure, the four sites are nested within source, and neither unseen-source generalization nor all-160 behavior follows from these measurements. Existing TEAL video evidence is separate from this CPU support diagnostic.
