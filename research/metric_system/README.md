# Academic families and execution profiles

[Chinese reference](../../docs/77_FFNDS学术指标族与评测协议_20261023.md) · [93-entry JSON](academic_metric_families.json) · [CSV](academic_metric_families.csv) · [evaluation protocol template](evaluation_protocol_template.json). These define 10 measurement families; `catalog.json` remains the active 34-metric observation/claim registry. Catalog entries specify academic origin versus project adaptation; official evaluators perform the actual video scoring.

The two records in `records/` map 17 frozen R14N0/R14 observations. Check with `python tools/check_metric_contract.py`; run `python -m unittest discover -s tests -p "test_metric*.py"` and the candidate-rollout tests. A full independent I2V claim uses nine dimensions and linked reviewed rollout; partial suites, failures and synthetic timing have dedicated diagnostic claims.

# FFNDS metric registry

Read [the literature-aligned metric hierarchy](../../docs/76_基于学术文献的FFNDS分层指标体系_20261022.md). `catalog.json` defines measurement levels L0–L5, ordinary video metrics and strict claim separation. `metric_record_template.json` is a **synthetic schema example**, not new experimental evidence.

A new record is append-only per declared scope, explicitly states measurement grain, population, reduction/denominator, baseline, source exposure, model state propagation, source/implementation revision and physical evidence. Do not retroactively reinterpret R14N0 local SSE, R11 forced-flow or R14 random tensors as free video.

Run `python3 tools/check_metric_contract.py` and `python3 -m unittest discover -s tests -p 'test_metric_contract.py' -v` before publishing new metric claims. Static checks cannot verify human video judgement, actual data or statistical methodology. Video promotion also requires `tools/check_candidate_rollout_gate.py`, separately approved GPU execution and full-video human review.

## R16: using the metrics, not just listing them

See [metric usage guide](../../docs/80_R15指标实际使用验收与Agent执行指引_20261023.md), [deterministic report](../20261023/R16_metric_usage_acceptance.json), and `python tools/verify_r15_metric_usage.py --check`. The verifier replays native R14N0 scalar mappings and checks small synthetic PSNR/ROI/stripe/aggregation fixtures. Official VBench and actual new-video/native-speed metrics remain unmeasured and must not be inferred.
