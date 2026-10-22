# FFNDS metric registry

Read [the literature-aligned metric hierarchy](../../docs/76_基于学术文献的FFNDS分层指标体系_20261022.md). `catalog.json` defines measurement levels L0–L5, ordinary video metrics and strict claim separation. `metric_record_template.json` is a **synthetic schema example**, not new experimental evidence.

A new record is append-only per declared scope, explicitly states measurement grain, population, reduction/denominator, baseline, source exposure, model state propagation, source/implementation revision and physical evidence. Do not retroactively reinterpret R14N0 local SSE, R11 forced-flow or R14 random tensors as free video.

Run `python3 tools/check_metric_contract.py` and `python3 -m unittest discover -s tests -p 'test_metric_contract.py' -v` before publishing new metric claims. Static checks cannot verify human video judgement, actual data or statistical methodology. Video promotion also requires `tools/check_candidate_rollout_gate.py`, separately approved GPU execution and full-video human review.
