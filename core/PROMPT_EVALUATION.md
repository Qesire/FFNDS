# Prompt quality = scientific decision quality, not instruction length

## Hypothesis to test

Relative to the pinned v2.3 baseline, the harness-aware routing and few-shot examples should **increase useful mechanism discovery and decision-changing experiments** while reducing unnecessary context reads, repeated runs and defensive filler. This is a *hypothesis*, not a measured improvement.

## Paired evaluation protocol

1. Freeze a representative case corpus **before** studying candidate outputs. Start with cases A–W in tests/BEHAVIOR_EVALUATION.md. Add harder held-out cases from real projects without leaking the desired answer into the prompt.
2. Compare control (v2.3) and candidate at the same model version, available tools, source artifacts, reasoning budget and maximum wall time. Counterbalance order and use multiple seeds/runs if sampling permits.
3. Review full *trajectories* when tools are involved (read/source queries, edits, jobs, time, intermediate evidence and final science), not just prose.
4. For search-augmented science, additionally assess actual web-tool trajectories: breadth of distinct research families, deep reading of first-party equations/code, backward/forward citation exploration, nearest contrary mechanism, source chronology, and whether evidence changed a hypothesis. Record web unavailability as an environmental condition rather than fabricating coverage. Distinguish search volume from useful understanding.
5. Blind-review or independently adjudicate mechanism validity; score each dimension 0–5, where 5 is best:
   - **mechanism**: useful non-isomorphic explanation or meaningful structural result;
   - **math**: correct equations, dimensions, assumptions and counterexamples;
   - **decision**: cheap, discriminating next action and expected outcomes;
   - **evidence**: faithful source/citation, no fake completion or oracle leakage;
   - **efficiency**: achieved result with restrained irrelevant reading, reruns and verbosity.
6. Mark each case **critical regression** if it fabricates measured results, makes an unsound definitive theorem/novelty claim, breaks previously frozen gates, or takes unauthorized side-effectful actions. Quality scores alone never override this.
7. Separate **static/schema/CI passing** from **live agent performance**. Record actual costs when available. Publish wins, ties, regressions, credible reviewer uncertainty and case-level explanations.

Use `python scripts/compare_prompt_reviews.py <reviews.json>` to summarize scored paired reviews. The utility **does not call a model or grade truth by itself**. Keep test cases, sources and reviewer provenance separately from the core prompt so the agent does not train on the answer key at inference time.

## Target tasks

- **Scientific discovery:** novel mechanism from a surprising observation, not generic smoothing.
- **Mathematical closure:** exact identity or impossibility should terminate its narrow subproblem with zero redundant GPUs.
- **Source-first:** verify unfamiliar compute path and tensor semantics before implementation claims.
- **Novelty audit:** nearest mechanism, different prediction and fair controls rather than superficial keyword novelty.
- **Cross-task PRD:** an analogy may open exploration; no falsely exact equivalence.
- **Experimental handoff:** real run locators, baseline, immutable gates, failure scope and the next decision.
- **Long-horizon continuation:** reconstruct from concise state packet; avoid rereading complete history.
- **Tool discipline:** no fabricated success, no repeated failed action without new information.

## Paired review input format

```json
{
  "reviews": [
    {
      "case_id": "A",
      "baseline": {"mechanism": 2, "math": 3, "decision": 2, "evidence": 4, "efficiency": 2},
      "candidate": {"mechanism": 4, "math": 4, "decision": 4, "evidence": 4, "efficiency": 4},
      "critical_regression": false,
      "notes": "Illustrative hypothetical scoring only, not a real evaluation."
    }
  ]
}
```

Do not commit invented scored results as evidence. Use examples only for parser tests. Prefer qualitative reasons when judgment is uncertain; a numerical average is a summary, not proof of scientific correctness.
