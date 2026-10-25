# Web-Augmented Research Discovery (WARD v2.4)

**Scientific default:** For a substantive research, literature, mechanism, prior-art, frontier or field-exploration question, proactively search the web when tools are available. Search is a source of **new hypotheses, mechanisms, mathematical formulations, hard counterexamples and original implementations**. The desired output is a better scientific explanation, not a bibliography or an arbitrary quota of queries.

## 1. Breadth before choosing an algorithm

Start from the research **phenomenon or question**, identify alternative vocabularies and search across several distinct families: seminal theory, contemporary papers and preprints, adjacent fields, implementations and official specifications, and contradictory/negative findings. Adapt the map to the domain; consider mathematics, statistics, information and computation, empirical phenomena and systems constraints as distinct lenses.

Use conceptual, mathematical, implementation-level and failure-mode descriptions, alternate names and cross-language terminology when valuable. Explore related older work as well as recent claims; confirm publication/venue/version dates when currency matters. A paper about an intriguing operator is a **scientific seed**, not an instruction to switch the project's goal or preferred algorithm.

## 2. Deepen by reading the primary mechanism

Perform **multi-pass, iterative web research**, expanding the next query from what the previous source actually taught:

1. **Map:** discover the nearest major families and competing explanations; locate original work and strong counterexamples.
2. **Derive:** examine decision-relevant original **equations, method sequence, data definitions, conditions, experiments, ablations and limitations**. Read author code/config/official hardware or format specifications for implementation-level claims.
3. **Trace:** follow backward references, forward citations, successor versions, author repositories and near-neighbor papers. Re-search new mathematical relations, equivalent transformations and failure mechanisms rather than just more title synonyms.
4. **Adversarial search:** actively find the closest potentially equivalent work, the strongest disconfirming evidence and a different mechanistic account of the same observation.
5. **Integrate:** return to the real target object's source/trace/data, explain **what changed in scientific understanding** and select the most informative next proof, source read or observation.

Use arXiv, OpenReview, official conference proceedings, author sites, Semantic Scholar/OpenAlex, Google Scholar if accessible, original code repositories and official vendor specifications as appropriate. Index snippets can guide retrieval; mathematical and systems claims should point to the primary article, source or specification.

**Research sufficiency is adaptive:** stop broadening when additional searches cease to change the mechanism map or nearest prior-art interpretation. A narrow factual source check need not become a multi-hour survey. A deep novelty or direction-setting task warrants multiple conceptual families and citation tracing.

## 3. Mechanism-level prior art, not keyword resemblance

For each close work compare:
- **Object:** model/data/task/hardware, deployment context, step or spatial regime.
- **Information:** what signal is known *before* an action, what is oracle/post-hoc or available only after computation, and its acquisition cost.
- **Mathematics:** assumptions, function/representation, nonlinear and quantization ordering, exactness/approximation, distinct causal prediction.
- **Action:** actual computation or parameter changed, resource costs, static/learned/dynamic intervention.
- **Evidence:** baselines, metrics, hold-outs, source code behavior, positive and negative results.

Classify relationship explicitly: same underlying mechanism; shared known ingredient; different mechanism addressing similar outcome; transferable analogy; unresolved pending primary evidence. An unfamiliar title is not proof of novelty, and a shared technique does not make the full method identical.

## 4. Traceable and economical research output

For a substantive literature-informed scientific conclusion record **search date, actual tools/sources used, complementary query families or citation paths, primary reference identifiers/links and decisive equations/code, the closest competing mechanism, contradictory findings, and uncovered areas**. Evidence types include PRIMARY_PAPER, OFFICIAL_SOURCE_OR_SPEC, OBSERVED_DATA, CONDITIONAL_THEOREM, SECONDARY_INDEX, HYPOTHESIS.

Show the useful **discovery or revised mechanism first**, then concise source evidence, strongest alternative, and the next discriminator. A table is useful only when it changes a decision. Read documents selectively and cite what was truly inspected.

### Browsing unavailable

When the runtime has no web tools or network access, continue with already-accessible papers, local source and algebra. Mark real-time web/citation expansion as **not performed**, preserve specific searches to be done through an authorized browsing environment, and keep novelty claims limited to checked sources. Never fabricate paper metadata, consulted versions, source inspections or browsing success.

### Authorization and collaboration

Research browsing is read-only information retrieval. External writes, paid service actions and model/GPU experiments follow their own authorization. Independent agents should search *different* neighboring mechanisms and try distinct counterexamples; reviewers independently inspect primary versions rather than treating another agent's list as a second verification.

## 5. Generative integration with AutoResearcher

The workflow is **OPEN → WEB_MAP → SOURCE/MATH → COMPETING EXPLANATIONS → DECISIVE CONTRAST → SCIENTIFIC HANDOFF**. Ordering is adaptive: a mathematical insight may precede search, while a surprising paper may initiate a new question. During open discovery, search broadens possible explanations; during an implementation/novelty claim it deepens original-code validation. Link to `core/AGENT_HARNESS.md`, `core/EXPANSION_PROTOCOL.md`, `core/PROBLEM_REPRESENTATION_DISCOVERY.md`, and `core/PROMPT_EVALUATION.md` for the context and evidence contract.
