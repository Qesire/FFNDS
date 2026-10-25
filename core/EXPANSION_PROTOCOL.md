# Mechanism Expansion Through Constructive Inquiry (MEAR v2.3)

## Literature as a generator of new scientific questions

During scientific expansion, use [Web Research Discovery](WEB_RESEARCH_DISCOVERY.md) by default when tools are available. Follow seminal/current papers and adjacent mathematical or engineering communities, search contradictory observations and differently named equivalent mechanisms, and trace references in both directions. Extract the structural observation or reformulation from a paper first; preserve direct algorithm copies as known baselines and develop new competing predictions from that insight. Record actual consulted papers/code and change in mechanism understanding.


## Principle

When evidence surprises us, **generate an explanation that unlocks a new experiment, representation or algorithmic degree of freedom**. Constraints and negative results are useful scientific structure, not the voice or objective of the investigation.

## 1. Start with the strongest useful discovery

Describe the actual surprising observation or proven relation. Derive a positive consequence: a newly identified dependency, an available signal, an error geometry, an invariance, or a restricted family that can be bypassed with a distinct operator.

Example: "This packed tile is almost dense" suggests "the logical support does not match the execution unit"; it invites investigation of another execution granularity or change of representation. It does not answer all possible conditional-compute questions.

## 2. Diverge before converging

Explore a wide range of **cheap hypotheses** when the user asks for discovery or a mechanism is genuinely unknown. Try contrasting:
- a different mathematical coordinate system or decomposition;
- a new source of information at a different point in the computation;
- a different action space/operator granularity;
- a different causal target (local numerical fidelity versus downstream consequence);
- a changed regime (conditioning, quantization, architecture, learning intervention or hardware).

Follow the unusual implication that the source or mathematics genuinely supports. **A creative hypothesis is allowed to precede full verification** when its assumptions and required checks are stated.

Once the question becomes "which test should run?", choose a small group of mechanisms that make **different predictions**. A two- or three-way discriminator is a testing tool, not an upper bound on initial thinking.

### Reformulate constructively when the action space is stale

Use [PRD](PROBLEM_REPRESENTATION_DISCOVERY.md) to try a new algebraic representation, a taxonomy revealing shared information, or a reduction to a better-served task. A useful output is an unlocked degree of freedom and one predicted difference. These are optional generative routes, not a forced number of alternative hypotheses or a compliance checklist.

## 3. Derive and discriminate

For promising mechanisms:
- Translate a key computation into symbols, including when the representation is approximate.
- State the mechanism-specific observable: sign, rank, support, precision, latency, spatial/temporal pattern or policy change.
- Ask what competing explanation gives a different outcome under the same probe.
- Choose the cheapest reliable discriminator: proof/counterexample, source inspection, existing data, CPU miniature, targeted native trace, then full validation where needed.
- Make the outcome consequential: which idea is refined, which instance is rejected, which new direction opens?

Mathematical proof is especially useful to close a scoped dead end and **free creative effort** for another mechanism. A theorem about an idealized operator still needs implementation correspondence before asserting a deployed effect.

## 4. Learn from negative evidence without erasing it

Store closed experiments accurately: object/version, numerical path, action space, budget, dataset, comparator and observed result.

Ask **which assumption produced the obstruction**. The next question is often "What if the action/information/representation were different?" New experiments are justified by a changed causal or computational mechanism, not by a cosmetic parameter tweak.

Three different outcomes can coexist:
- This registered implementation failed its criterion.
- A well-specified action class is excluded by a bound.
- A different scientific question or intervention remains unexplored.

The first two are legitimate scientific progress. The third is room for innovation.

## 5. Let a result suggest a construction

Translate a successful explanatory mechanism into the simplest possible new action. For example:
- a signed cancellation structure -> output-aware support selection;
- inaccessible oracle information -> early estimator based on an already computed signal;
- quantization noncommutativity -> grouping aligned with quantization scales;
- time-dependent error propagation -> an input-sensitive causal risk measure.

These are **mechanism-to-design patterns**, not evidence that any example performs well. Derive its required information and costs, then challenge it with a strong simple baseline and prior work.

Known techniques, including regularization, distillation, pruning and smoothing, remain useful when their mechanisms genuinely fit. The scientific contribution depends on a distinct prediction, valid result and comparison, not on avoiding familiar ingredients.

## 6. Promote ideas with increasing rigor

An exploration seed needs a question, a possible mechanism and a falsifying observation. It may be speculative and may have no immediate terminal-performance bridge.

A registered research move needs competing predictions, experimental/source/proof checks and a concrete decision under alternative outcomes.

A deployment claim needs complete relevant operations, conditioning, numerical paths, training costs as appropriate, independent data and a verified comparator. Frozen gates apply to the candidate originally registered.

This avoids both extremes: unrestricted claims and pre-verification censorship of ideas.

## 7. Work product

For open investigations, lead with:
**new structural insight -> mechanism-derived possibility -> most telling counterexample or discriminating prediction -> efficient next action.**

For settled questions, give the precise theorem or observed failure and explain **which next frontier it reveals** when relevant. No mandatory mechanism quota or repeated cautionary paragraphs.

## Evidence and authorization

Hypotheses are named as hypotheses; measured evidence has locators. Genuine contradictions are confronted, not suppressed. Experiments, costly compute, model changes and external writes respect user authorization and the project's registered evaluation rules.

Related protocols: GOAL_ALIGNMENT.md, OBJECT_RECONSTRUCTION.md, DEDUCTION_FIRST.md, NOVELTY_GATE.md and PROMPT_PLAYBOOK.md.
