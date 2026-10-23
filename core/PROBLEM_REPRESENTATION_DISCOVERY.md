# Problem and Representation Discovery (PRD) — constructive research operator

**Purpose:** find a more productive mathematical description, computational basis or *problem to solve* before optimizing a familiar algorithm. PRD extends CR/DF/MEAR; it does not replace source reconstruction, a registered experimental gate, or the scientific north star.

## Research move

**Observation -> alternative representation/problem -> newly exposed freedom -> constructive mechanism -> decisive contrast.**

A useful PRD result is a new *question, operator or falsifiable explanation*, not a collection of equivalent formulas. Start by giving the strongest constructive insight. Explore freely at low cost; verify the exact conversion before asserting source-grounded equivalence or deployment benefit.

PRD has three complementary routes (choose whichever opens the research question, rather than running all three mechanically).

### E — Equivalent or conditional reparameterization

Find another computation of the same function that exposes different parameters, coordinates, interactions or dependencies. Examples: dynamic-weight networks, explicit operator sums, output-space geometry, kernel expansions, low-rank or state-space forms.

For a promising rewrite, annotate:
- original operator, shapes, axis semantics and source/trace;
- new expression and *scope* (exact / conditional / approximation / conjectural);
- current-input, context, time, state and random-number dependencies of each apparent "weight";
- masks, positional transformations, normalization, nonlinear ordering, quantization and numerical mode;
- the new assumption or apparent constraint revealed by the representation.

**Example (single causal attention head):** for a prefix X in R^(n x d), current token x in R^(1 x d), projections Wq,Wk,Wv and Wo, write
q=xWq, K=XWk, V=XWv, o=softmax(q K^T / sqrt(dk)) V Wo.
Then o=softmax(x A(X)) B(X), where A(X)=Wq Wk^T X^T/sqrt(dk) and B(X)=X Wv Wo. This is an *exact softmax-coupled dynamic two-layer representation* for the pinned computation. If the prefix includes x, A and B can depend on the current token. Replacing softmax by independent ReLU/GLU changes the function; adding sequence mixing designs a new operator.

Alternative exponential representation: o=[exp(q K^T/sqrt(dk))/Z(q,K)] V Wo, Z=sum_j exp(q k_j^T/sqrt(dk)). Placing 1/Z into the second weight matrix makes that "weight" query dependent. Recognize dependency, not only symbol-level algebra.

### T — Mechanism taxonomy and constructive combination

Group existing methods by *what operation they perform* and *what information and state they require*, not by paper label. Build a small set of independent dimensions, locate constraints or missing interactions, and derive a construction from the mechanism.

Useful axes include: compression versus routing; source and timing of the decision; basis or unit of work; conditional versus fixed parameters; local detail versus global summary; coupling of routing and readout; exact versus approximate computation; cost of forming/reusing the policy.

The MiTA example uses landmark queries to construct both compressed global summaries and routed top-k KV experts, sharing the search signal. Its insight is **reuse of a decision across two operations**, not a generic instruction to apply top-k everywhere. A new combination becomes an experiment only if its predicted benefit or failure differs from individually matched components.

### R — Cross-problem reduction / trajectory reformulation

Ask which mature solver already possesses the prior or geometry needed for the original task. Define an encoding T from the original instance to another task, the borrowed solver S, and a readout/selector D:

    answer_A = D(S(T(instance_A)))

T need not be invertible; D need not be an inverse. Describe which constraints are gained or relaxed and what downstream target determines correctness.

The Frame2Frame example maps static image editing to an image-to-video trajectory, then selects a frame satisfying the edit while preserving the source. "Image manifold" is a geometric interpretation, **not** a demonstrated exact Riemannian geodesic solver. The reduction is a learned heuristic task transfer, not algebraic equivalence.

## Extract the actionable freedom

After proposing a representation/reduction, ask:
1. What *formerly implicit choice* is now explicit (basis, rank, topology, memory unit, timing, path or objective)?
2. Can that choice change a real operator or solver, or does it only re-label computation?
3. What new mechanism follows, and what is a sharp counterexample or different prediction?
4. What information would be available at the *actual decision time*, and what computation must be paid for?
5. Which minimal proof, CPU miniature, source check or existing artifact discriminates the idea?

A successful theorem can close an old search direction and reveal a new one by changing one stated assumption. An exploratory idea need not yet have a fast GPU kernel or a proven terminal bridge. Keep the exact established boundary while pursuing the new question.

## Use with the existing six capabilities

- **CR:** ground the original and transformed computation in the actual focal source graph. A source fragment can still inspire a labeled exploratory PRD seed.
- **DF:** derive identities, assumptions, counterexamples, invariants and lower bounds; no GPU run for settled algebra.
- **MEAR:** change representation, action space, information timing or causal target to generate a genuinely different explanation.
- **CRL:** select the least expensive discriminator once a promising few competing mechanisms emerge.
- **MNG:** compare nearest *mechanism signatures*, not merely titles or keyword overlap. Check whether the new construction, regime or prediction differs from prior work.
- **GAPI:** keep the north star visible; do not confuse symbolic equivalence, expressivity, data quality or a local proxy with deployable performance.

A compact PRD handoff may state: original question -> alternative representation or task -> relation type and assumption -> unlocked freedom -> constructive candidate -> distinct prediction -> next check -> prior-art link. Use the optional [representation ledger](../schemas/representation_ledger.schema.json) for reusable results; research_state schema v2.1 remains unchanged.

## Anti-defensive creativity by construction

**Start with the possibility:** "This operator can also be seen as __; that reveals __; we could therefore try __." Put the consequential assumption next to the equation, rather than producing a gatekeeping preamble. Find rich candidates before narrowing expensive experiments. A valid impossibility theorem settles its scope; an unsupported analogy stays a productive hypothesis. More text, more checklists, or a higher count of reworded hypotheses are not research improvements.

See [prompt operators](PROMPT_PLAYBOOK.md), [paper discovery cases](../examples/PAPER_DISCOVERY_CASES.md), [novelty](NOVELTY_GATE.md) and [reconstruction](OBJECT_RECONSTRUCTION.md).
