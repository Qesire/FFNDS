# Deduction First (DF)

Use algebra and constructive counterexamples to **generate and sharpen hypotheses** as well as close settled subquestions. Before allocating an experiment, determine which questions can already be settled by exact algebra, logical consequences or static source analysis. Derivations should carry exact assumptions, dimensional semantics, proof steps, valid regime, failure examples and implications. Use ACCEPT_CONDITIONAL when the scoped theorem is proved; do not spend GPU time re-establishing it.

Example: if h is a hidden activation, y=W_d h, y_m=W_d(m elementwise-multiplied by h) and S indexes discarded coordinates, then Delta y = W_d[:,S]h[S] exactly. Hence ||Delta y||_2 <= ||W_d[:,S]||_2 ||h[S]||_2. Small h[S] alone cannot guarantee small output error without a constraint on W_d[:,S]. Exact cancellation occurs iff h[S] lies in the null space of W_d[:,S]. None of these algebraic statements needs a GPU run.

By contrast, actual h distributions, real masked-kernel performance, numerical effects and the continuation of perturbations through denoising depend on implementation and data: design a smallest targeted check only for those unproved quantities.

An unfavorable conclusion must identify its precise structural obstruction, applicability and remaining opportunity, not end at 'insufficient evidence'.

**Generative use of a boundary:** once a theorem closes the scoped operator family, ask which assumption changing the information timing, operator, conditioning or representation would produce a *different* scientific question. A conjectured alternative remains exploratory until its conditions are checked. Mathematics directs exploration as well as constrains claims.
