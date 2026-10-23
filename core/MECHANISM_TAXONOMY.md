# Mechanism Taxonomy: from literature to missing algorithmic degrees of freedom

A taxonomy is a **generative coordinate system**: represent what prior methods compute, then discover testable combinations or gaps. It is not a categorical veto on new ideas.

## Build a useful coordinate system

For a focal computational problem, extract a compact set of meaningfully independent dimensions:
- **Information:** what signals exist before the decision, and at what stage are they computed?
- **Representation:** token position, feature, channel, rank-one operator, segment, trajectory, learned slot, state.
- **Action:** exact compute, approximation, compression, routing, caching, omission, solver transfer.
- **Reuse:** per token / query, shared group, persistent cache, global model weights.
- **Output geometry:** routing and readout coupled/decoupled; local metric versus downstream state.
- **Execution:** algebraic FLOPs, policy overhead, memory traffic, actual kernel work and hardware occupancy.
- **Scientific regime:** model, data, conditioning, training and inference assumptions.

Only retain dimensions that distinguish concrete mechanisms for the current object.

## Mechanism signature and nearest work

For each source or candidate, write one substantive signature:

    (object / operator, signal & timing, dynamic state, transformed basis,
     decision & readout, action granularity, downstream prediction, cost)

Compare candidates against the nearest source at four levels:
1. Mathematically identical under a reparameterization?
2. Same mechanism but different approximation / regime / budget?
3. Genuinely different causal edge, available signal, unit of work or predicted behavior?
4. If hybrid: what interaction predicts a result not obtainable by independent components?

Examples: HyperMLP exposes learned sequence-mixing and separated dynamic routing/readout from a context-conditioned MLP representation; MiTA discovers reusable compression-plus-routing of fast-weight memory; Frame2Frame changes the target task to temporal generation with frame selection. They are **inspiration for distinct transformations**, not interchangeable innovations.

## Construct and test

1. Formulate an unexpected missing combination or assumption change in one sentence.
2. Derive the simplest proposed operator, including dimensions and cost of creating its policy/state.
3. Predict how it differs from its closest existing component, matched baseline and no-mechanism control.
4. Prefer a cheap discriminator, then register the real experiment with frozen target metrics.
5. Record a negative result for the *specific* mechanism and budget; use it to locate a different coordinate or problem reduction.

For FFN dynamic sparsity, a possible shift is from thresholding independent hidden scalars to selecting output-space conditional rank-one contributions: F(x)=sum_i h_i(x) w_{d,i}. The exact discarded-output error is h_S^T (W_{d,S} W_{d,S}^T) h_S in squared Euclidean norm (with W_{d,S} rows). This exposes signed interactions; it does not by itself produce a practical pre-W1 selector or sparse GPU speedup. Those are separate design questions.

Deliver the **new degree of freedom and discriminating prediction first**, not a survey table without scientific action.
