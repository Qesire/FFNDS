# High-stage action necessity after the actual factorial contrast

Proposer: `/root`. Version 1. Evidence: ALG + completed VIDEO metric observations; next numerical action is OPEN and not registered by this note.

The [actual four-arm study](../../../20261030/CHIPMUNK_P64_FACTORIAL_QUALITY_RESULTS_20261030.md) reveals an actionable stage asymmetry in two exposed image sources, seed42: HIGH_ONLY continuous-flow ratios are 1.01549/0.97506, while LOW_ONLY ratios are 0.95883/0.82624. HIGH_ONLY imaging differences are −0.008/+1.912, LOW_ONLY −2.120/−1.483. Source interactions have opposite signs. This ranks the tested student policies and quality axes; block20 teacher error does not provide that ordering.

## Competing mechanisms and distinct predictions

1. **High-stage residual updates carry useful information, with a comparatively forgiving Dense suffix.** P64 corrects a required subset of channel changes even though its local residual error is large. Removing those corrections at high step1 should measurably worsen motion/semantic quality relative to HIGH_ONLY P64. Dense low steps may still reduce some effects, but do not establish an irrelevant high FFN.
2. **At the tested high step1, the cached same-expert FFN output is already sufficient for the measured video functions.** The selected P64 corrections have little downstream action value, and their paid observer/cache machinery can be avoided. A same-source high-only output-reuse control should stay near HIGH_ONLY P64 while making no current FFN W1/GELU/W2 or P64 proxy calls at those40 sites.

Both mechanisms allow LOW_ONLY degradation. Source-conditioned mixed outcomes are a meaningful alternative to either universal prediction. Two development images cannot decide a population-wide rule.

## Algebra and the physical strong control

In real arithmetic, let h(x)=GELU(W1 x+b1), Y(x)=W2 h(x)+b2. For cached same-expert old state x0 and selected channel set S, delta reuse is

`Y_S(x)=Y(x0)+W2[:,S]*(h(x)[S]−h(x0)[S])`.

Its omitted real-arithmetic term is `W2[:,complement(S)]*(h(x)[complement(S)]−h(x0)[complement(S)])`. The empty-set control S=∅ is simply Y(x0), so it needs neither current h nor old h. This is an algebraic identity, not a byte identity for the BF16-clone/delta-round/FP32-accumulate implementation.

The proposed **FFN-only** control caches pre-gate Y at all40 high-expert step0 blocks, clones/releases the matching Y at step1, and leaves the current original modulation/gate/residual, all attention, low-expert FFNs, all four denoising steps and decoding in place. This skips two actual dense FFN GEMMs at each of40 sites. It is an untrained, existing-mechanism strong baseline, not an innovation claim. The old logical Y buffer is 40×31668×5120×2 = 12,971,212,800 bytes (about12.08GiB), compared with old h+Y 47,993,487,360 bytes (about44.70GiB); native allocation/reservation and total savings remain unmeasured. Original weight prefetch remains paid unless independently changed in another explicitly registered action.

## Smallest next evidence

Use same-source/seed numerical recipe and real full4×40 videos: new hooked Dense and Y-cache Dense neutral controls must match all10 original states and the full Dense video before accepting the high-only output-reuse candidate. Compare the new candidate with existing same-recipe HIGH_ONLY P64 and Dense, charge Y cloning/release/cleanup and unchanged prefetch, and use the same eight axes, continuous40-pair RAFT and full81-frame evidence. The controls and candidate require a new finite independently reviewed ACTIVE scope, exact source closure and fresh outputs. No model is run by this note, and no old failure is retried. Broader source/seed confirmation follows only if this strong control is informative.

A complementary saved-state discriminator can compare the actual Dense/high/low/both final perturbations and the signed policy interaction. Changes of mask with the high-stage student prefix prevent interpreting that interaction as fixed-mask linear superposition. This observation analysis would be a separate bounded registered CPU scope.
