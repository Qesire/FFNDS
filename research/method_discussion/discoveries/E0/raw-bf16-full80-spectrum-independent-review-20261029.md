# Independent review: raw BF16 full80 spectrum

**SUPPORT — source and synthetic controls only.** This review is bound to the following SHA-256 identities:

- Driver `tools/e0_raw_bf16_full80_spectrum.py`: `8b034b8e11705a78956c0564640e8b5be37e1c8b35b7e4571f649fe20313ad01`.
- Numerical core `tools/raw_bf16_full80_spectrum.py`: `91199f8f70eebccde66ce75ea6bfc0492ba6008f8e8a1106f3c3e8520a9988e8`.
- Driver tests: `8bb4f101a252c90ba75838e7df78f2313cb4aae84c3ebe17c6edbd4658d63f1c`; core tests: `635f51f5e04dbd2e1643451643c00392e93b7aa70e68b32248b49c5fde15c040`.
- Draft `config/e0_raw_bf16_full80_spectrum_20261029_draft.json`: `b1892c9521f936658d3bd585c8669a7d82820c03bed3c66b5bb8fac0046c72c3`. All 926 repository source hashes matched at review.

The audited eight-source manifest and actual CPU exit are required before numerical execution. All 1,280 teacher postwrite byte-neutral checks precede any fit. The design uses six calibration sources, two exposed development sources, and source-held-out folds only at eight fixed anchors. The sample Gram is bias-anchored and source-weighted; shared RMS gating is identified as a feasible surrogate, and the rounding triangle is explicitly a continuous relaxation, not a native BF16 floor. The BF16 factor forward uses two staged GEMMs, the original bias, and the actual rounded effective gate for readout. Full-M performance, student states and video quality remain unmeasured.

The fixed ledger specifies 384 eigendecompositions, 8 lower-spectrum eigvalsh calls and 1,280 FP64 original W2 references. GPU reservation precedes source snapshots and the existing monitor is checked before each copy. The revised wall budget is 2,700 seconds with a 3,000-second outer limit; the 10 GiB output budget and scientific scope remain unchanged. Independent driver tests passed 7/7 and synthetic CPU core tests passed 8/8, including an explicit hidden-moment commutator comparison, rounding-triangle clamp, and typed-gain metric. No real operand, model or GPU was run in this review. This supports the fixed source-level design and controls, not experimental success.
