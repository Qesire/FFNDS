# B-H / A0 matched terminal head · 2026-10-28

**A0 is more accurate than the B-H H128 pilot on this matched terminal-head comparison.** Against the same native-captured BF16 target, equal-source heldout8 head relative SSE is 0.003958 for B-H versus 0.000609 for A0 energy94 and 0.000843 for A0 random94 in the high expert; the corresponding low-expert values are 0.003615, 0.000661 and 0.000752. B-H is lower than either A0 comparator on 0/8 heldout sources in each expert. The 8 heldout sources were historically exposed.

| Expert / source split | B-H saved | A0 energy94 | A0 random94 |
| --- | ---: | ---: | ---: |
| High / train16 | 0.004661 | 0.000720 | 0.000980 |
| High / heldout8 exposed | 0.003958 | 0.000609 | 0.000843 |
| Low / train16 | 0.003668 | 0.000824 | 0.000945 |
| Low / heldout8 exposed | 0.003615 | 0.000661 | 0.000752 |

For each source, the two expert steps are averaged before equal weighting across sources; metric errors use FP64 reductions on the same captured BF16 head target. All 96 full-M geometry neutral selected-row head outputs matched their captured hashes. Both A0 arms then used their own Q2 in 192 M256 FFN evaluations, and every output hash matched its original final diagnostic; 288 full-M LN/head calls comprise the neutral and A0 evaluations. The 96 B-H outputs were reused from saved predictions, without new B-H forwards. The actual worker exited 0 and the operation ledger closed.

The comparisons use the same source split and update budget, while A0 trains against gated Y and B-H against statistics; B-H has 672,386 trainable parameters per expert versus 712,704 A0 LoRA parameters (5.66% fewer; deployment parameter counts differ separately). These differences prevent attribution to a single design choice. B-H's lower logical arithmetic cost remains an unvalidated execution opportunity, not a measured speedup. The experiment covers two terminal functions of 80; full-video quality and end-to-end speed remain unmeasured.

[Scalar audit and per-source values](BH_A0_MATCHED_TERMINAL_HEAD_RESULTS_20261028.json) include receipt, protocol and asset references. The prior B-H final-metrics sidecars were not bound by its original RUN; this audit compared their recorded hashes without loading tensor payloads.
