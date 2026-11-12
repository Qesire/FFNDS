# Independent design review · G20 roadmap v3

Reviewer: `/root/numeric_audit` (independent child agent).
Scope: static mathematics, estimands, controls, and execution semantics; not model/GPU execution approval.
Reviewed snapshot: `review_inputs/roadmap_v3_first_snapshot.md`.
Snapshot SHA256: `1509da52025fbaade960651e729b3901e7d513bb9a3d00db98926a8abc51e646`.
Decision: **REQUEST_CHANGES**.

## Required corrections

1. **L0 uses an unarchived mathematical D.** R1b saved the actually tested BF16 A matrices and BF16 D of the old solver; the stable F64/FP32 D was not archived. With no refit authorized in L0, define D_eff as the difference of the two actual stored BF16 matrices lifted to FP32, or define c as their actual gated output difference. Name the resulting estimand correctly; do not claim to have read the exact stable mathematical D. The quadratic identity describes output interpolation; deployment after BF16 matrix/cache/gate rounding needs its own endpoint checks.

2. **N1's delta-only control can collapse to a linear control.** Oddifying GELU gives [GELU(v)-GELU(-v)]/2=v/2, including the tanh approximation. Freeze a truly nonlinear odd feature such as tanh, its CAL-derived scale, and its linear readout. The endpoint and delta-only models should have corresponding linear plus nonlinear features and matched readout dimensions; fixed r alone is not equal capacity. If h is a learned nonlinear network, the promise of fixed features and convex readout no longer applies.

3. **N1 PASS/INCONCLUSIVE rules conflict.** A point improvement >=2% plus upper CI<0 can pass even when the CI crosses -2%. Either require the upper CI below -2%, or explicitly distinguish the 2% point-estimate budget gate from a zero-effect uncertainty gate. Use one unambiguous rule.

4. **ICC needs the missing stage-2 repair counterfactual before a causal readout claim.** Add the frozen-input 2x2 native/approximate first stage by native/approximate second stage. Dense, ICC, and native-h/approximate-second are already available or computed; approximate-h/native-W2 is the decisive missing arm. Replacing W2 changes the propagated first-stage error as well, so a large stage-2 norm share alone does not imply that giving W2 more rank helps. SVD-to-CD-SVD changes to both factors establish a whole-recipe effect; a mixed SVD-first/CD-second arm is needed if the paper attributes that gain specifically to the readout.

5. **Student calibration must collect the failing candidate's own states.** If N's free rollout loses its local benefit, CAL24 Q1 states alone do not identify N-state drift. Prefer the same CAL24 run with the actual failed N and paid native shadow targets, paired against Dense calibration at equal counts. Reuse Q1 states when available; a Q1-only additional capture is a separate question. Bound the loop to one student recalibration before a new scope.

6. **Motion protection requires direction/trajectory.** A lower bound on motion magnitude retention allows excess oscillation or wrong-direction movement. Restore a frozen RAFT direction and/or CoTracker trajectory protection, including valid coverage/lost tracks. Magnitude alone must not decide motion preservation.

## Already satisfactory in this snapshot

- R1/R1b solver rechecks are treated as closed; omega is not promoted to a target-error bound.
- CAL breadth and row-density contrasts are separated, with same-row-budget comparisons and fold-local learned priors.
- D1 uses new CAL24/DEV24 source clusters and reuses D4 trajectories; O40 has an explicit 48-trajectory cap and is not treated as equal cost to D4.
- D1 now has a fixed CAL mixture transfer table and separate equal-budget adaptation table, fixed G_ref, conditional-branch identity, original CFG caveat, and expert/time support gaps.
- Seven system arms include ICC and author-faithful Chipmunk under the same Attention background. An eighth N-only arm is tied to standalone/synergy claims.
- Full-M candidate cache semantics, downstream cache refresh, source-level statistics, and one frozen confirmation stage are present.

## Small efficiency clarification

The C96 full CAL set is identical under three CAL subset orderings; reuse its fit rather than counting it as three independent or useful repetitions. New low-dimensional or full-rank solvers should reuse factorizations by exact input/prior identity and report a finite fit/evaluation count in RUN_SPEC.

## Recheck contract

Recheck only these corrections and the final O/D/arm consistency against a new supplied snapshot SHA. Do not treat this review as authorization for new model execution or as a claim that the candidate has passed.
