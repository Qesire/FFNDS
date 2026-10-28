# S1: a common left factor and shrunk step readouts

2026-10-29 · ALG/SRC preparation only. No saved numerical tensors were read,
no fitting/decomposition/model/GPU call was made. This derives the existing
[S1 proposal](../../../20261029/ASSET_FIRST_STRUCTURE_DISCOVERY_AND_METHOD_DERIVATION_20261029.md#s1--csfb共享核心基--收缩的step特化读出);
it adds no literature claim or change to the running video scopes.

**The identifiable new action is to change the step readout while keeping the
same input representation.** Existing shared80 and step160 exports change both
factors. They therefore do not isolate whether a transferable common hidden
representation needs different readouts at its two steps.

For one physical expert/block, inherit the exact BF16 shared bank:
`U=left.weight.T`, `V0=right.weight.T`, and original `b2=right.bias`.
For each step, the deployable function is

\[
 z=\mathrm{BF16MM}(h,U),\qquad
 y_\alpha=\mathrm{BF16Addmm}(b_2,z,\mathrm{BF16}(V_0+\alpha_t\Delta_t)).
\]

The original W1, tanh-GELU, BF16 gate multiplication and residual writeback stay
in place. `alpha=0` must reproduce the original shared-bank operand bytes;
factors from unrelated bases must never be subtracted to construct `Delta`.
The original two low-rank GEMMs remain; the step is known before W1.

## Source-equal conditional readout equations

Use only the six calibration image clusters, with their original 256 rows per
source/step. `Z_st` is the actual first-factor BF16 result converted to FP64
for statistics; using FP64 `hU` would define a different training feature.
Let `G_t=diag(g_t)`, `E_st=Y_st-1*b2-Z_st*V0`, and
`w_st=1/(6*(||Y_st G_t||_F^2+epsilon))`. A constant half-weight for the two
steps can be absorbed into the corresponding ridge coefficient. No empirical
per-source centering is introduced; the original bias remains fixed.

\[
 C_t=\sum_s w_{st}Z_{st}^TZ_{st},\qquad
 T_t=\sum_s w_{st}Z_{st}^TE_{st}.
\]

For the proposal's literal unweighted ridge
`sum_s w_st ||(E_st-Z_st Delta_t)G_t||_F^2 + lambda_t ||Delta_t||_F^2`,
each output column satisfies

\[
 (g_{tj}^2 C_t+\lambda_t I)\Delta_{tj}=g_{tj}^2T_{tj}.
\]

This is **not** one ordinary ridge solve shared across all output channels.
It is nevertheless inexpensive to solve exactly: diagonalize the 960-by-960
matrix once, `C=Q diag(d) Q.T`, then

\[
 \Delta_t=Q\left[(Q^TT_t)\odot
 \frac{g_t^2}{d[:,None]g_t^2+\lambda_t}\right].
\]

With positive lambda, zero-gate columns give zero corrections directly, with
no division by a zero gate. A **different explicit prior**, penalizing
`||Delta_t G_t||_F^2`, permits one Cholesky solve
`(C_t+lambda_t I)Delta_t=T_t` for nonzero-gate columns. That variant must be
named as a changed regularizer rather than silently substituted for the S1 loss.

## Shrinkage and honest source-held evidence

For a fixed correction, define training-only scalars
`A_t=sum_s w_st ||Z_st Delta_t G_t||_F^2` and
`B_t=sum_s w_st <E_st G_t,Z_st Delta_t G_t>`. Continuous squared risk along
the correction is `L(alpha)=L(0)-2 alpha B_t+alpha^2 A_t`, so its constrained
minimum is `clip(B_t/A_t,0,1)` when `A_t>0`; use zero otherwise.
This is a **training** optimum, not a source-held shrinkage estimate or a BF16
native optimum. Export rounding can change this curve.

For an exact positive-ridge correction fitted on these same data, the normal
equations imply `B_t=A_t+lambda_t||Delta_t||_F^2`. Thus training-only alpha is
normally **one after clipping**: it cannot provide additional generalization
shrinkage. The gate-weighted ridge has the analogous identity with
`||Delta_t G_t||_F^2`. Choosing a smaller alpha requires an independent prior or
cross-fitted evidence; minimizing the same training loss does not supply it.

A source-held estimate of alpha requires cross-fitting the correction on five
sources and evaluating signed `A/B` on the excluded source. If the common `U`
was itself learned from all six, such a calculation is conditional readout
validation and leaks the excluded source into representation learning. Honest
whole-method LOO must refit `U` in each fold. Existing all6 exported banks do not
provide those fold bases; the eight-anchor LOO numerical reports did not export
their factors. Two exposed development sources and the video20 cohort are not
used to choose alpha/lambda. One fixed alpha/lambda candidate, preregistered
before new candidate videos, avoids a width/rank/learning-rate grid.

## Native implementation and the next discriminating observation

At rank960, a deduplicated bank needs 80 common left factors, 160 step right
factors and 80 original biases: **3,697,049,600 tensor bytes**, versus
2,910,617,600 shared80 and 5,821,235,200 independently factored step160 bytes.
Per-token W2 MAC remains `(13824+5120)*960=18,186,240`; full-FFN MAC remains
88,965,120. Runtime dictionaries can share each left/bias tensor across its
two step keys, using the existing factor dispatcher. Factor preload, original
buffer residency, BF16 rounding and actual video/system costs still count.

For each step the sufficient statistics are a 960-by-960 Gram and a
960-by-5120 cross-moment, about 44.5 MiB together in FP64. The literal ridge
needs one small eigendecomposition per step, not one per output column. Actual
first-factor BF16 feature GEMMs, teacher-range I/O, all statistics, solves,
export/readback and native replay need a separately frozen paid scope.

**Competing predictions:** if the common representation is adequate but the
step readout is biased, a shrunk readout can improve transfer while preserving
the shared-bank MAC. If independent step factors mainly exploit unstable input
directions, common-U readout fitting will retain sharing's stability. If the
unrecoverable information lies outside the common hidden features or changes
on student prefixes, this action can improve teacher-state training loss yet
fail in full160 free videos. M256 factor rounding is not full-M equality;
neither local SSE nor this conditional optimum predicts motion acceptance.
