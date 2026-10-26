# Whole-network finite response separates two indistinguishable drift mechanisms

The proposed dense-at-Student-state flow makes a useful finite scientific object observable: the response of the **complete native dense step** to an actual displaced latent. Existing total-flow anti-alignment admits both a contracting dense map with an outward candidate error and an expanding dense map partly compensated by that error. The extra whole-network flow distinguishes these explanations; individual FFN shadows do not provide it.

Author: independent `whole_flow_native_review` agent, 2026-10-26. This record concerns source, algebra and CPU synthetic checks. Implementation snapshots and final test status are recorded at audit closure below. Execution authority remains the separate live registry/protocol.

## DISCOVERY: identical observed drift, different dense responses

Let `delta = zS-zD`, `e = vS(zS)-vD(zS)`, `R = vD(zS)-vD(zD)` and `h = sigma-sigma_next > 0`. Then

```
d = vS(zS)-vD(zD) = e+R
P = delta-h*R
delta_next_ideal = P-h*e
||delta_next_ideal||² = ||P||² + h²||e||² - 2h<P,e>.
```

An independently executed two-coordinate example sets `delta=(1,0)`, `h=.5`, `vD(zD)=(0,0)` and `vS(zS)=(-2,1)`. Both following cases have the same old observable `d=(-2,1)`, the same outward signed relation `<delta,d>=-2`, and the same ideal next error energy `4.25`:

| Dense-at-Student flow | `||P||²/||delta||²` | `h²||e||²` | `-2h<P,e>` | Interpretation at this finite pair |
| --- | ---: | ---: | ---: | --- |
| `(1,0)` | .25 | 2.5 | 1.5 | Dense response contracts; candidate effect increases the resulting separation. |
| `(-3,2)` | 7.25 | .5 | -3.5 | Dense response expands; candidate effect partly compensates it. |

These are exact finite arithmetic constructions, not observations of Wan. They establish why measuring only the old total `d` cannot identify the dense-response explanation. They also show why squared norms are not additive causal percentages: the signed cross term materially changes the answer.

For the deployed boundary, the new component additionally evaluates the unchanged native `step_post` on a copied Student entry scheduler. Its outputs define

```
P_native = zD_at_Student_after - zD_after
N_native = zS_after - zD_at_Student_after
delta_next_actual = P_native + N_native.
```

This finite native-map geometry includes the FP32 scheduler operations and final BF16 cast. It is distinct from the FP64 ideal identity. A finite-pair gain establishes response at the sampled state pair, with neither a Jacobian estimate nor a universal stability guarantee.

Graph anchors are **P23** (already measured local native four-arm errors), **P24** (whole-network function/state-response question) and **G02** (four-step expert/scheduler semantics). The new P24 interpretation is conditional on a later successful authorized observation. Existing P23/P24 traces remain measured evidence at their published scope.

## EVIDENCE: reconstruction of the actual frozen path

I personally read the local frozen scheduler base, Wan scheduler, Wan22 distilled scheduler, pre-infer, post-infer, full transformer infer, block-offload infer, stream manager, model scheduler binding, both relevant runner lifecycle implementations, SDPA implementation and HiF4 runtime/hook adapter. The original paired controller, observer, residency policy and trace contract were read alongside the new code. All four original paired files still matched their frozen v3 protocol SHA256 values during review.

- `Wan22StepDistillScheduler.step_post` reads the current latent, flow, current step index and sigmas, computes two FP32 operations, then replaces `latents` with a cast output. It does not mutate the copied entry's time-conditioning tensor or the saved latent clone.
- `WanPreInfer.infer` reads the bound scheduler's latent, time input and conditioning flag. Under the frozen ordinary I2V/NoCaching/torch_sdpa branch, the shallow scheduler copy plus cloned post-`step_pre` latent supplies the intended native input. This conclusion relies on those exact source/config bindings; a generic caching scheduler could mutate shared state.
- The third flow uses the original HiF4 runtime apply implementation and the same native pre, 40 blocks, head and unpatchify path. Temporarily replacing the paired runtime interception prevents candidate substitution and local four-arm shadows on this path. The observer must be inactive with no pending site at entry, and the original interception is restored in `finally`.
- At each block, compute waits for default-stream pre-infer work. Original offload prefetch/swap synchronizes load and compute streams; the explicit policy fence then releases current-block decoded FFN weights. After block39, native prefetch/swap leaves block0 ready for the next pass. A final wait/synchronize protects head consumption and cleanup. This is a source/lifetime argument, not a measured GPU equivalence claim.
- The actual main dense/candidate latent hashes are checked against historical complete trajectories after replay. The added step0 dense flow supplies a stricter identical-input neutral control before interpreting displaced states. The shadow native scheduler never replaces a live trajectory.

The driver retains original source, input, bank and historical video receipt validation via paired preflight, then adds v3 RUN/TRACE/strict-bundle bindings. Its default operation performs provenance inspection; native execution first requires an exact separately registered protocol and canonical one-attempt ledger. A prior running/failed attempt prevents retry or the next pair.

Each source should account for paired native linear/QDQ `3200` plus `4*400` whole-dense calls, yielding **4800** each. The current-block decoded FFN transfers should be paired `320` plus whole-dense `4*80`, yielding **640**. Original paired observer counts remain separately checked; native total runtime counts are not a count of FFN MAC reduction.

The eight stored tensors per step contain five BF16 latent arrays and three FP32 flow arrays. At the actual `(16,21,58,104)` shape, eight states across two sources require **356,708,352 raw tensor bytes** before file headers, traces and source copies. The fixed `.5 GiB` aggregate output cap has room for this design but remains a runtime check. The latent-only path deliberately omits VAE decode and media creation; existing full-video metric bundles are retained references.

## Actionable findings and their disposition

1. **Fresh-run latent-only lifecycle defect, repaired.** The original new `LatentOnlyMain` called frozen `BaseRunner.end_run_segment` without creating `gen_video`. That hook reads `self.gen_video`, which ordinary fresh runners only create during VAE decode. The repaired path assigns `gen_video=None` before calling the original hook; its synthetic lifecycle check now uses the real hook behavior.
2. **Last monitor error could retain PASS, repaired.** `monitor.close` can add a failure itself, and the sampling thread can record a final isolation failure. The revised driver checks the monitor after closing, records finalization failures, marks the attempt failed and preserves the receipt/ledger before raising.
3. **Numerical pathology needed raw evidence, repaired.** Initial code saved only after successful geometry reductions. The revised path saves eight native tensors and pending metadata before geometry/neutral checks, then updates metadata only. Raw bytes remain available if finite checks fail.
4. **Ideal/native scope wording, repaired.** The component now distinguishes never committing a counterfactual to a live trajectory from executing `step_post` on its isolated shadow scheduler.
5. **Native neutral and old-trace cross-check, added.** The final component checks equal step0 native post outputs and matches actual next-delta energy to the existing trace, independently covering shadow-scheduler wiring. A whole dense flow can be correct while a wrong shadow step index produces an incorrect native state, so these are useful separate checks.
6. **Copied protocol prose, corrected before freezing.** Initial top-level scope still described four new videos and per-branch video hashes. The final scope explicitly describes four latent paths, eight additional complete dense flows, 16 original latent hashes and zero new videos. It binds the four existing v3 metric bundles as prior evidence and preserves their unaccepted quality status. The preparation base commit is current rather than the inherited historical v3 base.

Remaining evidence limits are direct: this review executes no native model, asserts no deployed equivalence or speed improvement, and grants no formal video-quality acceptance. The new flow is an observer paid for research; it does not itself remove W1/W2 work.

## Actual search and verification scope

On 2026-10-26 I used primary-source web queries for `PyTorch CUDA wait_stream record_stream` and `ModelTC LightX2V Wan22StepDistillScheduler step_post`, then opened the official [CUDA semantics](https://docs.pytorch.org/docs/main/notes/cuda.html), [record_stream documentation](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.record_stream.html), and [upstream scheduler source](https://github.com/ModelTC/LightX2V/blob/main/lightx2v/models/schedulers/wan/step_distill/scheduler.py). The latter moving main page differs from the local frozen Wan22 source; local SHA-bound source is the experimental contract. Official CUDA documentation supports the need to distinguish execution ordering from allocation lifetime; the local implementation's explicit synchronization supplies both at these boundaries. This was a focused software/source review, with no claim of a broad novelty search or a newly invented error decomposition.

The independent two-coordinate example ran with Torch `2.10.0+cu128` and printed `torch.cuda.is_initialized() == False`. I also AST-extracted the actual frozen `BaseRunner.end_run_segment` and executed it on a fresh minimal runner: the pre-fix lifecycle raised `AttributeError`, while the repaired latent-only lifecycle returned its latent with `gen_video_final=None`. No native imports or model execution were needed for that check.

## Audit closure

Independently reran `PYTHONPATH=tools .venv/bin/python -m unittest discover -s tests -p 'test_e0_whole_flow_counterfactual.py' -v`: **9 tests passed**, including nonfinite-flow raw preservation, full block call order, original synthetic latent identity, exception hook restoration, real media-hook behavior and one-attempt authorization/ledger guards.

Final reviewed SHA256 bindings:

| Object | SHA256 |
| --- | --- |
| `tools/e0_whole_flow_counterfactual.py` | `0592cfe2a3b92b64b36ecc6e12b99ca653d5939f20305ae02b5ce35d7115306b` |
| `tools/run_e0_whole_flow_counterfactual.py` | `c75901e09bb9e53f2304992163089686c77bc84417c38e4b73894667758662c8` |
| `tests/test_e0_whole_flow_counterfactual.py` | `8e80f397005450b57e02902bb58a0cd4285b7ce62788a607e6aa02d9e49906c7` |
| `config/e0_whole_flow_counterfactual_20261026_v1.json` | `cebee73f6acd646de6b8eddcc03bb7a3982fca652a273cf77f6d5e259ab10b2c` |

I independently compared every driver/native source hash in this final protocol to the actual local file and all matched. The parent reported a successful full read-only asset/80-pack/retained-strict-bundle preflight with `model_scope_authorized=false` and `GPU_initialized=false`; that complete asset preflight was not rerun by this reviewer.

At these exact snapshots, I found no remaining source/math defect that blocks presenting the fixed discriminator for explicit user authorization. Native bitwise neutrality, GPU memory fit, finite flow values, actual call ledgers and trajectory invariance remain empirical runtime checks. A successful run would establish scoped causal-response evidence, with formal video acceptance still a separate question. The result should guide whether future compute-saving FFN operators need state-sensitive corrections or different functional coordinates; this observer itself is additional dense work.
