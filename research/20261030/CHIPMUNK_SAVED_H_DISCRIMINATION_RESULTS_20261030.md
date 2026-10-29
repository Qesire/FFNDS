# Chipmunk saved-h discrimination: one teacher group

**OBS / NATIVE saved operands:** On one previously exposed source, the fixed paid proxy omitted a projected change with 0.388245 times the full-change squared norm. Two retrospective selections at the same width reduced that ratio to 0.215900 using activation-change energy and 0.209964 using gate-weighted W2 column energy. The latter reduces omitted projected energy by 2.749% relative to the activation oracle in this group. This makes missing activation-change information a useful next question; it does not identify a unique causal failure mechanism or establish an online improvement.

The fixed case is source 898, seed 42, high expert block 20, step 0→1, first 128 matching teacher rows. These are selected **nonraster** coordinates, not a contiguous 128-row tile. Each selection retains 5,120 of 13,824 channels. The original GPU proxy scores and indices are reused, including the original BF16 subtraction before conversion to float. The two alternatives are hindsight oracles using the current saved `h`; they cannot be chosen at the proxy's information time without acquiring and paying for that information.

The diagnostic converts the two saved BF16 `h` tensors to FP64 before subtraction. It projects the omitted change through the original W2 and the step-1 gate in FP64. The denominator is the squared norm of the full projected change, **39,519.206077161754**, not the current output energy. Diagonal energy sums the individual gated column contributions; signed cross is the projected energy minus that diagonal.

| Selection | Omitted projected energy | Diagonal energy | Signed cross | Omitted / full squared norm |
|---|---:|---:|---:|---:|
| Frozen proxy | 15,343.125550 | 9,683.117666 | +5,660.007884 | 0.388245 |
| Activation-energy oracle | 8,532.201239 | 5,353.925499 | +3,178.275741 | 0.215900 |
| Gate-weighted consumer oracle | 8,297.621859 | 5,273.420578 | +3,024.201280 | 0.209964 |

Signed cross is positive and contributes approximately 36–37% of omitted projected energy in each arm. It shows reinforcement in these particular omitted sets; diagonal column energy alone does not reproduce the summed projection. The oracle masks differ, so the modest weighted-oracle improvement over the activation oracle is a comparison of two fixed rankings, not a proof that consumer direction is unimportant. Likewise, the lower activation-oracle result is consistent with information missing from the mean proxy but does not by itself establish that within-group variance, rather than other ranking differences, caused the gap.

This FP64 saved-teacher-h measurement is **not directly comparable** to the prior 0.279554 gate-relative L2: that earlier experiment recomputed its same-M128 BF16 cache and used a different target and denominator. Neither is a full-M native replay, a free-running student video, or a physical speed result.

A useful next fixed comparison would acquire or compute a causal, paid estimate of within-group activation variation for matched step pairs on additional sources, then compare it at equal selection width with the original mean proxy and fixed random controls. The design must account for the W1/observer and selection work, use actual contiguous raster groups if making tile claims, preserve the current-step information boundary, and compare projected and native BF16 outputs separately. A distinct follow-up on the other available source/step pairs needs its own proxy computation and scope; these hindsight masks cannot be relabeled as that proxy.

The CPU job exited 0 in 22.26455 s; its RUN and EXIT receipts were independently checked for closure. Its ledger records six source hashes, five HF asset hashes, one checkpoint hash, two payload tensor reads, one W2 read and one diagnostic. Source hash bytes were fully completed. The [whitelist JSON](CHIPMUNK_SAVED_H_DISCRIMINATION_RESULTS_20261030.json) binds the exact RUN, EXIT and protocol, and retains scalar results without copying the channel arrays. No checkpoint or tensor was reread for this report.
