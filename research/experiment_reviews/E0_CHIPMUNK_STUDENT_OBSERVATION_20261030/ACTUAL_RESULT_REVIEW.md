# Independent actual-result review: student observation V2

Reviewer `/root/sync_abba_actual_review_terra`, independent of the producer authors. I inspected actual CPU, GPU and independent-verifier receipts and the frozen implementation, and independently recomputed pooled statistics from the bound GPU RUN metadata. I did not read tensor payloads, run a model, or rerun the independent verifier. **Both scoped source runs and their separate 310-artifact verifications are actually closed; the byte gates and descriptive measurements below are supported for these recorded runs.**

## Actual receipt identities and gates

| Source | Protocol SHA256 | GPU RUN / EXIT SHA256 | Verifier RUN / EXIT SHA256 |
|---|---|---|---|
| 898 | `f78ac3261267a2e36508bc553e195965c55f61b701df9c75213ce0f3d98ca7be` | `a0dba0eab24972ed6ef4da413181c1202d161d5486ab9d480068499797fa1c2f` / `dd59eb8e250c06a524a432554bb99647995c5f65b4b9d77c894127446179ff8f` | `f4f56db21ff33beadc2937c49b3fd745b35afead4cfd32c84b4669aa6cf23cf6` / `bf8722fe2725bf506d7a45cef5d2eceb32ecb3a5837f574b60c112f6897a0f1f` |
| 979 | `7f93533de9c732f1fb1c6909292f79d37c5ff6ed83dd86718a352d92f9e89753` | `0c427adb5b24232b520162e757b201deed913bb2a9a346c3f51655253efc43db` / `6b16e80b39c78d4c01f8a3fd9d9c742efef9bc9fb7ef55f27c5ef42acce7c674` | `8c9a5a282df2e4fa02bcf525c076426edc93aa60fbb0c1999a39a272b654fa2c` / `a98a69276a25a8d3de1d6859e738069f3848d6fdc9e5f8b424bde9bf7b6277d9` |

Each source's CPU preflight, native GPU job and verifier supervisor report actual EXIT 0 bound to the relevant RUN and protocol. CPU RUN / EXIT SHA256 are `37c4f24e92aaf0c4fca020c7867629c815b083a933d839179edd976c1cd001ef` / `76db109c694d8a829d2b0822d5b47c28ee1b83b35634263bcf1aa75022d3bc0a` for 898, and `824f04372958f0f037b4ef37078e5fa1e7f84de240e910edfed3678be80a99a7` / `36d1dda09f1ae7e2e729b8393830b4d9a00332fe4d40fcba78f55c06e455b6c9` for 979. Verifier RUNs explicitly bind their corresponding actual GPU RUN and EXIT. The two independent verifier jobs each report 310 unique artifacts with predebited and completed payload bytes equal and no inflight read: 25,581,786,506 and 25,579,560,126 bytes, in 175.798 and 174.659 seconds respectively. The 620 artifacts include typed tensor payloads **and** metadata, statistics, videos, state indexes and journals; they are not 620 tensor files.

Each source's producer reports all 20/20 state and 2/2 complete-video byte comparisons true, paired to historical input-video part0/part3 phases `h_cache_dense` and `high_h_readonly_p64`. The producer performed the streaming comparisons with historical files. The independent verifier reread this run's typed outputs and checked their hashes, metadata and historical reference identities; it did **not** reread historical payloads. Each of the two phases has 160 FFN sites; the student phase applies 40 high-step1 candidate sites. The ledger predebited and observed counts match every frozen nonzero limit: per source this includes 40 Dense-shadow h/Y bitwise checks, 80 actual post-process write bitwise checks across both arms, 40 Dense and 40 student full-M W1 shadows and 4,960 student diagnostic chunks. The frozen verifier checks the per-site manifest, 84 exact group counts, persisted mask/score and coordinate identities, and both journals; success is not inferred solely from phase completion.

## Observed student-state statistics

For each source the full 31,668 rows are reduced across 40 student high-step1 sites into 21 latent times × four quadrants. All sites have counts `[364,364,390,390]` per time. The retained sample has 64 actual selector rows and 256 disjoint stratified evaluation rows; they are nested within each source, not additional independent sources. The pooled relative L2 is the square root of summed error energy divided by summed same-state Dense-reference energy, so sites are weighted by their reference energy. The separately reported site mean is a descriptive equal-site average.

| Source | Pre-gate relative L2 | Gate-weighted relative L2 | Actual same-student-residual write relative L2 | Ideal total / diagonal / signed cross | Actual vs ideal SSE |
|---|---:|---:|---:|---:|---:|
| 898 | 0.473566 | 0.360532 | 0.169600 | 172,012,713 / 93,985,056 / 78,027,657 | 9,979.18 |
| 979 | 0.485746 | 0.342973 | 0.164050 | 153,162,657 / 85,817,189 / 67,345,468 | 9,674.44 |

These signed cross values are positive pooled FP32 ideal omitted-projection diagnostics; the actual BF16 candidate/reference comparison follows a different rounded path. The measured identity total − diagonal − cross is about 0.0267 and −0.0236 respectively, consistent with finite-precision reduction, not bit-exact algebra. The actual-vs-ideal SSE uses the corrected opposite direction. Write denominator includes the shared student residual and therefore is not the FFN output norm alone or a motion metric.

The previously closed [input-video quality report](../../20261030/CHIPMUNK_INPUT_VIDEO_QUALITY_RESULTS_20261030.md) reports candidate/Dense motion ratios 1.015495 for 898 and 0.787582 for 979, with 979 MUSIQ −6.382042. This new pooled gate-weighted local error is higher on 898 than on 979. Thus its simple two-source ranking does not account for the ordering of that prior quality observation. This comparison does not identify a causal mechanism or prove H3; source, trajectory, direction and downstream sensitivity can differ. Historical byte-matched videos provide observer neutrality, not a new quality evaluation or recovery.

## Resources and reviewed identities

Observed whole-run wall times were 1,787.218 / 1,788.974 s. Cumulative GPU allocated peaks were 58,947,437,568 bytes for both, reserved peaks 70,871,154,688 / 70,493,667,328 bytes, and peak RSS 117,022,822,400 / 117,026,140,160 bytes. Peak tracked output including reserves was 26,655,163,403 / 26,654,080,985 bytes. These are run-wide peaks with no reset, not per-arm costs. Both GPU isolation monitors passed with zero violations and both output monitors stopped without fault. Physical I/O and bus traffic remain unknown; observer timings are not an uninstrumented performance benchmark.

The final summary was regenerated in memory from the bound receipts and exactly matches the checked JSON source objects; independently summing the RUN's 40 partition arrays reproduces the table. Reviewed SHA256 identities:

- `tools/e0_chipmunk_student_observation_results.py`: `34104c3664eed5c99c4327070c91986662dffc422e56e7d7e169cbfd6d393875`.
- `tools/e0_chipmunk_student_observation_owner.py`: `e6674d508c698bd3ea407a31be63cfe2bd2af5afb95d2365b8d3aedff8fcf787`.
- `research/20261030/CHIPMUNK_STUDENT_OBSERVATION_RESULTS_20261030.json`: `0d2edfb9e15bdb11e0b880e905ae152ce7a1d8f31ffe8cb6c62aea48d15e6760`.
- `research/20261030/CHIPMUNK_STUDENT_OBSERVATION_RESULTS_20261030.md`: `3369ccde8676942df74f9a06ecf407be02dbeac97b53c001d17eeae4ebbaba95`.
