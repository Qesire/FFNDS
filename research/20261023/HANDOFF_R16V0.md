# R16V0 — actual full160 dynamic FFN and human smoke completed

One implementable full-model input-conditioned operator now exists and has actually run through the four-step student's own full free trajectory. It uses existing frozen UB1 parameters plus pre-W1 gatedR256 magnitude to send onequarter ofrows to fullFFN andthreequarters to packedUB1,with actual compact GEMMs. Read [construction/results](../../docs/81_全层动态行分派的可执行构造_20261023.md), [raw public scalar receipt](R16V0_native_results.json), [analysis](R16V0_analysis.json), [all-frame observations](R16V0_video_review.json), [full review](R16V0_full_duration_review.json) and [human reply](R16V0_human_review.json).

## Completed facts

Exactly5fullvideos and164standaloneFFNs; no training/factorization/official evaluation. Five arms denseDQ,UB1,stridefixed,hashfixed,dynamic. All4x40sites run for every compressed arm,each ofthefivevideos832x464/81frames/16fps. Original UB1 independent formula controls pass byte-for-byte atfour originaldense-stateinputs. Dense initialnoise andfourfree latents matchR15V0 original; allarms sameinitialnoise,ownstate,no resets/splice. MainMAC .625 plus fullR256 .0185185. OneGPU,peak64.350GiB,HF generation/review materials15.46MB.

The complete81frames ofallfivearms were inspected through7contactpages. Bear/scene stay intact;after~1.5s,dense turns more left whiledynamic/hash remainmorefrontal. UB1 ismorestationary;fishdetailsdriftinmultiplearms includingdense. The human fullyplayedthe5sleftdense/rightdynamicpair andanswered “已完整观看，没有明显崩坏”. ROLLOUT_PASSED_SMOKE is supported; motion equivalence/finegeometry/flicker/officialquality are not. The automatic resource export's defaultpendinghumanfield was subsequently supplemented bythe explicit humanreceipt in the derived currentresource summary; original model/debit/media receipts are untouched.

Four native same-input step/block20 observations give dynamic localSSE gains3.236%,8.243%,11.365%,11.103%versus each position'sbestof2fixed. Pooledgain vs bestsinglefixed6.128%. Two warmups/five counterbalanced timings atsamefourpositions produce1.26273xratio of summedmedian nativecoreFFNtime. Not a full160or systemspeed benchmark.

Crucial contrary evidence: wholefree dynamic RGBMSE1122.578 vs bestpixelfixed1037.256,ratio1.0822575,despite betterlocalSSE. MeanframePSNRrankingreverses comparedwith pooledMSE. Fidelity is not official/perceptualquality. Single oldsource cluster only; no sourceCI. Allwholegenerationwall values are observations,not comparativebenchmark;densewall additionallycontains164nativebench/controlcalls.

## Next work toward the actual goal

Keep this algorithm/quarterfraction/parameters fixed. Freeze newsource-disjoint images/prompts with exposureaudit,including appropriateofficialcamera/motion strata. Pin/stage allnine officialI2Vevaluators and per-dimension manifests; do not reuse historical20source three-metric filename-copy harness as an invented nine-dimensional independent result. Prepare separately scoped freshpairedquality and warm/counterbalanced160FFN/systemtiming. Report strongfixed controls andfrozenUB1,plus missing optimized/public baselines. Inspect true motion and stability ratherthan selecting a favorable PSNRaggregation.

Do not locally retune thisbear,addresidualranks,seeds,fractions or silentlyrepeat. The exact5video/164FFNbudget is spent, allcompleteddebits preserved,model eligibilityempty. Newmodel/evaluator execution needs its own concrete immutable protocol anddirectauthority. No automatic followup was authorized. Source/static/CPU preparation cancontinue immediately. Earlier R15V0 four-site smoke, R14N0/D0/R4/R11 closures remainunchanged.

## Reproduce without a model

`python3 tools/check_research_routes.py`
`python3 tools/check_full160_row_dispatch.py`
`python3 tools/analyze_full160_row_dispatch.py`
`python3 tools/check_candidate_rollout_gate.py`
`python3 -m unittest discover -s tests -p 'test_full160_records.py' -v`

The registered entry tools/run_full160_row_dispatch.py nowrefuses furtherexecution;useactualartifacts,not reruns. CPU compaction/partitiontests raninthe existing workerTorch environment andpassed;three stdlibdebit tests also passed. Model sourcebindings remain exactlyasexecuted.

## R17 non-model preparation added after publication

Fresh23 original-metadata/image-byte cohort frozen against98priorimages:subject8/background8/camera7,eachcameraonadifferentimage. The exact historical exclusion inventory was independently replayed and published as hashed identities. Official per-dimension subsets preserved. Nineofficialfunctions+CoTracker import pass withcheckpoint/NNforward forbiddenandCUDAhidden. A missingwheel third_partysubtree was repaired with30exactupstreamsourcefiles; neitherexistingvenvchanged.

All19 evaluator assets,227 package source files and pinned DINO/CoTracker source repositories are staged and hashed in HF,5,642,963,942 logical bytes. Exact source-view/environment checks pass without importingTorch. Nine-scoring-source audit preserves MUSIQ detail/aggregate100xunits and consistency frame-pair weighting; paired bootstrap definitions have synthetic math tests,not new scores. Offline workflow44commands/190testcases pass; the subsequent precise instrumentation-field change passes its3repeated event tests. Boolean-index synchronization in the frozen dispatcher remains charged; only the observer adds no per-block sync. See [actual preparation receipt](R17_preparation_checks.json).

The user then directly instructed “快跑啊” in response to the fixed145generation/45metric request. [Separate authority](R17E0_authorization.json) and [immutable registered protocol](R17E0_registered_protocol.json) are saved; the method-specific guard passes. One locked idleGPU is executing the145generation phase. Do not launch another generator or treat partial observations as final measurements. The45officialmetric phase remains mutually exclusive and requires its immutable asset/source leaf before entry. Draft hashes were refreshed after preparation; earlier reviewed draft bytes remain inHF with unchanged scope. No training/factorization/additionalFFN/automaticretry or expandedfollowup is authorized. Earlier closures unchanged.

All5warmups passed originalfourlatent/initialnoise identity. The background finite coordinator waits for the one existing generator,then decodes115freshvideos onCPU,binds the immutable metric phase,runs exactly45officialcalls and reconstructs scores/timing; it never starts a second generator or retries failures. Current generation progress is in ignored runs/r17_execution and originalHF receipts.

An additional no-forward CPU read of the old warmup masks shows mean firstlatent-time density66.9674%versus22.9016%later,strictglobal25%; the firstbin uses12.7557%ofall dense-rowbudget.128/160sites exceeduniformfirstbin25%. NativeI2Vconditionmask/pre-infer flatten source confirms firstbinstructuralasymmetry. See [actual allocation/source readout](R17E0_time_allocation_readout.json). This is not freshquality or proof of causal motiondamage. Frozen currentbatch unchanged; no normalized-score/temporalquota experiment triggered.

The completed5warm/25counterbalanced timing prefix is now [recorded with all actual repetitions](R17E0_system_timing_prefix.json). Oneoldsource,eacharm5times. Dense generation median50.8582s,dynamic48.6824s,systemratio1.04469x(latency reduction4.28%); full160FFN plusall sharedmodulationratio1.25646x. Fixedstride/hash about48.645s,UB147.2799s. Allrepeat/warm fourlatents match originalR16byarm. Freshquality stillrunning; complete-session resource audit pending. Do not promote this timing prefix to full completed145/45scope or a population/Pareto claim.
