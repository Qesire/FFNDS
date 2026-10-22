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
