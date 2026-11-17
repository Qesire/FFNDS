# True-M guarded replay candidate

Do not dispatch without independent review, an ACTIVE entry, and a completed Stage-3 grid. The launcher does not submit a job.

The stage must be a fresh child of `/public/home/scc/pb23061276/projects/wan-quant/work/ffnds_strict_reproduction_20261116_336b7e1`. Stage `scope.json`, `REVIEW.json`, `ACTIVE_RESEARCH.json`, the exact bound files, and `SHA256SUMS` there. `output` must equal `stage/output`; the shell creates `receipt` exclusively.

The scope must bind these values:

* `scope_id`, `operation="chip_true_m"`, `selector_seed=42`, `max_predictions=24`, `max_arithmetic_pairs=4`, `num_videos=0`, `num_model_nfe=0`.
* `python_executable`, `python_executable_sha256`, `torch_version`, `triton_version`; `runtime_guard` as `{path,sha256}`.
* `implementation_sha256`, mapping absolute paths to SHA-256 for all stage Python files required by the guard and `chip_true_m_job.sh`. Include `chip_true_m_job.py`, `chip_true_m_replay.py`, `chip_provider.py`, `chip_wan_voxel_tail_v1.py`, `chip_tail52_valid_candidate.py`, `icc_stage3_assets.py`, and `icc_local_replay.py`.
* `provider`: `pinned_src`, the full nine-entry `source_sha256` map from the bounded module receipt, `library`, and `library_sha256` from the successful safe-selector receipt.
* `prerequisites`: hash-bound `{path,sha256}` specs named `module_run`, `module_exit`, `module_result`, `selector_run`, `selector_exit`, `selector_result`, `grid_run`, `grid_exit`, and `grid_scores`. The first six are job 168787 and job 168713 evidence, staged as byte-identical copies or referenced at immutable locations. `grid_scores` must be the completed grid's 792-cell `scores.json`.
* `canonical_grid_scope` and `canonical_cache_manifest` as hash-bound specs from that same grid. The manifest's `cache` directory must remain adjacent to its manifest. True-M verifies each native tensor receipt, its grid-scope identity, full pair identity and checkpoint identity; it never recomputes local512 targets.
* `dev_pairs`, `voxel`, `capture_manifest`, `gp64_proxy_source`, `gp64_action_source`, and `tail_candidate_source`: hash-bound specs. The GP64 source artifacts are the pinned baseline `chipmunk_fullm_proxy_math.py` and `chipmunk_wan_packed_pregate.py`; the tail source must match the stage copy. `dev_pairs` is the assembled 66-pair manifest; only its 12 two-neutral pairs are used.
* `checkpoints` keyed by `high` and `low` with `{path,sha256}`, identical to the grid's checkpoint specs; `fixed64_token_ids` copied exactly from the captured manifest.
* `resources`: `gpu_memory_gib<=78`, `host_memory_gib<=160`, `wall_seconds<=10800`, `0<soft_seconds<wall_seconds`, `0<output_bytes<=2147483648`, and positive `min_free_disk_gib`. Guard requires one A100 80GB, partition `P107-A100`, QoS `qos_p107-a100`, account `competition`, and the independently checked one-node allocation.

No scope is supplied here: actual canonical grid receipt/file hashes, grid scope, runtime version, code hashes, output path, and reviewer-approved budget must come from the completed grid and reviewed staged bundle. Torch seed 42 is not the compiled selector's CURAND seed. The output records the per-row proxy words used by the pinned selector's separate CURAND initialization, along with the formula.

`test_chip_true_m_guard.py` exercises receipt identity rejection on CPU only. It is not a GPU, shell, resource or scientific validation. The true-M status is candidate until independent review and actual scoped execution.
