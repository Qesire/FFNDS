# Candidate full-video smoke registry

One JSON file per **selected** backup algorithm/version goes in registry/. Use record_template.json as a starting shape. Exploration seeds without a selected implementation do not need entries.

Status progression:
- LOCALLY_SUPPORTED: confirmed for further candidate development, must set next_required to COMPLETE_PAIRED_FREE_RUNNING_VIDEO_SMOKE.
- WAITING_FOR_ROLLOUT_AUTHORIZATION / ROLLOUT_RUNNING: must report video smoke pending, **not** video validated.
- ROLLOUT_FAILED_QUALITY: complete paired candidate video contains visible catastrophic collapse; do not promote this version.
- ROLLOUT_PASSED_SMOKE: at least one genuine complete free-running video, same-setting D-Q pair, with full-duration human review recording no catastrophic collapse.
- READY_FOR_PARETO_BENCHMARK: smoke passed and next validation plan exists, but not automatically a published method or independent-quality pass.
- CLOSED: old/negative result kept as historical evidence.

Do not create an accepted record from only flow, synthetic output, local/standalone FFN, or free-looking videos created by reverting intermediate latents to dense. Each selected candidate is expected to progress to paired full generation **after a separately approved model job**. Do not claim a method is viable while it remains pending; waiting for permission is transparent, not a quality waiver.

Run python3 tools/check_candidate_rollout_gate.py to validate honest metadata invariants. It validates schema/evidence locators, not the visual truth or genuineness of video. The artifacts can remain in verified HF cache with sanitized relative locators; committed metadata includes SHA, full paired configuration and a report with timecodes. The candidate may select an initial low-cost representative source; independent sources and comparable published baselines follow.

Policy: ../../docs/75_备选算法必须完整生成与防止视频质量崩溃_20261022.md.
