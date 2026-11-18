# P2 DEV3 scoring and playback

Strict historical Stage 4 root on school:

```
$HOME/projects/wan-quant/work/ffnds_strict_reproduction_20261116_336b7e1/quality_stage4_main_v1
```

The source order is vbi_0392, vbi_1013, vbi_0800. The strict Dense, Q1r2 and Global-P64 MP4 paths exist for all three. Their historical per-video metric cells and EXIT receipts are under batches/main__DIM/. The nine metric protocols are preserved there and their exact package/model/config digests are reused. Imaging quality already uses raw MUSIQ points.

After all four P2 generation stages have EXIT0, pass a JSON map of L0, L+, Q32, and N32 to their generation stage roots:

```bash
python3 quality_paired.py prepare "$strict_stage4" "$new_stage4" "$producer_stage_map" "$author_agent_id" \
  --ffprobe "$HOME/miniconda3/envs/vbench_i2v/bin/ffprobe"
sha256sum "$new_stage4"/*/scope.json
```

Have the independent reviewer review the nine concrete scopes, record the actual recheck in each local REVIEW.json and register the finite matching scope/ACTIVE identity under the stage's active_key. Submit each through the repository's authorized school scheduler path with the scope SHA as the second argument to quality_paired_job.sh. The original Stage 4 wrapper contract uses partition P107-A100, account competition, qos qos_p107-a100, one A100 with 80 GiB, 4 CPUs, 160 GiB host cap, 78 GiB device cap and 2 h scheduler ceiling. Historical 55-video full batches took 98–265 s each excluding queue wait. The new non-RAFT batches contain only 12 new videos each. The RAFT batch includes all 21 existing/new MP4s because the archived baseline scores retain 40 pairwise amplitude scalars per video but no same-grid vector fields. That batch retains the 21 fields in host memory during comparison (approximately 2.6 GB additional resident field storage) and writes only scalar summaries. No extra generation is required. Keep the reviewed session GPU concurrency limit.

After scoring, build the provenance-checked paired table and preview manifest:

```bash
python3 quality_paired.py report "$strict_stage4" "$new_stage4" "$playback_and_report"
```

QUALITY_PAIRED.csv preserves individual sources and differences against Dense, Q1r2, Global-P64 and L0. DEV3_SUMMARY.json reports equal-source DEV deltas and extremes. Dynamic degree remains descriptive. Motion smoothness cannot alone support a choice. RAFT amplitude uses the original native dynamic-degree worker's mean top-5%-of-grid displacement, averaged over forty two-frame pairs. The added paired RAFT columns compare same frame-pair index and same pixel, provide mean endpoint error and mean absolute magnitude error in px, and report average angular error in degrees on pixels where both magnitudes exceed 0.001 px. The angular support fraction is also recorded. These are fixed-grid comparisons in independently generated videos, with no object correspondence claim. Missing cells and undefined angular support are explicit.

Use PLAYBACK_MANIFEST.json to retrieve hash-verified original MP4s from school. For each source deliver an 81-frame 16-fps synchronized preview in fixed order with a baseline row (Dense, Q1r2, Global-P64) and a candidate row (L0, L+, Q32, N32), plus all original MP4s for native-resolution viewing. Encode the synchronized view with FFmpeg xstack using frame-zero aligned inputs; label each tile, and verify 81 output frames at 16 fps with ffprobe. View the entire clip at native playback timing and scrub identified intervals against originals. Record timecode, method, source, severity, action/motion, subject identity, background consistency, flicker, and any disagreement with metrics. Contact sheets can index passages but carry no acceptance decision. Synchronization and preview outputs are pending the actual twelve videos and user review.

## Frame-grid contract

The frozen ffprobe is conda-forge 8.1.2 with SHA-256 `040709d6ae5123136c2c397e7164e4d2cc909b81e1357263795feb3c09a07d7d`. The scope records the executable path and checks its bytes at preparation and at evaluation. Each clip is decoded and checked for zero start PTS, exactly 81 frame PTS in exact rational order 0/16,…,80/16, both declared rates of 16 fps, matching best-effort timestamps, and 832×464 pixel grids. Time bases may differ if the rational presentation times and declared rates are identical. Declared codec, stream/frame pixel format, and stream/frame color tags must match across all 21 clips in the RAFT batch. Absent and explicitly unknown color tags are recorded as null and match only absent/unknown tags. VIDEO_FRAME_CONTRACT.json contains the proven time base, ordered frame grid and declarations for each clip. Its digest is in the quality EXIT receipt.

The pinned vbench.VBench.build_full_info_json constructs prompt filenames with slot indices 0 through 4. The adapter verifies that exact native ordered list and adds only the two reviewed existing links at indices 5 and 6 for the dynamic-degree batch. From pinned vbench.dynamic_degree.DynamicDegree.get_frames, decoder order goes through OpenCV, the interval is max(1, round(16/8)) = 2, extract_frame returns frames 0:81:2, and infer uses consecutive elements of the subsampled list. The RAFT adapter checks selection by tensor identity, then checks both actual native forward image tensors against each expected sampled input. RAFT flow arrays are paired only after validating all ordered 0→2, 2→4, …, 78→80 slots. The exact frame-index pairs are included in paired_raft.json and verified by the report reader.
