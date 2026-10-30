"""Explicit proposed work units, not execution authorization."""
GRID=(21,29,52);SITES=40;SNAPSHOT_BLOCKS=(0,13,26,39);SELECTOR_ROWS=64;EVAL_ROWS=256
# All per student video; storage and observer numeric reads require future scope.
WORK={'full_shadow_W1':40,'full_shadow_GELU':40,'full_shadow_W2':40,
 'selector_identity':40,'selector_identity_D2H':40,'selector_identity_D2H_bytes':20480,'mask_validation':40,'mask_unique':40,'mask_bounds_scalar_checks':80,
 'W2_FP32_and_gather':40,'diagonal_weight_reduction':40,'observer_chunk':4960,'omitted_projection_matmul':4960,'diagonal_gemv':4960,
 'observer_output_gate_and_write_comparisons':4960,'finite_all_scalar_checks':14880,'partition_bincounts':69440,'partition_stat_D2H':69440,'chunk_stats_write':4960,
 'mask_index_body_copies':40,'mask_index_D2H_bytes':1638400,'selector_score_copies':40,'selector_score_D2H_bytes':2211840,'gate_copies':40,'gate_D2H_bytes':819200,'metadata_selector_D2H_calls':40,'metadata_selector_D2H_bytes':20480,'coordinate_rows_64_and_256':40,
 'student_D2H_bytes':13425016832,'student_raw_sha_bytes':13425016832,'raw_tensor_sha':264,'typed_header_reads':64,'sample_320_gather_and_write':40,'sample_selector_D2H_calls':40,'sample_selector_D2H_bytes':20480,'student_rows_H2D_calls':40,'student_rows_H2D_bytes':102400,'sample_tensor_D2H':240,'sample_file_hash':40,'site_stats_write':40,'metadata_json_write':40,
 'snapshot_blocks':4,'snapshot_tensor_D2H':24,'snapshot_file_hash':24,'observer_sync_before_release':40,'GPU_model_generations':1,'training':0}

SNAPSHOT_TENSORS=('old_h','current_h','old_Y','current_dense_Y','candidate_Y','gate','selector_rows','evaluation_rows','mask_indices','selector_score')

def reserve(ledger,rid,cost,fn):
 """Uses existing charged ledger; caller supplies frozen limits and sync."""
 if not cost or any(k not in WORK or type(v) is not int or v<=0 for k,v in cost.items()):raise ValueError('invalid observer cost')
 return ledger.call(rid,cost,fn)

DENSE_WORK={'dense_D2H_bytes':11996823552,'dense_raw_sha_bytes':11996823552,'dense_shadow_W1':40,'dense_shadow_GELU':40,'dense_shadow_W2':40,'dense_h_Y_bitwise_checks':40,'dense_metadata_write':40,'dense_gate_D2H':40,'dense_gate_D2H_bytes':819200,'dense_sample_write':40,'dense_rows_H2D_calls':40,'dense_rows_H2D_bytes':102400,'dense_sample_tensor_D2H':200,'dense_raw_tensor_sha':220,'dense_sample_file_sha':40,'dense_header_reads':60,'dense_snapshot_blocks':4,'dense_snapshot_tensor_D2H':20,'dense_snapshot_file_sha':20,'postprocess_shadow_write':40,'postprocess_actual_bitwise_checks':40}
STUDENT_WORK={k:v for k,v in WORK.items() if k not in ('GPU_model_generations','training')}
STUDENT_WORK.update(postprocess_shadow_write=40,postprocess_actual_bitwise_checks=40)
MODES=('h_cache_dense','high_h_readonly_p64')
