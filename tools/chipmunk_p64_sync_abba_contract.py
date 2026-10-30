"""Static seven-video action contract; no experiment authorization."""
MODES={0:('none','A_warm','B_warm','A1','B1','B2','A2'),1:('none','B_warm','A_warm','B1','A1','A2','B2')}
SOURCES=(898,979)
STATE_COUNT=10
ARTIFACT_COUNT=7*(STATE_COUNT+1)+2

def schedule(part):
 if type(part) is not int or part not in MODES:raise ValueError('part differs')
 return MODES[part]

def expected_counts(part):
 schedule(part)
 return {'video_generations':7,'full_FFN_sites':1120,'logical_FFN_linear_dispatches':2240,'candidate_sites':240,'dense_FFN_GEMMs':1760,'selected_W1':240,'selected_W2':240,'proxy_W1':240,'old_selected_W1':240,'state_files':70,'video_files':7,'journals':2,'verification_artifacts':ARTIFACT_COUNT,'mask_SHA':240,'cache_create':240,'cache_release':240,'prefetch_loads':1134,'stage_event_pairs':49,'deferred_completion_barriers':120}
