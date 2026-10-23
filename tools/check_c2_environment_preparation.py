"""Reproduce the fixed import-environment repair from saved CPU/native evidence."""
import argparse
import ast
import hashlib
from pathlib import Path
from c2_contract import ARMS, CAPS, ROOT, atomic, read, sha
from c2_environment_contract import METHOD
from check_c2_reserved_preflight import generation_loop

OUT = ROOT / 'research/20261023/C2N2_environment_source_preflight.json'


def analyze():
    parent = read(ROOT / 'research/20261023/C2N1_native_results.json')
    bad = read(ROOT / 'research/20261023/C2N2_bad_launcher_CPU_reproduction.json')
    good = read(ROOT / 'research/20261023/C2N2_environment_CPU_preflight.json')
    draft_path = ROOT / 'research/20261023/C2N2_environment_smoke_DRAFT.json'
    draft = read(draft_path)
    assert parent['requests'] == [] and not any(parent['budget']['used'].values())
    assert parent['error_message'] == "No module named 'HiFloat4'"
    assert parent['early_CUDA_reservation']['own_process_visible_to_NVML']
    assert parent['resources']['sampled_isolation_pass'] and parent['resources']['violations'] == []
    assert bad['status'] == 'PASS_REPRODUCED_BAD_LAUNCHER_CPU_ONLY'
    assert bad['module_specs'] == {'HiFloat4': False, 'lightx2v': False, 'transformers': True}
    assert good['status'] == 'PASS_NATIVE_IMPORTS_ASSETS_AND_ALL160_METADATA_CPU_ONLY'
    assert all(good['module_specs'].values()) and good['metadata_sites'] == 160
    assert good['combined_DQ_cache_SHA256_verified'] and good['CUDA_initialized'] is False
    assert good['source_supported_platform_check_skipped_for_CPU_import_only'] is True
    assert good['new_GPU_FFN_norm_selector_generation_calls'] == 0
    assert good['source_sha256'] == sha(ROOT / 'tools/check_c2_launch_environment.py')
    assert draft['id'] == METHOD and draft['caps'] == CAPS and draft['arms'] == ARMS
    assert draft['authorization'] is None and draft['automatic_retry'] is False
    assert draft['preserve_project_model_PYTHONPATH'] and draft['native_platform_check_enabled']
    assert 'worker_eval_env.sh' not in draft['launch_chain']
    for name, expected in draft['source_sha256'].items():
        assert sha(ROOT / name) == expected, name
    old = generation_loop(ROOT / 'tools/run_c2_reserved_smoke.py')
    new = generation_loop(ROOT / 'tools/run_c2_environment_smoke.py')
    assert old == new
    tree = ast.parse((ROOT / 'tools/run_c2_environment_smoke.py').read_text())
    guard = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'verify_model_environment']
    torch_line = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Import)
                  and any(a.name == 'torch' for a in n.names)]
    assert max(guard) < min(torch_line)
    return {'id': METHOD, 'status': 'PASS_FIXED_ENVIRONMENT_CPU_SOURCE_PREFLIGHT_NOT_GPU_AUTHORITY',
            'source_sha256': sha(__file__), 'draft_sha256': sha(draft_path),
            'parent_native_result_sha256': sha(ROOT / 'research/20261023/C2N1_native_results.json'),
            'positive_CPU_proof_sha256': sha(ROOT / 'research/20261023/C2N2_environment_CPU_preflight.json'),
            'negative_CPU_proof_sha256': sha(ROOT / 'research/20261023/C2N2_bad_launcher_CPU_reproduction.json'),
            'generation_loop_AST_identical': True,
            'generation_loop_AST_sha256': hashlib.sha256(old.encode()).hexdigest(),
            'model_import_guard_before_Torch_and_CUDA': True,
            'native_platform_check_enabled': True,
            'CPU_import_platform_check_explicitly_skipped': True,
            'new_GPU_FFN_norm_selector_generation_calls': 0,
            'actual_C2N2_model_or_native_numeric_validation': False,
            'independent_execution_authority_required': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(); result = analyze()
    if args.check:
        assert read(OUT) == result
    else:
        if OUT.exists(): raise RuntimeError('Preserve first CPU source proof')
        atomic(OUT, result)
    print({'status': result['status'], 'new_model_calls': 0})


if __name__ == '__main__':
    main()
