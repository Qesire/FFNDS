"""Reproduce CPU/source evidence for an unexecuted startup-only C2 repair."""
import argparse
import ast
import hashlib
from pathlib import Path
from c2_contract import ROOT, atomic, read, sha
from c2_reserved_contract import METHOD

OUT = ROOT / 'research/20261023/C2N1_startup_CPU_preflight.json'


def generation_loop(path):
    nodes = [n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.For)
             and isinstance(n.iter, ast.Name) and n.iter.id == 'ARMS']
    assert len(nodes) == 1
    return ast.dump(nodes[0], include_attributes=False)


def analyze():
    parent = ROOT / 'research/20261023/C2N0_registered_protocol_v1.json'
    p = read(parent); result = read(ROOT / 'research/20261023/C2N0_native_results.json')
    assert result['status'] == 'FAILED_C2_GPU_ISOLATION' and result['requests'] == []
    assert not any(result['budget']['used'].values()) and result['budget']['debits'] == []
    original = ROOT / 'tools/run_c2_norm_scaled_smoke.py'
    repaired = ROOT / 'tools/run_c2_reserved_smoke.py'
    assert sha(original) == p['source_sha256']['tools/run_c2_norm_scaled_smoke.py']
    old_loop, new_loop = generation_loop(original), generation_loop(repaired)
    assert old_loop == new_loop
    tree = ast.parse(repaired.read_text()); calls = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name): calls.setdefault(n.func.id, []).append(n.lineno)
    assert max(calls['reserve_before_assets']) < min(calls['verified_restored_assets'])
    math = ['tools/norm_scaled_row_dispatch.py', 'tools/ub1_row_dispatch.py', 'tools/row_dispatch_kernels.py']
    for name in math: assert sha(ROOT / name) == p['source_sha256'][name]
    setup = ['tools/startup_gpu_reservation.py', 'tools/c2_reserved_contract.py',
             'tools/run_c2_reserved_smoke.py', 'scripts/worker_wait_uuid_idle_exec.sh']
    return {'id': METHOD, 'status': 'PASS_CPU_SOURCE_PREFLIGHT_NOT_NATIVE_VALIDATION',
            'parent_protocol_sha256': sha(parent),
            'parent_native_result_sha256': sha(ROOT / 'research/20261023/C2N0_native_results.json'),
            'parent_actual_calls': result['budget']['used'], 'parent_algorithm_verdict': 'NOT_MEASURED',
            'generation_loop_AST_identical': True,
            'generation_loop_AST_sha256': hashlib.sha256(old_loop.encode()).hexdigest(),
            'unchanged_scientific_operator_sha256': {n: sha(ROOT / n) for n in math},
            'setup_source_sha256': {n: sha(ROOT / n) for n in setup},
            'reservation_precedes_CPU_asset_validation': True,
            'new_native_model_calls': 0, 'new_GPU_executions': 0,
            'live_CUDA_reservation_validated': False,
            'CPU_tests_use_mock_CUDA_and_NVML': True,
            'execution_authority_granted_by_this_check': False,
            'scope': 'Same C2 scientific computation; separate startup-only engineering attempt. Tiny same-process CUDA allocation and UUID binding are implemented but not run on a GPU.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--check', action='store_true')
    args = parser.parse_args(); result = analyze()
    if args.check:
        assert read(OUT) == result, 'Saved CPU preflight differs'
    else: atomic(OUT, result)
    print({'status':result['status'], 'same_scientific_loop':True, 'new_GPU_executions':0})


if __name__ == '__main__': main()
