"""CPU preparation for R14N0: imports, source branch arithmetic and budget tests.

No model construction, weights, quantizer invocation or CUDA initialization.
The source-extracted writeback is tested only on tiny synthetic CPU tensors.
"""
import argparse
import ast
import json
from pathlib import Path
import types
import unittest

from r14_stripe_contract import ROOT, sha
from worker_identity import require_worker

SOURCES = [
    'tools/run_complete_stripe_value.py', 'tools/stripe_native_operator.py',
    'tools/stripe_teacher_capture.py', 'tools/r14_stripe_contract.py',
    'tools/preflight_complete_stripe_value.py', 'tools/check_research_routes.py',
    'tools/analyze_stripe_native_values.py', 'tools/spatial_contrast_kernels.py',
    'tools/probe_three_direction_backend.py', 'tools/variable_teacher_capture.py',
    'tools/collect_ub1_native_states.py', 'tools/ub1_native_data.py', 'tools/ub1_native_replay.py',
    'tools/step_budget_runtime.py', 'tools/trained_packed_runtime.py',
    'tools/folded_adapter.py', 'tools/onpolicy_residual.py', 'tools/reduced_rank_residual.py',
    'tools/restored_assets.py', 'tools/historical_source_snapshot.py',
    'tools/run_ar2_restart.py', 'tools/ar2_checkpoint.py', 'tools/worker_identity.py',
    'config/worker.env', 'scripts/worker_exec.sh',
    '.runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/lightx2v_wan_distill.py',
    '.runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/wan_i2v.py',
    '.runtime/code/LightX2V/lightx2v/models/networks/wan/infer/transformer_infer.py',
    '.runtime/code/LightX2V/lightx2v/models/schedulers/wan/step_distill/scheduler.py',
    '.runtime/code/Wan2.2-I2V-A14B-W4A4/quant/hif4_backend.py',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require_worker()
    if args.output.exists():
        raise RuntimeError('Preserve first CPU preparation receipt')
    import torch
    import stripe_native_operator
    import stripe_teacher_capture
    import run_complete_stripe_value
    from prepare_complete_stripe_operand_draft import select_stripes
    from onpolicy_residual import same_bits
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU preflight cannot initialize CUDA')
    for name in SOURCES:
        if name.endswith('.py'):
            ast.parse((ROOT / name).read_text())
    draft_path = ROOT / 'research/20261022/R14_native_value_DRAFT.json'
    draft = json.loads(draft_path.read_text())
    groups, rows = select_stripes(0)
    if (groups != draft['coordinates']['planned_stripe_ids']
            or rows != draft['coordinates']['planned_native_row_ids']):
        raise RuntimeError('Shared complete-stripe coordinates do not reproduce')
    native_source = ROOT / '.runtime/code/LightX2V/lightx2v/models/networks/wan/infer/transformer_infer.py'
    module = ast.parse(native_source.read_text())
    klass = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == 'WanTransformerInfer')
    post = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == 'post_process')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[post], type_ignores=[]), str(native_source), 'exec'), namespace)
    original = namespace['post_process']
    controls = []
    torch.manual_seed(20261022)
    for residual_dtype, sensitive in ((torch.bfloat16, torch.bfloat16), (torch.float32, torch.float32),
                                       (torch.float32, torch.bfloat16)):
        residual = torch.randn((8, 5120)).to(residual_dtype)
        y, gate = torch.randn((8, 5120)).bfloat16(), torch.randn((1, 5120)).bfloat16()
        infer = types.SimpleNamespace(infer_dtype=torch.bfloat16, sensitive_layer_dtype=sensitive,
                                      clean_cuda_cache=False)
        expected = original(infer, residual.clone(), y, gate)
        actual = stripe_teacher_capture.writeback(residual, y, gate, torch.bfloat16, sensitive)
        if not same_bits(actual, expected):
            raise RuntimeError('Reconstructed native residual branch differs from original source')
        controls.append({'residual_dtype': str(residual_dtype), 'sensitive_dtype': str(sensitive),
                         'source_extracted_post_process_bitwise_equal': True, 'synthetic_CPU_only': True})
    patterns = ('test_stripe_contract.py', 'test_stripe_value_analysis.py', 'test_r14_backend_records.py')
    suite = unittest.TestSuite(unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern=p) for p in patterns)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful() or result.testsRun != 13 or torch.cuda.is_initialized():
        raise RuntimeError('CPU software/resource checks failed')
    report = {'status': 'PASS_R14N0_CPU_PREFLIGHT_NOT_NATIVE_PROOF',
              'source_sha256': {name: sha(ROOT / name) for name in SOURCES},
              'draft_sha256': sha(draft_path), 'CPU_tests': result.testsRun,
              'synthetic_writeback_controls': controls, 'shared_native_stripes': len(groups),
              'planned_rows_per_source': len(rows), 'CUDA_initialized': False,
              'new_model_generations': 0, 'model_weights_loaded': 0, 'quantizer_calls': 0,
              'limits': 'CPU imports, syntax, isolated resource/registration tests and synthetic source-extracted residual arithmetic. No native model, fused native input, replay or quality proof.'}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(report['status'], flush=True)


if __name__ == '__main__':
    main()
