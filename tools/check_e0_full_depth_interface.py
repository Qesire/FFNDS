"""Read-only AST/config checks; never imports native model modules or launches jobs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CONFIG = {
    'model_cls': 'wan2.2_moe_distill', 'scheduler': 'Wan22StepDistillScheduler',
    'infer_steps': 4, 'num_layers': 40, 'dim': 5120, 'ffn_dim': 13824,
    'boundary_step_index': 2, 'attention': 'torch_sdpa', 'feature_caching': 'NoCaching',
    'enable_cfg': False, 'seq_parallel': False, 'parallel': False,
    'use_image_encoder': False, 'rope_type': 'torch',
    'cpu_offload': True, 'offload_granularity': 'block',
}
NATIVE_ROOT = '.runtime/code/LightX2V/lightx2v/'
REQUIRED_APIS = [
    ('models/networks/wan/infer/transformer_infer.py', 'WanTransformerInfer',
     'infer_block', ['self', 'block', 'x', 'pre_infer_out']),
    ('models/networks/wan/infer/transformer_infer.py', 'WanTransformerInfer',
     'post_process', ['self', 'x', 'y', 'c_gate_msa', 'pre_infer_out']),
    ('models/networks/wan/infer/offload/transformer_infer.py', 'WanOffloadTransformerInfer',
     'infer_with_blocks_offload', ['self', 'blocks', 'x', 'pre_infer_out']),
    ('models/networks/base_model.py', 'BaseTransformerModel', 'set_scheduler', ['self', 'scheduler']),
    ('models/schedulers/scheduler.py', 'BaseScheduler', 'step_pre', ['self', 'step_index']),
    ('models/schedulers/wan/step_distill/scheduler.py', 'Wan22StepDistillScheduler', 'step_post', ['self']),
    ('common/offload/manager.py', 'WeightAsyncStreamManager', 'swap_blocks', ['self']),
]


def validate_design(design, root=ROOT):
    if design.get('status') != 'INTERFACE_DESIGN_ONLY_NOT_A_MODEL_PROTOCOL':
        raise ValueError('This checker accepts a static design, never an execution protocol')
    if design.get('model_execution_authority') != 'NONE' or design.get('new_generations') != 0:
        raise ValueError('Static source checks cannot authorize model execution')
    config = design.get('required_effective_config', {})
    for key, expected in REQUIRED_CONFIG.items():
        if key not in config or type(config[key]) is not type(expected) or config[key] != expected:
            raise ValueError('Unsupported native configuration: ' + key)
    hashes = design['source_sha256']
    for relative, expected in hashes.items():
        data = (root / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Source drift: ' + relative)
    for relative, class_name, method, arguments in REQUIRED_APIS:
        relative = NATIVE_ROOT + relative
        if relative not in hashes:
            raise ValueError('Required native source not bound: ' + relative)
        tree = ast.parse((root / relative).read_text())
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name]
        functions = [n for c in classes for n in c.body if isinstance(n, ast.FunctionDef) and n.name == method]
        if len(functions) != 1 or [a.arg for a in functions[0].args.args] != arguments:
            raise ValueError('Native API changed: ' + class_name + '.' + method)
    if design.get('residency_policy') != 'NON_FFN_RETAIN_CURRENT_BLOCK_FFN_ONLY_UNIMPLEMENTED':
        raise ValueError('Unbounded dense FFN residency is not this design')
    if design.get('offload_swap_policy') != 'ONCE_AFTER_BOTH_TRAJECTORIES_AND_SHADOWS':
        raise ValueError('Both trajectories must finish before the current buffer is swapped')
    if design.get('scheduler_boundaries') != ['before_step_pre', 'after_step_pre', 'after_step_post']:
        raise ValueError('Entry cast must be distinguished from scheduler update')
    missing = design.get('unimplemented_native_components', [])
    if not missing:
        raise ValueError('Static design must not masquerade as a completed native producer')
    return {'status': 'STATIC_SOURCE_INTERFACE_PASS_NATIVE_UNIMPLEMENTED',
            'checked_sources': len(hashes), 'checked_APIs': len(REQUIRED_APIS),
            'unimplemented_native_components': missing, 'GPU_calls': 0, 'new_generations': 0,
            'model_execution_authority': 'NONE', 'native_neutrality': 'NOT_MEASURED',
            'peak_memory_fit': 'UNKNOWN', 'quality_accepted': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, default=ROOT / 'config/e0_full_depth_trace_design_20261026.json')
    args = parser.parse_args()
    print(json.dumps(validate_design(json.loads(args.design.read_text())), indent=2))


if __name__ == '__main__':
    main()
