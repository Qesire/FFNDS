"""Verify original D0 assets/TF2 operands and native signatures without a model."""
import ast
import argparse
import hashlib
from pathlib import Path
import time

from r12d0_contract import ROOT


def require_argument(relative, function, argument):
    tree = ast.parse((ROOT / relative).read_text())
    definitions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function]
    if not any(argument in [a.arg for a in n.args.args] for n in definitions):
        raise RuntimeError('Frozen native callable signature differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'research/20261022/R12_D0_asset_preflight.json')
    args = parser.parse_args()
    import torch
    from ar2_checkpoint import atomic_json, digest
    from historical_source_snapshot import HistoricalSources
    from restored_assets import verified_restored_assets
    from run_cross_layer_transport import references
    import paired_budget_runtime, paired_budget_observer, run_paired_budget_value, analyze_paired_budget_value
    output = args.output
    if output.exists():
        raise RuntimeError('Preserve prior operand check')
    started = time.monotonic()
    history = HistoricalSources()
    _, _, assets = verified_restored_assets(history)
    refs, bindings = references(history, assets)
    require_argument('.runtime/code/LightX2V/lightx2v/models/schedulers/wan/scheduler.py', 'step_pre', 'step_index')
    require_argument('.runtime/code/LightX2V/lightx2v/models/networks/wan/infer/transformer_infer.py', 'post_process', 'pre_infer_out')
    rows = []
    for index in [0, 5, 10, 15]:
        case = refs['DQ'][index]['case']
        sample = case['input']
        if digest(sample['image_path']) != sample['input_provenance']['image_sha256'] or sample['seed'] != 42:
            raise RuntimeError('Frozen source/seed changed')
        rows.append({'index': index, 'image_sha256': sample['input_provenance']['image_sha256'],
            'seed': 42, 'prompt_sha256': hashlib.sha256(sample['prompt'].encode()).hexdigest(),
            'DQ_input_sha256': [s['input_sha256'] for s in case['steps']]})
    if torch.cuda.is_initialized():
        raise RuntimeError('Operand preflight initialized CUDA')
    result = {'status': 'PASS_R12_D0_ORIGINAL_ASSETS_INPUTS_STATIC_SIGNATURES_CPU_ONLY',
        'CUDA_initialized': False, 'new_model_generations': 0, 'assets': assets,
        'original_TF2_reports': bindings, 'inputs': rows, 'source_sha256': digest(__file__),
        'elapsed_seconds': time.monotonic() - started}
    atomic_json(output, result)
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()
