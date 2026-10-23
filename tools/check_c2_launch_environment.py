"""CPU-only negative/positive proof of the exact native C2 import environment."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(16 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expect', choices=['missing_project_paths', 'complete_project_environment'], required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('This environment proof requires explicitly hidden GPUs')
    started = time.monotonic()
    specs = {name: importlib.util.find_spec(name) is not None for name in ['HiFloat4', 'lightx2v', 'transformers']}
    result = {'mode': args.expect, 'module_specs': specs, 'CUDA_VISIBLE_DEVICES_empty': True,
              'new_GPU_FFN_norm_selector_generation_calls': 0, 'source_sha256': sha(__file__)}
    if args.expect == 'missing_project_paths':
        if specs != {'HiFloat4': False, 'lightx2v': False, 'transformers': True}:
            raise RuntimeError('Negative reproduction did not match the actual path-clearing failure')
        result['status'] = 'PASS_REPRODUCED_BAD_LAUNCHER_CPU_ONLY'
    else:
        if not all(specs.values()):
            raise RuntimeError('Native project modules still not importable')
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU proof unexpectedly initialized CUDA')
        # LightX2V provides this explicit flag for CPU-only source inspection.
        # Native model launching never sets it; the model environment unsets it.
        os.environ['SKIP_PLATFORM_CHECK'] = 'True'
        result['source_supported_platform_check_skipped_for_CPU_import_only'] = True
        from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
        from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
        from historical_source_snapshot import HistoricalSources
        from restored_assets import verified_restored_assets
        from ub1_native_replay import UB1Positions
        from ub1_row_dispatch import load_metadata
        history = HistoricalSources()
        kwargs, cache, assets = verified_restored_assets(history)
        policy = ROOT / 'runs/worker_restart_20261020/ub1_step_budget_fit/UB1_policy.json'
        positions = UB1Positions(policy, history,
            expected_policy_sha256='d8840b9328af18c9f3d07b4570130e58467d0f45cae12d1dd13e8db89ebd7a6a')
        metadata = load_metadata(positions)
        if set(metadata) != {(s, b) for s in range(4) for b in range(40)} or torch.cuda.is_initialized():
            raise RuntimeError('Incomplete all160 CPU metadata or unexpected CUDA context')
        result.update(status='PASS_NATIVE_IMPORTS_ASSETS_AND_ALL160_METADATA_CPU_ONLY',
                      torch_version=torch.__version__, adapter_imported=True, native_transformer_imported=True,
                      metadata_sites=len(metadata), combined_DQ_cache_SHA256_verified=True,
                      fixed_assets=assets, CUDA_initialized=False)
    result['CPU_seconds'] = time.monotonic() - started
    out = Path(os.environ['HF_HOME']) / 'derived/ffnds/C2N2_launch_CPU_preflight'
    out.mkdir(parents=True, exist_ok=True)
    path = out / (args.expect + '.json')
    if path.exists():
        raise RuntimeError('Preserve first environment proof')
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
