"""Two-thread CPU import/signature audit; checkpoint loads/NN forwards denied."""
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
from unittest import mock

from fresh_quality_contract import CORE, ROOT, atomic, sha


def main():
    from worker_identity import require_worker
    require_worker()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Import readiness must hide every GPU')
    import torch
    torch.set_num_threads(2)
    target = ROOT / 'research/20261023/R17_metric_import_preflight.json'
    if target.exists():
        raise RuntimeError('Preserve the first import readiness attempt')
    hf = Path(os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface')))
    cotracker = hf / 'derived/ffnds/R17_metric_assets/source/cotracker'
    sys.path.insert(0, str(cotracker))
    import vbench2_beta_i2v
    supplement = hf / 'derived/ffnds/R17_metric_source_supplements/vbench2_beta_i2v'
    vbench2_beta_i2v.__path__.append(str(supplement))
    def forbidden(*args, **kwargs):
        raise RuntimeError('Readiness forbids loading checkpoints or executing any NN module')
    rows = []
    with mock.patch.object(torch, 'load', forbidden), mock.patch.object(torch.jit, 'load', forbidden), \
            mock.patch.object(torch.hub, 'load', forbidden), mock.patch.object(torch.nn.Module, '_call_impl', forbidden):
        for dimension in CORE:
            package = 'vbench2_beta_i2v' if dimension in ('i2v_subject', 'i2v_background', 'camera_motion') else 'vbench'
            module = importlib.import_module(f'{package}.{dimension}')
            function = getattr(module, f'compute_{dimension}')
            signature = inspect.signature(function)
            if list(signature.parameters)[:3] != ['json_dir', 'device', 'submodules_list']:
                raise RuntimeError('Official metric callable signature differs')
            rows.append({'dimension': dimension, 'module': f'{package}.{dimension}',
                         'module_sha256': sha(module.__file__),
                         'function_source_sha256': hashlib.sha256(inspect.getsource(function).encode()).hexdigest(),
                         'signature': str(signature)})
        importlib.import_module('cotracker.predictor')
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU-only import preflight initialized CUDA')
    report = {'id': 'R17_OFFICIAL_METRIC_IMPORT_READINESS',
              'status': 'PASS_NINE_OFFICIAL_FUNCTIONS_AND_COTRACKER_IMPORT_NO_CHECKPOINTS_OR_FORWARDS',
              'source_sha256': sha(__file__), 'functions': rows,
              'versions': {n: importlib.metadata.version(n) for n in ('vbench', 'dreamsim', 'torch', 'torchvision', 'pyiqa', 'timm')},
              'checkpoint_loads': 0, 'NN_module_forwards': 0, 'CUDA_initialized': False,
              'CPU_threads': 2, 'model_execution_authority': False,
              'concurrency_scope': 'Read-only two-thread import check alongside bounded I/O-only asset downloads; generation/evaluation environments unchanged.'}
    atomic(target, report)
    print(json.dumps({'status': report['status'], 'official_dimensions': len(rows), 'CUDA_initialized': False}), flush=True)


if __name__ == '__main__':
    main()
