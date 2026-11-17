"""Load the pinned author MLP/Triton source for the separate safe Wan port."""

import ast
import hashlib
import importlib
from pathlib import Path
import sys
import types

import torch

from chip_wan_voxel_tail_v1 import install_safe_selector_boundary


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def install_default_stream_bridge(ops, cuda):
    """Order the unmodified default-stream CUDA launchers with caller work.

    Both extension operators enqueue on the device default stream, while their
    upstream/downstream PyTorch and Triton work uses the active caller stream.
    Record allocator use for every tensor passed to either asynchronous launch.
    """
    def wrap(original):
        def invoke(*args, **kwargs):
            tensors = [x for x in (*args, *kwargs.values()) if getattr(x, 'is_cuda', False)]
            if not tensors:
                raise ValueError('default-stream bridge requires CUDA tensor arguments')
            device = tensors[0].device
            if any(x.device != device for x in tensors) or cuda.current_device() != device.index:
                raise RuntimeError('default-stream bridge device mismatch')
            caller = cuda.current_stream(device=device)
            default = cuda.default_stream(device=device)
            for tensor in tensors:
                tensor.record_stream(default)
            producer_done = cuda.Event()
            producer_done.record(caller)
            default.wait_event(producer_done)
            consumer_ready = cuda.Event()
            try:
                with cuda.stream(default):
                    return original(*args, **kwargs)
            finally:
                consumer_ready.record(default)
                caller.wait_event(consumer_ready)
        return invoke
    ops.topk_indices = wrap(ops.topk_indices)
    ops.copy_indices = wrap(ops.copy_indices)


def load_author_provider(spec):
    root = Path(spec['pinned_src']).resolve()
    if any(name == 'chipmunk' or name.startswith('chipmunk.') for name in sys.modules):
        raise RuntimeError('another Chipmunk provider is already loaded')
    sources = {
        'modules/mlp.py', 'util/layer_counter.py', 'util/config.py',
        'util/storage/layer_storage.py', 'util/storage/offloaded_tensor.py',
        'ops/mlp.py', 'ops/indexed_io.py', 'triton/mlp_csp_mm1_bf16.py', 'triton/mlp_csp_mm2_bf16.py',
    }
    if set(spec['source_sha256']) != sources:
        raise RuntimeError('incomplete author source identity')
    for name, identity in spec['source_sha256'].items():
        if digest(root / name) != identity:
            raise RuntimeError(f'changed Chipmunk source {name}')
    library = Path(spec['library']).resolve()
    if digest(library) != spec['library_sha256']:
        raise RuntimeError('changed A100-safe selector library')
    torch.ops.load_library(str(library))

    def package(name, path):
        value = types.ModuleType(name)
        value.__path__ = [str(path)]
        sys.modules[name] = value
        return value

    namespace = package('chipmunk', root)
    util = package('chipmunk.util', root / 'util')
    cfg = importlib.import_module('chipmunk.util.config')
    util.GLOBAL_CONFIG = cfg.GLOBAL_CONFIG
    util.get_kernel_config_mlp = cfg.get_kernel_config_mlp
    util.MlpStorage = importlib.import_module('chipmunk.util.storage.layer_storage').MlpStorage
    namespace.triton = package('chipmunk.triton', root / 'triton')
    for module, export in (('mlp_csp_mm1_bf16', 'csp_mlp_mm1_bf16'), ('mlp_csp_mm2_bf16', 'csp_mlp_mm2_bf16')):
        path = root / 'triton' / (module + '.py')
        fullname = 'chipmunk.triton.' + module
        handle = types.ModuleType(fullname)
        handle.__file__, handle.__package__ = str(path), 'chipmunk.triton'
        sys.modules[fullname] = handle
        syntax = ast.parse(path.read_bytes(), filename=str(path))
        demo = [node for node in syntax.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == 'csp_mlp_mm2_bf16_function_ptr' for target in node.targets)]
        if len(demo) != int(module == 'mlp_csp_mm2_bf16'):
            raise RuntimeError('unexpected author import-time driver')
        syntax.body = [node for node in syntax.body if node not in demo]
        exec(compile(syntax, str(path), 'exec'), handle.__dict__)
        setattr(namespace.triton, export, getattr(handle, export))
    ops = package('chipmunk.ops', root / 'ops')
    namespace.ops = ops
    indexed = importlib.import_module('chipmunk.ops.indexed_io')
    ops.topk_indices, ops.copy_indices = indexed.topk_indices, indexed.copy_indices
    ops.mlp = importlib.import_module('chipmunk.ops.mlp').run_e2e
    install_default_stream_bridge(ops, torch.cuda)
    install_safe_selector_boundary(ops)
    cfg.GLOBAL_CONFIG['mlp']['provider'] = 'triton'
    if any(value for key, value in cfg.GLOBAL_CONFIG['offloading'].items() if key.startswith('mlp.')):
        raise RuntimeError('unexpected author cache offload defaults')
    sparse = importlib.import_module('chipmunk.modules.mlp').SparseDiffMlp
    counter = importlib.import_module('chipmunk.util.layer_counter').LayerCounter
    return sparse, counter
