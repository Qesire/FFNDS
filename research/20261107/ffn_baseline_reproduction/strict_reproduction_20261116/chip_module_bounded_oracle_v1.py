"""Bounded upstream Chipmunk module probe. GPU execution requires independent approval.
Pass --source to pinned src/chipmunk and --library to original compiled selector .so.
This file instruments operator boundaries only; it never changes upstream source.
"""
import argparse, ast, hashlib, importlib, json, os, sys, types
from pathlib import Path
import torch

p = argparse.ArgumentParser()
p.add_argument('--source', required=True, help='pinned src/chipmunk directory')
p.add_argument('--library', required=True, help='original selector shared library')
p.add_argument('--output', required=True)
p.add_argument('--library-sha256', required=True)
a = p.parse_args()
src = Path(a.source).resolve()
expected = {
'modules/mlp.py':'0e1e63fb40b273782e7face0f94ba72cd81906d36af24f24934ea3e3209177fc',
'util/layer_counter.py':'cb9ffab57832ce69141ecd914fbb3f2c873c26bdca31eee1f7e2b41c01764288',
'util/config.py':'1ba524ec3a265f99d4788de7d0bb2988690c84415a7f459b4ab4836201e6a315',
'util/storage/layer_storage.py':'ee56637830f41b4732b4f287caca6b27f1055a8566b09e1bc92ac14ccd89912a',
'util/storage/offloaded_tensor.py':'4e2de98dd363f58a9947b2e59d527d2a94b15240ec187960ab34e3e1808befe1',
'ops/mlp.py':'ad4e035ebddb84effda7eac9df7035378a06a5a4bcd36a5996498f80cd2746ce',
'ops/indexed_io.py':'48eed56ec068b04942adc0cce370458fa24524b633752c71a5f4fae4b078bc48',
'triton/mlp_csp_mm1_bf16.py':'429fa45b341fc0bfc0c0665f701571b54263df7cbaa62828a1889f84c83a6397',
'triton/mlp_csp_mm2_bf16.py':'a656a87864f554c9198f1dd2ca78a79674302e79660ed0aa9bf7c1734bf2513f'}
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
for path, sha in expected.items():
    if digest(src / path) != sha: raise RuntimeError('source identity mismatch: ' + path)
if digest(Path(a.library)) != a.library_sha256.lower(): raise RuntimeError('library identity mismatch')
if not os.environ.get('SLURM_JOB_ID') or torch.cuda.device_count() != 1 or torch.__version__ != '2.10.0+cu128':
    raise RuntimeError('one Slurm GPU with frozen Torch runtime required')
props = torch.cuda.get_device_properties(0)
if (props.major, props.minor) != (8, 0) or 'A100' not in props.name: raise RuntimeError('A100 required')
torch.ops.load_library(str(Path(a.library).resolve()))

def pkg(name, path):
    m = types.ModuleType(name); m.__path__ = [str(path)]; sys.modules[name] = m
    return m
chip = pkg('chipmunk', src)
util = pkg('chipmunk.util', src / 'util')
config = importlib.import_module('chipmunk.util.config')
util.GLOBAL_CONFIG = config.GLOBAL_CONFIG
util.get_kernel_config_mlp = config.get_kernel_config_mlp
util.MlpStorage = importlib.import_module('chipmunk.util.storage.layer_storage').MlpStorage
tri = pkg('chipmunk.triton', src / 'triton'); chip.triton = tri
for name, export in (('mlp_csp_mm1_bf16', 'csp_mlp_mm1_bf16'), ('mlp_csp_mm2_bf16', 'csp_mlp_mm2_bf16')):
    path = src / 'triton' / (name + '.py')
    module_name = 'chipmunk.triton.' + name
    mod = types.ModuleType(module_name)
    mod.__file__, mod.__package__ = str(path), 'chipmunk.triton'
    sys.modules[module_name] = mod
    tree = ast.parse(path.read_bytes(), filename=str(path))
    # Leave all kernel and function bodies unchanged. The author mm2 import
    # demo is a separate invocation on a fully selected last block with no
    # lookahead slack. Skip only that bootstrap, unused by provider=triton.
    imports = [node for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'csp_mlp_mm2_bf16_function_ptr' for t in node.targets)]
    if len(imports) != int(name == 'mlp_csp_mm2_bf16'):
        raise RuntimeError('unexpected exact-author module bootstrap')
    tree.body = [node for node in tree.body if node not in imports]
    exec(compile(tree, str(path), 'exec'), mod.__dict__)
    setattr(tri, export, getattr(mod, export))
ops = pkg('chipmunk.ops', src / 'ops'); chip.ops = ops
indexed = importlib.import_module('chipmunk.ops.indexed_io')
ops.mlp = importlib.import_module('chipmunk.ops.mlp').run_e2e
ops.topk_indices = indexed.topk_indices
ops.copy_indices = indexed.copy_indices
SparseDiffMlp = importlib.import_module('chipmunk.modules.mlp').SparseDiffMlp
LayerCounter = importlib.import_module('chipmunk.util.layer_counter').LayerCounter
cfg = config.GLOBAL_CONFIG
cfg['steps'] = 16
if any(cfg['offloading'][k] for k in cfg['offloading'] if k.startswith('mlp.')):
    raise RuntimeError('the resident fixture requires author MLP offloading defaults')
cfg['mlp'].update(is_enabled=True, provider='triton', first_n_dense_layers=2,
                  full_step_every=10, block_mask_cache=2, top_keys=.3, random_keys=.05)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
records = []
selector_calls = []
orig_topk, orig_copy, orig_mlp = ops.topk_indices, ops.copy_indices, ops.mlp
active_proxy = None

def validate(indices, counts):
    # Reject all bad prefixes before either author copy or Triton kernel can read them.
    for row, count in zip(indices.reshape(-1, indices.shape[-1]), counts.flatten()):
        n = int(count.item()); assert 0 <= n < row.numel() and n % 256 == 0
        prefix = row[:n]
        assert bool(((prefix >= 0) & (prefix < 1024)).all().item()), 'unsafe selector prefix'
        assert prefix.unique().numel() == n, 'duplicate selector indices'

def topk(*args):
    if not torch.equal(args[0], active_proxy):
        raise AssertionError('selector input differs from native block-128 preactivation proxy')
    selector_calls.append(counter.cur_inference_step - 1)
    args[1].fill_(-1); args[2].fill_(-1)
    orig_topk(*args); validate(args[1], args[2])
def copy(*args):
    validate(args[2], args[3])
    proxy, cache, indices, counts = args
    expected_cache = cache.clone()
    for block, count in enumerate(counts[0].tolist()):
        columns = indices[0, block, :count].long()
        expected_cache[0, block, columns] = proxy[0, block, columns]
    orig_copy(*args)
    if not torch.equal(cache, expected_cache):
        raise AssertionError('proxy selective writeback mismatch')
def mlp(**kw):
    validate(kw['indices'], kw['counts'])
    orig_mlp(**kw)
ops.topk_indices, ops.copy_indices, ops.mlp = topk, copy, mlp

def numerical_witnesses():
    # Run the unchanged author kernels with exact BF16 operands; the alternatives
    # differ by a whole BF16 ULP, not by a tolerance applied to random data.
    x = torch.zeros((256,256), device='cuda', dtype=torch.bfloat16); x[:,0] = 1
    w1 = torch.zeros((1024,256), device='cuda', dtype=torch.bfloat16); w1[:,0] = 1
    bias = torch.full((1024,), 2**-8, device='cuda', dtype=torch.bfloat16)
    ind = torch.arange(1024, device='cuda', dtype=torch.int32).repeat(2,1)
    cnt = torch.full((2,), 256, device='cuda', dtype=torch.int32)
    packed = torch.zeros((256,1024), device='cuda', dtype=torch.bfloat16)
    h = torch.zeros((1024,256), device='cuda', dtype=torch.bfloat16)
    tri.csp_mlp_mm1_bf16(x,w1,packed,bias,h,ind,cnt)
    f = torch.tensor(1 + 2**-8,device='cuda',dtype=torch.float32)
    correct = torch.nn.functional.gelu(f, approximate='tanh').to(torch.bfloat16)
    rounded_first = torch.nn.functional.gelu(f.to(torch.bfloat16).float(), approximate='tanh').to(torch.bfloat16)
    if correct.item() == rounded_first.item() or h[0,0].item() != correct.item():
        raise AssertionError('MM1 bias/activation witness failed')
    packed.zero_(); packed[:,0] = 2**-8; packed[:,1] = 2**-16
    w2 = torch.zeros((1024,256),device='cuda',dtype=torch.bfloat16); w2[0,0] = 1; w2[1,0] = 1
    y = torch.ones((256,256),device='cuda',dtype=torch.bfloat16)
    tri.csp_mlp_mm2_bf16(packed,w2,ind,cnt,y)
    actual = y[0,0].item()
    correct_y = torch.tensor(1.0 + float(torch.tensor(2**-8 + 2**-16,dtype=torch.bfloat16)),dtype=torch.bfloat16).item()
    alternative_y = torch.tensor(1.0 + 2**-8 + 2**-16,dtype=torch.bfloat16).item()
    if correct_y == alternative_y or actual != correct_y: raise AssertionError('MM2 accumulation witness failed')
    return {'mm1_actual':h[0,0].item(), 'mm1_expected':correct.item(), 'mm1_round_first':rounded_first.item(),
            'mm2_actual':actual, 'mm2_expected':correct_y, 'mm2_add_first':alternative_y}

witnesses = numerical_witnesses()

torch.manual_seed(842)
fc1 = torch.nn.Linear(256, 1024, device='cuda', dtype=torch.bfloat16)
fc2 = torch.nn.Linear(1024, 256, device='cuda', dtype=torch.bfloat16)
counter = LayerCounter(3, 1)
layers = [SparseDiffMlp(i, counter, fc1, torch.nn.GELU(approximate='tanh'), fc2) for i in range(3)]
module = layers[2]
base = torch.randn((1, 256, 256), device='cuda', dtype=torch.bfloat16)
with torch.no_grad():
    masks = {}
    for step in range(13):
        assert counter.cur_inference_step == step
        x = (base.float() + step * .01 * torch.sin(base.float())).to(torch.bfloat16)
        for layer in layers[:2]:
            dense = layer(x)
            assert torch.equal(dense, fc2(torch.nn.functional.gelu(fc1(x), approximate='tanh')))
            assert layer.storage.get_out_cache() is None
        before_h = module.storage.get_sparse_act_T()
        before_y = module.storage.get_out_cache()
        before_proxy = module.storage.get_blockmean_mid_cache()
        before_h = None if before_h is None else before_h.clone()
        before_y = None if before_y is None else before_y.clone()
        before_proxy = None if before_proxy is None else before_proxy.clone()
        active_proxy = None if before_proxy is None else (fc1(x.reshape(1, 2, 128, 256).mean(dim=2)) - before_proxy).abs()
        y = module(x).clone()
        inds, counts = module.storage.get_indices(), module.storage.get_counts()
        rec = {'step': step, 'finite': bool(torch.isfinite(y).all().item())}
        if not rec['finite']:
            raise AssertionError('nonfinite author output')
        if step in (0, 10):
            z = fc1(x)
            h = torch.nn.functional.gelu(z, approximate='tanh')
            assert torch.equal(y, fc2(h))
            assert torch.equal(module.storage.get_sparse_act_T(), h.transpose(-1, -2))
            assert torch.equal(module.storage.get_blockmean_mid_cache(), z.reshape(1, 2, 128, 1024).mean(dim=2))
            rec['mode'] = 'anchor'
            if step == 10:
                assert torch.equal(inds, masks[9][0]) and torch.equal(counts, masks[9][1])
        else:
            if inds is None:
                raise AssertionError('missing author sparse mask')
            masks[step] = (inds.clone(), counts.clone())
            if step == 1:
                row0 = set(inds[0, 0, :int(counts[0, 0])].tolist())
                row1 = set(inds[0, 1, :int(counts[0, 1])].tolist())
                if row0 == row1:
                    raise AssertionError('fixture failed to distinguish block masks')
            h = module.storage.get_sparse_act_T().clone()
            # Independent arithmetic uses the same actual indices and cached states.
            ref_h, ref_y = before_h.clone(), before_y.clone()
            for block in range(2):
                ids = inds[0, block, :int(counts[0, block])].long()
                sl = slice(block * 128, (block + 1) * 128)
                keep = torch.ones(1024, device=x.device, dtype=torch.bool)
                keep[ids] = False
                assert torch.equal(h[0, keep, sl], before_h[0, keep, sl])
                actual_h = torch.nn.functional.gelu((x[0, sl].float() @ fc1.weight[ids].float().T + fc1.bias[ids].float()), approximate='tanh').to(torch.bfloat16)
                old_h = before_h[0, ids, sl].T
                delta = (actual_h.float() - old_h.float()).to(torch.bfloat16)
                ref_h[0, ids, sl] = actual_h.T
                ref_y[0, sl] = (before_y[0, sl].float() + (delta.float() @ fc2.weight[:, ids].float().T).to(torch.bfloat16).float()).to(torch.bfloat16)
            rec.update(mode='reuse' if step == 11 else 'recompute', counts=counts.flatten().tolist(),
                       h_max_abs=float((h.float()-ref_h.float()).abs().max().item()),
                       y_max_abs=float((y.float()-ref_y.float()).abs().max().item()),
                       direct_max_abs=float((y.float()-fc2(torch.nn.functional.gelu(fc1(x), approximate='tanh')).float()).abs().max().item()))
            if rec['h_max_abs'] > .015625 or rec['y_max_abs'] > .03125:
                raise AssertionError(f'same-mask author arithmetic differs at step {step}: {rec}')
            if step == 11:
                assert torch.equal(before_proxy, module.storage.get_blockmean_mid_cache())
        records.append(rec)
    assert torch.equal(masks[9][0], masks[11][0]) and torch.equal(masks[9][1], masks[11][1]), 'step 11 did not preserve step 9 mask'
    assert selector_calls == list(range(1, 10)) + [12], selector_calls
    counter.reset(); module.storage = util.MlpStorage(2)
    assert module.storage.get_out_cache() is None and counter.cur_inference_step == 0
    layers[0](base); layers[1](base); module(base)
    assert module.storage.get_out_cache() is not None
    torch.cuda.synchronize()
with Path(a.output).open('x', encoding='utf-8') as output:
    json.dump({'status':'PASS', 'records':records, 'new_request_seeded':True, 'selector_steps': selector_calls,
               'source_sha256': expected, 'library_sha256': a.library_sha256, 'gpu': props.name, 'witnesses': witnesses,
               'limits': 'aligned author MLP width1024; exclude unsafe full-count lookahead; skip unused CUDA-provider import demo'},
              output, indent=2)
    output.write('\n')


