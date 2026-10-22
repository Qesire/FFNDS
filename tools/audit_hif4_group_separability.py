"""Source/CPU discriminator for native HiF4 group and row boundaries.

Runs the unchanged original Python reference body on CPU, never a model or
CUDA. This supports a scoped algebraic property, not native GEMM identity.
"""
import argparse
import ast
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
QROOT = '.runtime/code/HiF4_Sp/HiFloat4/hif4_gpu/quant_cy/base/'


def reference():
    # Import only the original pure-Python QType and reference function bodies,
    # without importing/building the CUDA-extension package on the CPU worker.
    types = {}
    path = ROOT / QROOT / 'QType.py'
    exec(compile(path.read_text(), str(path), 'exec'), types)
    path = ROOT / QROOT / 'QFuncs/hifx.py'
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == 'QType')]
    namespace = {'QType': types['QType']}
    exec(compile(tree, str(path), 'exec'), namespace)
    return types['QType']('hifx4').dim(-1), namespace['quant_hifx']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Preserve the original audit')
    import torch
    from safetensors.torch import load_file
    from ar2_checkpoint import atomic_json, digest
    from run_ar2_restart import tensor_sha
    from worker_identity import require_worker
    require_worker()
    torch.set_num_threads(8)
    started = time.monotonic()
    qtype, original = reference()
    if qtype.blk_size != 64 or qtype.blk_outer_size != 1:
        raise RuntimeError('Original native quantizer grouping differs')

    def quant(x):
        return original(x.float(), qtype, -1).to(x.dtype)

    def equal(a, b):
        return a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))

    audit_path = ROOT / 'research/20261022/R12_E0_operand_audit.json'
    audit = json.loads(audit_path.read_text())
    # The public HF_DERIVED locator intentionally drops the physical HF root;
    # OP2 and the node-local model cache need not share that root. Resolve from
    # the original producer receipts, never infer paths from neutral locators.
    original_paths = {}
    for shard in (0, 1):
        report = json.loads((ROOT / f'reports/worker_20261020/OP2_UB1_calibration24_shard{shard}_RUN.json').read_text())
        for entry in report['completed']:
            if entry['index'] not in (0, 5, 10, 15):
                continue
            if digest(entry['input_report']) != entry['input_report_sha256']:
                raise RuntimeError('Original OP2 producer receipt changed')
            case = json.loads(Path(entry['input_report']).read_text())
            for scope in case['scopes']:
                if scope['step'] == 3 and scope['block'] in (30, 31):
                    original_paths[entry['index'], scope['block']] = Path(scope['path'])
    rows = []
    for o in audit['observations']:
        if o['family'] != 'OP2' or o['index'] not in (0, 5, 10, 15):
            continue
        path = original_paths[o['index'], o['block']]
        if digest(path) != o['sha256']:
            raise RuntimeError('Actual saved operand bytes changed')
        x = load_file(str(path), device='cpu')['V1']
        if tensor_sha(x) != o['tensors']['V1']['sha256'] or x.shape != (256, 5120):
            raise RuntimeError('Actual saved V1 decode differs')
        q = quant(x)
        group_order = torch.arange(79, -1, -1)
        channels = (group_order[:, None] * 64 + torch.arange(64)).reshape(-1)
        selected = (torch.arange(0, 80, 2)[:, None] * 64 + torch.arange(64)).reshape(-1)
        checks = {'whole64_group_reordering': equal(quant(x[:, channels]), q[:, channels]),
                  'whole64_group_selection': equal(quant(x[:, selected]), q[:, selected]),
                  'row_partition': equal(torch.cat([quant(x[:85]), quant(x[85:])]), q),
                  'row_reordering': equal(quant(x.flip(0)), q.flip(0))}
        if not all(checks.values()):
            raise RuntimeError('Reference group/row separability counterexample found')
        rows.append({'index': o['index'], 'block': o['block'], 'locator': o['locator'],
                     'payload_sha256': o['sha256'], 'input_sha256': tensor_sha(x),
                     'input_shape': list(x.shape), 'reference_output_sha256': tensor_sha(q), 'checks': checks})
    if len(rows) != 8:
        raise RuntimeError('Incomplete fixed saved-input scope')
    generator = torch.Generator(device='cpu').manual_seed(20261022)
    x = torch.randn((3, 13824), generator=generator).bfloat16()
    x[:, ::13] *= 12
    q = quant(x)
    packs = []
    for o in audit['observations']:
        if o['family'] != 'ORIGINAL_DQ_WEIGHTS':
            continue
        for n in (108, 162):
            ids = torch.tensor(o['CPU_pack_variants'][str(n)]['groups'])
            channels = (ids[:, None] * 64 + torch.arange(64)).reshape(-1)
            ok = equal(quant(x[:, channels]), q[:, channels])
            if not ok:
                raise RuntimeError('Audited108/162 reference pack violates group separability')
            packs.append({'block': o['block'], 'groups': n, 'reference_bitwise_equal': ok})
    # This reorders channels INSIDE every64 group across its4/8 hierarchy.
    within = torch.arange(64)
    within[0], within[4] = within[4].clone(), within[0].clone()
    columns = (torch.arange(216)[:, None] * 64 + within).reshape(-1)
    inverse = columns.argsort()
    permuted_back = quant(x[:, columns])[:, inverse]
    mismatch = int((permuted_back.view(torch.int16) != q.view(torch.int16)).sum())
    if not mismatch:
        raise RuntimeError('Expected within-group hierarchical counterexample did not occur')
    if torch.cuda.is_initialized():
        raise RuntimeError('This check must stay CPU-only')
    sources = ['QType.py', 'QTensor.py', 'QFuncs/hifx.py', 'cusrc/hifxg_quant_cuda.cu']
    result = {'id': 'R12_HIF4_GROUP_AND_ROW_SEPARABILITY',
        'status': 'PASS_SOURCE_AND_ORIGINAL_PYTHON_REFERENCE_CPU_CHECK',
        'source_sha256': {QROOT + p: digest(ROOT / QROOT / p) for p in sources},
        'audit_source_sha256': digest(__file__), 'operand_audit_sha256': digest(audit_path),
        'CUDA_initialized': False, 'new_model_generations': 0, 'new_raw_tensor_bytes': 0,
        'reference_loading': 'Original QType and quant_hifx bodies unchanged; only relative QType import resolved directly, no CUDA extension loaded.',
        'native_source_observation': 'hifx_quant_cuda_inner reads/writes exactly thread_idx*64+i, i=0..63. Its4/8/64 hierarchy uses no other group or row. All registered widths are multiples of64.',
        'derived_property': 'For complete ordered64-channel groups S and row selection/partition P, Q(P X[:,S])=P Q(X)[:,S], when the retained input values and group boundaries are identical.',
        'saved_post_Q1_reference_cases': rows, 'synthetic_hidden_pack_cases': packs,
        'within64_hierarchy_counterexample': {'input_shape': list(x.shape), 'swapped_offsets': [0, 4],
            'input_sha256': tensor_sha(x), 'reference_original_output_sha256': tensor_sha(q),
            'permuted_back_output_sha256': tensor_sha(permuted_back), 'BF16_element_mismatches': mismatch},
        'scope': 'Source deduction plus CPU Python counterpart. Does not prove CUDA/Python equality, packed W1/W2 BF16 GEMM equality, native tile quality or speed. No new method registration.',
        'elapsed_seconds': time.monotonic() - started}
    atomic_json(args.output, result)
    print(json.dumps({'status': result['status'], 'saved_input_cases': len(rows),
                      'synthetic_pack_cases': len(packs), 'within_group_mismatches': mismatch,
                      'elapsed_seconds': result['elapsed_seconds']}))


if __name__ == '__main__':
    main()
