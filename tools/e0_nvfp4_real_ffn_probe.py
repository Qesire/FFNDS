"""Fixed real-Wan W4A4 FFN probe, with a separate closed CPU asset producer.

M256 uses true exposed raw-teacher V1 as operator input. M31668 repeats those
rows deterministically for shape cost. Neither is an NVFP4 model rollout.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'tools') not in sys.path:
    sys.path.insert(0, str(ROOT/'tools'))
import nvfp4_real_ffn_probe as core
import nvfp4_real_ffn_assets as assets
from raw_bf16_assigned_gpu_monitor_v4 import startup

ID = 'E0_NVFP4_REAL_WAN_FFN_PACKED_PROBE_20261029'
RESOURCES = {'GPU': 1, 'CPU_threads': 4, 'GPU_reserved_max_bytes': 24 << 30,
    'wall_seconds': 600, 'outer_seconds': 900, 'new_HF_bytes': 512 << 20,
    'single_file_RLIMIT_FSIZE_bytes': 128 << 20, 'session_max_GPUs': 2, 'automatic_retry': False}
WEIGHT_BYTES = 2*(2*(core.D*core.H//2+core.D*core.H//16)+(core.D+core.H)*2+16)
LIMITS = {**core.COUNTS, 'checkpoint_tensor_ranges': 20, 'checkpoint_tensor_bytes': WEIGHT_BYTES,
    'checkpoint_H2D_tensors': 20, 'checkpoint_H2D_bytes': WEIGHT_BYTES,
    'real_V1_H2D_tensors': 2, 'real_V1_H2D_bytes': 2*256*core.D*2,
    'fullM_native_output_finiteness_checks': 2, 'saved_M256_capsules': 2}
DESIGN = {'experts': ['high', 'low'], 'block': 0, 'native_source_ordinal': 894,
    'raw_teacher_steps': [0, 2], 'rows': 256, 'M_cost': [256, 31668],
    'fullM_input': 'ROW_i_IS_TRUE_V1_ROW_i_MOD_256_REPEATED_OPERATOR_COST_NOT_FULL_STATE',
    'checkpoint': 'PUBLIC_NVFP4_QAT_DISTILLED_IDENTITY_NOT_BF16_ORIGINAL_OR_HIF4',
    'pipeline': 'DYNAMIC_Q1_PACKED_W1_BIAS_BF16_TANH_GELU_DYNAMIC_Q2_PACKED_W2_BIAS_BF16',
    'Q2_producer': 'THIS_CALL_NATIVE_W1_AND_GELU_ONLY_NO_TEACHER_OR_DENSE_H',
    'W': 'STATIC_CHECKPOINT_PACKED_U8_E2M1_AND_SWIZZLED_E4M3_SCALE',
    'A': 'EACH_REAL_CALL_DYNAMIC_GROUP16_PACKED_E2M1_AND_SWIZZLED_E4M3_SCALE',
    'global_scales': 'CHECKPOINT_F32_INPUT_GLOBAL_SCALE_AND_ALPHA_NO_REFIT',
    'reference': 'M256_ACTUAL_PACKED_A_SA_W_SW_FP32_GEMM_ALPHA_BIAS',
    'Dense': 'SAME_CHECKPOINT_DECODED_BF16_W_EQUALS_CODE_TIMES_SF_TIMES_ALPHA_TIMES_INPUT_GLOBAL_SCALE',
    'timing': {'warmup': 2, 'repeats': 5, 'modes': ['Q1', 'Q2', 'packed_W1', 'packed_W2', 'NVFP4_FFN', 'decoded_BF16_FFN'],
        'ledger_IO': 'BEFORE_AND_AFTER_CUDA_EVENT_WINDOW'},
    'kernel_reference_BF16_relative_SSE_max': 1e-4,
    'quality_acceptance': 'PREPARATORY_OPERATOR_PROBE_NOT_FULL160_MULTISOURCE_VIDEO_OR_SYS_ACCEPTANCE',
    'automatic_retry': False}
REQUIRED = {'tools/e0_nvfp4_real_ffn_probe.py', 'tools/nvfp4_real_ffn_probe.py',
    'tools/nvfp4_real_ffn_assets.py', 'tools/nvfp4_checkpoint_byte_producer.py',
    'tools/worker_identity.py', 'tools/raw_bf16_assigned_gpu_monitor_v4.py',
    'tools/startup_gpu_reservation.py', 'tools/assigned_gpu_resources.py',
    'tests/test_nvfp4_real_ffn_probe.py', 'tests/test_e0_nvfp4_real_ffn_probe.py'}


class Output:
    """Owned explicit files only; no recursive HF directory scans."""
    def __init__(self, out, limit):
        self.out, self.limit, self.files = Path(out), limit, {}
        self.journal = (self.out/'OPERATIONS.jsonl').open('xb', buffering=0)
        self.journal_bytes, self.journal_hash = 0, hashlib.sha256()
    def used(self):
        return sum(v['bytes'] for v in self.files.values())+self.journal_bytes
    def check(self, extra=0):
        assets.need(self.used()+extra <= self.limit, 'Actual owned output budget exceeded')
    def event(self, value):
        data = (json.dumps(value, separators=(',', ':'))+'\n').encode(); self.check(len(data))
        count = self.journal.write(data); self.journal_bytes += count; self.journal_hash.update(data[:count])
        assets.need(count == len(data), 'Short operation journal write')
    def json(self, name, value):
        raw = (json.dumps(value, indent=2, allow_nan=False)+'\n').encode(); self.check(len(raw))
        path = self.out/name; tmp = path.with_suffix(path.suffix+'.tmp')
        try:
            with tmp.open('xb') as stream:
                assets.need(stream.write(raw) == len(raw), 'Short evidence JSON write')
            os.replace(tmp, path)
        except BaseException:
            if tmp.exists(): self.observe(tmp)
            raise
        self.files[path] = {'bytes': len(raw), 'sha256': assets.sha(raw)}
    def observe(self, path):
        path = Path(path); raw = path.read_bytes(); self.files[path] = {'bytes': len(raw), 'sha256': assets.sha(raw)}; self.check()
    def binding(self, path, hf):
        return {'path': str(Path(path).relative_to(hf)), **self.files[Path(path)]}
    def close(self):
        self.journal.close()


class Ledger:
    """Predebit compound work outside timing; actual calls count in memory."""
    def __init__(self, report, event):
        self.report, self.event, self.active = report, event, None
        report.update(predebited={}, invoked={}, returned={})
    def invoke(self, key, fn, *args, **kwargs):
        return self.units(key, 1, fn, *args, **kwargs)
    def units(self, key, count, fn, *args, **kwargs):
        assets.need(self.active is not None and type(count) is int and count > 0
            and self.active.get(key, 0) >= count, 'Actual operation was not predebited: '+key)
        self.active[key] -= count
        self.report['invoked'][key] = self.report['invoked'].get(key, 0)+count
        result = fn(*args, **kwargs)
        self.report['returned'][key] = self.report['returned'].get(key, 0)+count
        return result
    def work(self, costs, fn):
        assets.need(self.active is None and costs and all(k in LIMITS and type(n) is int and n > 0
            and self.report['predebited'].get(k, 0)+n <= LIMITS[k] for k, n in costs.items()), 'Registered probe budget exceeded')
        for k, n in costs.items(): self.report['predebited'][k] = self.report['predebited'].get(k, 0)+n
        self.event({'kind': 'reserved_before_call', 'costs': costs}); self.active = dict(costs)
        try:
            result = fn(); assets.need(not any(self.active.values()), 'Compound operation count differs')
        finally:
            pending = self.active; self.active = None
            self.event({'kind': 'actual_call_snapshot', 'pending_reserved_calls': pending,
                'invoked': dict(self.report['invoked']), 'returned': dict(self.report['returned'])})
        return result
    def one(self, key, fn, *args, **kwargs):
        return self.work({key: 1}, lambda: self.invoke(key, fn, *args, **kwargs))
    def finish(self):
        assets.need(all(self.report[side].get(key, 0) == n for side in ('predebited', 'invoked', 'returned') for key, n in LIMITS.items()), 'Probe exact operation ledger incomplete')


class API:
    def __init__(self, torch, gemm, ledger): self.torch, self.module, self.ledger = torch, gemm, ledger
    def quant(self, x, g): return self.ledger.invoke('dynamic_NVFP4_quant', self.module.scaled_nvfp4_quant, x, g)
    def gemm(self, a, w, sa, sw, alpha, bias): return self.ledger.invoke('packed_NVFP4_GEMM', self.module.cutlass_scaled_nvfp4_mm, a, w, sa, sw, alpha, bias)
    def gelu(self, x): return self.ledger.invoke('tanh_GELU', self.torch.nn.functional.gelu, x, approximate='tanh')
    def dense(self, x, w, bias): return self.ledger.invoke('decoded_BF16_GEMM', self.torch.addmm, bias, x, w.T)


def validate(plan, frozen=False):
    cpu = plan['id'] == assets.CPU_ID
    assets.need(cpu or plan['id'] == ID, 'Unknown probe protocol identity')
    assets.need(plan['resources'] == (assets.CPU_RESOURCES if cpu else RESOURCES)
        and plan['design'] == DESIGN and plan.get('automatic_retry') is False, 'Fixed probe design/resources differ')
    assets.need(REQUIRED <= {x['path'] for x in plan['repo_sources']}, 'Incomplete new source closure')
    relative = Path(plan['output_HF_relative'])
    assets.need(not relative.is_absolute() and '..' not in relative.parts and relative.parts[0] == 'ffnds-derived', 'Output escaped HF')
    if not cpu: assets.need(plan['limits'] == LIMITS, 'Fixed native operation budget differs')
    if frozen:
        assets.need(plan.get('status') == 'FROZEN_REGISTERED' and plan.get('assets_finalized') is True
            and plan.get('private_worker_config_sha256') not in (None, 'NOT_BOUND'), 'Final fixed assets/worker not bound')
        assets.need(isinstance(plan.get('software_export' if cpu else 'CPU_assets'), dict), 'Independent closed prerequisite NOT_BOUND')


def authorize(plan, digest):
    validate(plan, True); active = assets.read(ROOT/'research/ACTIVE_RESEARCH.json')
    scopes = [s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id') == plan['id']]
    policy = active['resource_scheduling_policy_20261028']
    assets.need(active['user_continuation_authorization_20261027'].get('new_scope_registration_allowed') is True
        and policy.get('mode') == 'DYNAMIC_MAX_ACTUALLY_CALLABLE' and policy.get('global_gpu_card_cap') is None
        and len(scopes) == 1 and scopes[0].get('protocol_sha256') == digest
        and scopes[0].get('execution_status') in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE')
        and scopes[0].get('resources') == plan['resources'], 'Exact probe ACTIVE authority missing')
    if plan['id'] == ID:
        assets.need(scopes[0].get('limits') == LIMITS and scopes[0].get('max_gpu_cards') == 1, 'Native budget/card assignment differs')
    for item in plan['repo_sources']: assets.bound(ROOT, item)
    assets.need(assets.private_worker_sha() == plan['private_worker_config_sha256'], 'Wrong private execution worker')
    return scopes[0]


def require_CPU(plan, hf):
    parent, run = assets.closed(plan['CPU_assets'], hf, assets.CPU_ID, 'COMPLETE_FRESH_NVFP4_CHECKPOINT_SHA_STAT_ISOLATED_PACKAGE_TRUE_V1')
    assets.need(parent['design'] == DESIGN and parent['resources'] == assets.CPU_RESOURCES
        and parent['private_worker_config_sha256'] == run['private_worker_config_sha256'] == plan['private_worker_config_sha256']
        and run['predebited_full_SHA_bytes'] == run['actual_full_SHA_read_bytes'] == assets.CPU_RESOURCES['checkpoint_full_SHA_bytes']
        and run['model_calls'] == run['GPU_calls'] == 0 and run['Torch_or_NumPy_imported'] is False, 'Fresh CPU asset proof differs')
    assets.need(list(sys.version_info[:3]) == run['Python_version']
        and {k: importlib.metadata.version(k) for k in run['runtime_versions']} == run['runtime_versions'], 'Prepared ABI/runtime versions changed')
    for item in run['models']:
        assets.need(assets.stat_record(hf/item['cache_path']) == item['stat'], 'Full-SHA-proved NVFP4 checkpoint stat changed')
    for item in run['software_files']:
        path = Path(item['path']); assets.need(assets.stat_record(path) == item['stat'] and assets.sha(path.read_bytes()) == item['sha256'], 'Isolated exact software changed')
    assets.bound(hf, run['inputs']['binding'])
    return run


def load_layer(torch, hf, model, number, ledger, records):
    layer = {}; path = hf/model['cache_path']; rows = model['header']['ffn_tensors']
    types = {'U8': torch.uint8, 'F8_E4M3': torch.float8_e4m3fn, 'BF16': torch.bfloat16, 'F32': torch.float32}
    n, k = (core.H, core.D) if number == 0 else (core.D, core.H)
    shapes = {'weight': [n, k//2], 'weight_scale': [n, k//16], 'bias': [n]}
    for name in ('weight', 'weight_scale', 'bias', 'alpha', 'input_global_scale'):
        key = f'blocks.0.ffn.{number}.{name}'; info = rows[key]; a, b = info['data_offsets']
        expected_dtype = {'weight': 'U8', 'weight_scale': 'F8_E4M3', 'bias': 'BF16'}.get(name, 'F32')
        assets.need(info['dtype'] == expected_dtype and (info['shape'] == shapes[name] if name in shapes else math.prod(info['shape']) == 1), 'Actual NVFP4 linear header differs')
        def read_tensor():
            assets.need(assets.stat_record(path) == model['stat'], 'Checkpoint changed before range read')
            with path.open('rb') as stream:
                stream.seek(8+model['header']['header_bytes']+a); raw = stream.read(b-a)
            assets.need(len(raw) == b-a and assets.stat_record(path) == model['stat'], 'Checkpoint range read truncated/changed')
            cpu = torch.frombuffer(bytearray(raw), dtype=types[info['dtype']]).reshape(info['shape']).clone()
            assets.need(assets.sha(cpu.reshape(-1).view(torch.uint8).numpy().tobytes()) == assets.sha(raw), 'Loaded tensor bytes differ')
            records.append({'key': key, 'shape': info['shape'], 'dtype': info['dtype'], 'bytes': len(raw),
                'raw_tensor_sha256': assets.sha(raw), 'master_sha256': model['sha256'], 'master_stat': model['stat']})
            return cpu
        # Transfer-byte keys use explicit units; range operation reserves the entire fixed byte amount first.
        costs = {'checkpoint_tensor_ranges': 1, 'checkpoint_tensor_bytes': b-a,
            'checkpoint_H2D_tensors': 1, 'checkpoint_H2D_bytes': b-a}
        def operation():
            cpu = ledger.units('checkpoint_tensor_bytes', b-a,
                lambda: ledger.invoke('checkpoint_tensor_ranges', read_tensor))
            return ledger.units('checkpoint_H2D_bytes', b-a,
                lambda: ledger.invoke('checkpoint_H2D_tensors', cpu.to, 'cuda'))
        layer[name] = ledger.work(costs, operation)
    for name in ('alpha', 'input_global_scale'):
        layer[name] = layer[name].reshape(1)
        assets.need(bool(torch.isfinite(layer[name]).all()) and bool((layer[name] > 0).all()), 'Invalid checkpoint global scale/alpha')
    unscaled = ledger.one('static_weight_decodes', core.decode_unscaled, layer['weight'], layer['weight_scale'])
    layer['decoded_unscaled_FP32'] = unscaled
    layer['decoded_FP32'] = unscaled*(layer['alpha']*layer['input_global_scale'])
    layer['decoded_BF16'] = layer['decoded_FP32'].to(torch.bfloat16).contiguous()
    return layer


def time_mode(torch, name, costs, fn, ledger, check):
    for _ in range(core.WARMUP): check(); ledger.work(costs, fn)
    torch.cuda.synchronize(); start_wall = time.monotonic(); samples = []
    for _ in range(core.REPEATS):
        check(); begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        def measured():
            begin.record(); result = fn(); end.record(); end.synchronize()
            return result, begin.elapsed_time(end)
        result, elapsed = ledger.work({**costs, 'CUDA_timed_samples': 1},
            lambda: ledger.invoke('CUDA_timed_samples', measured))
        samples.append(elapsed); del result
    return {'mode': name, **core.timing_summary(samples, time.monotonic()-start_wall)}


def numeric(torch, x, original_rows, native, dense, layers, api, ledger):
    cases, refs, extras = [], {}, {}
    for index, (value, codes, scales, got) in enumerate(((x, native['A1'], native['S1'], native['pre_GELU']),
            (native['h'], native['A2'], native['S2'], native['Y']))):
        layer = layers[index]
        decoded_a = ledger.one('M256_activation_decodes', core.decode_unscaled, codes, scales)
        reference = ledger.one('FP32_reference_GEMM', torch.addmm, layer['bias'].float(), decoded_a,
            layer['decoded_unscaled_FP32'].T, alpha=float(layer['alpha']))
        unquantized = ledger.one('FP32_reference_GEMM', torch.addmm, layer['bias'].float(), value.float(), layer['decoded_FP32'].T)
        same_dense = dense['pre_GELU'] if index == 0 else ledger.work({'decoded_BF16_GEMM': 1}, lambda: api.dense(value, layer['decoded_BF16'], layer['bias']))
        def compare():
            result = {'linear': index, 'packed_kernel_vs_same_packed_FP32': core.error_metrics(got, reference),
                'packed_kernel_vs_same_packed_reference_cast_BF16': core.error_metrics(got, reference.to(torch.bfloat16)),
                'A_quantization_same_W_FP32': core.error_metrics(reference, unquantized),
                'BF16_decoded_weight_and_GEMM_vs_FP32': core.error_metrics(same_dense, unquantized)}
            assets.need(result['packed_kernel_vs_same_packed_reference_cast_BF16']['relative_SSE'] <= DESIGN['kernel_reference_BF16_relative_SSE_max'], 'Actual packed kernel/reference threshold failed')
            return result
        cases.append(ledger.one('numerical_linear_cases', compare))
        refs[f'W{index+1}_samepacked_FP32'] = reference; refs[f'W{index+1}_unquantized_A_FP32'] = unquantized
        if index == 1: extras['W2_decoded_BF16_same_native_h'] = same_dense
    ffn = ledger.one('numerical_FFN_cases', core.error_metrics, native['Y'], dense['Y'])
    capsule = {'V1': x, 'rows': original_rows, 'A1': native['A1'], 'S_A1_raw_U8': native['S1'].view(torch.uint8),
        'A2': native['A2'], 'S_A2_raw_U8': native['S2'].view(torch.uint8), 'W1_output_native': native['pre_GELU'],
        'h_native': native['h'], 'Y_native': native['Y'], 'W1_output_decoded_BF16': dense['pre_GELU'],
        'Y_decoded_BF16': dense['Y'], **refs, **extras}
    return {'linears': cases, 'FFN_native_vs_same_checkpoint_decoded_BF16': ffn}, capsule


def run_gpu(plan, hf, out, output, report, save, check):
    cpu = require_CPU(plan, hf)
    assets.need(not {'torch', 'numpy'} & set(sys.modules) and len(os.sched_getaffinity(0)) <= 4,
        'Fresh numerical worker/four-CPU affinity required after independent CPU gate')
    assets.need(os.environ.get('NVIDIA_TF32_OVERRIDE') != '1' and os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE') != '1', 'FP32 reference precision override')
    import torch
    reservation, monitor, report['CUDA_reservation'] = startup.reserve_before_assets(torch)
    try:
        assets.need(torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (12, 0)
            and 'RTX PRO 6000' in torch.cuda.get_device_name(0), 'One assigned RTX PRO6000 SM120 required')
        torch.set_num_threads(4); torch.set_num_interop_threads(4); torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        report['matmul_policy'] = {'allow_tf32': False, 'allow_bf16_reduced_precision_reduction': False,
            'reference': 'FP32_ADDMM_SAME_PACKED_OPERANDS_NO_AUTOCAST'}
        package = Path(cpu['software_package'])
        assets.need('lightx2v_kernel' not in sys.modules, 'Unexpected preimported kernel fallback')
        spec = importlib.util.spec_from_file_location('lightx2v_kernel', package/'__init__.py', submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec); sys.modules['lightx2v_kernel'] = module; spec.loader.exec_module(module)
        from lightx2v_kernel import gemm
        assets.need(Path(gemm.__file__).resolve() == package/'gemm.py' and Path(module.common_ops.__file__).resolve() == package/'common_ops.abi3.so', 'Kernel import escaped exact isolated overlay')
        report['kernel_origins'] = {'package': str(package), 'gemm': gemm.__file__, 'ELF': module.common_ops.__file__}
        ledger = Ledger(report.setdefault('operations', {}), output.event); api = API(torch, gemm, ledger)
        from safetensors.torch import load_file, save_file
        inputs = load_file(str(assets.bound(hf, cpu['inputs']['binding'])), device='cpu')
        def check_gpu():
            check(); monitor.check(); assets.need(torch.cuda.max_memory_reserved() <= RESOURCES['GPU_reserved_max_bytes'], 'Probe reserved24GiB exceeded')
        report['cells'] = []
        for kind in ('high', 'low'):
            check_gpu(); model = next(item for item in cpu['models'] if item['kind'] == kind); source_records = []
            report.setdefault('checkpoint_tensors', []).append({'expert': kind, 'tensors': source_records})
            layers = [load_layer(torch, hf, model, number, ledger, source_records) for number in (0, 2)]
            # The byte keys describe transfers, not CUDA kernels.
            size = 256*core.D*2
            def upload():
                return ledger.units('real_V1_H2D_bytes', size,
                    lambda: ledger.invoke('real_V1_H2D_tensors', inputs[kind+'_V1'].to, 'cuda'))
            x = ledger.work({'real_V1_H2D_tensors': 1, 'real_V1_H2D_bytes': size}, upload)
            rows = inputs[kind+'_rows']
            for m in core.M_VALUES:
                check_gpu(); value = x if m == 256 else ledger.one('repeat_real_rows_to_fullM', lambda: x.index_select(0, torch.arange(m, device='cuda') % 256))
                native = ledger.work({'dynamic_NVFP4_quant': 2, 'packed_NVFP4_GEMM': 2, 'tanh_GELU': 1}, lambda: core.packed_ffn(value, layers, api, capture=True))
                cell = {'expert': kind, 'M': m, 'input_identity': 'TRUE_256_SAVED_V1_ROWS' if m == 256 else DESIGN['fullM_input'], 'timings': []}
                if m == 256:
                    dense = ledger.work({'decoded_BF16_GEMM': 2, 'tanh_GELU': 1}, lambda: core.decoded_ffn(value, layers, api, capture=True))
                    cell['numerical'], capsule = numeric(torch, value, rows, native, dense, layers, api, ledger)
                    path = out/(kind+'_M256.safetensors')
                    def store():
                        cpu_values = {k: v.detach().cpu().contiguous() for k, v in capsule.items()}
                        output.check(sum(v.numel()*v.element_size() for v in cpu_values.values())+(1 << 20))
                        try: save_file(cpu_values, str(path))
                        finally:
                            if path.exists(): output.observe(path)
                    ledger.one('saved_M256_capsules', store); cell['capsule'] = output.binding(path, hf)
                    del dense, capsule
                else:
                    ledger.one('fullM_native_output_finiteness_checks', lambda: assets.need(bool(torch.isfinite(native['Y']).all()), 'Nonfinite fullM native output'))
                jobs = [('Q1', {'dynamic_NVFP4_quant': 1}, lambda: api.quant(value, layers[0]['input_global_scale'])),
                    ('Q2', {'dynamic_NVFP4_quant': 1}, lambda: api.quant(native['h'], layers[1]['input_global_scale'])),
                    ('packed_W1', {'packed_NVFP4_GEMM': 1}, lambda: api.gemm(native['A1'], layers[0]['weight'], native['S1'], layers[0]['weight_scale'], layers[0]['alpha'], layers[0]['bias'])),
                    ('packed_W2', {'packed_NVFP4_GEMM': 1}, lambda: api.gemm(native['A2'], layers[1]['weight'], native['S2'], layers[1]['weight_scale'], layers[1]['alpha'], layers[1]['bias'])),
                    ('NVFP4_FFN', {'dynamic_NVFP4_quant': 2, 'packed_NVFP4_GEMM': 2, 'tanh_GELU': 1}, lambda: core.packed_ffn(value, layers, api)),
                    ('decoded_BF16_FFN', {'decoded_BF16_GEMM': 2, 'tanh_GELU': 1}, lambda: core.decoded_ffn(value, layers, api))]
                for name, costs, fn in jobs:
                    cell['timings'].append(time_mode(torch, name, costs, fn, ledger, check_gpu))
                report['cells'].append(cell); save(report); del native, value
            del layers, x; torch.cuda.empty_cache()
        check_gpu(); ledger.finish(); report.update(status='COMPLETE_REAL_OPERAND_NATIVE_W4A4_FFN_PREPARATORY_PROBE',
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            CPU_assets=plan['CPU_assets'], input_provenance=cpu['inputs']['provenance'])
    finally:
        monitor.close(); report['assigned_GPU_resources'] = monitor.report()
        if report['assigned_GPU_resources']['violations']:
            report['status'] = 'FAILED_PRESERVED_NO_AUTOMATIC_RETRY'; report['resource_error'] = 'Assigned GPU monitor violation'
            if sys.exc_info()[0] is None: raise RuntimeError(report['resource_error'])


def drafts():
    source = assets.read(ROOT/'research/20261029/NVFP4_CHECKPOINT_BYTES_RESULTS_20261029.json')
    corpus = assets.read(ROOT/'research/20261029/ORIGINAL_BF16_FULL_DEPTH_CORPUS_RESULTS_20261029.json')
    convert = lambda value: {key: copy.deepcopy(value[key]) for key in ('protocol', 'RUN', 'exit')}
    shared = {'design': DESIGN, 'automatic_retry': False, 'assets_finalized': False,
        'private_worker_config_sha256': 'fa8a9da592292ec1560135e40ae1a6e0222a23c08af023e8e87f310580c7070d',
        'repo_sources': [assets.binding(ROOT/p, ROOT) for p in sorted(REQUIRED)]}
    cpu = {**copy.deepcopy(shared), 'id': assets.CPU_ID, 'status': 'DRAFT_ARCHIVE_AND_EXECUTION_WORKER_NOT_BOUND',
        'resources': assets.CPU_RESOURCES, 'checkpoint_download': convert(source), 'corpus_audit': {**convert(corpus), 'manifest': corpus['manifest']},
        'software_export': None, 'local_HF_root': '/var/tmp/ffnds-xiang/huggingface',
        'software_cache_relative': 'software/lightx2v_kernel_nvfp4_probe_20261029_v1',
        'output_HF_relative': 'ffnds-derived/e0_nvfp4_real_ffn_inputs_cpu_20261029/attempt1/worker/results'}
    gpu = {**copy.deepcopy(shared), 'id': ID, 'status': 'DRAFT_ACTUAL_CPU_ASSETS_EXIT0_NOT_BOUND', 'resources': RESOURCES,
        'limits': LIMITS, 'CPU_assets': None,
        'output_HF_relative': 'ffnds-derived/e0_nvfp4_real_ffn_packed_probe_20261029/attempt1/worker/results'}
    proof_path = ROOT/'.runtime/sm120_overlay_export_20261029/PROOFS.json'
    if proof_path.exists():
        proof = assets.read(proof_path)
        cpu['software_export'] = {key: copy.deepcopy(proof[key]) for key in ('protocol', 'RUN', 'exit')}
        cpu['status'] = 'DRAFT_ALL_INPUT_LINEAGE_BOUND_REVIEW_AND_FREEZE_PENDING'
    return cpu, gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('command', choices=('draft', 'inspect', 'prepare', 'run'))
    parser.add_argument('--protocol', type=Path, required=True); parser.add_argument('--hf-root', type=Path, required=True)
    parser.add_argument('--cpu-protocol', type=Path)
    args = parser.parse_args(); hf = args.hf_root.resolve()
    if args.command == 'draft':
        assets.need(args.cpu_protocol is not None and not args.protocol.exists() and not args.cpu_protocol.exists(), 'Fresh GPU and CPU draft paths required')
        cpu, gpu = drafts(); args.cpu_protocol.write_text(json.dumps(cpu, indent=2)+'\n'); args.protocol.write_text(json.dumps(gpu, indent=2)+'\n'); return
    plan = assets.read(args.protocol); validate(plan, args.command != 'inspect')
    if args.command == 'inspect': print(json.dumps({'id':plan['id'], 'resources':plan['resources'], 'status':plan['status']})); return
    assets.need((args.command == 'prepare') == (plan['id'] == assets.CPU_ID), 'CPU/native command identity mismatch')
    digest = assets.sha(args.protocol.read_bytes()); authorize(plan, digest)
    out = (hf/plan['output_HF_relative']).resolve(); assets.need(out.is_relative_to(hf) and not out.exists(), 'Fresh confined output required'); out.mkdir(parents=True)
    output = Output(out, plan['resources']['new_HF_bytes']); started = time.monotonic()
    report = {'id': plan['id'], 'protocol_sha256': digest, 'status': 'STARTED_NO_AUTOMATIC_RETRY',
        'automatic_retry': False, 'design': DESIGN, 'quality_or_full_model_acceptance': False}
    save = lambda value: output.json('RUN.json', value)
    def check():
        assets.need(time.monotonic()-started <= plan['resources']['wall_seconds'], 'Fixed probe wall exhausted'); output.check()
    def timeout(*_): raise TimeoutError('Fixed probe wall exhausted; no automatic retry')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(plan['resources']['wall_seconds']); save(report)
    try:
        if plan['id'] == assets.CPU_ID:
            def store_input(path, raw):
                output.check(len(raw))
                try:
                    with path.open('xb') as stream:
                        assets.need(stream.write(raw) == len(raw), 'Short CPU input capsule write')
                finally:
                    if path.exists(): output.observe(path)
            value = assets.prepare(plan, hf, out, lambda current: save({**current, 'protocol_sha256':digest}),
                check, report=report, store_input=store_input)
            report.update(value)
        else:
            soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE); cap = RESOURCES['single_file_RLIMIT_FSIZE_bytes']
            assets.need(hard == resource.RLIM_INFINITY or hard >= cap, 'Single-file hard limit too small')
            resource.setrlimit(resource.RLIMIT_FSIZE, (cap, cap))
            run_gpu(plan, hf, out, output, report, save, check)
    except BaseException as error:
        report.update(status='FAILED_PRESERVED_NO_AUTOMATIC_RETRY', error_type=type(error).__name__, error=str(error)); raise
    finally:
        signal.alarm(0); report['wall_seconds'] = time.monotonic()-started
        report['operation_journal'] = {'path': str((out/'OPERATIONS.jsonl').relative_to(hf)), 'bytes': output.journal_bytes, 'sha256': output.journal_hash.hexdigest()}
        report['actual_tracked_output_bytes_before_final_RUN'] = output.used(); save(report); output.close()
    print(json.dumps({'id':plan['id'], 'status':report['status']}))


if __name__ == '__main__':
    main()
