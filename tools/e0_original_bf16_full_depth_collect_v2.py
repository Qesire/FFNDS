"""Versioned raw-BF16 collection repair: incremental output accounting.

The frozen v1 capture/FFN/runner functions are reused without source edits.
Only output I/O bookkeeping, per-packet monitor checks and the independently
proved execution-worker identity differ. No recursive output traversal occurs.
"""
import argparse
from contextlib import contextmanager
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import stat
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'tools') not in sys.path:
    sys.path.insert(0, str(ROOT/'tools'))


def private_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy = private_module('_ffnds_full_depth_v2_parent', 'tools/e0_original_bf16_full_depth_collect.py')
native = private_module('_ffnds_full_depth_v2_native', 'tools/e0_original_bf16_videos.py')
legacy.native = native
import raw_bf16_worker_mirror_proof as mirror

V1_SOURCE_SHA = '3079c7ce5cc40747dbedcb64419026d708973b1e723e558bf292ebaa7d8e4dcf'
REPAIR = {
    'version': 2,
    'capture_and_model_operations': 'IDENTICAL_FROZEN_V1',
    'output_accounting': 'ACTUAL_WRITTEN_BYTES_INCREMENTAL_NO_RECURSIVE_WALK',
    'packet_monitor': 'CHECK_EXISTING_THREAD_RESULT_NO_NEW_GPU_QUERY',
    'execution_worker': 'FRESH_INDEPENDENT_RAW_FULL_SHA_PROOF_THEN_STAT_REUSE',
    'maximum_single_video_file_bytes': 128 << 20,
    'process_single_file_hard_bound': 'RLIMIT_FSIZE_SOFT_128MiB_INHERITED_BY_OPAQUE_WRITER',
    'safetensors_header_and_small_metadata_reserve_bytes': 1 << 20,
    'automatic_retry': False,
}
RESOURCES = copy.deepcopy(legacy.RESOURCES)
limits = legacy.limits
run_id = legacy.run_id
drawings = legacy.drawings
COHORT = legacy.COHORT
EXECUTION_PROOF = '.runtime/original_bf16_new_worker_20261029/EXECUTION_RAW_MASTER_CPU_PROOF.json'
EXTRA_SOURCES = {
    'tools/e0_original_bf16_full_depth_collect_v2.py',
    'tests/test_e0_original_bf16_full_depth_collect_v2.py',
    'tools/raw_bf16_worker_mirror_proof.py',
    'tests/test_raw_bf16_worker_mirror_proof.py',
}
legacy.REQUIRED = legacy.REQUIRED | EXTRA_SOURCES


def scope_id(part):
    native.need(type(part) is int and part in (0, 1), 'Exactly two independent parts')
    return legacy.BASE_ID + '_PART' + str(part) + '_V2'


legacy.scope_id = scope_id


def output_relative(part):
    return f'ffnds-derived/e0_original_bf16_full_depth_collect_v2_20261028/part{part}/attempt1/worker/results'


def hex_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate(plan, frozen=False):
    legacy.validate(plan, frozen)
    native.need(native.sha(ROOT/'tools/e0_original_bf16_full_depth_collect.py') == V1_SOURCE_SHA
        and plan.get('repair') == REPAIR and plan['output_HF_relative'] == output_relative(plan['part']),
        'Fixed v2 repair/output identity differs')
    native.need(plan.get('execution_private_worker_config_sha256') is None or hex_sha(plan['execution_private_worker_config_sha256']),
        'Execution worker SHA format differs')
    native.need(isinstance(plan.get('failed_predecessor'), dict), 'Failed v1 predecessor missing')
    native.need(all(item['bytes'] <= REPAIR['maximum_single_video_file_bytes'] for item in plan['repo_sources'])
        and legacy.PACKET_BYTES + (1 << 20) < REPAIR['maximum_single_video_file_bytes'],
        'Registered snapshot/payload exceeds fixed process single-file bound')
    if frozen:
        native.need(hex_sha(plan.get('execution_private_worker_config_sha256'))
            and isinstance(plan.get('execution_raw_master_CPU_proof'), dict)
            and {'protocol', 'RUN', 'launch_exit'} <= set(plan['execution_raw_master_CPU_proof']),
            'Current execution worker raw proof NOT_BOUND')


def verify_predecessor(plan, hf):
    old = plan['failed_predecessor']
    protocol = native.read(native.bound(ROOT, old['protocol']))
    run = native.read(native.bound(hf, old['RUN']))
    exit_record = native.read(native.bound(hf, old['launch_exit']))
    old_id = legacy.BASE_ID + '_PART' + str(plan['part'])
    native.need(protocol.get('id') == old_id and run.get('id') == old_id
        and run.get('protocol_sha256') == old['protocol']['sha256']
        and run.get('status') == 'FAILED_FULL_DEPTH_PART_PRESERVED_NO_RETRY'
        and exit_record.get('id') == old_id and exit_record.get('protocol_sha256') == old['protocol']['sha256']
        and exit_record.get('run_sha256') == old['RUN']['sha256'] and exit_record.get('process_exited') is True
        and type(exit_record.get('child_returncode')) is int and exit_record['child_returncode'] == -2
        and exit_record.get('automatic_retry') is False, 'Actual failed v1 RUN/exit identity differs')
    native.need(protocol['cases'] == plan['cases'] and protocol['limits'] == limits()
        and protocol['resources'] == RESOURCES and protocol['output_HF_relative'] != plan['output_HF_relative'],
        'V2 sampling/model operation scope differs from failed v1')
    return old


def execution_worker_sha():
    from worker_identity import private_config_path, require_worker
    require_worker()
    return native.sha(private_config_path())


def require_execution_master(plan, hf):
    """One independent 69GB proof is shared; each consumer rechecks four stats."""
    proof = plan.get('execution_raw_master_CPU_proof')
    native.need(isinstance(proof, dict) and {'protocol', 'RUN', 'launch_exit'} <= set(proof),
        'Current execution raw proof NOT_BOUND')
    protocol = native.read(native.bound(ROOT, proof['protocol']))
    mirror.validate(protocol)
    native.need(protocol.get('status') == 'FROZEN_REGISTERED'
        and protocol['private_worker_config_sha256'] == plan['execution_private_worker_config_sha256']
        and protocol['models'] == plan['models'], 'Execution raw proof model/worker protocol differs')
    run = native.read(native.bound(hf, proof['RUN']))
    legacy.exited(native.read(native.bound(hf, proof['launch_exit'])), proof['RUN'], proof['protocol']['sha256'], mirror.ID)
    expected = {
        'id': mirror.ID, 'protocol_sha256': proof['protocol']['sha256'],
        'status': 'PASS_FRESH_WORKER_RAW_SHA_STAT_HEADER_NO_MODEL_NO_NUMPY_NO_TORCH',
        'parent_protocol_sha256': legacy.PARENT_SHA, 'models_sha256': native.canonical(plan['models']),
        'private_worker_config_sha256': plan['execution_private_worker_config_sha256'],
        'predebited_full_payload_bytes': mirror.TOTAL, 'actual_full_payload_read_bytes': mirror.TOTAL,
        'all_CUDA_devices_hidden': True, 'NumPy_or_Torch_imported': False, 'model_calls': 0,
        'GPU_calls': 0, 'automatic_retry': False,
    }
    native.need(all(run.get(k) == v for k, v in expected.items()) and 0 <= run.get('wall_seconds', -1) <= 300,
        'Independent current worker proof identity/count differs')
    native.need(execution_worker_sha() == plan['execution_private_worker_config_sha256'], 'Actual execution worker configuration differs')
    rows = run['model_payloads']
    native.need(len(rows) == 4 and {r['path'] for r in rows} == {r['path'] for r in plan['models']}, 'Four current raw masters required')
    for item in plan['models']:
        row = next(r for r in rows if r['path'] == item['path'])
        native.need(row.get('kind') == item['kind'] and row.get('sha256') == item['sha256']
            and row.get('status') == 'PASS_FRESH_FULL_SHA'
            and row.get('predebited_bytes') == row.get('actual_read_bytes') == item['bytes']
            and native.stat_record(item['path']) == row.get('stat'), 'Execution master changed after fresh full SHA proof')
        if item['kind'] in ('high', 'low'):
            native.need(run['native_headers'][item['kind']]['header_sha256'] == item['header_sha256'], 'Execution raw header differs')
    return proof


def verify_lineage(plan, hf):
    result = legacy.verify_lineage(plan, hf, stat_check=False)
    verify_predecessor(plan, hf)
    require_execution_master(plan, hf)
    return result


def cpu_identity(plan, digest):
    result = legacy.cpu_identity(plan, digest)
    result.update(status='PASS_V2_INCREMENTAL_OUTPUT_REUSED_CURRENT_WORKER_RAW_STAT_NO_NUMERIC_IMPORTS',
        execution_raw_master_CPU_proof=plan['execution_raw_master_CPU_proof'],
        execution_private_worker_config_sha256=plan['execution_private_worker_config_sha256'],
        failed_predecessor=plan['failed_predecessor'], repair=REPAIR)
    return result


def preflight(args):
    start = time.monotonic()
    native.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not {'torch', 'numpy'} & set(sys.modules), 'Fresh hidden-CUDA stdlib proof required')
    plan = native.read(args.protocol); validate(plan, True)
    native.need(len(os.sched_getaffinity(0)) <= 4, 'Preflight affinity exceeds4')
    verify_lineage(plan, args.hf_root)
    for item in plan['repo_sources']:
        native.bound(ROOT, item)
    for case in plan['cases']:
        native.bound(args.hf_root, case['image'])
    for item in plan['cases'][0]['controls'].values():
        native.bound(args.hf_root, item)
    native.model_metadata(plan)
    native.verify_local_package(plan, {'repo': ROOT, 'hf': args.hf_root}, args.local_root)
    result = cpu_identity(plan, native.sha(args.protocol)); result['wall_seconds'] = time.monotonic() - start
    native.need(result['wall_seconds'] <= 300 and not {'torch', 'numpy'} & set(sys.modules), 'CPU proof wall/import gate differs')
    return result


def authorize(plan, digest):
    validate(plan, True)
    return legacy.authorize(plan, digest)


def require_cpu(scope, plan, digest, hf):
    proof = scope.get('cpu_preflight')
    native.need(isinstance(proof, dict) and {'RUN', 'launch_exit'} <= set(proof), 'Independent v2 part proof NOT_BOUND')
    run = native.read(native.bound(hf, proof['RUN']))
    native.need(all(run.get(k) == v for k, v in cpu_identity(plan, digest).items()) and 0 <= run.get('wall_seconds', -1) <= 300,
        'Independent v2 byte proof differs')
    legacy.exited(native.read(native.bound(hf, proof['launch_exit'])), proof['RUN'], digest, plan['id'])
    return proof


class WrittenBytes:
    """Track exact bytes from owned writes, with one stat/hash per external file.

    This process owns a fresh output directory. Snapshot streams and JSON use
    their actual write return counts; append journals have an incremental hash.
    Safetensors and mp4 retain the original serializers. Their single completed
    file is observed once, never by traversing the output/source tree.
    """
    def __init__(self, out, limit, monitor_check=lambda: None):
        self.out = Path(os.path.abspath(out))
        self.limit, self.monitor_check = limit, monitor_check
        self.files, self.holds = {}, {}
        self.used = self.peak_accounted_with_holds = 0
        self.external_file_observations = self.packet_monitor_checks = 0
        self.atomic_json_writes = self.snapshot_files = 0
        self.accounting_errors = []

    def path(self, value):
        path = Path(os.path.abspath(value))
        native.need(path.is_relative_to(self.out) and '..' not in Path(value).parts, 'Owned output escaped fresh directory')
        return path

    def check(self, additional=0):
        size = self.used + sum(self.holds.values()) + additional
        native.need(type(additional) is int and additional >= 0 and size <= self.limit, 'Incremental part HF byte budget exceeded')
        self.peak_accounted_with_holds = max(self.peak_accounted_with_holds, size)

    def hold(self, key, amount):
        native.need(key not in self.holds and type(amount) is int and amount >= 0, 'Output hold identity/bytes differ')
        self.check(amount); self.holds[key] = amount

    def reserve_payload(self, amount):
        self.packet_monitor_checks += 1
        self.monitor_check()  # Existing background samples only; zero new nvidia-smi calls.
        self.hold('safetensors', amount + REPAIR['safetensors_header_and_small_metadata_reserve_bytes'])

    def record(self, path, size, digest, *, replace=False, hold=None):
        path = self.path(path)
        native.need(type(size) is int and size >= 0 and hex_sha(digest), 'Actual output byte/hash metadata differs')
        old = self.files.get(path)
        native.need(replace or old is None, 'Unregistered output overwrite')
        allowance = None
        if hold is not None:
            native.need(hold in self.holds, 'External output lacks pre-reserved bytes')
            allowance = self.holds.pop(hold)
        self.used += size - (old['bytes'] if old else 0)
        self.files[path] = {'bytes': size, 'sha256': digest}
        native.need(allowance is None or size <= allowance, 'External output exceeded pre-reserved bytes; actual bytes retained')
        self.check()

    def write_json(self, path, value):
        path = self.path(path)
        data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
        # Atomic replacement temporarily retains both old target and new temp.
        self.check(len(data))
        temporary = path.with_suffix(path.suffix + '.tmp')
        written = 0
        try:
            with temporary.open('xb', buffering=0) as stream:
                written = stream.write(data)
                native.need(written == len(data), 'Short atomic JSON write')
            os.replace(temporary, path)
        except BaseException:
            # A partial temp belongs to this scope and stays in the failure inventory.
            if temporary.exists():
                self.observe_external(temporary, allow_unreserved=True)
            raise
        self.record(path, written, hashlib.sha256(data).hexdigest(), replace=True)
        self.atomic_json_writes += 1

    def copy_bound(self, source, destination, item, *, hold='frozen_metadata'):
        destination = self.path(destination)
        native.need(hold in self.holds and item['bytes'] <= self.holds[hold], 'Snapshot not covered by one frozen metadata reserve')
        digest, size, created = hashlib.sha256(), 0, False
        try:
            with Path(source).open('rb') as inp, destination.open('xb', buffering=0) as out:
                created = True
                for data in iter(lambda: inp.read(1 << 20), b''):
                    native.need(size + len(data) <= item['bytes'], 'Snapshot source grew beyond frozen bytes')
                    count = out.write(data)
                    size += count; digest.update(data[:count])
                    native.need(count == len(data), 'Short source snapshot write')
        finally:
            # Return counts are actual bytes, including any partial failed copy.
            if created:
                self.holds[hold] -= size
                self.record(destination, size, digest.hexdigest())
        native.need(size == item['bytes'] and digest.hexdigest() == item['sha256'], 'Snapshot source changed after binding')
        self.snapshot_files += 1

    def observe_external(self, path, *, hold=None, allow_unreserved=False, known=None):
        path = self.path(path)
        if known is None:
            # One explicitly named file, not an output-tree scan.
            info = path.lstat()
            native.need(stat.S_ISREG(info.st_mode), 'External output must be a regular owned file')
            digest, read_bytes = hashlib.sha256(), 0
            with path.open('rb') as stream:
                for data in iter(lambda: stream.read(8 << 20), b''):
                    digest.update(data); read_bytes += len(data)
            native.need(read_bytes == info.st_size, 'External output changed while hashing')
            size, digest = read_bytes, digest.hexdigest()
        else:
            # inspect_video already read/hash/stat-checked this exact completed file.
            size, digest = known['bytes'], known['sha256']
        if hold is None and not allow_unreserved:
            native.need('safetensors' in self.holds, 'External payload was not pre-reserved')
            hold = 'safetensors'
        self.record(path, size, digest, hold=hold)
        self.external_file_observations += 1

    def binding(self, path, root):
        path, root = self.path(path), Path(os.path.abspath(root))
        native.need(path.is_relative_to(root) and path in self.files, 'Output binding lacks actual completed write')
        return {'path': str(path.relative_to(root)), **self.files[path]}

    def summary(self):
        return {'mode': REPAIR['output_accounting'], 'recursive_output_walks': 0,
            'actual_tracked_live_bytes': self.used, 'tracked_files': len(self.files),
            'peak_tracked_bytes_plus_reserved_and_atomic_temp': self.peak_accounted_with_holds,
            'unconsumed_reserved_bytes': sum(self.holds.values()), 'atomic_JSON_writes': self.atomic_json_writes,
            'source_snapshot_files': self.snapshot_files, 'external_file_observations': self.external_file_observations,
            'payload_monitor_checks': self.packet_monitor_checks,
            'unconsumed_reservations': dict(self.holds), 'accounting_errors': list(self.accounting_errors)}

    def open_journal(self, path):
        return CountedJournal(self, path)


class CountedJournal:
    def __init__(self, counter, path):
        self.counter, self.path = counter, counter.path(path)
        native.need(self.path not in counter.files, 'Journal overwrite forbidden')
        self.stream = self.path.open('xb', buffering=0); self.size = 0; self.digest = hashlib.sha256()
        counter.record(self.path, 0, self.digest.hexdigest())

    def write(self, text):
        data = text.encode('utf-8'); self.counter.check(len(data))
        count = self.stream.write(data); self.stream.flush()
        self.size += count; self.digest.update(data[:count])
        self.counter.record(self.path, self.size, self.digest.hexdigest(), replace=True)
        native.need(count == len(data), 'Short predebit journal write')
        return len(text)

    def close(self):
        self.stream.close()


@contextmanager
def counted_io(counter):
    """Contextual I/O observation; original tensor/video serializers run once."""
    import safetensors.torch as safetensors_torch
    import run_ub1_consumer_videos as video_module
    old_write, old_binding, old_reserve = native.write, native.binding, legacy.reserve_output_bytes
    old_save, old_inspect = safetensors_torch.save_file, video_module.inspect_video

    def binding(path, root):
        owned = Path(os.path.abspath(path)).is_relative_to(counter.out)
        return counter.binding(path, root) if owned else old_binding(path, root)

    def save_file(tensors, filename, *a, **kw):
        try:
            result = old_save(tensors, filename, *a, **kw)
        except BaseException:
            # Preserve a partial file/cost on failure; no retry or hidden second save.
            path = counter.path(filename)
            if path.exists():
                try:
                    counter.observe_external(path)
                except BaseException as error:
                    counter.accounting_errors.append(type(error).__name__ + ': ' + str(error))
            raise
        counter.observe_external(filename)
        return result

    def inspect_video(path, contract):
        result = old_inspect(path, contract)
        counter.observe_external(path, hold=str(counter.path(path)), known=result)
        return result

    native.write, native.binding = counter.write_json, binding
    legacy.reserve_output_bytes = lambda _out, amount: counter.reserve_payload(amount)
    safetensors_torch.save_file, video_module.inspect_video = save_file, inspect_video
    try:
        yield
    finally:
        native.write, native.binding, legacy.reserve_output_bytes = old_write, old_binding, old_reserve
        safetensors_torch.save_file, video_module.inspect_video = old_save, old_inspect


@contextmanager
def single_file_bound():
    """Bound the opaque ffmpeg writer as well as all owned regular files."""
    old_limit = resource.getrlimit(resource.RLIMIT_FSIZE)
    old_signal = signal.getsignal(signal.SIGXFSZ)
    maximum = REPAIR['maximum_single_video_file_bytes']
    native.need(old_limit[1] == resource.RLIM_INFINITY or old_limit[1] >= maximum,
        'Execution hard RLIMIT_FSIZE is below frozen128MiB')
    def exceeded(*_):
        raise OSError('Fixed128MiB per-file output limit reached; partial output preserved, no retry')
    signal.signal(signal.SIGXFSZ, exceeded)
    resource.setrlimit(resource.RLIMIT_FSIZE, (maximum, old_limit[1]))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, old_limit)
        signal.signal(signal.SIGXFSZ, old_signal)


def run(args):
    started = time.monotonic(); plan = native.read(args.protocol); digest = native.sha(args.protocol)
    scope = authorize(plan, digest); cpu = require_cpu(scope, plan, digest, args.hf_root)
    native.need(not {'torch', 'numpy'} & set(sys.modules), 'Fresh numerical worker required')
    verify_lineage(plan, args.hf_root)
    for item in plan['repo_sources']:
        native.bound(ROOT, item)
    native.model_metadata(plan)
    local, rows = native.verify_local_package(plan, {'repo': ROOT, 'hf': args.hf_root}, args.local_root)
    native.need({k: importlib.metadata.version(k) for k in plan['runtime_versions']} == plan['runtime_versions'], 'Actual package versions differ')
    native.need(execution_worker_sha() == plan['execution_private_worker_config_sha256'], 'Private execution identity differs')
    native.need(len(os.sched_getaffinity(0)) <= 4 and not any(os.environ.get(k) == '1' for k in ('NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')), 'CPU affinity/TF32 override differs')
    out = (args.hf_root/plan['output_HF_relative']).resolve()
    native.need(out.is_relative_to(args.hf_root.resolve()) and not out.exists(), 'Fresh confined v2 output required; no retry')
    out.mkdir(parents=True)
    report = {'id': plan['id'], 'protocol_sha256': digest, 'numeric_path': 'BF16_ORIGINAL', 'part': plan['part'],
        'status': 'STARTED_RAW_FULL_DEPTH_PART_V2_NO_RETRY', 'automatic_retry': False, 'CPU_preflight': cpu, 'phases': [],
        'execution_raw_master_CPU_proof': plan['execution_raw_master_CPU_proof'], 'failed_predecessor': plan['failed_predecessor'],
        'repair': REPAIR, 'source_cohort': plan['source_cohort'], 'native_part_image_clusters': 4,
        'new_calibration_image_clusters': 3, 'old_development_image_clusters': 1, 'complete_cohort_claimed': False,
        'cohort_completion': 'PART_ONLY_REQUIRES_TWO_INDEPENDENT_ACTUAL_EXIT0_RECEIPTS',
        'state_identity': 'Teacher states only; no student states captured', 'source_history': 'HISTORICALLY_EXPOSED_CONSUMER20_DEVELOPMENT',
        'official_quality': 'NOT_RUN', 'training_updates': 0, 'decompositions': 0, 'quality_or_speed_acceptance': False,
        'capture_runtime_identity': 'Native generation plus row capture/D2H/serialization/CPU checks; not a throughput measurement'}
    monitor = None
    counter = WrittenBytes(out, RESOURCES['new_HF_bytes'], lambda: monitor.check() if monitor else None)
    # Reserve all opaque native video writers before the first model call.
    for case in plan['cases'][1:]:
        counter.hold(str(counter.path(out/run_id(case)/'output.mp4')), REPAIR['maximum_single_video_file_bytes'])
    journal = counter.open_journal(out/'OPERATIONS.jsonl')
    persist = lambda: counter.write_json(out/'RUN.json', report)
    ledger = legacy.Ledger(report.setdefault('operation_ledger', {}), plan['run_ids'], lambda event: journal.write(json.dumps(event, separators=(',', ':'))+'\n'))
    persist()

    def check():
        authorize(plan, digest)
        if monitor:
            monitor.check()
        available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        native.need(available >= RESOURCES['host_available_min_bytes'] and shutil.disk_usage(out).free >= RESOURCES['free_disk_min_bytes'], 'Host/disk resource reserve exceeded')
        counter.check()
        native.need(time.monotonic()-started <= RESOURCES['wall_seconds'], 'Full-depth wall1200 exhausted')
        if 'torch' in sys.modules:
            native.need(sys.modules['torch'].cuda.max_memory_reserved() <= RESOURCES['GPU_reserved_max_bytes'], 'GPU reserved32GiB exceeded')

    def timeout(*_):
        raise TimeoutError('Fixed v2 full-depth wall1200 exhausted; no retry')

    signal.signal(signal.SIGALRM, timeout); signal.alarm(RESOURCES['wall_seconds'])
    from e0_b_final_block_capture_local_package import CheckedPackageFinder
    finder, no_hif4 = CheckedPackageFinder(local, rows), native.NoHiF4()
    file_guard = single_file_bound()
    try:
        file_guard.__enter__()
        check(); snapshot = out/'EXECUTED_SOURCES'; snapshot.mkdir()
        copies = [(ROOT/item['path'], snapshot/item['path'], item) for item in plan['repo_sources']]
        copies += [(args.protocol, snapshot/'protocol.json', native.binding(args.protocol, ROOT)),
            (ROOT/plan['source_cohort']['path'], snapshot/'source_cohort.json', plan['source_cohort'])]
        counter.hold('frozen_metadata', sum(item['bytes'] for _, _, item in copies))
        for source, destination, item in copies:
            destination.parent.mkdir(parents=True, exist_ok=True)
            counter.copy_bound(source, destination, item)
        native.need(counter.holds.pop('frozen_metadata') == 0, 'Frozen snapshot byte reserve not exhausted exactly')
        sys.dont_write_bytecode = True
        sys.path[:0] = [str(local), str(ROOT/'.runtime/code/LightX2V')]
        sys.meta_path[:0] = [no_hif4, finder]
        os.environ.update(DTYPE='BF16', SENSITIVE_LAYER_DTYPE='BF16', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        import torch
        from startup_gpu_reservation import reserve_before_assets
        reservation, monitor, report['CUDA_reservation'] = reserve_before_assets(torch)
        native.need(torch.cuda.device_count() == 1 and 'A100' in torch.cuda.get_device_name(0), 'Single assigned A100 required')
        torch.set_num_threads(4); torch.set_num_interop_threads(4); torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        report['runtime_precision'] = {'matmul_TF32': False, 'float32_matmul_precision': 'highest', 'bf16_reduced_precision_reduction': True}
        with counted_io(counter), native.OriginalGuard(torch, plan, ledger) as guard:
            begun = time.perf_counter(); pipe = native.build_pipeline(plan); setup_seconds = time.perf_counter()-begun
            report.update(raw_model_setup_seconds=setup_seconds, original_loader_paths=guard.loaded,
                BF16_matrix_residency='Original raw CPU masters and four CUDA block buffers; original offload including W2 unchanged')
            registered = legacy.registration(plan, args.hf_root)
            native.write(snapshot/'registered_video_plan.json', registered); persist(); check()
            for index, case in enumerate(plan['cases']):
                native.need(index == 0 or report.get('local_development_neutral_passed') is True, 'Calibration before local neutral')
                phase = legacy.native_phase(torch, pipe, guard, plan, case, out, args.hf_root, digest, report, persist, check)
                if index == 0:
                    native.need(phase['all_four_original_flow_and_post_bytes_passed'] is True, 'Local neutral did not pass')
                    report['local_development_neutral_passed'] = True; persist(); check(); continue
                from ffnds_video_metrics import evaluate, validate as validate_core
                import cv2
                cv2.setNumThreads(4)
                identity = {'source_id': case['image_sha256'], 'seed': 42, 'expected_frames': 81,
                    'input_prompt_digest': case['prompt_sha256'], 'model_digest': native.canonical(plan['models']),
                    'protocol_sha256': digest, 'run_sha256': phase['generation_receipt']['sha256']}
                cost = {'generation_wall_seconds': phase['source_capture_wall_seconds'], 'ffn_boundary_seconds': phase['FFN_boundary_CUDA_seconds'],
                    'setup_seconds': setup_seconds, 'peak_gpu_memory_mib': phase['peak_cuda_allocated_bytes']/(1 << 20)}
                core = ledger.call(run_id(case), {'core_CPU_reports': 1}, evaluate, Path(phase['video']['path']), None, {'identity': identity, 'cost': cost})
                native.need(not validate_core(core), 'New81frame CORE incomplete')
                core_path = out/run_id(case)/'CORE_METRICS.json'; native.write(core_path, core)
                phase['core_metrics'] = native.binding(core_path, args.hf_root)

                def save_registration():
                    folder = out/f'job{index-1:06d}'; folder.mkdir()
                    native.write(folder/'RUN.json', {'schema': 'ffnds.registered.native.video.v1', 'status': 'COMPLETE_NATIVE_GENERATION_ONLY',
                        'job': registered['samples'][index-1], 'source': registered['sources'][index-1], 'native_ffn_sites': 160,
                        'scheduler_generator_reset': True, 'native_ffn_counters': phase['native_counter_delta'],
                        'actual_initial_noise_file': str(args.hf_root/phase['states']['initial_noise']['path']),
                        'video_path': phase['video']['path'], 'decoded_frames': 81, 'decoded_size': [832, 464], 'decoded_fps': 16,
                        'immutable_generation_receipt': phase['generation_receipt'], 'numeric_path': 'BF16_ORIGINAL'})
                ledger.call(run_id(case), {'source_registration_writes': 1}, save_registration); persist(); check()
            ledger.finish()
            from ffnds_stage_registered_i2v import collect
            verified = collect(registered, out); native.write(out/'VERIFIED_GENERATIONS.json', verified)
            report.update(status='COMPLETE_RAW_BF16_FULL_DEPTH_PART_FOUR_SOURCES',
                verified_cohort=native.binding(out/'VERIFIED_GENERATIONS.json', args.hf_root), checked_transformers_imports=finder.origins,
                captured_native_sites=sum(len(p['packets']) for p in report['phases']), peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
            native.need(report['captured_native_sites'] == 640 and not counter.holds, 'Incomplete part sites/output reservations')
            check()
    except BaseException as error:
        report.update(status='FAILED_FULL_DEPTH_PART_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        signal.alarm(0); monitor_error = None
        if monitor:
            try:
                monitor.close(); report['assigned_GPU_resources'] = monitor.report()
                native.need(not report['assigned_GPU_resources'].get('violations'), 'Final GPU monitor violations')
            except BaseException as error:
                monitor_error = error; report.update(status='FAILED_FULL_DEPTH_PART_PRESERVED_NO_RETRY', resource_error=str(error))
        for item in (finder, no_hif4):
            if item in sys.meta_path:
                sys.meta_path.remove(item)
        # A failed native video writer may leave one named partial mp4. Account
        # those three possible files explicitly, never scan any directory.
        for case in plan['cases'][1:]:
            path = counter.path(out/run_id(case)/'output.mp4')
            if path not in counter.files and path.exists():
                counter.observe_external(path, hold=str(path))
        try:
            report['wall_seconds'] = time.monotonic()-started
            journal.close()
            inventory = {str(path.relative_to(out)): value for path, value in counter.files.items() if path != out/'RUN.json'}
            audit = {'schema': 'ffnds.raw_bf16.output_written_bytes.v2', 'id': plan['id'], 'protocol_sha256': digest,
                'files_excluding_top_level_RUN_and_this_manifest': inventory, 'incremental_checkpoint': counter.summary(),
                'complete_live_bytes_formula': 'sum(inventory bytes) + actual RUN.json bytes + actual OUTPUT_ACCOUNTING.json bytes',
                'scope': 'Owned fresh results directory; source snapshots, journals, completed/partial outputs; directories have no payload bytes'}
            counter.write_json(out/'OUTPUT_ACCOUNTING.json', audit)
            report['output_accounting'] = {'receipt': counter.binding(out/'OUTPUT_ACCOUNTING.json', args.hf_root),
                'checkpoint_before_final_RUN_write': counter.summary(), 'recursive_output_walks': 0}
            persist()
        finally:
            file_guard.__exit__(None, None, None)
        if monitor_error is not None and sys.exc_info()[0] is None:
            raise monitor_error
    return report


def draft(hf, part, cohort_path=COHORT, execution_proof_path=EXECUTION_PROOF):
    # Start from the exact failed part, retaining historical model/worker lineage.
    old_path = ROOT/f'config/e0_original_bf16_full_depth_collect_part{part}_20261028_v1.json'
    plan = copy.deepcopy(native.read(old_path))
    native.need(plan['source_cohort']['path'] == cohort_path, 'Cohort changed during v2 I/O repair')
    old_id = plan['id']; active = native.read(ROOT/'research/ACTIVE_RESEARCH.json')
    matches = [s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id') == old_id]
    native.need(len(matches) == 1 and {'RUN', 'actual_worker_exit'} <= set(matches[0]), 'Actual failed v1 receipts NOT_BOUND')
    failed = matches[0]
    plan.update(id=scope_id(part), status='DRAFT_V2_INCREMENTAL_IO_NEW_EXECUTION_RAW_PROOF_NOT_BOUND', assets_finalized=False,
        binding_status='NOT_BOUND_CURRENT_EXECUTION_RAW_PROOF', repair=REPAIR, output_HF_relative=output_relative(part),
        failed_predecessor={'protocol': native.binding(old_path, ROOT), 'RUN': failed['RUN'], 'launch_exit': failed['actual_worker_exit']},
        execution_private_worker_config_sha256=None, execution_raw_master_CPU_proof=None)
    extra = set(EXTRA_SOURCES) | {str(old_path.relative_to(ROOT))}
    if (ROOT/execution_proof_path).is_file():
        proof = native.read(ROOT/execution_proof_path)
        proof_protocol = native.read(native.bound(ROOT, proof['protocol']))
        mirror.validate(proof_protocol)
        plan.update(execution_raw_master_CPU_proof=proof,
            execution_private_worker_config_sha256=proof_protocol['private_worker_config_sha256'],
            status='DRAFT_V2_BOUND_ASSETS_SOFTWARE_REVIEW_AND_NEW_WORKER_CPU_STAT_PENDING',
            binding_status='BOUND_ASSETS_REVIEW_AND_FREEZE_PENDING')
        extra.add(proof['protocol']['path'])
    plan['repo_sources'] = [native.binding(ROOT/p, ROOT) for p in sorted({item['path'] for item in plan['repo_sources']} | extra)]
    validate(plan); verify_predecessor(plan, hf)
    legacy.verify_lineage(plan, hf, stat_check=False)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('draft', 'inspect', 'preflight', 'run'))
    parser.add_argument('--part', type=int); parser.add_argument('--protocol', type=Path)
    parser.add_argument('--hf-root', type=Path, required=True); parser.add_argument('--local-root', type=Path)
    parser.add_argument('--out', type=Path); parser.add_argument('--cohort', default=COHORT)
    parser.add_argument('--execution-proof', default=EXECUTION_PROOF)
    args = parser.parse_args(); args.hf_root = args.hf_root.resolve()
    if args.command == 'draft':
        native.need(args.part in (0, 1) and args.out is not None and not args.out.exists(), 'Part/fresh draft required')
        value = draft(args.hf_root, args.part, args.cohort, args.execution_proof); native.write(args.out, value)
        print(json.dumps({'id': value['id'], 'status': value['status'], 'model_calls': 0}))
    elif args.command == 'inspect':
        plan = native.read(args.protocol); validate(plan)
        print(json.dumps({'id': plan['id'], 'part': plan['part'], 'limits': limits(), 'resources': RESOURCES, 'repair': REPAIR, 'model_calls': 0}, indent=2))
    elif args.command == 'preflight':
        native.need(args.out is not None and args.local_root is not None, 'Fresh CPU output/local package required')
        out = args.out.resolve(); native.need(out.is_relative_to(args.hf_root) and not out.exists(), 'Fresh confined CPU output required')
        out.mkdir(parents=True)
        value = {'id': native.read(args.protocol)['id'], 'protocol_sha256': native.sha(args.protocol), 'status': 'STARTED_V2_BYTE_PREFLIGHT_NO_RETRY'}
        native.write(out/'RUN.json', value)
        try:
            value = preflight(args)
        except BaseException as error:
            value.update(status='FAILED_V2_BYTE_PREFLIGHT_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error)); raise
        finally:
            native.write(out/'RUN.json', value)
        print(json.dumps({'status': value['status'], 'model_calls': 0}))
    else:
        native.need(args.local_root is not None, 'Exact local package required')
        print(json.dumps({'status': run(args)['status']}))


if __name__ == '__main__':
    main()
