"""Independent v8 imports-only recovery using the completed v2 environment.

No package installer, resolver, downloader, checkpoint loader or score runner.
Every import-side cache path belongs to the new v8 scope.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import time
import errno
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ID = 'E0_OFFICIAL_I2V_IMPORT_COMPATIBILITY_V8'
PTH_SHA = '2638ce9e2500e572a5e0de7faed6661eb569d1b696fcba07b0dd223da5f5d224'
RELATIVE = 'derived/ffnds/E0_official_metric_environment_v8'
DIMENSIONS = ['i2v_subject', 'i2v_background', 'camera_motion',
              'subject_consistency', 'background_consistency', 'motion_smoothness',
              'dynamic_degree', 'aesthetic_quality', 'imaging_quality',
              'temporal_flickering']
BOUND_HELPERS = ('tools/offline_i2v_metric_loaders.py', 'tools/rollout_contract.py')
ACTIVATION_SOURCE_SHA = '37450e9d543aef0495ed67a6fd2017db63938e9995aa2cff16c79b4533ba10bd'
IPV6_SOURCE_SHA = '2633bbdb69731e5ccb5cf4e4afd65605d86c7979cc5633126f50c92d5ad74a74'
FONT_SOURCE_SHA = 'e630f2e5a443227d06af60d9c3f3869cd815ab0c4958742f711e95095f31a868'
FONT_BINARY = Path('/usr/bin/fc-list')
FONT_BINARY_SHA = '30ced2d92eb30d217b509070e5b38c5d061294f0e7da93cc2f2a8f4583799422'
FONT_COMMANDS = [['fc-list', '--help'], ['fc-list', '--format=%{file}\\n']]
ACTIVATION_DEFAULTS = [
    {'global_name': 'gelu_python', 'type_name': 'GELUActivation', 'line': 244},
    {'global_name': 'gelu_new', 'type_name': 'NewGELUActivation', 'line': 245},
    {'global_name': 'gelu', 'type_name': 'GELUActivation', 'line': 246},
    {'global_name': 'gelu_fast', 'type_name': 'FastGELUActivation', 'line': 247},
    {'global_name': 'quick_gelu', 'type_name': 'QuickGELUActivation', 'line': 248},
    {'global_name': 'silu', 'type_name': 'SiLUActivation', 'line': 249},
    {'global_name': 'mish', 'type_name': 'MishActivation', 'line': 250},
    {'global_name': 'linear_act', 'type_name': 'LinearActivation', 'line': 251},
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def norm(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def two_cpus():
    if hasattr(os, 'sched_getaffinity'):
        allowed = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, allowed[:2])


def cache_root(hf):
    return Path(hf).resolve() / RELATIVE


def tree_bytes(directory, missing=None):
    """Count regular files without following symlinks; tolerate only ENOENT races.

    Directory snapshots are intentionally non-atomic while uv creates/removes
    temporary directories. EACCES, EIO, ENOTDIR and all other errors propagate.
    """
    directory = Path(directory)
    if not stat.S_ISDIR(directory.lstat().st_mode):
        raise NotADirectoryError(errno.ENOTDIR, 'Accounting root is not a directory', str(directory))
    pending = [directory]
    total = 0

    def vanished(error):
        if error.errno != errno.ENOENT:
            raise error
        if missing is not None:
            missing.append(str(error.filename))

    while pending:
        current = pending.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except OSError as error:
                        vanished(error)
                        continue
                    if stat.S_ISDIR(info.st_mode):
                        pending.append(Path(entry.path))
                    elif stat.S_ISREG(info.st_mode):
                        total += info.st_size
        except OSError as error:
            vanished(error)
    return total


def budget_check(directory, plan, started):
    limits = plan['limits']
    total = tree_bytes(directory)
    if (total > limits['new_files_bytes_max']
            or shutil.disk_usage(directory).free < limits['free_disk_reserve_bytes']
            or time.monotonic() - started > limits['wall_seconds_max']):
        raise RuntimeError('Environment scope resource limit reached')
    return total


def validate_closure(rows, shared):
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    all_rows = {r['name']: r for r in rows + shared['distributions']}
    if len(all_rows) != len(rows) + len(shared['distributions']):
        raise RuntimeError('Shared runtime and local package overlap')
    for row in all_rows.values():
        for raw in row['requires_dist']:
            req = Requirement(raw)
            if req.marker is not None and not req.marker.evaluate(default_environment()):
                continue
            target = all_rows.get(norm(req.name))
            if target is None or target['version'] not in req.specifier:
                raise RuntimeError('Unclosed runtime dependency: ' + row['name'] + ' -> ' + raw)


def effective_distributions(lock, overlay):
    rows = lock['artifacts'] + lock['shared_runtime']['distributions']
    overlay_dist = list(metadata.distributions(path=[str(overlay)]))
    if len(overlay_dist) != 1 or norm(overlay_dist[0].metadata['Name']) != 'setuptools' or overlay_dist[0].version != '80.9.0':
        raise RuntimeError('Overlay distribution identity differs')
    selected = metadata.distribution('setuptools')
    selected_path = Path(selected._path).resolve()
    if not selected_path.is_relative_to(overlay.resolve()) or selected_path != Path(overlay_dist[0]._path).resolve():
        raise RuntimeError('Effective setuptools metadata origin differs')
    actual_rows = []; report = {}; shared = []
    for row in rows:
        dist = metadata.distribution(row['name'])
        expected = '80.9.0' if row['name'] == 'setuptools' else row['version']
        if dist.version != expected: raise RuntimeError('Effective distribution version differs')
        updated = dict(row, version=expected, requires_dist=dist.requires or [])
        (shared if row in lock['shared_runtime']['distributions'] else actual_rows).append(updated)
        report[row['name']] = {'base_version':row['version'],'effective_version':dist.version,
                               'overlay':row['name']=='setuptools'}
    validate_closure(actual_rows, {'distributions':shared})
    return report


def verify_original_sources(hf, lock):
    for name, digest in lock['original_manifest_sha256'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('Original metric manifest changed')
    assets = read(ROOT / '.runtime/R17_metric_assets.json')
    supplement = read(ROOT / 'research/20261023/R17_metric_source_supplement.json')
    base = hf / assets['HF_relative_root']
    groups = [(base / 'installed_source_view', assets['installed_source_inventory']),
              (hf / supplement['HF_relative_root'], supplement['files'])]
    groups += [(base / 'source' / k, r['files']) for k, r in assets['source_repositories'].items()]
    for directory, rows in groups:
        for row in rows:
            if sha(directory / row['path']) != row['sha256']:
                raise RuntimeError('Pinned original source file changed')
    return assets, supplement


def verify_shared(shared):
    base = ROOT / shared['site_project_relative']
    for row in shared['distributions']:
        for relative, digest in row['metadata_sha256'].items():
            if sha(base / relative) != digest:
                raise RuntimeError('Shared project runtime metadata changed')
    return base


def verify_bound_helpers(plan):
    bindings = plan.get('helper_sha256', {})
    if set(bindings) != set(BOUND_HELPERS):
        raise RuntimeError('Exact repository helper bindings required')
    for relative, digest in bindings.items():
        if sha(ROOT / relative) != digest:
            raise RuntimeError('Bound repository helper changed: ' + relative)


def load_bound_helper(plan):
    """Temporarily expose only the verified helper directory, after runtime paths."""
    import importlib
    import importlib.util
    verify_bound_helpers(plan)
    helper_path = str(ROOT / 'tools')
    added = helper_path not in sys.path
    if added:
        sys.path.append(helper_path)
    try:
        # Refuse a shadowing installed module before executing any helper code.
        for relative in BOUND_HELPERS:
            name = Path(relative).stem
            found = importlib.util.find_spec(name)
            if found is None or found.origin is None or Path(found.origin).resolve() != (ROOT / relative).resolve():
                raise RuntimeError('Bound helper import origin differs: ' + name)
        helper = importlib.import_module('offline_i2v_metric_loaders')
        for relative in BOUND_HELPERS:
            actual = sys.modules.get(Path(relative).stem)
            if actual is None or Path(actual.__file__).resolve() != (ROOT / relative).resolve():
                raise RuntimeError('Actual bound helper origin differs')
        verify_bound_helpers(plan)
        return helper
    finally:
        if added:
            sys.path.remove(helper_path)


def verify_import_exceptions(site, plan):
    activation = plan['stateless_activation_allowlist']
    if (activation['module'] != 'transformers.activations'
            or activation['site_relative'] != 'transformers/activations.py'
            or activation['source_sha256'] != ACTIVATION_SOURCE_SHA
            or activation['instances'] != ACTIVATION_DEFAULTS):
        raise RuntimeError('Exact eight stateless activation defaults required')
    ipv6 = plan['disabled_ipv6_probe']
    if (ipv6['site_relative'] != 'urllib3/util/connection.py'
            or ipv6['source_sha256'] != IPV6_SOURCE_SHA):
        raise RuntimeError('Exact urllib3 capability probe source required')
    pool = plan['stateless_pool_allowlist']
    if pool != {'mad_site_relative':'pyiqa/archs/mad_arch.py','mad_sha256':'336f28ffbeb99f762dfb826dc1e78ead154208521ba26450e1a7f0e1f1dffa71','pool_site_relative':'torch/nn/modules/pooling.py','pool_sha256':'7a66ceba372d3e7bf4dda94871b9b730b8d0c2bf9f6f5051874917bb5dcac826','line':23,'max_instances':1}: raise RuntimeError('Pool allowlist differs')
    if sha(Path(site)/pool['mad_site_relative']) != pool['mad_sha256'] or sha(Path(site)/pool['pool_site_relative']) != pool['pool_sha256']: raise RuntimeError('Pool source binding differs')
    for item in (activation, ipv6):
        if sha(Path(site) / item['site_relative']) != item['source_sha256']:
            raise RuntimeError('Bound import-exception source changed')
    if (sha(Path(site) / 'matplotlib/font_manager.py') != FONT_SOURCE_SHA
            or FONT_BINARY.resolve() != FONT_BINARY or sha(FONT_BINARY) != FONT_BINARY_SHA
            or Path(shutil.which('fc-list') or '').resolve() != FONT_BINARY):
        raise RuntimeError('Pinned font source or executable changed')


def activation_module_guard(original_init, source, source_digest, specs, result, denied,
                            module_name='transformers.activations'):
    """Allow only ordered module-global defaults; keep live references for checks."""
    source = Path(source).resolve()
    instances = []
    events = result.setdefault('stateless_activation_wrappers', [])

    def guarded_init(instance, *args, **kwargs):
        cls = type(instance)
        module = sys.modules.get(module_name)
        frame = sys._getframe(1)
        global_line = None
        while frame is not None:
            if frame.f_code.co_name == '<module>' and Path(frame.f_code.co_filename).resolve() == source:
                global_line = frame.f_lineno
                break
            frame = frame.f_back
        position = len(events)
        if (position >= len(specs) or cls.__module__ != module_name
                or module is None or getattr(module, cls.__name__, None) is not cls
                or Path(module.__file__).resolve() != source or sha(source) != source_digest
                or cls.__name__ != specs[position]['type_name']
                or global_line != specs[position]['line'] or args or kwargs):
            denied('NN_MODULE_CONSTRUCTION_OUTSIDE_EXACT_ACTIVATION_DEFAULTS')()
        original_init(instance)
        if any(getattr(instance, name, None) for name in ('_parameters', '_buffers', '_modules')):
            denied('ACTIVATION_DEFAULT_HAS_PARAMETER_BUFFER_OR_CHILD')()
        instances.append(instance)
        events.append(dict(specs[position], module=module_name, source_sha256=source_digest))

    def finalize():
        module = sys.modules.get(module_name)
        if len(events) != len(specs) or len({id(x) for x in instances}) != len(specs):
            denied('ACTIVATION_DEFAULT_COUNT_DIFFERS')()
        for spec, instance in zip(specs, instances):
            if (getattr(module, spec['global_name'], None) is not instance
                    or any(getattr(instance, name, None) for name in ('_parameters', '_buffers', '_modules'))):
                denied('ACTIVATION_DEFAULT_STATE_OR_GLOBAL_CHANGED')()
        return len(instances)

    return guarded_init, finalize


def stateless_pool_guard(activation_guard, original_init, pool_type, pool_source, pool_digest,
                         mad_source, mad_digest, result, denied):
    mad_source = Path(mad_source).resolve(); pool_source = Path(pool_source).resolve()
    instances = []
    def guarded(instance, *args, **kwargs):
        if type(instance) is not pool_type:
            return activation_guard(instance, *args, **kwargs)
        frame = sys._getframe(1); module_frame = None
        while frame is not None:
            if frame.f_code.co_name == '<module>' and Path(frame.f_code.co_filename).resolve() == mad_source:
                module_frame = frame; break
            frame = frame.f_back
        module = sys.modules.get('pyiqa.archs.mad_arch')
        if (instances or len(result.get('stateless_activation_wrappers', [])) != 8 or args or kwargs
                or module is None or Path(module.__file__).resolve() != mad_source
                or module_frame is None or module_frame.f_lineno != 23
                or sha(mad_source) != mad_digest or sha(pool_source) != pool_digest
                or pool_type.__module__ != 'torch.nn.modules.pooling'
                or getattr(sys.modules.get('torch.nn.modules.pooling'), 'MaxPool2d', None) is not pool_type):
            denied('POOL_OUTSIDE_EXACT_GLOBAL')()
        original_init(instance)
        if any(getattr(instance, n, None) for n in ('_parameters','_buffers','_modules')):
            denied('POOL_INITIAL_STATE_NOT_EMPTY')()
        instances.append(instance)
        result['stateless_pool_wrappers'] = [{'module':'pyiqa.archs.mad_arch','line':23,'type':'torch.nn.MaxPool2d','source_sha256':mad_digest}]
    def finalize():
        module = sys.modules.get('pyiqa.archs.mad_arch')
        if len(instances) != 1 or module is None or getattr(module,'MAX',None) is not instances[0]: denied('POOL_GLOBAL_COUNT_OR_IDENTITY')()
        obj = instances[0]
        expected={'kernel_size':(2,2),'stride':1,'padding':1,'dilation':1,'return_indices':False,'ceil_mode':False}
        if any(getattr(obj,k,None) != v for k,v in expected.items()) or any(getattr(obj,n,None) for n in ('_parameters','_buffers','_modules')): denied('POOL_FINAL_STATE_DIFFERS')()
        return 1
    return guarded, finalize


def disable_local_ipv6_probe(site, plan):
    """The pinned urllib3 _has_ipv6 function checks this flag before any socket."""
    import socket
    item = plan['disabled_ipv6_probe']
    source = Path(site) / item['site_relative']
    if sha(source) != item['source_sha256'] or item['source_sha256'] != IPV6_SOURCE_SHA:
        raise RuntimeError('IPv6 capability-probe source changed')
    previous = socket.has_ipv6
    socket.has_ipv6 = False
    return {'socket_has_ipv6_before': previous, 'socket_has_ipv6_during_imports': False,
            'source_sha256': item['source_sha256'], 'socket_creation_allowed': False}


def verify_installed_environment(hf, plan, lock):
    """Inspect metadata only; no site initialization or third-party import."""
    binding = plan['installed_environment']
    for relative, digest in binding['preserved_source_sha256'].items():
        if sha(ROOT / relative) != digest:
            raise RuntimeError('Executed v1/v2 source/protocol/lock changed')
    for relative, digest in binding['preserved_receipt_sha256'].items():
        if sha(Path(hf) / relative) != digest:
            raise RuntimeError('Executed v2 receipt/log changed')
    site = Path(hf) / binding['site_HF_relative']
    for relative, digest in binding['site_metadata_sha256'].items():
        if sha(site / relative) != digest:
            raise RuntimeError('Reused installed distribution metadata changed')
    found = {norm(d.metadata['Name']): d for d in metadata.distributions(path=[str(site)])}
    expected = lock['artifacts'] + lock['shared_runtime']['distributions']
    for row in expected:
        if row['name'] not in found or found[row['name']].version != row['version']:
            raise RuntimeError('Reused installed distribution version differs')
    installed = read(Path(hf) / binding['installed_receipt_HF_relative'])
    expected_archives = {r['name']: (r['sha256'], r['bytes']) for r in lock['artifacts']}
    observed = {r['name']: (r['sha256'], r['bytes']) for r in installed['package_artifacts']}
    if expected_archives != observed or len(installed['package_artifacts']) != len(expected_archives):
        raise RuntimeError('Complete v2 installation artifact receipt required')
    if sha(Path(hf) / binding['python_HF_relative']) != binding['python_sha256']:
        raise RuntimeError('Reused Python interpreter changed')
    return site


def inspect_wheel(path, expected_sha, expected_bytes, max_unpacked):
    if sha(path) != expected_sha or Path(path).stat().st_size != expected_bytes:
        raise RuntimeError('Wheel identity differs')
    records = {}; total = 0
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            name = info.filename
            mode = (info.external_attr >> 16) & 0o170000
            if (not name or name.startswith('/') or '\\' in name or any(p in ('..', '.') for p in name.split('/'))
                    or mode == stat.S_IFLNK or (mode not in (0, stat.S_IFREG, stat.S_IFDIR))
                    or name in records or info.flag_bits & 1):
                raise RuntimeError('Unsafe or duplicate wheel member')
            if info.is_dir():
                records[name] = None; continue
            total += info.file_size
            if total > max_unpacked: raise RuntimeError('Wheel unpacked budget exceeded')
            data = z.read(info)
            if len(data) != info.file_size: raise RuntimeError('Wheel member size mismatch')
            records[name] = hashlib.sha256(data).hexdigest()
    if 'pkg_resources/__init__.py' not in records or 'setuptools/_vendor/' not in ''.join(records):
        raise RuntimeError('Wheel missing expected package or vendored dependencies')
    return records


def prepare_overlay(hf, directory, plan):
    spec = plan['overlay']; wheel = (Path(hf) / spec['wheel_HF_relative']).resolve()
    if not wheel.is_relative_to(Path(hf).resolve()): raise RuntimeError('Wheel escaped HF root')
    records = inspect_wheel(wheel, spec['sha256'], spec['bytes'], spec['max_unpacked_bytes'])
    overlay = directory / 'setuptools_overlay'; overlay.mkdir(exist_ok=False)
    with zipfile.ZipFile(wheel) as z:
        for name, digest in records.items():
            path = overlay / name
            if digest is None: path.mkdir(parents=True, exist_ok=True); continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as f: f.write(z.read(name))
            if sha(path) != digest: raise RuntimeError('Extracted member SHA differs')
            path.chmod(0o444)
    for root, dirs, files in os.walk(overlay, topdown=False): Path(root).chmod(0o555)
    write(directory / 'OVERLAY_MANIFEST.json', {'wheel_sha256':spec['sha256'], 'members':records})
    return overlay


def verify_overlay(directory, plan, hf):
    spec = plan['overlay']; receipt = read(directory / 'OVERLAY_MANIFEST.json')
    wheel = (Path(hf) / spec['wheel_HF_relative']).resolve()
    if not wheel.is_relative_to(Path(hf).resolve()) or receipt['wheel_sha256'] != spec['sha256'] or receipt['members'] != inspect_wheel(wheel, spec['sha256'], spec['bytes'], spec['max_unpacked_bytes']): raise RuntimeError('Overlay receipt mismatch')
    if spec.get('distutils_precedence_pth_sha256') != PTH_SHA or receipt['members'].get('distutils-precedence.pth') != PTH_SHA:
        raise RuntimeError('Bound text metadata identity differs')
    root = directory / 'setuptools_overlay'
    actual = set()
    for item in root.rglob('*'):
        if item.is_symlink(): raise RuntimeError('Overlay symlink denied')
        if item.is_file(): actual.add(item.relative_to(root).as_posix())
    expected = {k for k,v in receipt['members'].items() if v is not None}
    if actual != expected or any(sha(root/k) != receipt['members'][k] for k in expected):
        raise RuntimeError('Overlay contents differ')
    return root


def scope_environment(directory, hf, assets):
    """Create scope-local caches before launching the isolated import child."""
    paths = {
        'TMPDIR': directory / 'tmp', 'TEMP': directory / 'tmp', 'TMP': directory / 'tmp',
        'TORCHINDUCTOR_CACHE_DIR': directory / 'torchinductor',
        'TORCH_COMPILE_DEBUG_DIR': directory / 'torch_compile_debug',
        'MPLCONFIGDIR': directory / 'mplconfig', 'XDG_CACHE_HOME': directory / 'xdg_cache',
        'TORCH_HOME': directory / 'torch_cache', 'HF_HOME': directory / 'hf_runtime',
    }
    for path in set(paths.values()):
        path.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
               NUMEXPR_NUM_THREADS='2', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               VBENCH_CACHE_DIR=str(Path(hf) / assets['HF_relative_root']))
    env.update({name: str(path) for name, path in paths.items()})
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    return env, paths


def authorized(protocol):
    plan = read(protocol)
    if plan['id'] != ID or sha(__file__) != plan['tool_sha256']:
        raise RuntimeError('Frozen tool/protocol identity differs')
    verify_bound_helpers(plan)
    exact = {'attempts_max': 1, 'wall_seconds_max': 1200, 'cpu_threads_max': 2,
             'new_files_bytes_max': 4 * 1024**3, 'free_disk_reserve_bytes': 10 * 1024**3,
             'gpu_assignments_max': 0, 'model_constructions_max': 0,
             'parameter_creations_max': 0, 'checkpoint_loads_max': 0,
             'model_forwards_max': 0, 'official_scores_max': 0,
             'stateless_activation_wrapper_constructions_max': 8, 'buffer_creations_max': 0,
             'font_queries_max': 2, 'stateless_pooling_wrapper_constructions_max': 1,
             'allowed_text_metadata_hash_reads_max': 1}
    if plan['limits'] != exact:
        raise RuntimeError('Fixed imports-only limits differ')
    lock = ROOT / plan['lock_path']
    if sha(lock) != plan['lock_sha256']:
        raise RuntimeError('Frozen package lock changed')
    scopes = read(ROOT / 'research/ACTIVE_RESEARCH.json')['experiment_authority']['current_user_authorized_scopes']
    if not any(s.get('id') == ID and s.get('protocol_sha256') == sha(protocol)
               and s.get('execution_status') in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE') for s in scopes):
        raise RuntimeError('Exact final protocol SHA is not registered in ACTIVE')
    locked = read(lock)
    if locked['python'] != sys.version.split()[0]:
        raise RuntimeError('Pinned Python version differs')
    return plan, locked


def run_checked(command, env, directory, plan, started, log):
    with log.open('ab') as output:
        process = subprocess.Popen(command, env=env, stdout=output, stderr=output,
                                   start_new_session=True, cwd=directory)
        try:
            while process.poll() is None:
                budget_check(directory, plan, started)
                time.sleep(0.25)
            if process.returncode:
                raise RuntimeError('Bounded subprocess failed; inspect immutable log')
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise


def run(args):
    plan, lock = authorized(args.protocol)
    two_cpus()
    hf = Path(args.hf_root).resolve()
    directory = cache_root(hf)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'ATTEMPT.json').open('x') as f:
        json.dump({'id': ID, 'protocol_sha256': sha(args.protocol), 'attempt': 1,
                   'status': 'STARTED_NO_RETRY'}, f)
    started = time.monotonic()
    report = {'id': ID, 'status': 'RUNNING_IMPORTS_ONLY_REUSED_INSTALLED_ENVIRONMENT',
              'protocol_sha256': sha(args.protocol), 'lock_sha256': plan['lock_sha256'],
              'package_installs': 0, 'package_network_fetches': 0, 'checkpoint_loads': 0,
              'model_constructions': 0, 'pretrained_model_constructions': 0,
              'parameter_creations': 0, 'buffer_creations': 0,
              'stateless_activation_wrapper_constructions': 0, 'model_forwards': 0,
              'official_scores': 0, 'gpu_assignments': 0}
    try:
        assets, _ = verify_original_sources(hf, lock)
        verify_shared(lock['shared_runtime'])
        site = verify_installed_environment(hf, plan, lock)
        overlay = prepare_overlay(hf, directory, plan)
        env, paths = scope_environment(directory, hf, assets)
        report['scope_cache_HF_relative'] = {key: str(path.relative_to(hf)) for key, path in paths.items()}
        report['reused_installed_distribution_count'] = len(lock['artifacts']) + len(lock['shared_runtime']['distributions'])
        budget_check(directory, plan, started)
        command = [str(hf / plan['installed_environment']['python_HF_relative']), '-I', '-S',
                   str(Path(__file__).resolve()), '_import', '--hf-root', str(hf), '--site', str(site),
                   '--lock', str(ROOT / plan['lock_path']), '--protocol', str(Path(args.protocol).resolve())]
        run_checked(command, env, directory, plan, started, directory / 'import.log')
        imported = read(directory / 'IMPORT_RESULT.json')
        if imported['status'] != 'PASS_IMPORTS_ONLY':
            raise RuntimeError('Import child did not pass')
        report.update(status='PASS_IMPORTS_ONLY', import_result_sha256=sha(directory / 'IMPORT_RESULT.json'))
    except BaseException as error:
        report.update(status='FAILED_SCOPE_DEBIT_PRESERVED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        import_receipt = directory / 'IMPORT_RESULT.json'
        if import_receipt.exists():
            child = read(import_receipt)
            report['import_result_sha256'] = sha(import_receipt)
            report['completed_official_function_imports'] = len(child['functions'])
            report['stateless_activation_wrapper_constructions'] = len(child.get('stateless_activation_wrappers', []))
            report['stateless_pool_wrapper_constructions'] = len(child.get('stateless_pool_wrappers', []))
            report['denied_attempts'] = child['denied_attempts']
            report['font_queries'] = child.get('font_queries', [])
            report['allowed_text_metadata_hash_reads'] = child.get('allowed_text_metadata_hash_reads', 0)
        accounting_error = None
        try:
            report['new_files_bytes'] = tree_bytes(directory)
        except OSError as error:
            accounting_error = error
            report.update(status='FAILED_ACCOUNTING_ERROR_DEBIT_PRESERVED', new_files_bytes=None,
                          final_accounting_error={'type': type(error).__name__, 'errno': error.errno,
                                                  'message': str(error)})
        try:
            verify_installed_environment(hf, plan, lock)
            verify_shared(lock['shared_runtime'])
            report['reused_environment_metadata_unchanged'] = True
        except BaseException as error:
            report.update(status='FAILED_REUSED_ENVIRONMENT_IDENTITY',
                          reused_environment_error={'type': type(error).__name__, 'message': str(error)})
            accounting_error = accounting_error or error
        write(directory / 'RUN.json', report)
        if accounting_error is not None:
            raise accounting_error
    print(json.dumps({'status': report['status'], 'elapsed_seconds': report['elapsed_seconds']}))


def import_audit_guard(directory, result):
    """Application guard; kernel/native I/O confinement is a separate boundary."""
    def denied(kind):
        def fail(*unused, **unused_kw):
            result['denied_attempts'].append(kind)
            import traceback
            result.setdefault('denied_attempt_details', []).append({'kind': kind, 'stack': [
                {'file': f.filename, 'line': f.lineno, 'function': f.name}
                for f in traceback.extract_stack(limit=12)[:-1]]})
            raise RuntimeError('Import preflight denied: ' + kind)
        return fail

    font_pending = []

    def font_check_output(original, source, executable=FONT_BINARY, binary_sha=FONT_BINARY_SHA):
        source = Path(source).resolve()
        def checked(argv, *args, **kwargs):
            frame = sys._getframe(1)
            n = len(result.setdefault('font_queries', []))
            if (n >= 2 or argv != FONT_COMMANDS[n] or args or kwargs
                    or frame.f_code.co_name != '_get_fontconfig_fonts'
                    or Path(frame.f_code.co_filename).resolve() != source
                    or frame.f_lineno != (256, 260)[n]
                    or sha(source) != FONT_SOURCE_SHA
                    or Path(shutil.which('fc-list') or '').resolve() != executable
                    or sha(executable) != binary_sha):
                denied('FONT_QUERY_IDENTITY')()
            item = {'argv': argv, 'binary': str(executable), 'binary_sha256': binary_sha,
                    'font_manager_sha256': FONT_SOURCE_SHA, 'exit_status': None}
            result['font_queries'].append(item)
            font_pending.append(argv)
            try:
                output = original(argv)
                item['exit_status'] = 0
                return output
            except subprocess.CalledProcessError as error:
                item['exit_status'] = error.returncode
                raise
            finally:
                font_pending.clear()
        return checked

    def audit(event, values):
        if event == 'subprocess.Popen' and font_pending:
            if values[0] == 'fc-list' and values[1] == font_pending[-1] and values[2] is None and values[3] is None:
                font_pending.clear()
                return
            denied('FONT_QUERY_POPEN_MISMATCH')()
        if event.startswith('socket.') or event in ('subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn'):
            denied('NETWORK_OR_SUBPROCESS:' + event)()
        if event == 'open' and isinstance(values[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(values[0])).resolve()
            mode = values[1] or ''
            flags = values[2] or 0
            writing = any(c in mode for c in 'wax+') or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if writing and path.is_relative_to(directory / 'setuptools_overlay'): denied('OVERLAY_WRITE')()
            if writing and not path.is_relative_to(directory):
                denied('WRITE_OUTSIDE_SCOPE')()
            if not writing and (path.suffix.lower() in ('.pt', '.pth', '.ckpt', '.safetensors', '.bin', '.npz')
                                or '/R17_metric_assets/dreamsim/' in str(path)):
                frame = sys._getframe(1)
                codes = set()
                while frame is not None:
                    codes.add(frame.f_code)
                    frame = frame.f_back
                if (path == (directory / 'setuptools_overlay/distutils-precedence.pth').resolve()
                        and sha.__code__ in codes and verify_overlay.__code__ in codes
                        and result.get('allowed_text_metadata_hash_reads', 0) == 0):
                    result['allowed_text_metadata_hash_reads'] = 1
                    return
                denied('CHECKPOINT_FILE_READ')()
        if event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.rename', 'os.symlink', 'os.link', 'os.chmod', 'os.truncate'):
            candidates = values[:2] if event in ('os.rename', 'os.link', 'os.symlink') else values[:1]
            for value in candidates:
                if isinstance(value, (str, bytes, os.PathLike)) and Path(os.fsdecode(value)).resolve().is_relative_to(directory / 'setuptools_overlay'): denied('OVERLAY_MUTATION')()
                if isinstance(value, (str, bytes, os.PathLike)) and not Path(os.fsdecode(value)).resolve().is_relative_to(directory):
                    denied('FILESYSTEM_MUTATION_OUTSIDE_SCOPE')()
    return denied, audit, font_check_output


def import_preflight(args):
    # No site initialization: all imports come from this explicit isolated path.
    sys.dont_write_bytecode = True
    site = Path(args.site).resolve()
    directory = cache_root(args.hf_root)
    plan, lock = authorized(args.protocol)
    expected_site = verify_installed_environment(Path(args.hf_root), plan, lock).resolve()
    if site != expected_site:
        raise RuntimeError('Only the exact read-only reused v2 metric site may be imported')
    sys.path.insert(0, str(site))
    overlay = verify_overlay(directory, plan, args.hf_root)
    sys.path.insert(0, str(overlay))
    verify_import_exceptions(site, plan)
    with (directory / 'IMPORT_ATTEMPT.json').open('x') as f:
        json.dump({'id': ID, 'attempt': 1, 'status': 'DEBIT_BEFORE_IMPORTS'}, f)
    result = {'id': ID, 'status': 'STARTED_IMPORTS_ONLY', 'denied_attempts': [], 'functions': [],
              'checkpoint_loads': 0, 'model_constructions': 0, 'parameter_creations': 0,
              'buffer_creations': 0, 'model_forwards': 0,
              'official_scores': 0, 'CUDA_initialized': False,
              'allowed_text_metadata_hash_reads': 0}
    denied, audit, font_check_output = import_audit_guard(directory, result)

    subprocess.check_output = font_check_output(subprocess.check_output, site / 'matplotlib/font_manager.py')

    sys.addaudithook(audit)
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise RuntimeError('Every GPU must be hidden')
        if Path(args.lock).resolve() != (ROOT / plan['lock_path']).resolve():
            raise RuntimeError('Child lock path differs')
        assets, supplement = verify_original_sources(Path(args.hf_root), lock)
        shared = verify_shared(lock['shared_runtime'])
        result['disabled_ipv6_probe'] = disable_local_ipv6_probe(site, plan)
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError('Torch import initialized CUDA')
        torch.set_num_threads(2)
        torch.load = denied('CHECKPOINT_DESERIALIZATION')
        torch.serialization.load = denied('CHECKPOINT_DESERIALIZATION')
        torch.jit.load = denied('JIT_DESERIALIZATION')
        for name in ('load', 'load_state_dict_from_url', 'download_url_to_file'):
            setattr(torch.hub, name, denied('TORCH_HUB:' + name))
        torch.cuda._lazy_init = denied('CUDA_INITIALIZATION')
        if hasattr(torch._C, '_cuda_init'):
            torch._C._cuda_init = denied('CUDA_NATIVE_INITIALIZATION')
        activation = plan['stateless_activation_allowlist']
        guarded_init, finalize_activations = activation_module_guard(
            torch.nn.Module.__init__, site / activation['site_relative'], activation['source_sha256'],
            activation['instances'], result, denied)
        pool=plan['stateless_pool_allowlist']
        guarded_init, finalize_pool = stateless_pool_guard(guarded_init, torch.nn.Module.__init__, torch.nn.MaxPool2d, site/pool['pool_site_relative'], pool['pool_sha256'], site/pool['mad_site_relative'], pool['mad_sha256'], result, denied)
        torch.nn.Module.__init__ = guarded_init
        torch.nn.Parameter.__new__ = staticmethod(denied('PARAMETER_CREATION'))
        torch.nn.UninitializedParameter.__new__ = staticmethod(denied('PARAMETER_CREATION'))
        torch.nn.Module.register_parameter = denied('PARAMETER_REGISTRATION')
        torch.nn.Module.register_buffer = denied('BUFFER_REGISTRATION')
        for name in ('Buffer', 'UninitializedBuffer'):
            if hasattr(torch.nn, name):
                getattr(torch.nn, name).__new__ = staticmethod(denied('BUFFER_CREATION'))
        torch.nn.Module._call_impl = denied('NN_FORWARD')
        torch.nn.Module._wrapped_call_impl = denied('NN_FORWARD')
        import safetensors
        import safetensors.torch
        safetensors.safe_open = denied('SAFETENSORS_DESERIALIZATION')
        safetensors.torch.safe_open = denied('SAFETENSORS_DESERIALIZATION')
        safetensors.torch.load_file = denied('SAFETENSORS_DESERIALIZATION')
        safetensors.torch.load = denied('SAFETENSORS_DESERIALIZATION')
        for name in ('safe_open', 'deserialize'):
            if hasattr(safetensors._safetensors_rust, name):
                setattr(safetensors._safetensors_rust, name, denied('SAFETENSORS_DESERIALIZATION'))
        source = Path(args.hf_root) / assets['HF_relative_root'] / 'installed_source_view'
        sys.path.insert(0, str(source))
        sys.path.insert(0, str(Path(args.hf_root) / assets['HF_relative_root'] / 'source/cotracker'))
        import importlib
        import inspect
        import vbench2_beta_i2v
        vbench2_beta_i2v.__path__.append(str(Path(args.hf_root) / supplement['HF_relative_root'] / 'vbench2_beta_i2v'))
        for dimension in DIMENSIONS:
            result['current_import'] = dimension
            package = 'vbench2_beta_i2v' if dimension in DIMENSIONS[:3] else 'vbench'
            module = importlib.import_module(package + '.' + dimension)
            fn = getattr(module, 'compute_' + dimension)
            if list(inspect.signature(fn).parameters)[:3] != ['json_dir', 'device', 'submodules_list']:
                raise RuntimeError('Original official signature changed')
            if Path(module.__file__).resolve() != (source / package / (dimension + '.py')).resolve():
                raise RuntimeError('Metric source-view import precedence differs')
            result['functions'].append({'dimension': dimension, 'module_sha256': sha(module.__file__),
                                        'signature': str(inspect.signature(fn)),
                                        'module_HF_relative': str(Path(module.__file__).relative_to(args.hf_root))})
        for name in ('cotracker.predictor', 'peft', 'open_clip', 'pyiqa.archs.musiq_arch'):
            result['current_import'] = name
            importlib.import_module(name)
        result['stateless_activation_wrapper_constructions'] = finalize_activations()
        result['stateless_pool_wrapper_constructions'] = finalize_pool()
        import socket
        connection = sys.modules.get('urllib3.util.connection')
        if socket.has_ipv6 is not False or (connection is not None and connection.HAS_IPV6 is not False):
            raise RuntimeError('Disabled local IPv6 capability changed during imports')
        result['effective_distribution_versions'] = effective_distributions(lock, overlay)
        result['actual_installed_metadata_closure_checked'] = True
        result['distribution_locations'] = {}
        for row in lock['artifacts'] + lock['shared_runtime']['distributions']:
            path = Path(metadata.distribution(row['name']).locate_file('')).resolve()
            result['distribution_locations'][row['name']] = (
                {'project_relative': str(path.relative_to(ROOT))} if path.is_relative_to(ROOT)
                else {'HF_relative': str(path.relative_to(args.hf_root))})
        # Pure path/JSON loader contracts only: no context manager/model loader is entered.
        helper = load_bound_helper(plan)
        result['bound_repository_helpers_sha256'] = plan['helper_sha256']
        result['loader_path_contracts'] = {}
        for dimension in DIMENSIONS:
            setting = [] if dimension == 'temporal_flickering' else helper.submodules(dimension, assets, Path(args.hf_root))
            result['loader_path_contracts'][dimension] = setting
        canonical = ROOT / plan['input_manifest_path']
        if sha(canonical) != plan['input_manifest_sha256']:
            raise RuntimeError('Actual condition metadata differs')
        inputs = read(canonical)
        result['source_eligibility'] = []
        for index in plan['source_indices']:
            row = inputs[index]
            image = Path(row.get('image_path') or row['img_paths'][0])
            if sha(image) != row['input_provenance']['image_sha256']:
                raise RuntimeError('Original condition image differs')
            if any(label in row['prompt'] for label in plan['camera_instruction_labels']):
                raise RuntimeError('This cohort was frozen without camera instructions')
            result['source_eligibility'].append({'source': index, 'dimension': row['dimension'],
                'condition_image_sha256': sha(image), 'camera_motion': 'NOT_APPLICABLE'})
        result['module_origins'] = {}
        for name in ('torch', 'torchvision', 'numpy', 'cv2', 'transformers', 'tokenizers',
                     'huggingface_hub', 'peft', 'open_clip', 'pyiqa', 'dreamsim', 'timm', 'decord'):
            module = importlib.import_module(name)
            path = Path(module.__file__).resolve()
            if path.is_relative_to(shared):
                origin = {'kind': 'READ_ONLY_PROJECT_RUNTIME', 'project_relative': str(path.relative_to(ROOT))}
            elif path.is_relative_to(site):
                origin = {'kind': 'READ_ONLY_REUSED_V2_METRIC_SITE', 'HF_relative': str(path.relative_to(args.hf_root))}
            elif path.is_relative_to(directory):
                origin = {'kind': 'ISOLATED_METRIC_ENV', 'HF_relative': str(path.relative_to(args.hf_root))}
            elif path.is_relative_to(source):
                origin = {'kind': 'PINNED_ORIGINAL_SOURCE_VIEW', 'HF_relative': str(path.relative_to(args.hf_root))}
            else:
                raise RuntimeError('Unexpected module import origin: ' + name)
            result['module_origins'][name] = origin
        pkg = importlib.import_module('pkg_resources'); packaging = importlib.import_module('packaging')
        if getattr(pkg, 'packaging', None) is not packaging: raise RuntimeError('pkg_resources.packaging identity differs')
        if not Path(pkg.__file__).resolve().is_relative_to(overlay) or not Path(packaging.__file__).resolve().is_relative_to(site): raise RuntimeError('Compatibility module origin mismatch')
        vendor = overlay / 'setuptools/_vendor'
        if str(vendor) not in sys.path: raise RuntimeError('Expected pinned vendored path missing')
        for name, module in list(sys.modules.items()):
            if name == 'jaraco' or name == 'pkg_resources' or name.startswith(('pkg_resources.', 'jaraco.', 'platformdirs', 'packaging.')):
                origin = getattr(module, '__file__', None)
                paths = list(getattr(module, '__path__', []))
                if any(not (Path(p).resolve().is_relative_to(overlay) or Path(p).resolve().is_relative_to(site) or Path(p).resolve().is_relative_to(shared)) for p in paths): raise RuntimeError('Unexpected compatibility namespace origin: ' + name)
                if origin and not (Path(origin).resolve().is_relative_to(overlay) or Path(origin).resolve().is_relative_to(site) or Path(origin).resolve().is_relative_to(shared)): raise RuntimeError('Unexpected compatibility dependency origin: ' + name)
        result['compatibility_origins'] = {'pkg_resources': str(Path(pkg.__file__).resolve().relative_to(directory)), 'packaging': str(Path(packaging.__file__).resolve().relative_to(site)), 'vendor': str(vendor.relative_to(directory))}
        verify_overlay(directory, plan, args.hf_root)
        if result['denied_attempts'] or torch.cuda.is_initialized():
            raise RuntimeError('A denied operation was attempted during imports')
        result['status'] = 'PASS_IMPORTS_ONLY'
    except BaseException as error:
        result.update(status='FAILED_IMPORTS_ONLY', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write(directory / 'IMPORT_RESULT.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run_parser = sub.add_parser('run')
    run_parser.add_argument('--protocol', required=True)
    internal = sub.add_parser('_import')
    for key in ('site', 'lock', 'protocol'):
        internal.add_argument('--' + key, required=True)
    for item in (run_parser, internal):
        item.add_argument('--hf-root', default=os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface')))
    args = parser.parse_args()
    {'run': run, '_import': import_preflight}[args.command](args)

if __name__ == '__main__':
    main()
