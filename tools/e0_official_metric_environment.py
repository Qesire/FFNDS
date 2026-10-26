"""Frozen E0 metric environment preparation; imports-only, never scoring.

`resolve` performs metadata resolution and reads source archives without executing
their setup code. `run` requires exact protocol-SHA registration in ACTIVE. It
installs only locked artifacts into a new cache venv, then runs a guarded child.
No legacy R17 entry point or generation environment is mutated.
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
import tarfile
import time
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ID = 'E0_OFFICIAL_I2V_IMPORT_COMPATIBILITY_V1'
RELATIVE = 'derived/ffnds/E0_official_metric_environment_v1'
PINS = {
    'vbench': '0.1.5', 'dreamsim': '0.2.1', 'pyiqa': '0.1.10',
    'timm': '1.0.12', 'numpy': '1.26.4', 'opencv-python': '4.11.0.86',
    'transformers': '4.33.2', 'tokenizers': '0.13.3',
    'huggingface-hub': '0.25.2', 'peft': '0.6.2', 'accelerate': '0.23.0',
    'open-clip-torch': '2.24.0', 'decord': '0.6.0', 'omegaconf': '2.3.0',
    'Pillow': '10.4.0', 'scipy': '1.13.1', 'matplotlib': '3.9.2',
    'scikit-image': '0.24.0', 'scikit-learn': '1.5.2', 'pandas': '2.2.3',
    'imgaug': '0.4.0', 'facexlib': '0.3.0', 'lmdb': '1.5.1',
    'yapf': '0.40.2', 'tensorboard': '2.18.0',
}
# Reviewed setup/requirements text from these exact PyPI source archives.
# Only the Python package subtree is extracted; setup/build code is never run.
SOURCE_ONLY = {
    'dreamsim': ('0.2.1', 'dreamsim',
                 ['numpy', 'open-clip-torch', 'peft', 'Pillow', 'torch', 'timm',
                  'scipy', 'torchvision', 'transformers']),
    'openai-clip': ('1.0.1', 'clip', ['ftfy', 'regex', 'tqdm']),
    'fairscale': ('0.4.13', 'fairscale', ['torch>=1.8.0', 'numpy>=1.22.0']),
    'fvcore': ('0.1.5.post20221221', 'fvcore',
               ['numpy', 'yacs>=0.1.6', 'pyyaml>=5.1', 'tqdm', 'termcolor>=1.1',
                'Pillow', 'tabulate', 'iopath>=0.1.7']),
    'iopath': ('0.1.10', 'iopath', ['tqdm', 'typing_extensions', 'portalocker']),
    'antlr4-python3-runtime': ('4.9.3', 'src/antlr4', []),
    'filterpy': ('1.4.5', 'filterpy', ['numpy', 'scipy', 'matplotlib']),
}
DIMENSIONS = ['i2v_subject', 'i2v_background', 'camera_motion',
              'subject_consistency', 'background_consistency', 'motion_smoothness',
              'dynamic_degree', 'aesthetic_quality', 'imaging_quality',
              'temporal_flickering']
BOUND_HELPERS = ('tools/offline_i2v_metric_loaders.py', 'tools/rollout_contract.py')


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


def pypi(name, version):
    url = f'https://pypi.org/pypi/{name}/{version}/json'
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def budget_check(directory, plan, started):
    limits = plan['limits']
    total = sum(p.lstat().st_size for p in directory.rglob('*')
                if p.is_file() and not p.is_symlink())
    if (total > limits['new_files_bytes_max']
            or shutil.disk_usage(directory).free < limits['free_disk_reserve_bytes']
            or time.monotonic() - started > limits['wall_seconds_max']):
        raise RuntimeError('Environment scope resource limit reached')
    return total


def download(row, directory, check=lambda: None):
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / row['filename']
    if target.exists():
        if target.stat().st_size != row['bytes'] or sha(target) != row['sha256']:
            raise RuntimeError('Cached package artifact changed: ' + row['name'])
        return target
    url = urllib.parse.urlparse(row['url'])
    if url.scheme != 'https' or url.hostname != 'files.pythonhosted.org':
        raise RuntimeError('Only locked PyPI artifact URLs accepted')
    tmp = target.with_suffix(target.suffix + '.part')
    with urllib.request.urlopen(row['url'], timeout=30) as response, tmp.open('xb') as f:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            f.write(block)
            if f.tell() > row['bytes']:
                raise RuntimeError('Package exceeds locked artifact size')
            check()
    if tmp.stat().st_size != row['bytes'] or sha(tmp) != row['sha256']:
        raise RuntimeError('Package artifact digest differs: ' + row['name'])
    tmp.replace(target)
    return target


def artifact(name, version, kind='wheel'):
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename
    data = pypi(name, version)
    if kind == 'source':
        choices = [r for r in data['urls'] if r['packagetype'] == 'sdist']
    else:
        tags = {tag: i for i, tag in enumerate(sys_tags())}
        choices = []
        for row in data['urls']:
            if row['packagetype'] != 'bdist_wheel' or row.get('yanked'):
                continue
            matching = [tags[t] for t in parse_wheel_filename(row['filename'])[3] if t in tags]
            if matching:
                choices.append((min(matching), row))
        choices = [r for _, r in sorted(choices, key=lambda pair: pair[0])]
    if not choices:
        raise RuntimeError('No compatible locked artifact: ' + name)
    row = choices[0]
    return {'name': norm(name), 'version': version, 'kind': kind,
            'filename': row['filename'], 'url': row['url'], 'bytes': row['size'],
            'sha256': row['digests']['sha256'],
            'requires_dist': data['info']['requires_dist'] or []}


def shared_runtime():
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    todo = ['torch']
    found = set()
    while todo:
        name = norm(todo.pop())
        if name in found:
            continue
        found.add(name)
        for raw in metadata.distribution(name).requires or []:
            req = Requirement(raw)
            if req.marker is None or req.marker.evaluate(default_environment()):
                todo.append(req.name)
    found.add('torchvision')  # NumPy and Pillow belong to the isolated metric site.
    rows = []
    base = (ROOT / '.venv/lib/python3.11/site-packages').resolve()
    for name in sorted(found):
        dist = metadata.distribution(name)
        if Path(dist.locate_file('')).resolve() != base:
            raise RuntimeError('Shared runtime distribution outside project venv')
        files = list(dist.files or [])
        roots = sorted({str(p).split('/')[0] for p in files if not str(p).startswith('..')})
        metadata_files = [str(p) for p in files if str(p).endswith(('/METADATA', '/RECORD', '/WHEEL'))]
        rows.append({'name': name, 'version': dist.version, 'roots': roots,
                     'requires_dist': dist.requires or [],
                     'metadata_sha256': {p: sha(base / p) for p in metadata_files}})
    return {'site_project_relative': str(base.relative_to(ROOT)), 'distributions': rows,
            'identity_scope': 'Installed METADATA/RECORD/WHEEL hashes; RECORD content hashes are declared, not all recomputed.'}


def archive_members(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                if not item.is_dir():
                    if (item.external_attr >> 16) & 0o170000 == 0o120000:
                        raise RuntimeError('Source archive symlink refused')
                    yield item.filename, archive.read(item)
    else:
        with tarfile.open(path) as archive:
            for item in archive:
                if item.isfile():
                    yield item.name, archive.extractfile(item).read()
                elif not item.isdir():
                    raise RuntimeError('Source archive special member refused')


def source_recipe(row, path):
    prefix = SOURCE_ONLY[row['name']][1]
    selected = {}
    setup = {}
    for name, content in archive_members(path):
        parts = PurePosixPath(name).parts
        if '..' in parts or name.startswith('/'):
            raise RuntimeError('Unsafe source archive path')
        relative = '/'.join(parts[1:])
        if relative in ('setup.py', 'requirements.txt'):
            setup[relative] = hashlib.sha256(content).hexdigest()
        if relative.startswith(prefix + '/'):
            selected[relative[len(prefix.rsplit('/', 1)[0]) + 1:] if '/' in prefix else relative] = {
                'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
    if not selected or 'setup.py' not in setup:
        raise RuntimeError('Reviewed source package layout differs')
    row.update(package_prefix=prefix, package_files=selected,
               reviewed_setup_sha256=setup, requires_dist=SOURCE_ONLY[row['name']][2],
               installation='UNCHANGED_PYTHON_SUBTREE_WITH_GENERATED_DIST_INFO_NO_SETUP_NO_BUILD')


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


def resolve(args):
    two_cpus()
    directory = cache_root(args.hf_root)
    directory.mkdir(parents=True, exist_ok=True)
    shared = shared_runtime()
    reqs = directory / 'metadata_requirements.in'
    roots = [f'{n}=={v}' for n, v in PINS.items()]
    roots += [req for _, _, reqs_ in SOURCE_ONLY.values() for req in reqs_]
    reqs.write_text('\n'.join(roots) + '\n')
    excluded = directory / 'metadata_excluded.txt'
    excluded.write_text('\n'.join([r['name'] for r in shared['distributions']] + list(SOURCE_ONLY)) + '\n')
    compiled = directory / 'metadata_resolution.txt'
    env = dict(os.environ, UV_CACHE_DIR=str(directory / 'resolution_cache'),
               UV_NO_PROGRESS='1', UV_CONCURRENT_DOWNLOADS='2',
               CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    command = [shutil.which('uv'), 'pip', 'compile', str(reqs), '--python', str(ROOT / '.venv/bin/python'),
               '--only-binary', ':all:', '--excludes', str(excluded),
               '--generate-hashes', '--default-index', 'https://pypi.org/simple', '-o', str(compiled)]
    subprocess.run(command, env=env, check=True, timeout=600, stdout=subprocess.DEVNULL)
    resolved = re.findall(r'^([A-Za-z0-9_.-]+)==([^\s\\]+)', compiled.read_text(), re.M)
    rows = [artifact(n, v) for n, v in resolved]
    for name, (version, _, _) in SOURCE_ONLY.items():
        row = artifact(name, version, 'source')
        path = download(row, directory / 'metadata_sources')
        source_recipe(row, path)
        rows.append(row)
    validate_closure(rows, shared)
    source_files = ['.runtime/R17_metric_assets.json', 'research/20261023/R17_metric_source_supplement.json']
    lock = {'schema': 'ffnds.e0.metric_environment_lock.v1', 'id': ID,
            'status': 'RESOLVED_NOT_INSTALLED_NOT_IMPORTED', 'python': sys.version.split()[0],
            'platform': sys.platform, 'shared_runtime': shared, 'artifacts': sorted(rows, key=lambda r: r['name']),
            'metadata_resolution_sha256': sha(compiled),
            'resolver_uv_sha256': sha(shutil.which('uv')),
            'original_manifest_sha256': {p: sha(ROOT / p) for p in source_files},
            'source_setup_executions': 0, 'dependency_closure_checked': True,
            'metric_module_imports': 0, 'model_constructions': 0, 'checkpoint_loads': 0,
            'official_scores': 0}
    write(args.output_lock, lock)
    print(json.dumps({'status': lock['status'], 'locked_artifacts': len(rows),
                      'shared_distributions': len(shared['distributions']), 'lock_sha256': sha(args.output_lock)}))


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


def authorized(protocol):
    plan = read(protocol)
    if plan['id'] != ID or sha(__file__) != plan['tool_sha256']:
        raise RuntimeError('Frozen tool/protocol identity differs')
    verify_bound_helpers(plan)
    exact = {'attempts_max': 1, 'wall_seconds_max': 1200, 'cpu_threads_max': 2,
             'new_files_bytes_max': 4 * 1024**3, 'free_disk_reserve_bytes': 10 * 1024**3,
             'gpu_assignments_max': 0, 'model_constructions_max': 0,
             'parameter_creations_max': 0, 'checkpoint_loads_max': 0,
             'model_forwards_max': 0, 'official_scores_max': 0}
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
    if sha(shutil.which('uv')) != locked['resolver_uv_sha256']:
        raise RuntimeError('Pinned package installer/resolver changed')
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


def extract_python_source(row, source, site):
    count = 0
    for name, content in archive_members(source):
        relative = '/'.join(PurePosixPath(name).parts[1:])
        prefix = row['package_prefix']
        if not relative.startswith(prefix + '/'):
            continue
        target_name = relative[len(prefix.rsplit('/', 1)[0]) + 1:] if '/' in prefix else relative
        expected = row['package_files'].get(target_name)
        if expected is None or hashlib.sha256(content).hexdigest() != expected['sha256']:
            raise RuntimeError('Reviewed source subtree differs')
        target = site / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as f:
            f.write(content)
        count += 1
    if count != len(row['package_files']):
        raise RuntimeError('Incomplete source subtree installation')
    info = site / (row['name'].replace('-', '_') + '-' + row['version'] + '.dist-info')
    info.mkdir()
    text = f"Metadata-Version: 2.1\nName: {row['name']}\nVersion: {row['version']}\n"
    text += ''.join('Requires-Dist: ' + r + '\n' for r in row['requires_dist'])
    (info / 'METADATA').write_text(text)
    (info / 'INSTALLER').write_text('ffnds-reviewed-source-subtree-no-build\n')
    write(info / 'direct_url.json', {'url': row['url'], 'archive_info': {'hashes': {'sha256': row['sha256']}}})


def run(args):
    plan, lock = authorized(args.protocol)  # before any cache mutation or third-party import
    two_cpus()
    hf = Path(args.hf_root).resolve()
    directory = cache_root(hf)
    directory.mkdir(parents=True, exist_ok=True)
    attempt = directory / 'ATTEMPT.json'
    # Exclusive debit is preserved across failure; no implicit retries.
    with attempt.open('x') as f:
        json.dump({'id': ID, 'protocol_sha256': sha(args.protocol), 'attempt': 1,
                   'status': 'STARTED_NO_RETRY'}, f)
    started = time.monotonic()
    report = {'id': ID, 'status': 'RUNNING_ENVIRONMENT_AND_IMPORTS_ONLY',
              'protocol_sha256': sha(args.protocol), 'lock_sha256': plan['lock_sha256'],
              'package_artifacts': [], 'checkpoint_loads': 0, 'model_constructions': 0,
              'model_forwards': 0, 'official_scores': 0, 'gpu_assignments': 0}
    try:
        assets, _ = verify_original_sources(hf, lock)
        shared = verify_shared(lock['shared_runtime'])
        budget_check(directory, plan, started)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   NUMEXPR_NUM_THREADS='2', HF_HOME=str(hf),
                   UV_CACHE_DIR=str(directory / 'uv_cache'), UV_NO_PROGRESS='1',
                   UV_COMPILE_BYTECODE='0', UV_CONCURRENT_DOWNLOADS='2',
                   MPLCONFIGDIR=str(directory / 'mplconfig'),
                   XDG_CACHE_HOME=str(directory / 'xdg_cache'),
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   TORCH_HOME=str(directory / 'torch_cache'),
                   VBENCH_CACHE_DIR=str(hf / assets['HF_relative_root']))
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        venv = directory / 'venv'
        if venv.exists():
            raise RuntimeError('Fresh isolated venv required')
        run_checked([str(ROOT / '.venv/bin/python'), '-I', '-m', 'venv', '--without-pip', str(venv)],
                    env, directory, plan, started, directory / 'bootstrap.log')
        site = venv / 'lib/python3.11/site-packages'
        for row in lock['shared_runtime']['distributions']:
            for name in row['roots']:
                if name.endswith('.pth'):
                    continue  # guarded child uses -S; no executable .pth processing
                target = site / name
                if target.is_symlink() and target.resolve() == (shared / name).resolve():
                    continue  # shared namespace root, e.g. nvidia/cuda
                if target.exists() or target.is_symlink():
                    raise RuntimeError('Shared package root collision: ' + name)
                target.symlink_to(shared / name, target_is_directory=(shared / name).is_dir())
        artifact_dir = directory / 'artifacts'
        wheels = []
        sources = []
        for row in lock['artifacts']:
            path = download(row, artifact_dir, lambda: budget_check(directory, plan, started))
            report['package_artifacts'].append({'name': row['name'], 'sha256': sha(path), 'bytes': path.stat().st_size})
            (wheels if row['kind'] == 'wheel' else sources).append((row, path))
        shared_names = {name for r in lock['shared_runtime']['distributions'] for name in r['roots']}
        for _, path in wheels:
            with zipfile.ZipFile(path) as archive:
                if any(PurePosixPath(name).parts[0] in shared_names for name in archive.namelist()):
                    raise RuntimeError('A local wheel would overlap shared runtime files')
        # uv sees only local, hash-bound wheels; source distributions never reach a backend.
        wheel_list = directory / 'install_locked_wheels.txt'
        wheel_list.write_text('\n'.join(f"{row['name']} @ {path.as_uri()} --hash=sha256:{row['sha256']}"
                                        for row, path in wheels) + '\n')
        run_checked([shutil.which('uv'), 'pip', 'install', '--python', str(venv / 'bin/python'),
                     '--offline', '--no-index', '--no-deps', '--only-binary', ':all:',
                     '--require-hashes', '-r', str(wheel_list)], env, directory, plan, started,
                    directory / 'bootstrap.log')
        for row, path in sources:
            extract_python_source(row, path, site)
        verify_shared(lock['shared_runtime'])
        write(directory / 'INSTALLED_ARTIFACTS.json', report)
        child = [str(venv / 'bin/python'), '-I', '-S', str(Path(__file__).resolve()), '_import',
                 '--hf-root', str(hf), '--site', str(site), '--lock', str(ROOT / plan['lock_path']),
                 '--protocol', str(Path(args.protocol).resolve())]
        run_checked(child, env, directory, plan, started, directory / 'import.log')
        imported = read(directory / 'IMPORT_RESULT.json')
        if imported['status'] != 'PASS_IMPORTS_ONLY':
            raise RuntimeError('Import child did not pass')
        report.update(status='PASS_IMPORTS_ONLY', import_result_sha256=sha(directory / 'IMPORT_RESULT.json'))
    except BaseException as error:
        report.update(status='FAILED_SCOPE_DEBIT_PRESERVED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        report['new_files_bytes'] = sum(p.lstat().st_size for p in directory.rglob('*')
                                       if p.is_file() and not p.is_symlink())
        write(directory / 'RUN.json', report)
    print(json.dumps({'status': report['status'], 'elapsed_seconds': report['elapsed_seconds']}))


def import_audit_guard(directory, result):
    """Application guard; kernel/native I/O confinement is a separate boundary."""
    def denied(kind):
        def fail(*unused, **unused_kw):
            result['denied_attempts'].append(kind)
            raise RuntimeError('Import preflight denied: ' + kind)
        return fail

    def audit(event, values):
        if event.startswith('socket.') or event in ('subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn'):
            denied('NETWORK_OR_SUBPROCESS:' + event)()
        if event == 'open' and isinstance(values[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(values[0])).resolve()
            mode = values[1] or ''
            flags = values[2] or 0
            writing = any(c in mode for c in 'wax+') or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if writing and not path.is_relative_to(directory):
                denied('WRITE_OUTSIDE_SCOPE')()
            if not writing and (path.suffix.lower() in ('.pt', '.pth', '.ckpt', '.safetensors', '.bin', '.npz')
                                or '/R17_metric_assets/dreamsim/' in str(path)):
                denied('CHECKPOINT_FILE_READ')()
        if event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.rename', 'os.symlink', 'os.link', 'os.chmod', 'os.truncate'):
            candidates = values[:2] if event in ('os.rename', 'os.link', 'os.symlink') else values[:1]
            for value in candidates:
                if isinstance(value, (str, bytes, os.PathLike)) and not Path(os.fsdecode(value)).resolve().is_relative_to(directory):
                    denied('FILESYSTEM_MUTATION_OUTSIDE_SCOPE')()
    return denied, audit


def import_preflight(args):
    # No site initialization: all imports come from this explicit isolated path.
    sys.dont_write_bytecode = True
    site = Path(args.site).resolve()
    sys.path.insert(0, str(site))
    directory = cache_root(args.hf_root)
    if site != (directory / 'venv/lib/python3.11/site-packages').resolve():
        raise RuntimeError('Only the fixed isolated metric site may be imported')
    plan, lock = authorized(args.protocol)
    with (directory / 'IMPORT_ATTEMPT.json').open('x') as f:
        json.dump({'id': ID, 'attempt': 1, 'status': 'DEBIT_BEFORE_IMPORTS'}, f)
    result = {'id': ID, 'status': 'STARTED_IMPORTS_ONLY', 'denied_attempts': [], 'functions': [],
              'checkpoint_loads': 0, 'model_constructions': 0, 'model_forwards': 0,
              'official_scores': 0, 'CUDA_initialized': False}
    denied, audit = import_audit_guard(directory, result)

    sys.addaudithook(audit)
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise RuntimeError('Every GPU must be hidden')
        if Path(args.lock).resolve() != (ROOT / plan['lock_path']).resolve():
            raise RuntimeError('Child lock path differs')
        assets, supplement = verify_original_sources(Path(args.hf_root), lock)
        shared = verify_shared(lock['shared_runtime'])
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
        torch.nn.Module.__init__ = denied('NN_MODULE_CONSTRUCTION')
        torch.nn.Parameter.__new__ = staticmethod(denied('PARAMETER_CREATION'))
        torch.nn.UninitializedParameter.__new__ = staticmethod(denied('PARAMETER_CREATION'))
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
        result['distribution_versions'] = {r['name']: metadata.version(r['name'])
                                           for r in lock['artifacts'] + lock['shared_runtime']['distributions']}
        for row in lock['artifacts'] + lock['shared_runtime']['distributions']:
            if result['distribution_versions'][row['name']] != row['version']:
                raise RuntimeError('Installed distribution version differs from lock')
        actual_rows = []
        for row in lock['artifacts']:
            actual_rows.append(dict(row, requires_dist=metadata.distribution(row['name']).requires or []))
        validate_closure(actual_rows, lock['shared_runtime'])
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
            elif path.is_relative_to(directory):
                origin = {'kind': 'ISOLATED_METRIC_ENV', 'HF_relative': str(path.relative_to(args.hf_root))}
            elif path.is_relative_to(source):
                origin = {'kind': 'PINNED_ORIGINAL_SOURCE_VIEW', 'HF_relative': str(path.relative_to(args.hf_root))}
            else:
                raise RuntimeError('Unexpected module import origin: ' + name)
            result['module_origins'][name] = origin
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
    resolve_parser = sub.add_parser('resolve')
    resolve_parser.add_argument('--output-lock', required=True)
    run_parser = sub.add_parser('run')
    run_parser.add_argument('--protocol', required=True)
    internal = sub.add_parser('_import')
    for key in ('site', 'lock', 'protocol'):
        internal.add_argument('--' + key, required=True)
    for item in (resolve_parser, run_parser, internal):
        item.add_argument('--hf-root', default=os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface')))
    args = parser.parse_args()
    {'resolve': resolve, 'run': run, '_import': import_preflight}[args.command](args)


if __name__ == '__main__':
    main()
