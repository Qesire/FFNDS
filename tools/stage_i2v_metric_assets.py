"""Stage and hash official nine-dimension evaluator assets in HF cache.

Downloads/copies only: no torch import, model construction, forward or CUDA.
The generation environment and existing evaluation environment are untouched.
"""
from concurrent.futures import ThreadPoolExecutor
import ast
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import urllib.request
import zipfile

from rollout_contract import ROOT, atomic, read, sha


def request_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'FFNDS-evaluator-preparation'}), timeout=45) as response:
        return json.load(response)


def download(url, target, expected=None):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix(target.suffix + '.incomplete')
        offset = temporary.stat().st_size if temporary.exists() else 0
        headers = {'User-Agent': 'FFNDS-evaluator-preparation'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
            mode = 'ab' if offset and response.status == 206 else 'wb'
            with temporary.open(mode) as stream:
                written = offset if mode == 'ab' else 0
                announced = written // (64 << 20)
                for block in iter(lambda: response.read(4 << 20), b''):
                    stream.write(block)
                    written += len(block)
                    if written > 4 << 30:
                        raise RuntimeError('Single evaluator asset exceeds frozen staging limit')
                    if written // (64 << 20) > announced:
                        announced = written // (64 << 20)
                        print(f'Staging {target.name}: {written//(1<<20)}MiB', flush=True)
        if expected and sha(temporary) != expected:
            raise RuntimeError('Published evaluator checksum mismatch')
        temporary.replace(target)
    actual = sha(target)
    if expected and actual != expected:
        raise RuntimeError('Cached evaluator checksum mismatch')
    return {'url': url, 'sha256': actual, 'bytes': target.stat().st_size}


def hardlink(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(target) != sha(source):
            raise RuntimeError('Existing staged evaluator asset differs')
    else:
        try:
            target.hardlink_to(source.resolve())
        except OSError:
            shutil.copyfile(source, target)


def stage_repo(owner_repo, target):
    target = Path(target)
    receipt = target / 'SOURCE.json'
    if receipt.exists():
        data = read(receipt)
        for item in data['files']:
            if sha(target / item['path']) != item['sha256']:
                raise RuntimeError('Pinned evaluator source changed')
        return data
    commit = request_json(f'https://api.github.com/repos/{owner_repo}/commits/main')['sha']
    tree = request_json(f'https://api.github.com/repos/{owner_repo}/git/trees/{commit}?recursive=1')
    if tree.get('truncated'):
        raise RuntimeError('Incomplete evaluator source inventory')
    selected = [row['path'] for row in tree['tree'] if row['type'] == 'blob' and
                (row['path'].endswith(('.py', '.yaml', '.json')) or row['path'] == 'LICENSE')]
    target.mkdir(parents=True, exist_ok=True)
    def one(relative):
        path = target / relative
        info = download(f'https://raw.githubusercontent.com/{owner_repo}/{commit}/{relative}', path)
        return {'path': relative, 'sha256': info['sha256'], 'bytes': info['bytes']}
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(one, selected))
    data = {'repository': f'https://github.com/{owner_repo}', 'commit': commit, 'files': files,
            'source_only_no_execution': True}
    atomic(receipt, data)
    return data


def source_snapshot(package_name, destination):
    package = Path(importlib.util.find_spec(package_name).origin).parent
    destination = Path(destination)
    files = []
    for source in sorted(package.rglob('*')):
        if not source.is_file() or '__pycache__' in source.parts or source.suffix == '.pyc':
            continue
        if source.stat().st_size > 64 << 20:
            raise RuntimeError('Unexpected model tensor in installed evaluator source package')
        relative = source.relative_to(package)
        target = destination / relative
        if target.exists() and sha(target) != sha(source):
            raise RuntimeError('Existing source snapshot differs')
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        files.append({'path': f'{package_name}/{relative}', 'bytes': source.stat().st_size, 'sha256': sha(source)})
    return files


def main():
    from worker_identity import require_worker
    require_worker()
    hf = Path(os.environ['HF_HOME'])
    out = hf / 'derived/ffnds/R17_metric_assets'
    manifest = ROOT / '.runtime/R17_metric_assets.json'
    if manifest.exists():
        raise RuntimeError('Preserve the first complete evaluator asset registration')
    out.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out).free < 90 << 30:
        raise RuntimeError('Preserve80GiB reserve plus evaluator preparation allowance')
    assets = []
    old = read(ROOT / '.runtime/vbench_models.json')
    for item in old['models']:
        if sha(item['path']) != item['sha256']:
            raise RuntimeError('Original three-metric checkpoint changed')
        target = out / item['relative']
        hardlink(item['path'], target)
        assets.append({'name': item['name'], 'HF_relative': str(target.relative_to(hf)),
                       'sha256': sha(target), 'bytes': target.stat().st_size, 'origin': 'previous_pinned_official_VBench3_asset'})
    source_view = out / 'installed_source_view'
    inventory = []
    for package in ('vbench', 'vbench2_beta_i2v', 'dreamsim'):
        inventory += source_snapshot(package, source_view / package)
    dino = stage_repo('facebookresearch/dino', out / 'source/dino')
    cotracker = stage_repo('facebookresearch/co-tracker', out / 'source/cotracker')
    urls = [
        ('DINO_base', 'dino_model/dino_vitbase16_pretrain.pth',
         'https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth', None),
        ('CLIP_B32', 'clip_model/ViT-B-32.pt',
         'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt',
         '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'),
        ('DreamSim_ensemble_archive', 'downloads/dreamsim_ensemble_checkpoint.zip',
         'https://github.com/ssundaram21/dreamsim/releases/download/v0.2.0-checkpoints/dreamsim_ensemble_checkpoint.zip', None),
        ('RAFT_official_archive', 'downloads/raft-models.zip',
         'https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip', None),
        ('CoTracker2', 'torch/checkpoints/cotracker2.pth',
         'https://huggingface.co/facebook/cotracker/resolve/60495999ebd33af25fc51d89f83d1d6de3bc18a5/cotracker2.pth',
         '362f5274376d610dc987b6daf2c2fefe63e06e1835f4ec1a10d0a15c5a4eef4f')]
    def asset(item):
        name, relative, url, expected = item
        target = out / relative
        record = download(url, target, expected)
        return {'name': name, 'HF_relative': str(target.relative_to(hf)), **record}
    with ThreadPoolExecutor(max_workers=4) as pool:
        assets += list(pool.map(asset, urls))
    for archive, directory in [('downloads/dreamsim_ensemble_checkpoint.zip', 'dreamsim'),
                                ('downloads/raft-models.zip', 'raft_model')]:
        destination = out / directory
        destination.mkdir(exist_ok=True)
        with zipfile.ZipFile(out / archive) as stream:
            if sum(item.file_size for item in stream.infolist()) > 6 << 30:
                raise RuntimeError('Evaluator archive expands beyond asset budget')
            for item in stream.infolist():
                relative = PurePosixPath(item.filename)
                if relative.is_absolute() or '..' in relative.parts:
                    raise RuntimeError('Unexpected evaluator archive path')
                if item.is_dir():
                    continue
                target = destination.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    with stream.open(item) as source, target.open('wb') as dest:
                        shutil.copyfileobj(source, dest, 4 << 20)
                assets.append({'name': f'{directory}:{item.filename}', 'HF_relative': str(target.relative_to(hf)),
                               'sha256': sha(target), 'bytes': target.stat().st_size, 'archive_sha256': sha(out / archive)})
    hardlink(out / 'dino_model/dino_vitbase16_pretrain.pth', out / 'dreamsim/checkpoints/dino_vitbase16_pretrain.pth')
    hardlink(out / 'dino_model/dino_vitbase16_pretrain.pth', out / 'torch/checkpoints/dino_vitbase16_pretrain.pth')
    total = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    if total > 10 << 30:
        raise RuntimeError('Logical evaluator cache exceeds10GiB preparation budget')
    report = {'id': 'R17_OFFICIAL_I2V_EVALUATOR_ASSETS', 'status': 'STAGED_NO_MODEL_CONSTRUCTION_OR_EXECUTION',
              'source_sha256': sha(__file__), 'assets': assets, 'installed_source_inventory': inventory,
              'source_repositories': {'dino': dino, 'cotracker': cotracker},
              'versions': {n: importlib.metadata.version(n) for n in ('vbench', 'dreamsim', 'torch', 'torchvision', 'pyiqa', 'timm')},
              'HF_relative_root': str(out.relative_to(hf)), 'logical_cache_bytes': total,
              'model_constructions': 0, 'model_forwards': 0, 'model_execution_authority': False,
              'remaining_readiness': 'Bind offline cache loader adapters, source-specific nine-dimension manifests and numeric execution controls before separately authorized evaluator jobs.'}
    atomic(manifest, report)
    atomic(out / 'ASSETS.json', report)
    atomic(ROOT / 'research/20261023/R17_metric_asset_preparation.json', report)
    print(json.dumps({'status': report['status'], 'assets': len(assets), 'logical_cache_bytes': total}), flush=True)


if __name__ == '__main__':
    main()
