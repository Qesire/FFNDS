"""Restore official source omitted from the wheel, without package changes."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

from rollout_contract import ROOT, atomic, sha
from stage_i2v_metric_assets import request_json, download


def main():
    revision = 'fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490'
    tree_sha = '1c671f135a20ac2c8ec9ea90df6d4c5943a07e51'
    tree = request_json(f'https://api.github.com/repos/Vchitect/VBench/git/trees/{tree_sha}?recursive=1')
    if tree.get('truncated'):
        raise RuntimeError('Incomplete original visualizer/helper source tree')
    hf = Path(os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface')))
    root = hf / 'derived/ffnds/R17_metric_source_supplements'
    prefix = 'vbench2_beta_i2v/third_party/cotracker'
    selected = [v for v in tree['tree'] if v['type'] == 'blob' and v['path'].endswith(('.py', '.yaml', '.json'))]
    def one(row):
        relative = prefix + '/' + row['path']
        path = root / relative
        result = download(f'https://raw.githubusercontent.com/Vchitect/VBench/{revision}/{relative}', path)
        data = path.read_bytes()
        if hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() != row['sha']:
            raise RuntimeError('Original Git tree source blob mismatch')
        return {'path': relative, 'sha256': result['sha256'], 'bytes': len(data), 'git_blob_sha': row['sha']}
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(one, selected))
    report = {'id': 'R17_OFFICIAL_WHEEL_SOURCE_SUPPLEMENT', 'status': 'PASS_EXACT_PINNED_ORIGINAL_SOURCE_NO_MODELS',
              'upstream': 'https://github.com/Vchitect/VBench', 'revision': revision, 'git_tree_sha': tree_sha,
              'source_sha256': sha(__file__), 'HF_relative_root': str(root.relative_to(hf)), 'files': files,
              'reason': 'vbench0.1.5 wheel camera module imports third_party.cotracker.utils.visualizer, but wheel omits third_party. Original source namespace supplied without modifying score equations or installed packages.',
              'model_constructions': 0, 'model_forwards': 0}
    target = ROOT / 'research/20261023/R17_metric_source_supplement.json'
    if target.exists():
        raise RuntimeError('Preserve the first source supplement')
    atomic(target, report)
    atomic(root / 'SOURCE.json', report)
    print(json.dumps({'status': report['status'], 'source_files': len(files)}), flush=True)


if __name__ == '__main__':
    main()
