"""Freeze a deterministic source-disjoint20 supplement for the frozen UB2."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

from huggingface_hub import hf_hub_download
from PIL import Image

from ar2_checkpoint import atomic_json, digest
from fetch_formal_inputs import ROOT, REPOSITORY, REVISION, ARCHIVE_FILE, ARCHIVE_SHA


def main():
    protocol_path = ROOT / 'config/ub2_confirmation20.json'
    protocol = json.loads(protocol_path.read_text())
    run_root = ROOT / 'runs/worker_restart_20261020'
    policy_path = run_root / 'terminal_dense_ub2_fit/UB2_policy.json'
    if (digest(policy_path) != protocol['frozen_candidate_policy_sha256']
            or digest(ROOT / 'tools/terminal_dense_runtime.py') != protocol['frozen_runtime_source_sha256']
            or digest(ROOT / 'tools/trained_packed_runtime.py') != protocol['frozen_parent_runtime_source_sha256']):
        raise RuntimeError('Exact pre-quality frozen UB2 candidate required')
    metadata_path = Path(hf_hub_download(REPOSITORY, protocol['metadata_file'], revision=REVISION))
    if digest(metadata_path) != protocol['metadata_sha256']:
        raise RuntimeError('Pinned metadata differs')
    metadata = json.loads(metadata_path.read_text())
    prior_paths = [ROOT / f'.runtime/inputs/{name}/canonical.json' for name in ['formal20', 'calibration24', 'disjoint20']]
    prior = [r for p in prior_paths for r in json.loads(p.read_text())]
    names = {r.get('image_name', r['input_provenance'].get('image_name')) for r in prior}
    prompts = {r['prompt'] for r in prior}
    groups = {r['source_group'] for r in prior}
    image_sha = {r['input_provenance']['image_sha256'] for r in prior}
    original12 = ROOT / '.runtime/inputs/calibration12/recovery_report.json'
    image_sha.update(r['image_sha256'] for r in json.loads(original12.read_text())['records'])
    research_paths = [ROOT / f'.runtime/inputs/task_{task:03d}_seed_20261001.json' for task in [0, 6]]
    image_sha.update(digest(json.loads(p.read_text())[0]['image_path']) for p in research_paths)
    selected = []
    for dimension, kind in [('i2v_subject', 'animal'), ('i2v_background', 'architecture')]:
        source_rows = [(i, r) for i, r in enumerate(metadata) if dimension in r['dimension']]
        eligible = []
        for filtered_index, (source_index, r) in enumerate(source_rows):
            group = f'{dimension}:{filtered_index}'
            if (r['image_type'] == kind and group not in groups and r['image_name'] not in names
                    and r['prompt_en'] not in prompts):
                eligible.append({**r, 'dimension': dimension, 'filtered_index': filtered_index,
                                 'source_index': source_index, 'source_group': group})
        if len(eligible) < 10:
            raise RuntimeError('Insufficient disjoint sources; stop without replacement')
        selected.extend(eligible[:10])
    wanted = {r['image_name'] for r in selected}
    if len(wanted) != 20 or len({r['source_group'] for r in selected}) != 20:
        raise RuntimeError('Duplicate selected image/source identities')
    archive = Path(hf_hub_download(REPOSITORY, ARCHIVE_FILE, revision=REVISION))
    if digest(archive) != ARCHIVE_SHA:
        raise RuntimeError('Pinned archive differs')
    cache = Path.home() / '.cache/huggingface/ffnds-derived/vbench-i2v' / REVISION / 'crop/16-9'
    cache.mkdir(parents=True, exist_ok=True)
    found = {}
    with tarfile.open(archive, 'r|gz') as handle:
        for member in handle:
            parts = PurePosixPath(member.name).parts
            if len(parts) < 3 or parts[-3:-1] != ('crop', '16-9') or parts[-1] not in wanted:
                continue
            name = parts[-1]
            if name in found or not member.isfile() or not 0 < member.size < 64 << 20:
                raise RuntimeError('Duplicate/invalid archive member')
            content = handle.extractfile(member).read((64 << 20) + 1)
            if len(content) != member.size:
                raise RuntimeError('Truncated archive image')
            with Image.open(io.BytesIO(content)) as image:
                size = list(image.size)
                image.verify()
            if abs(size[0] / size[1] - 16 / 9) > .02:
                raise RuntimeError('Source is not the registered16:9 crop')
            sha = hashlib.sha256(content).hexdigest()
            if sha in image_sha or sha in {v['sha256'] for v in found.values()}:
                raise RuntimeError('Source image SHA overlap; stop without replacement')
            path = cache / name
            if path.exists() and digest(path) != sha:
                raise RuntimeError('Existing HF derived crop differs')
            if not path.exists():
                temp = Path(str(path) + '.tmp')
                temp.write_bytes(content)
                temp.replace(path)
            found[name] = {'path': str(path), 'sha256': sha, 'size': size, 'archive_member': member.name}
    if set(found) != wanted:
        raise RuntimeError('Missing selected archive members')
    out = ROOT / '.runtime/inputs/confirmation20'
    out.mkdir(parents=True, exist_ok=True)
    canonical = []
    for index, row in enumerate(selected):
        image = found[row['image_name']]
        canonical.append({'task_id': 30000 + index, 'seed': 42, 'negative_prompt': '', 'prompt': row['prompt_en'],
                          'prompt_sha256': hashlib.sha256(row['prompt_en'].encode()).hexdigest(),
                          'source_group': row['source_group'], 'source_index': row['source_index'],
                          'filtered_index': row['filtered_index'], 'dimension': row['dimension'],
                          'image_name': row['image_name'], 'image_type': row['image_type'],
                          'image_path': image['path'], 'role': 'prospective_source_disjoint_confirmation',
                          'input_provenance': {'repository': REPOSITORY, 'revision': REVISION,
                                               'metadata_sha256': digest(metadata_path), 'archive_sha256': ARCHIVE_SHA,
                                               'image_sha256': image['sha256'], 'archive_member': image['archive_member']}})
    target = out / 'canonical.json'
    if target.exists() and json.loads(target.read_text()) != canonical:
        raise RuntimeError('Confirmation20 already frozen differently')
    atomic_json(target, canonical)
    report = {'status': 'PASS_FROZEN_PROSPECTIVE_CONFIRMATION20_INPUTS', 'protocol_sha256': digest(protocol_path),
              'source_sha256': digest(Path(__file__)), 'canonical_sha256': digest(target),
              'candidate_policy_sha256': digest(policy_path),
              'exclusion_bindings': {str(p): digest(p) for p in [*prior_paths, original12, *research_paths]},
              'metadata_sha256': digest(metadata_path), 'archive_sha256': ARCHIVE_SHA,
              'source_count': 20, 'image_sha_overlap': [], 'source_name_prompt_overlap': [],
              'records': [{'index': i, 'source_group': r['source_group'], 'image_sha256': r['input_provenance']['image_sha256']}
                          for i, r in enumerate(canonical)], 'model_execution': False, 'quality_or_speed_acceptance': False}
    atomic_json(out / 'recovery_report.json', report)
    print(json.dumps({'status': report['status'], 'sources': [r['source_group'] for r in canonical]}))


if __name__ == '__main__':
    main()
