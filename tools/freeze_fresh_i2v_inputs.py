"""Freeze unseen source identities for the unchanged full160 operator.

Only metadata, image bytes and dimensions are read. No model/evaluator runs.
The official stratum/metric assignment is preserved; camera sources are
distinct images rather than seven instructions on one pseudo-independent image.
"""
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile

from fetch_formal_inputs import ROOT, REPOSITORY, REVISION, ARCHIVE_FILE, ARCHIVE_SHA
from rollout_contract import atomic, read, sha

SUBJECT_TYPES = {'animal': 2, 'single-human': 2, 'multiple-human': 1,
                 'transportation': 1, 'food': 1, 'plant': 1}
BACKGROUND_TYPES = {'architecture': 2, 'indoor': 2, 'scenery': 2, 'abstract': 2}
CAMERA = ['camera pans left', 'camera pans right', 'camera tilts up', 'camera tilts down',
          'camera zooms in', 'camera zooms out', 'camera static']
CORE = ['i2v_subject', 'i2v_background', 'camera_motion', 'subject_consistency',
        'background_consistency', 'motion_smoothness', 'dynamic_degree',
        'aesthetic_quality', 'imaging_quality']


def collect_exposure(value, seen):
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str):
                if key in ('image_name',):
                    seen['names'].add(item)
                elif key == 'image_sha256':
                    seen['image_SHA'].add(item)
                elif key in ('prompt', 'prompt_en'):
                    seen['prompts'].add(item)
                elif key == 'source_group':
                    seen['groups'].add(item)
                elif key in ('image_path', 'img_path') and Path(item).is_file():
                    seen['image_SHA'].add(sha(item))
            collect_exposure(item, seen)
    elif isinstance(value, list):
        for item in value:
            collect_exposure(item, seen)


def select_rows(metadata, seen):
    selected = []
    used_names, used_prompts = set(seen['names']), set(seen['prompts'])
    for dimension, kinds in [('i2v_subject', SUBJECT_TYPES), ('i2v_background', BACKGROUND_TYPES)]:
        scope = [(i, row) for i, row in enumerate(metadata) if dimension in row['dimension']]
        for kind, count in kinds.items():
            eligible = []
            for filtered, (source_index, row) in enumerate(scope):
                group = f'{dimension}:{filtered}'
                if (row['image_type'] == kind and row['image_name'] not in used_names and
                        row['prompt_en'] not in used_prompts and group not in seen['groups']):
                    eligible.append((source_index, filtered, group, row))
            if len(eligible) < count:
                raise RuntimeError('Not enough unseen images in a frozen content stratum')
            for source_index, filtered, group, row in eligible[:count]:
                selected.append({**row, 'source_index': source_index, 'filtered_index': filtered,
                                 'source_group': group, 'stratum': dimension})
                used_names.add(row['image_name'])
                used_prompts.add(row['prompt_en'])
    scope = [(i, row) for i, row in enumerate(metadata) if 'camera_motion' in row['dimension']]
    for motion in CAMERA:
        eligible = [(i, filtered, row) for filtered, (i, row) in enumerate(scope)
                    if motion in row['prompt_en'].lower() and row['image_name'] not in used_names and
                    row['prompt_en'] not in used_prompts and f'camera_motion:{filtered}' not in seen['groups']]
        if not eligible:
            raise RuntimeError('Not enough source-disjoint images for each camera instruction')
        source_index, filtered, row = eligible[0]
        selected.append({**row, 'source_index': source_index, 'filtered_index': filtered,
                         'source_group': f'camera_motion:{filtered}', 'stratum': 'camera_motion',
                         'camera_instruction': motion})
        used_names.add(row['image_name'])
        used_prompts.add(row['prompt_en'])
    if len(selected) != 23 or len({r['image_name'] for r in selected}) != 23:
        raise RuntimeError('Require23distinctunderlyingimages')
    return selected


def main():
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from worker_identity import require_worker
    require_worker()
    out = ROOT / '.runtime/inputs/r17_fresh23'
    public = ROOT / 'research/20261023/R17_fresh23_input_audit.json'
    if out.exists() or public.exists():
        raise RuntimeError('Fresh cohort is immutable; preserve the first selection')
    seen = {k: set() for k in ('names', 'image_SHA', 'prompts', 'groups')}
    exposure = []
    for path in sorted((ROOT / '.runtime/inputs').rglob('*.json')):
        value = read(path)
        collect_exposure(value, seen)
        exposure.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path)})
    config = read(ROOT / 'config/calibration24_ffn_operands.json')
    meta_path = Path(hf_hub_download(REPOSITORY, config['metadata_file'], revision=REVISION, local_files_only=True))
    if sha(meta_path) != config['metadata_sha256']:
        raise RuntimeError('Pinned official source assignment metadata changed')
    metadata = read(meta_path)
    selected = select_rows(metadata, seen)
    wanted = {r['image_name'] for r in selected}
    archive = Path(hf_hub_download(REPOSITORY, ARCHIVE_FILE, revision=REVISION, local_files_only=True))
    if sha(archive) != ARCHIVE_SHA:
        raise RuntimeError('Pinned source image archive checksum mismatch')
    hf = Path(os.environ['HF_HOME'])
    cache = hf / 'derived/ffnds/R17_fresh23_inputs'
    if cache.exists():
        raise RuntimeError('Preserve existing staged input selection')
    cache.mkdir(parents=True)
    images = {}
    with tarfile.open(archive, 'r|gz') as stream:
        for member in stream:
            parts = PurePosixPath(member.name).parts
            if len(parts) < 3 or parts[-3:-1] != ('crop', '16-9') or parts[-1] not in wanted:
                continue
            name = parts[-1]
            if name in images or not member.isfile() or not 0 < member.size < 64 << 20:
                raise RuntimeError('Duplicate/nonregular/oversized requested image')
            data = stream.extractfile(member).read((64 << 20) + 1)
            if len(data) != member.size:
                raise RuntimeError('Truncated image')
            image_sha = hashlib.sha256(data).hexdigest()
            if image_sha in seen['image_SHA'] or image_sha in {r['sha256'] for r in images.values()}:
                raise RuntimeError('Image-byte overlap: preserve failure without replacing selected source')
            with Image.open(io.BytesIO(data)) as img:
                size = list(img.size)
                img.verify()
            if abs(size[0] / size[1] - 16 / 9) > .02:
                raise RuntimeError('Official16:9croprequired')
            path = cache / f'source{len(images):02d}' / 'input.jpg'
            path.parent.mkdir()
            path.write_bytes(data)
            images[name] = {'path': path, 'sha256': image_sha, 'size': size, 'archive_member': member.name}
    if set(images) != wanted:
        raise RuntimeError('Selected images missing from the original archive')
    out.mkdir(parents=True)
    canonical, public_rows = [], []
    for index, row in enumerate(selected):
        img = images[row['image_name']]
        sample = {'index': index, 'seed': 42, 'prompt': row['prompt_en'], 'image_path': str(img['path']),
                  'source_group': row['source_group'], 'source_index': row['source_index'],
                  'image_type': row['image_type'], 'image_name': row['image_name'], 'stratum': row['stratum'],
                  'official_dimensions': row['dimension'],
                  'core_dimensions': [d for d in row['dimension'] if d in CORE],
                  'camera_instruction': row.get('camera_instruction'),
                  'input_provenance': {'repository': REPOSITORY, 'revision': REVISION,
                                       'metadata_sha256': sha(meta_path), 'archive_sha256': ARCHIVE_SHA,
                                       'image_sha256': img['sha256'], 'archive_member': img['archive_member']}}
        canonical.append(sample)
        public_rows.append({**{k: v for k, v in sample.items() if k != 'image_path'},
                            'image_HF_relative': str(img['path'].relative_to(hf)), 'image_size': img['size']})
    atomic(out / 'canonical.json', canonical)
    report = {'id': 'R17_SOURCE_DISJOINT_FRESH23', 'status': 'PASS_NEW23_SOURCE_METADATA_AND_IMAGE_BYTE_AUDIT',
              'frozen_method_commit': '917cedc07c10189294187029b04aa06668486000',
              'source_sha256': sha(__file__), 'canonical_sha256': sha(out / 'canonical.json'),
              'source_count': 23, 'unique_image_SHA_count': 23,
              'strata': dict(Counter(r['stratum'] for r in canonical)),
              'per_core_dimension_source_count': {d: sum(d in r['core_dimensions'] for r in canonical) for d in CORE},
              'exposure_inventory': exposure, 'prior_exposure_counts': {k: len(v) for k, v in seen.items()},
              'prior_image_SHA_overlap': [], 'prior_name_prompt_group_overlap': [],
              'inputs': public_rows, 'model_or_evaluator_execution': False,
              'new_model_generations': 0, 'no_candidate_outcomes_used_for_selection': True,
              'independence_unit': 'underlying source image;camera instructions use seven different images',
              'selection_rule': {'subject': SUBJECT_TYPES, 'background': BACKGROUND_TYPES, 'camera': CAMERA,
                                 'order': 'firstunusedoriginalmetadataorderwithinpredeclaredstrata;bytecollisionfailswithoutreplacement'},
              'limitations': 'Prospective small stratified cohort,not fullofficialsuite or finalpaperquality. Official dimension assignment preserved; nine columns do not share a fabricated identical source population. No model authority is granted.'}
    atomic(public, report)
    atomic(cache / 'INPUT_AUDIT.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'source_count', 'strata', 'per_core_dimension_source_count', 'new_model_generations')}))


if __name__ == '__main__':
    main()
