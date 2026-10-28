"""CPU-only current-worker proof and exact isolated NVFP4 runtime staging."""
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import random
import struct
import sys
import tarfile
import time

import nvfp4_checkpoint_byte_producer as checkpoint
import worker_identity

ROOT = Path(__file__).resolve().parents[1]
CPU_ID = 'E0_NVFP4_REAL_FFN_INPUTS_CPU_20261029'
BUILD_ID = 'E0_SM120_LIGHTX2V_KERNEL_BUILD_CPU_20261029'
EXPORT_ID = 'E0_SM120_KERNEL_OVERLAY_EXPORT_20261029'
AUDIT_ID = 'E0_RAW_BF16_FULL_DEPTH_ASSET_AUDIT_20261029'
ELF_SHA = '474675037c50a97589731cae5225ddf23f9264803205cd538369a75233a3bb88'
NAMES = ('__init__.py', 'gemm.py', 'kv_cache.py', 'utils.py', 'version.py', 'common_ops.abi3.so')
CPU_RESOURCES = {'GPU': 0, 'CPU_threads': 2, 'wall_seconds': 1800, 'outer_seconds': 1860,
    'checkpoint_full_SHA_bytes': 16759667184, 'new_HF_bytes': 32 << 20,
    'node_local_software_cache_bytes': 4 << 20, 'automatic_retry': False}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def binding(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    need(path.is_relative_to(root), 'Binding escaped root')
    data = path.read_bytes()
    return {'path': str(path.relative_to(root)), 'bytes': len(data), 'sha256': sha(data)}


def bound(root, item):
    root = Path(root).resolve(); relative = Path(item['path'])
    need(not relative.is_absolute() and '..' not in relative.parts, 'Invalid bound relative path')
    path = (root/relative).resolve()
    need(path.is_relative_to(root), 'Bound path escaped root')
    data = path.read_bytes()
    need(len(data) == item['bytes'] and sha(data) == item['sha256'], 'Bound file bytes/SHA changed: '+str(relative))
    return path


def stat_record(path):
    value = Path(path).stat()
    return {'device': value.st_dev, 'inode': value.st_ino, 'bytes': value.st_size,
        'mtime_ns': value.st_mtime_ns, 'ctime_ns': value.st_ctime_ns}


def exited(value, run, digest, identity):
    need(value.get('id') == identity and value.get('protocol_sha256') == digest
        and value.get('run_sha256') == run['sha256'] and value.get('process_exited') is True
        and type(value.get('child_returncode')) is int and value['child_returncode'] == 0
        and value.get('timeout') is False and value.get('automatic_retry') is False, 'Actual independent exit0 required')


def closed(proof, hf, identity, success):
    need(isinstance(proof, dict) and {'protocol', 'RUN', 'exit'} <= set(proof), 'Closed proof NOT_BOUND')
    protocol = read(bound(ROOT, proof['protocol'])); run = read(bound(hf, proof['RUN']))
    exited(read(bound(hf, proof['exit'])), proof['RUN'], proof['protocol']['sha256'], identity)
    need(protocol.get('id') == run.get('id') == identity and run.get('protocol_sha256') == proof['protocol']['sha256']
        and run.get('status') == success, 'Closed successful parent identity differs')
    return protocol, run


def private_worker_sha():
    worker_identity.require_worker()
    return sha(worker_identity.private_config_path().read_bytes())


def expected_package(build):
    wrappers = {Path(item['path']).name: item for item in build['kernel_sources'] if '/python/lightx2v_kernel/' in item['path']}
    result = {'lightx2v_kernel/'+name: {key: wrappers[name][key] for key in ('bytes', 'sha256')}
        for name in NAMES if not name.endswith('.so')}
    result['lightx2v_kernel/common_ops.abi3.so'] = {'bytes': 1312256, 'sha256': ELF_SHA}
    return result


def extract_package(raw, expected, destination):
    """Six regular files only; never extract arbitrary archive paths or links."""
    destination = Path(destination).resolve()
    need(not destination.exists(), 'Fresh isolated node-local software cache required')
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:') as archive:
        members = archive.getmembers()
        need(len(members) == len(expected) and {m.name for m in members} == set(expected), 'Exact six-file overlay archive required')
        payloads = {}
        for member in members:
            path = Path(member.name)
            need(member.isfile() and not path.is_absolute() and '..' not in path.parts
                and member.size == expected[member.name]['bytes'], 'Unsafe or wrong-sized overlay member')
            payload = archive.extractfile(member).read()
            need(len(payload) == member.size and sha(payload) == expected[member.name]['sha256'], 'Overlay member SHA differs')
            if member.name.endswith('.so'):
                need(payload[:4] == b'\x7fELF', 'Native kernel is not ELF')
            payloads[member.name] = payload
    destination.mkdir(parents=True)
    for name, payload in payloads.items():
        path = destination/name; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            need(stream.write(payload) == len(payload), 'Short isolated package write')
    return [{'path': str((destination/name).resolve()), **value, 'stat': stat_record(destination/name)} for name, value in sorted(expected.items())]


def packet_v1(hf, entry, ordinal, step):
    """Validate the full original packet; retain V1 and its actual I64 rows."""
    item = entry['payload']; relative = Path(item['path'])
    need(not relative.is_absolute() and '..' not in relative.parts, 'Invalid source packet path')
    path = (hf/relative).resolve(); need(path.is_relative_to(hf.resolve()), 'Packet escaped HF')
    raw = path.read_bytes(); need(len(raw) == item['bytes'] and sha(raw) == item['sha256'], 'Original packet bytes/SHA changed')
    n = struct.unpack('<Q', raw[:8])[0]
    need(0 < n <= 1 << 20 and len(raw) >= 8+n, 'Saved operand header length differs')
    header = json.loads(raw[8:8+n]); body = raw[8+n:]
    shapes = {'V1': [256, 5120], 'h': [256, 13824], 'Y': [256, 5120], 'r': [256, 5120], 'u': [256, 5120],
        'gate': [5120], 'shift': [5120], 'scale': [5120], 'rows': [256]}
    need(set(header)-{'__metadata__'} == set(entry['tensor_metadata']) == set(shapes), 'Original packet tensor names differ')
    result, end = {}, 0
    for name in sorted(shapes, key=lambda key: header[key]['data_offsets'][0]):
        info, meta = header[name], entry['tensor_metadata'][name]; a, b = info['data_offsets']
        dtype = 'I64' if name == 'rows' else 'BF16'; size = math.prod(shapes[name])*(8 if name == 'rows' else 2)
        need(info['dtype'] == dtype and info['shape'] == meta['shape'] == shapes[name] and a == end and b-a == size, 'Packet tensor geometry differs')
        data = body[a:b]; need(len(data) == size and sha(data) == meta['raw_tensor_sha256'], 'Packet tensor raw SHA changed')
        if name in ('V1', 'rows'):
            result[name] = data
        end = b
    need(end == len(body), 'Packet payload trailing bytes')
    seed = 42+(ordinal+1)*1000003+1009*step
    expected = sorted(random.Random(seed).sample(range(31668), 512)[:256])
    need(list(struct.unpack('<256q', result['rows'])) == expected, 'Actual source/step rows differ')
    return result


def input_bundle(hf, corpus, check=lambda: None):
    _, audit = closed(corpus, hf, AUDIT_ID, 'COMPLETE_ASSET_BYTES_AUDITED')
    need(audit.get('manifest') == corpus.get('manifest'), 'Corpus manifest is not the actual completed audit')
    index = read(bound(hf, corpus['manifest']))
    cases = [case for case in index['cases'] if case['native_source_ordinal'] == 894]
    need(len(cases) == 1 and cases[0]['role'] == 'development' and cases[0]['source_state'] == 'teacher_input', 'Frozen exposed source894 missing')
    source = cases[0]; tensors, provenance = {}, []
    for kind, step in (('high', 0), ('low', 2)):
        check(); selected = [p for p in source['packets'] if p['step'] == step and p['block'] == 0]
        need(len(selected) == 1, 'Fixed true V1 site missing')
        values = packet_v1(hf, selected[0], 894, step)
        for key, data in values.items():
            tensors[f'{kind}_{key}'] = {'data': data, 'dtype': 'I64' if key == 'rows' else 'BF16',
                'shape': [256] if key == 'rows' else [256, 5120]}
        provenance.append({'expert': kind, 'step': step, 'block': 0, 'payload': selected[0]['payload'],
            'V1_raw_sha256': sha(values['V1']), 'rows_raw_sha256': sha(values['rows'])})
    header, body, cursor = {}, bytearray(), 0
    for name, value in sorted(tensors.items()):
        raw = value['data']; header[name] = {'dtype': value['dtype'], 'shape': value['shape'], 'data_offsets': [cursor, cursor+len(raw)]}
        body.extend(raw); cursor += len(raw)
    encoded = json.dumps(header, separators=(',', ':')).encode(); encoded += b' '*((-len(encoded)) % 8)
    return struct.pack('<Q', len(encoded))+encoded+body, {'source_id': source['source_id'], 'native_source_ordinal': 894,
        'consumer20_index': source['consumer20_index'], 'image_sha256': source['image_sha256'],
        'prompt_sha256': source['prompt_sha256'], 'history': source['history'], 'cases': provenance,
        'identity': 'RAW_BF16_TEACHER_V1_OPERATOR_INPUT_ONLY_NOT_NVFP4_NATIVE_STATE'}


def prepare(plan, hf, out, save, check, *, report=None, store_input=None):
    """Called only by the separately ACTIVE-registered CUDA-hidden CPU scope."""
    need(plan['resources'] == CPU_RESOURCES and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        and len(os.sched_getaffinity(0)) <= 2 and not {'torch', 'numpy'} & set(sys.modules), 'Fresh stdlib CPU worker required')
    need(private_worker_sha() == plan['private_worker_config_sha256'], 'Execution worker configuration differs')
    _, download = closed(plan['checkpoint_download'], hf, checkpoint.ID, 'COMPLETE_TWO_PUBLIC_NVFP4_CHECKPOINT_BYTES')
    export_plan, exported = closed(plan['software_export'], hf, EXPORT_ID, 'COMPLETE_OVERLAY_BYTES_EXPORTED')
    manifest = read(bound(hf, exported['manifest']))
    build, built = closed(manifest['build'], hf, BUILD_ID, 'COMPLETE_BUILD_NO_GPU_EXECUTION')
    need(built['artifact']['sha256'] == ELF_SHA and built['artifact']['bytes'] == 1312256, 'Actual built ELF differs')
    expected = expected_package(build)
    need({item['path']: {k: item[k] for k in ('bytes', 'sha256')} for item in manifest['files']} == expected, 'Exported wrapper/ELF closure differs')
    archive = bound(hf, manifest['archive']).read_bytes()
    relative = Path(plan['software_cache_relative'])
    need(not relative.is_absolute() and '..' not in relative.parts, 'Invalid isolated cache path')
    target = (Path(plan['local_HF_root']).resolve()/relative).resolve()
    need(target.is_relative_to(Path(plan['local_HF_root']).resolve()), 'Software escaped node-local HF cache')
    initial = {'id': CPU_ID, 'status': 'STARTED_CURRENT_SHA_STAT_PACKAGE_V1_NO_NUMERIC_IMPORTS',
        'model_calls': 0, 'GPU_calls': 0, 'Torch_or_NumPy_imported': False, 'automatic_retry': False,
        'predebited_full_SHA_bytes': 0, 'actual_full_SHA_read_bytes': 0, 'models': [],
        'private_worker_config_sha256': plan['private_worker_config_sha256'],
        'Python_version': list(sys.version_info[:3]),
        'runtime_versions': {key: importlib.metadata.version(key) for key in ('torch', 'safetensors')}}
    result = {} if report is None else report
    result.update(initial)
    need(result['Python_version'] == [3, 11, 15] and result['runtime_versions']['torch'] == '2.10.0+cu128',
        'Actual Python/Torch ABI metadata differs from successful kernel build')
    def count(n):
        result['actual_full_SHA_read_bytes'] += n; check()
    save(result)
    for item in checkpoint.FILES:
        record = next(value for value in download['files'] if value['filename'] == item['filename'])
        path = (hf/record['cache_path']).resolve(); need(path.is_relative_to(hf.resolve()), 'Checkpoint escaped HF cache')
        before = stat_record(path); result['predebited_full_SHA_bytes'] += item['bytes']; save(result)
        header = checkpoint.file_hash_header(path, item, count)
        need(before == stat_record(path) and header == record['header'], 'Fresh checkpoint stat/header differs from complete download')
        result['models'].append({'kind': 'high' if '_high.' in item['filename'] else 'low', 'cache_path': record['cache_path'],
            'bytes': item['bytes'], 'sha256': item['sha256'], 'stat': before, 'header': header})
        save(result)
    need(result['predebited_full_SHA_bytes'] == result['actual_full_SHA_read_bytes'] == CPU_RESOURCES['checkpoint_full_SHA_bytes'], 'Fresh full SHA bytes incomplete')
    check(); result['software_cache_target'] = str(target)
    need(sum(value['bytes'] for value in expected.values()) <= CPU_RESOURCES['node_local_software_cache_bytes'], 'Software cache byte budget exceeded')
    result['software_files'] = extract_package(archive, expected, target)
    result['software_package'] = str(target/'lightx2v_kernel')
    payload, provenance = input_bundle(hf, plan['corpus_audit'], check)
    need(callable(store_input), 'Owned input writer/pre-reservation required')
    store_input(out/'INPUTS.safetensors', payload)
    result.update(status='COMPLETE_FRESH_NVFP4_CHECKPOINT_SHA_STAT_ISOLATED_PACKAGE_TRUE_V1',
        inputs={'binding': binding(out/'INPUTS.safetensors', hf), 'provenance': provenance},
        software_export=plan['software_export'], checkpoint_download=plan['checkpoint_download'],
        corpus_audit=plan['corpus_audit'])
    need(not {'torch', 'numpy'} & set(sys.modules), 'Numerical package imported in asset-only CPU producer')
    return result
