"""Verify the exact completed DEV3 parent slots before an E4 or E6 launch."""
import hashlib
import json
from pathlib import Path


DEV3 = ('vbi_1013', 'vbi_0886', 'vbi_0800')


def hash_file(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def read_bound(spec):
    raw = Path(spec['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec['sha256']:
        raise ValueError('changed ablation parent evidence')
    return json.loads(raw)


def ensure_parent_metadata(parent, child):
    if (parent['arm'] != child['arm'] or parent['force_dense'] is not False or parent.get('ablation')
            or parent['frozen_comparison'] != child['frozen_comparison']
            or parent['dense_initial_noise']['sha256'] != child['dense_initial_noise']['sha256']):
        raise ValueError('ablation parent method, freeze or Dense initial noise differs')
    for key in ('seed', 'v2_reference_dir', 'reference_sha256', 'native_transformer_sha256', 'scheduler_sha256', 'schedule'):
        if parent[key] != child[key]:
            raise ValueError('ablation parent native contract differs: ' + key)
    if child['ablation'] == 'E4_SAME_FACTOR_2R':
        if (parent.get('cache_mode') != '4S' or child['cache_mode'] != '2R' or parent['parameters'] != child['parameters']):
            raise ValueError('E4 must reuse the actual main ICC factor bank')
        shared_code = ('wan_ffn_port.py',)
    elif child['ablation'] == 'E6_TAIL52':
        for key in ('chip_provider', 'voxel'):
            if parent[key] != child[key]:
                raise ValueError('E6 author provider or voxel map differs')
        if parent['chip_tail_mode'] != 'dense' or child['chip_tail_mode'] != 'valid52':
            raise ValueError('E6 differs outside the intended dense-to-valid52 tail switch')
        if parent['chip_stream_admission']['fixture'] != child['chip_stream_admission']['fixture']:
            raise ValueError('E6 compiled-stream boundary differs from its actual parent')
        shared_code = ('wan_chip_author_port.py', 'chip_provider.py', 'chip_wan_voxel_tail_v1.py',
                       'chip_tail52_valid_candidate.py')
    else:
        raise ValueError('unknown paired ablation variant')
    if any(parent['bundle_sha256'][key] != child['bundle_sha256'][key] for key in shared_code):
        raise ValueError('paired ablation and main parent use different operator bytes')


def read_parent(proof, child):
    stage = Path(proof['stage']).resolve()
    names = {'scope': 'scope.json', 'run': 'receipt/RUN.json', 'exit': 'receipt/EXIT.json', 'payload': 'output/SUMMARY.json',
             'nfe': 'output/NFE.jsonl'}
    if any(Path(proof[key]['path']).resolve() != stage / name for key, name in names.items()):
        raise ValueError('parent proof files belong to another stage')
    parent, start, stop, summary = (read_bound(proof[key]) for key in ('scope', 'run', 'exit', 'payload'))
    ensure_parent_metadata(parent, child)
    jid = str(proof['job_id'])
    if (start['scope_sha256'] != proof['scope']['sha256'] or stop['exit_code'] != 0
            or str(start['allocation']['job_id']) != jid or str(stop['slurm_job_id']) != jid
            or summary['status'] != 'COMPLETE' or summary['arm'] != child['arm']
            or [row['source'] for row in summary['records']] != parent['sources']
            or summary['nfe'] != parent['max_nfe'] or summary['nfe'] != len(parent['sources']) * 4):
        raise ValueError('parent whole producer is incomplete or outside reviewed arm')
    journal = Path(proof['nfe']['path']).read_bytes()
    expected = []
    for index, source in enumerate(parent['sources']):
        for step in range(4):
            expected.extend(({'event': 'NFE_ATTEMPT', 'source': source, 'step': step, 'attempt': 4 * index + step + 1},
                             {'event': 'NFE_COMPLETED', 'source': source, 'step': step, 'expert': step // 2}))
    if hashlib.sha256(journal).hexdigest() != proof['nfe']['sha256'] or [json.loads(line) for line in journal.splitlines()] != expected:
        raise ValueError('wrong actual parent forward journal')
    return stage, summary


def check_parents(scope):
    manifest = read_bound(scope['ablation_parent_evidence'])
    if (tuple(scope['sources']) != DEV3 or [row['source'] for row in manifest['parents']] != list(DEV3)
            or manifest['ablation'] != scope['ablation'] or manifest['frozen_comparison'] != scope['frozen_comparison']):
        raise ValueError('wrong actual paired parent cohort')
    producers = {str(row['job_id']): read_parent(row, scope) for row in manifest['producers']}
    if len(producers) != len(manifest['producers']):
        raise ValueError('duplicate ablation parent producer')
    noise = read_bound(scope['dense_initial_noise'])
    for row in manifest['parents']:
        source = row['source']
        stage, summary = producers[str(row['job_id'])]
        ref = row['receipt']
        if (Path(ref['path']).resolve() != stage / f'output/{source}.json'
                or ref['sha256'] != next(x['receipt_sha256'] for x in summary['records'] if x['source'] == source)):
            raise ValueError('parent per-source receipt was not written by this exact producer')
        receipt = read_bound(ref)
        if (receipt['source'] != source or receipt['arm'] != scope['arm'] or receipt['ablation'] is not None
                or receipt['initial_noise'] != noise[source] or receipt['schedule'] != scope['schedule']
                or receipt['seed'] != scope['seed'] or len(receipt['sites']) != 160
                or receipt['dispatch'] != [{'step': i, 'expert': i // 2} for i in range(4)]):
            raise ValueError('parent matched input or complete native trajectory differs')
        video = row['video']
        latent = row['latent']
        if (video['sha256'] != receipt['video']['mp4_sha256'] or latent != receipt['latent_file']
                or Path(video['path']).resolve() != stage / f'output/{source}.mp4'
                or Path(latent['path']).resolve() != stage / f'output/{source}_latent.pt'
                or hash_file(video['path']) != video['sha256'] or hash_file(latent['path']) != latent['sha256']):
            raise ValueError('parent adjoined raw video or final latent differs')
    if (scope['resources']['soft_seconds'] <= manifest['forecast']['bounded_seconds']
            or manifest['forecast']['native_parent_estimate_is_instrumented'] is not True):
        raise ValueError('ablation exceeds prospective bound from actual parent runs')
    return manifest
