"""Bind tested stream transport and exact untouched/retried Chip Wan slots."""
import hashlib
import json
from pathlib import Path


def read_bound(spec):
    raw = Path(spec['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec['sha256']:
        raise ValueError('changed Chip repair evidence')
    return json.loads(raw)


def check_transport_summary(data):
    labels = ('deterministic_all', 'aligned_support', 'production')
    expected = [(rows, label) for rows in (247, 1) for label in labels]
    if ((data['status'], data['selector_attempts'], data['copy_attempts'], data['wan_nfe'], data['videos'])
            != ('PASS_STREAM_TRANSPORT', 18, 18, 0, 0) or data['exceptions']
            or [(row['selector_rows'], row['case']) for row in data['cases']] != expected):
        raise ValueError('no successful exact compiled stream transport fixture')
    for row in data['cases']:
        if not all(row[key] is True for key in ('basic_valid', 'analytic_support_exact', 'padding_symdiff_within_510')):
            raise ValueError('invalid exact stream transport, support or copy witness')
        if row['case'] != 'production' and not row['mask_and_copy_parity']:
            raise ValueError('deterministic selector support or copy differs')
        if row['case'] == 'deterministic_all':
            if any(r['counts_values'] != [13824] * row['selector_rows']
                   for r in (row['default_first'], row['default_repeat'], row['nondefault'])):
                raise ValueError('deterministic full-support selector did not admit all columns')


def check_fixture(spec, scope):
    stage = Path(spec['stage']).resolve()
    required = {'scope': 'scope.json', 'run': 'receipt/RUN.json', 'exit': 'receipt/EXIT.json', 'payload': 'output/SUMMARY.json'}
    if any(Path(spec[key]['path']).resolve() != stage / name for key, name in required.items()):
        raise ValueError('Chip stream fixture evidence belongs to another stage')
    frozen, start, stop, data = (read_bound(spec[key]) for key in required)
    job = str(spec['job_id'])
    provider = scope['chip_provider']
    digest = scope['bundle_sha256']['chip_provider.py']
    if (frozen['operation'] != 'chip_stream_bridge_fixture' or frozen['provider'] != provider
            or frozen['stream_bridge_sha256'] != digest or start['scope_sha256'] != spec['scope']['sha256']
            or str(start['slurm_job_id']) != job or str(stop['slurm_job_id']) != job or stop['exit_code'] != 0
            or data['selector_sha256'] != provider['library_sha256'] or data['stream_bridge_sha256'] != digest):
        raise ValueError('no successful actual parity witness for the exact Chip stream transport')
    check_transport_summary(data)


def read_interrupted_producer(spec):
    stage = Path(spec['stage']).resolve()
    assets = {}
    for name, expected in spec['hashes'].items():
        path = (stage / name).resolve()
        if not path.is_relative_to(stage):
            raise ValueError('interrupted Chip evidence outside producer')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('changed exact interrupted Chip producer file')
        assets[name] = data
    previous = json.loads(assets['scope.json'])
    start, stop = (json.loads(assets[key]) for key in ('receipt/RUN.json', 'receipt/EXIT.json'))
    if (str(start['allocation']['job_id']) != spec['job_id'] or str(stop['slurm_job_id']) != spec['job_id']
            or start['scope_sha256'] != spec['scope_sha256'] or spec['hashes']['scope.json'] != spec['scope_sha256']
            or (stage / 'output/SUMMARY.json').exists()):
        raise ValueError('original Chip terminal identity changed')
    if spec['status'] == 'FAILED_SELECTOR_STREAM':
        journal = [json.loads(line) for line in assets['output/NFE.jsonl'].splitlines()]
        expected = [{'event': 'NFE_ATTEMPT', 'source': 'vbi_1013', 'step': 0, 'attempt': 1},
                    {'event': 'NFE_COMPLETED', 'source': 'vbi_1013', 'step': 0, 'expert': 0},
                    {'event': 'NFE_ATTEMPT', 'source': 'vbi_1013', 'step': 1, 'attempt': 2}]
        if (stop['exit_code'] != 1 or journal != expected
                or b'the author launcher uses the default stream' not in assets['receipt/runner.log']
                or (stage / 'output/vbi_1013.mp4').exists()):
            raise ValueError('prior Chip recording is not the fixed failed slot')
    elif spec['status'] != 'CANCELLED_BEFORE_ANY_NFE' or stop['exit_code'] != 143 or (stage / 'output/NFE.jsonl').exists():
        raise ValueError('canceled Chip cohort was not unused')
    return previous


def check_admission(scope):
    admission = scope['chip_stream_admission']
    if scope['arm'] != 'Chipmunk Author Safe' or scope['force_dense'] is not False:
        raise ValueError('Chip stream evidence applies only to the actual author arm')
    check_fixture(admission['fixture'], scope)
    if admission['kind'] == 'E6_PARENT_PORT':
        if scope['ablation'] != 'E6_TAIL52':
            raise ValueError('unexpected Chip E6 stream admission')
        return
    records = read_bound(admission['interruption'])['producers']
    if [row['job_id'] for row in records] != ['171340', '171342'] or scope.get('ablation'):
        raise ValueError('wrong actual interrupted Chip cohorts')
    old, cancelled = (read_interrupted_producer(row) for row in records)
    for name in ('arm', 'seed', 'schedule', 'frozen_comparison', 'chip_provider', 'voxel'):
        if old[name] != scope[name] or cancelled[name] != scope[name]:
            raise ValueError('repaired Chip recipe differs: ' + name)
    for previous in (old, cancelled):
        if (previous['chip_tail_mode'] != scope['chip_tail_mode'] or scope['chip_tail_mode'] != 'dense'
                or previous['dense_initial_noise']['sha256'] != scope['dense_initial_noise']['sha256']):
            raise ValueError('repaired Chip tail or seed42 noise differs')
    expected = {'FAILED_SLOT_RETRY': ['vbi_1013'], 'FIRST_SHARD_UNUSED': old['sources'][1:],
                'SECOND_SHARD_UNUSED': cancelled['sources']}
    kind = admission['kind']
    retries = int(kind == 'FAILED_SLOT_RETRY')
    if (scope['sources'] != expected[kind] or scope['retries'] != retries
            or admission['shared_retry_slots_before'] != 2 - retries
            or admission['shared_retry_slots_after_this_scope'] != 2):
        raise ValueError('not the exact failed or unexecuted Chip slot scope')
