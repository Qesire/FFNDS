"""Use the pinned scheduler guard with an explicitly reviewed Wan wall ceiling."""
import hashlib
import re
import types


REFERENCE_SHA256 = 'ba75a4c128dd258f841701f92d15630516a49162878a60db8135407802ae7bc1'
REFERENCE_PREDICATE = b'if not 0 < days*86400 + hours*3600 + minutes*60 + seconds <= 10800:'
REFERENCE_ERROR = b"raise RuntimeError('Slurm wall time exceeds 3 hours')"


def bind_guard(reference, max_wall_seconds=10800):
    if type(max_wall_seconds) is not int or not 0 < max_wall_seconds <= 21600:
        raise ValueError('Wan wall ceiling must be positive and at most six hours')
    source = reference.read_bytes()
    if (hashlib.sha256(source).hexdigest() != REFERENCE_SHA256
            or source.count(REFERENCE_PREDICATE) != 1 or source.count(REFERENCE_ERROR) != 1):
        raise ValueError('allocation reference is not the pinned original implementation')
    guard = types.ModuleType('_ffnds_reviewed_wan_runtime_guard')
    guard.__dict__['_reviewed_wall_seconds'] = max_wall_seconds
    source = source.replace(REFERENCE_PREDICATE, REFERENCE_PREDICATE.replace(b'10800', b'_reviewed_wall_seconds'))
    source = source.replace(REFERENCE_ERROR, b"raise RuntimeError('Slurm wall time outside reviewed scope')")
    exec(compile(source, str(reference), 'exec'), guard.__dict__)
    return guard


def require_wan_allocation(reference, resources):
    maximum = resources['wall_seconds']
    allocation = bind_guard(reference, maximum).require_compute_allocation()
    fields = dict(re.findall(r'(\w+)=([^\s]+)', allocation['scheduler_job_record']))
    expected = {'Partition': resources['partition'], 'Account': resources['account'], 'QOS': resources['qos']}
    if any(fields.get(key) != value for key, value in expected.items()):
        raise RuntimeError('actual Wan partition, account or QoS differs from reviewed scope')
    return allocation
