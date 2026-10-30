"""Metadata-only student observer owner draft; execution deliberately unavailable.

This module is not an experiment authorization. It validates parent closure and
builds a reviewable scope, but does not freeze a protocol or launch any work.
"""
import argparse
import copy
import hashlib
import json
import os
import struct
import sys
import importlib.metadata
import resource
import shutil
import signal
import stat
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import e0_chipmunk_input_video_owner as parent
import e0_sharq_wan_all160_video_owner_v2 as prior
import e0_teal_llama3_author_checkpoint_convert as support
import chipmunk_student_observation_contract as contract
import chipmunk_student_observation_driver as driver
import chipmunk_student_observation_tap as capture
import chipmunk_p64_deferred_ledger as deferred

native, HF = parent.native, parent.HF
vm, io = prior.vm, prior.io
PYTHON = ROOT / '.venv/bin/python'
SELF = 'tools/e0_chipmunk_student_observation_owner.py'
JOBS = 'tools/e0_chipmunk_student_observation_jobs.py'
TEST = 'tests/test_e0_chipmunk_student_observation_owner.py'
PARENT_PARTS = (0, 3)
SOURCES = (898, 979)
MODES = ('h_cache_dense', 'high_h_readonly_p64')
GROUP = 'E0_CHIPMUNK_STUDENT_OBSERVATION_GROUP_20261030'
RES = {'GPU': 1, 'CPU_threads': 4, 'wall_seconds': 10800, 'outer_seconds': 10920,
       'new_HF_bytes': 32 << 30, 'video_bytes_max': 1 << 30,
       'GPU_reserved_max_bytes': 76 << 30, 'host_available_min_bytes': 128 << 30,
       'max_RSS_bytes': 128 << 30, 'free_disk_min_bytes': 40 << 30,
       'max_concurrent_A100_models': 2, 'session_max_GPUs': 4,
       'single_file_bytes_max': 1 << 30, 'automatic_retry': False}
CPU = {'GPU': 0, 'CPU_threads': 4, 'wall_seconds': 900, 'outer_seconds': 1020,
       'new_HF_bytes': 16 << 20, 'automatic_retry': False}
VERIFY = {'GPU': 0, 'CPU_threads': 1, 'wall_seconds': 7200, 'outer_seconds': 7320,
          'read_payload_bytes_max': 32 << 30, 'new_HF_bytes': 32 << 20,
          'automatic_retry': False}
AUTHORS = ['/root', '/root/chipmunk_student_discrimination_terra',
           '/root/teal_full_llm_calibration_build_terra']
OBSERVATION_SOURCES = {SELF, JOBS, TEST, 'tools/chipmunk_student_observation_contract.py',
 'tools/chipmunk_student_observation_coordinates.py', 'tools/chipmunk_student_observation_core.py',
 'tools/chipmunk_student_observation_math.py', 'tools/chipmunk_student_observation_sink.py',
 'tools/chipmunk_student_observation_tap.py', 'tools/chipmunk_student_observation_write.py',
 'tools/chipmunk_student_observation_driver.py', 'tests/test_chipmunk_student_observation.py',
 'tests/test_chipmunk_student_observation_sink.py', 'tests/test_chipmunk_student_observation_driver.py',
 'tests/test_chipmunk_student_observation_tap.py', 'tools/chipmunk_p64_deferred_ledger.py'}

def need(value, message):
    if not value:
        raise ValueError(message)

def sid(part):
    need(type(part) is int and part in (0, 1), 'invalid part')
    return f'E0_CHIPMUNK_STUDENT_OBSERVATION_20261030_PART{part}'

def outpath(part):
    sid(part)
    return f'ffnds-derived/e0_chipmunk_student_observation_20261030/part{part}/attempt1/worker/results'

def run_ids(part):
    sid(part)
    return [f'chipmunk_student_observation_ord{SOURCES[part]}_seed42_{mode}' for mode in MODES]

def design(part):
    sid(part)
    return {'source_ordinal': SOURCES[part], 'seed': 42, 'modes': list(MODES),
            'steps': 4, 'blocks': 40, 'grid': [21,29,52], 'snapshot_blocks': [0,13,26,39],
            'selector_rows':64, 'evaluation_rows':256,
            'parent_state_and_video_byte_gates':'phases 1 and 3, ten states and complete video each',
            'dense_shadow_byte_gate':'40 native h/Y comparisons',
            'write_byte_gate':'one native post_process per layer, compare shadow',
            'source_role':'EXPOSED_DEVELOPMENT', 'quality_acceptance':False}

def payload_budget():
    # The sink currently records five BF16 fields in Dense and six in student.
    # Six fields: V1, old/current h, old Y, current dense Y, candidate Y.
    m, d, h, rows, blocks, snapshots = 31668, 5120, 13824, 320, 40, 4
    dense_row_bytes = 2 * (2 * h + 3 * d)
    student_row_bytes = 2 * (2 * h + 4 * d)
    sample = blocks * rows * (dense_row_bytes + student_row_bytes)
    snapshot = snapshots * m * (dense_row_bytes + student_row_bytes)
    return {'dense_sample_payload_bytes': blocks * rows * dense_row_bytes,
            'student_sample_payload_bytes': blocks * rows * student_row_bytes,
            'dense_full_payload_bytes': snapshots * m * dense_row_bytes,
            'student_full_payload_bytes': snapshots * m * student_row_bytes,
            'sample_and_full_raw_payload_bytes': sample + snapshot,
            'includes_file_headers_metadata_states_videos_journals': False}

def limits(part):
    sid(part)
    d = dict(vm.native_original_limits)
    m, w, h, k = 31668, 5120, 13824, 5120
    state = __import__('math').prod(native.VIDEO['latent_shape']) * 30
    d.update(core_CPU_reports=0, prefetch_transfer_observations=324, stage_CUDA_event_pairs=14,
      state_tensor_D2H=20, state_tensor_D2H_bytes=2*state, state_output_writes=20,
      video_geometry_checks=2, video_file_hash_calls=2, video_file_hash_reserved_bytes=2*RES['video_bytes_max'],
      authority_checks=2, state_file_hash_calls=20, state_file_hash_reserved_bytes=2*state+(20<<20),
      state_raw_sha_calls=20, state_raw_sha_bytes=2*state, p64_row_seed_and_H2D=1,
      cache_h_clone=80, cache_h_clone_bytes=80*m*h*2, cache_y_clone=80, cache_y_clone_bytes=80*m*w*2,
      cache_release_sync=80, cache_boundary_memory_observer=80, expert_boundary_cleanup=2,
      weight_materializations=80, weight_materialization_logical_bytes=40*2*w*h*2,
      p64_score=40, p64_gather_input=40, p64_gather_old_h=40, p64_W1_gemm=40, p64_gelu=40,
      p64_rms=40, p64_select=40, p64_input_gather_logical_bytes=40*64*w*2,
      p64_old_h_gather_logical_bytes=40*64*h*2, mask_sha_calls=40, mask_sha_logical_bytes=40*k*8,
      selected_old_h=40, old_selected_logical_bytes=40*m*k*2,
      packed_action=40, packed_W1_gemm=40, packed_W2_gemm_fp32=40, packed_clone_y=40,
      packed_gather_w1=40, packed_gather_w2=40, packed_gather_bias=40, packed_selected_gelu=40,
      packed_delta_round=40, packed_accumulate_y=40, packed_y_clone_bytes=40*m*w*2,
      packed_delta_bytes=40*m*k*2, packed_weight_gather_bytes=40*(k*w+w*k+k)*2,
      reference_state_checks=20, reference_video_checks=2,
      reference_state_read_reservation=20*(18<<20), reference_video_read_reservation=2*(64<<20))
    for costs in (contract.DENSE_WORK, contract.STUDENT_WORK):
        for name, amount in costs.items():
            d[name] = d.get(name, 0) + amount
    return d

def inspect_typed_payload(path, expected_file_sha, expected_tensors, max_file_bytes=1 << 30):
    """Single-pass file and raw-tensor hash verification, with bounded reads.

    expected_tensors maps tensor names to dtype (BF16/F32), shape and raw_sha256.
    Caller must predebit the full file length and bind the canonical allowed path.
    This helper does not authorize a verifier or establish an experiment result.
    """
    path = Path(path)
    before = path.stat()
    need(path.is_file() and 8 <= before.st_size <= max_file_bytes, 'payload size differs')
    need(isinstance(expected_tensors, dict) and bool(expected_tensors), 'tensor manifest missing')
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    whole = hashlib.sha256()
    with path.open('rb') as stream:
        need(identity(os.fstat(stream.fileno())) == identity(before), 'payload identity changed')
        prefix = stream.read(8)
        need(len(prefix) == 8, 'truncated header size')
        header_length = struct.unpack('<Q', prefix)[0]
        need(0 < header_length <= min(16 << 20, before.st_size - 8), 'header length differs')
        header_bytes = stream.read(header_length)
        need(len(header_bytes) == header_length, 'truncated header')
        whole.update(prefix)
        whole.update(header_bytes)
        header = json.loads(header_bytes)
        need(isinstance(header, dict) and set(header) == set(expected_tensors), 'tensor names differ')
        spans = []
        payload_bytes = before.st_size - 8 - header_length
        for name, expected in expected_tensors.items():
            actual = header[name]
            need(isinstance(actual, dict) and set(actual) == {'dtype', 'shape', 'data_offsets'},
                 'typed header differs')
            dtype, shape = expected['dtype'], expected['shape']
            need(dtype in ('BF16', 'F32') and actual['dtype'] == dtype and actual['shape'] == shape and
                 isinstance(shape, list) and all(type(n) is int and n >= 0 for n in shape), 'tensor type differs')
            count = 1
            for n in shape:
                count *= n
            offsets = actual['data_offsets']
            need(isinstance(offsets, list) and len(offsets) == 2 and
                 all(type(n) is int for n in offsets) and
                 0 <= offsets[0] <= offsets[1] <= payload_bytes and
                 offsets[1] - offsets[0] == count * (2 if dtype == 'BF16' else 4), 'tensor span differs')
            spans.append((offsets[0], offsets[1], name, expected['raw_sha256']))
        spans.sort()
        cursor = 0
        for start, end, name, raw_sha in spans:
            need(start == cursor, 'tensor spans overlap or have gaps')
            raw = hashlib.sha256()
            remaining = end - start
            while remaining:
                block = stream.read(min(8 << 20, remaining))
                need(bool(block), 'truncated tensor')
                remaining -= len(block)
                whole.update(block)
                raw.update(block)
            need(raw.hexdigest() == raw_sha, 'raw tensor hash differs')
            cursor = end
        need(cursor == payload_bytes and stream.read(1) == b'', 'payload extent differs')
        need(identity(os.fstat(stream.fileno())) == identity(before), 'payload mutated while reading')
    need(identity(path.stat()) == identity(before) and whole.hexdigest() == expected_file_sha,
         'file hash or identity differs')
    return {'bytes': before.st_size, 'tensor_count': len(spans)}

def parent_proof(part):
    sid(part)
    old = PARENT_PARTS[part]
    protocol = ROOT / f'config/e0_chipmunk_input_video_20261030_part{old}_pending_review_v1.json'
    plan = native.read(protocol)
    parent.validate(plan, True)
    verification = parent.inspect_verification(SimpleNamespace(protocol=protocol))
    need(verification.get('status') == 'PASS_ACTUAL_EXIT_CLOSED_CHIPMUNK_INPUT_VIDEO',
         'actual independent parent verification required')
    out = HF / plan['output_HF_relative']
    report = native.read(out / 'RUN.json')
    exit_receipt = native.read(out.parent / 'EXIT_RECEIPT.json')
    rb = native.binding(out / 'RUN.json', HF)
    parent.vm.exited(exit_receipt, rb, native.sha(protocol), plan['id'])
    need(report.get('job_spec_sha256') == exit_receipt.get('spec_sha256') and
         report.get('source_ordinal') == SOURCES[part] and report.get('seed') == 42 and
         report.get('status') == 'COMPLETE_CHIPMUNK_INPUT_VIDEO_EXIT_PENDING',
         'parent identity differs')
    phases = report['phases']
    need(len(phases) == 5 and phases[1]['mode'] == MODES[0] and phases[3]['mode'] == MODES[1],
         'parent phase identity differs')
    return {'parent_protocol': native.binding(protocol, ROOT), 'RUN': rb,
            'EXIT': native.binding(out.parent / 'EXIT_RECEIPT.json', HF),
            'verification_RUN': verification['verification_RUN'],
            'verification_EXIT': verification['verification_exit'],
            'case': copy.deepcopy(plan['case']), 'source_ordinal': SOURCES[part],
            'dense': {'states': phases[1]['states'], 'video': phases[1]['video']},
            'student': {'states': phases[3]['states'], 'video': phases[3]['video']}}

def verify_reference_metadata(entry, phases, state_comparisons, video_comparisons):
    """Validate recorded exact comparisons against the two closed parent arms.

    This checks metadata only. A future owner must actually read and compare
    independent files, and its verifier must audit the journal and receipts.
    """
    names = {'raw_noise_FP32', 'initial_noise', *(f'flow_step{i}' for i in range(4)),
             *(f'latent_step{i}' for i in range(4))}
    need(len(phases) == 2 and tuple(x['mode'] for x in phases) == MODES, 'phase order differs')
    need(len(state_comparisons) == 20 and
         {(x['mode'], x['state']) for x in state_comparisons} == {(mode, name) for mode in MODES for name in names} and
         all(x.get('equal') is True for x in state_comparisons), 'state byte gates incomplete')
    need(len(video_comparisons) == 2 and {x['mode'] for x in video_comparisons} == set(MODES) and
         all(x.get('equal') is True for x in video_comparisons), 'video byte gates incomplete')
    for phase, reference in zip(phases, (entry['dense'], entry['student'])):
        need(set(phase['states']) == names and set(reference['states']) == names, 'state identity differs')
        for name in names:
            need(all(phase['states'][name][key] == reference['states'][name][key]
                     for key in ('sha256', 'bytes', 'raw_tensor_sha256', 'shape', 'dtype')),
                 'state reference metadata differs')
        need(all(phase['video'][key] == reference['video'][key] for key in ('sha256', 'bytes')),
             'video reference metadata differs')
    return True

def draft(part, profile, affinity):
    entry = parent_proof(part)
    base = native.read(ROOT / parent.BASE)
    need(profile.is_file() and native.sha(profile) == base['execution_private_worker_config_sha256'],
         'worker profile differs')
    need(len(affinity) == 4 and len(set(affinity)) == 4 and set(affinity) <= os.sched_getaffinity(0),
         'four available distinct CPU cores required')
    inherited = ('models', 'model_view_root', 'model_view_metadata', 'runtime_versions',
                 'local_transformers', 'private_worker_config_sha256', 'runtime_overlay',
                 'execution_raw_master_CPU_proof')
    result = {k: copy.deepcopy(base[k]) for k in inherited}
    parent_plan = native.read(ROOT / entry['parent_protocol']['path'])
    paths = {x['path'] for x in parent_plan['repo_sources']} | OBSERVATION_SOURCES | {entry['parent_protocol']['path']}
    missing = sorted(path for path in paths if not (ROOT / path).is_file())
    # Do not assert a source freeze while the parallel component is unfinished.
    bindings = [native.binding(ROOT / x, ROOT) for x in sorted(paths) if x not in missing]
    result.update(id=sid(part), part=part, status='DRAFT_REVIEW_REQUESTED', assets_finalized=False,
      review=None, authors=AUTHORS, native_group_id=GROUP, case=entry['case'], source_parent=entry,
      execution_private_worker_config_sha256=native.sha(profile), cpu_affinity=affinity,
      resources=RES, CPU_preflight_resources=CPU, verification_resources=VERIFY,
      payload_budget=payload_budget(), run_ids=run_ids(part), output_HF_relative=outpath(part),
      design=design(part),
      repo_sources=bindings, missing_source_files=missing,
      limits=limits(part), execution_monitor_binary=native.binding(Path('/usr/bin/nvidia-smi'), Path('/usr/bin')),
      automatic_retry=False)
    result['execution_monitor_binary']['path'] = '/usr/bin/nvidia-smi'
    return result

def validate_draft(p):
    part = p.get('part')
    need(p.get('id') == sid(part) and p.get('status') == 'DRAFT_REVIEW_REQUESTED' and
         p.get('assets_finalized') is False and p.get('review') is None and
         p.get('resources') == RES and p.get('CPU_preflight_resources') == CPU and
         p.get('verification_resources') == VERIFY and p.get('payload_budget') == payload_budget() and
         p.get('run_ids') == run_ids(part) and p.get('output_HF_relative') == outpath(part) and
         p.get('authors') == AUTHORS and p.get('limits') == limits(part) and p.get('design') == design(part) and
         p.get('native_group_id') == GROUP and p.get('automatic_retry') is False,
         'draft identity differs')
    return True

def fingerprint(p):
    return native.canonical({k: v for k, v in p.items() if k not in ('status', 'assets_finalized', 'review')})

def validate(p, frozen=False):
    candidate = copy.deepcopy(p)
    if frozen:
        need(p.get('status') == 'FROZEN_PENDING_REGISTRATION' and p.get('assets_finalized') is True and
             p.get('review') is None, 'frozen protocol required')
        candidate['status'] = 'DRAFT_REVIEW_REQUESTED'
        candidate['assets_finalized'] = False
    validate_draft(candidate)
    need(not p.get('missing_source_files'), 'source closure incomplete')
    current = parent_proof(p['part'])
    need(p['source_parent'] == current and p['case'] == current['case'], 'parent closure differs')
    parent_plan = native.read(ROOT / current['parent_protocol']['path'])
    expected = {x['path'] for x in parent_plan['repo_sources']} | OBSERVATION_SOURCES | {current['parent_protocol']['path']}
    need({x['path'] for x in p['repo_sources']} == expected and len(p['repo_sources']) == len(expected),
         'source closure differs')
    for bound in p['repo_sources']:
        native.bound(ROOT, bound)
    base = native.read(ROOT / parent.BASE)
    for key in ('models', 'model_view_root', 'model_view_metadata', 'runtime_versions', 'local_transformers',
                'private_worker_config_sha256', 'runtime_overlay', 'execution_raw_master_CPU_proof',
                'execution_private_worker_config_sha256'):
        need(p[key] == base[key], 'base recipe differs')
    need(len(p['cpu_affinity']) == len(set(p['cpu_affinity'])) == 4, 'CPU affinity differs')
    return True

compare_reference = parent.compare_reference

def compare_video_reference(q, path, expected_bytes, expected_sha, actual_sha):
    """Reject a mismatched output size before starting the byte-comparison read."""
    info = Path(path).lstat()
    need(type(expected_bytes) is int and 0 <= expected_bytes <= (32 << 20) and
         stat.S_ISREG(info.st_mode) and info.st_size == expected_bytes,
         'new video size differs from historical reference')
    return parent.compare_reference(q, path, expected_bytes, expected_sha, actual_sha)

def observe_failed_video(counter, rid, path, hash_started, report, persist):
    """Predebit a distinct failure hash even if inspection already read it."""
    rows = report.setdefault('partial_video_observations', [])
    need(not any(row['run_id']==rid for row in rows), 'duplicate partial-video hash')
    row={'run_id':rid,'predebited_hash_bytes':RES['video_bytes_max'],
         'completed_hash_bytes':'UNKNOWN','inspection_hash_started':rid in hash_started}
    rows.append(row); persist()
    counter.observe_external(path, hold=str(path))
    binding=counter.binding(path,HF)
    row.update(binding=binding, completed_hash_bytes=binding['bytes']); persist()

def authorize(p,d):
 validate(p,True);a=native.read(ROOT/'research/ACTIVE_RESEARCH.json');rows=[x for x in a['experiment_authority']['current_user_authorized_scopes'] if x.get('id')==p['id']];need(a['user_continuation_authorization_20261027']['new_scope_registration_allowed'] is True and len(rows)==1,'ACTIVE required');s=rows[0];need(s.get('protocol_sha256')==d and s.get('resources')==RES and s.get('limits')==limits(p['part']) and s.get('run_ids')==p['run_ids'] and s.get('max_gpu_cards')==1 and s.get('session_max_GPUs')==4 and s.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') and s.get('native_group_id')==GROUP and s.get('native_group_active_slots') in (0,1),'ACTIVE differs');need(sum(x.get('native_group_active_slots',0) for x in a['experiment_authority']['current_user_authorized_scopes'] if x.get('native_group_id')==GROUP and x.get('execution_status')=='RUNNING_FIXED_SCOPE')<=2,'two model slot limit');r=native.read(native.bound(ROOT,s['review']));need(r.get('scope_id')==p['id'] and r.get('state')=='REVIEWER_RECHECK_APPROVED' and r.get('recheck_disposition')=='APPROVE' and r.get('review_schema')=='FFNDS_PREEXEC_REVIEW_V1' and r.get('author_agent_id') and r.get('reviewer_agent_id') and r['author_agent_id']!=r['reviewer_agent_id'] and r.get('approval_scope_fingerprint')==fingerprint(p) and r.get('critical_source_hashes')=={x['path']:x['sha256'] for x in p['repo_sources']} and r.get('cpu_preflight_spec_sha256')==s.get('cpu_preflight_spec_sha256') and r.get('gpu_launch_spec_sha256')==s.get('gpu_launch_spec_sha256') and r.get('responses_to_review') and r.get('adversarial_counterexample'),'independent review differs')
 for k in ('author_review','independent_review','reviewer_recheck'):native.bound(ROOT,r[k])
 need(r['reviewer_agent_id'] not in AUTHORS, 'reviewer must be independent of all authors')
 return s

def verify_job(a,p,s,kind):
 key='cpu_preflight_spec_sha256' if kind=='preflight' else 'gpu_launch_spec_sha256';need(a.job_spec and native.sha(a.job_spec)==s[key],'job SHA differs');j=native.read(a.job_spec);v=j['argv'];need(len(v)>1 and v[0]=='env' and v[1].startswith('FFNDS_WORKER_CONFIG='),'profile argument missing');profile=Path(v[1].split('=',1)[1]);need(profile.is_absolute() and native.sha(profile)==p['execution_private_worker_config_sha256'],'profile differs');aff=','.join(map(str,p['cpu_affinity']));env=['env',v[1],'OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','PATH=/usr/bin:/bin'];child=['taskset','-c',aff,str(PYTHON),'-I','-B',str(ROOT/SELF)];tail=['--protocol',str(a.protocol.resolve()),'--hf-root',str(HF),'--local-root',str(a.local_root.resolve()),'--job-spec',str(a.job_spec.resolve())];out=HF/p['output_HF_relative']
 if kind=='preflight':
  dest=out.parent.parent/'cpu_preflight/worker/results';expected=env+['CUDA_VISIBLE_DEVICES=']+child+['preflight','--out',str(dest)]+tail;need(a.out.resolve()==dest and Path(j['output_dir']).resolve()==dest.parent and Path(j['run_json']).resolve()==dest/'RUN.json' and j['external_timeout_seconds']==CPU['outer_seconds'] and os.environ.get('CUDA_VISIBLE_DEVICES')=='','preflight job differs')
 else:
  gpu=os.environ.get('FFNDS_PHYSICAL_GPU','');uuid=os.environ.get('CUDA_VISIBLE_DEVICES','');need(gpu in ('0','1','2','3') and uuid.startswith('GPU-'),'one UUID required');expected=env+['bash',str(ROOT/'scripts/worker_teal_assigned_uuid_exec.sh'),gpu,uuid]+child+['run']+tail;need(Path(j['output_dir']).resolve()==out.parent and Path(j['run_json']).resolve()==out/'RUN.json' and j['external_timeout_seconds']==RES['outer_seconds'],'GPU paths differ');b=p['execution_monitor_binary'];monitor=Path(b['path']);need(shutil.which('nvidia-smi')==str(monitor) and monitor.stat().st_size==b['bytes'] and native.sha(monitor)==b['sha256'],'monitor differs')
 need(v==expected and j['id']==p['id'] and j['protocol_sha256']==native.sha(a.protocol) and Path(j['protocol_path']).resolve()==a.protocol.resolve() and Path(j['cwd']).resolve()==ROOT and Path(j['hf_root']).resolve()==HF and sorted(os.sched_getaffinity(0))==sorted(p['cpu_affinity']) and sys.flags.isolated,'exact job differs');import worker_identity;worker_identity.require_worker()
def check_assets(p,local):
 for b in p['repo_sources']:native.bound(ROOT,b)
 native.bound(HF,p['case']['image']);import e0_s2_causal_worker_proof_v5 as proof
 proof.require_closed_proof(p,HF);native.model_metadata(p);return native.verify_local_package(p,{'repo':ROOT,'hf':HF},local)
def charged_assets(p,local,r,persist):
 with prior.AssetHashBudget(r,persist):return check_assets(p,local)
def preflight(a):
 p=native.read(a.protocol);digest=native.sha(a.protocol);s=authorize(p,digest);verify_job(a,p,s,'preflight');need(not {'torch','numpy'}&set(sys.modules) and not a.out.exists() and a.out.resolve().is_relative_to(HF),'fresh numerical-free CPU preflight');a.out.mkdir(parents=True);start=time.monotonic();r={'id':p['id'],'protocol_sha256':digest,'job_spec_sha256':s['cpu_preflight_spec_sha256'],'status':'STARTED_CPU_NO_RETRY','model_calls':0,'GPU_calls':0,'automatic_retry':False};native.write(a.out/'RUN.json',r);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU wall')));signal.alarm(CPU['wall_seconds'])
 signal.signal(signal.SIGUSR1,lambda *_:(_ for _ in ()).throw(RuntimeError('CPU output monitor failure')));monitor=support.UsageMonitor(a.out.parent,CPU['new_HF_bytes'],0);monitor.start()
 try:
  charged_assets(p,a.local_root,r,lambda:native.write(a.out/'RUN.json',r));need(not {'torch','numpy'}&set(sys.modules),'numerical import');r.update(status='PASS_CHIPMUNK_STUDENT_OBSERVATION_BYTE_PREFLIGHT',source_closure_sha256=native.canonical(p['repo_sources']),numerical_imports=False)
 except BaseException as e:r.update(status='FAILED_CPU_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e)[:500]);raise
 finally:
  signal.alarm(0);r['wall_seconds']=time.monotonic()-start;r['output_monitor']=monitor.stop()
  if r['output_monitor']['fault'] or not r['output_monitor']['thread_stopped']:r['status']='FAILED_CPU_MONITOR';native.write(a.out/'RUN.json',r);raise RuntimeError('CPU output monitor failed')
  native.write(a.out/'RUN.json',r)
def require_cpu(p,d,s):
 proof=s['cpu_preflight'];r=native.read(native.bound(HF,proof['RUN']));e=native.read(native.bound(HF,proof['exit']));vm.exited(e,proof['RUN'],d,p['id']);need(e.get('spec_sha256')==r.get('job_spec_sha256')==s['cpu_preflight_spec_sha256'] and r.get('status')=='PASS_CHIPMUNK_STUDENT_OBSERVATION_BYTE_PREFLIGHT' and r.get('source_closure_sha256')==native.canonical(p['repo_sources']) and r.get('model_calls')==r.get('GPU_calls')==0 and r.get('numerical_imports') is False and r.get('wall_seconds',CPU['wall_seconds']+1)<=CPU['wall_seconds'] and r['output_monitor']['thread_stopped'] and not r['output_monitor']['fault'] and r['output_monitor']['peak_apparent_bytes']<=CPU['new_HF_bytes'],'actual CPU preflight EXIT0 required');return proof

def final_memory(torch):
 try:
  return capture.memory_snapshot(torch) if torch.cuda.is_initialized() else {'status':'UNKNOWN_CUDA_NOT_INITIALIZED'}
 except BaseException as e:return {'status':'UNKNOWN','error_type':type(e).__name__}

def run(a):
 started=time.monotonic();p=native.read(a.protocol);digest=native.sha(a.protocol);s=authorize(p,digest);verify_job(a,p,s,'run');need(s.get('execution_status')=='RUNNING_FIXED_SCOPE' and s.get('native_group_active_slots')==1,'running slot required');cpu=require_cpu(p,digest,s);need(not {'torch','numpy'}&set(sys.modules),'fresh numerical process');need({k:importlib.metadata.version(k) for k in p['runtime_versions']}==p['runtime_versions'],'runtime versions differ');out=HF/p['output_HF_relative'];need(not out.exists(),'fresh output');out.mkdir(parents=True);native.ID,native.LIMITS,native.RUN_IDS,native.RESOURCE=p['id'],limits(p['part']),tuple(p['run_ids']),RES
 report={'id':p['id'],'protocol_sha256':digest,'job_spec_sha256':s['gpu_launch_spec_sha256'],'status':'STARTED_NO_RETRY','source_ordinal':p['case']['native_source_ordinal'],'seed':42,'quality_acceptance':False,'CPU_preflight':cpu,'automatic_retry':False,'phases':[],'captures':[],'current_phase':None,'current_state':None,'operation_ledger':{},'checkpoint_input_identity':'original BF16 two masters; physical mmap read bytes UNKNOWN'};emergency=support.reserve_emergency(out.parent);monitor=None;counter=io.WrittenBytes(out,RES['new_HF_bytes'],lambda:monitor.check() if monitor else None)
 for rid in p['run_ids']:counter.hold(str(counter.path(out/rid/'output.mp4')),RES['video_bytes_max'])
 video_hash_started=set()
 journal=counter.open_journal(out/'OPERATIONS.jsonl');events=counter.open_journal(out/'ACTIONS.jsonl');ledger=vm.Ledger(report['operation_ledger'],lambda x:journal.write(json.dumps(x,separators=(',',':'))+'\n'));persist=lambda:counter.write_json(out/'RUN.json',report)
 def check():
  if monitor:monitor.check()
  counter.check();usage=support.worker_usage(out.parent);need(usage['apparent_bytes']+counter.summary()['unconsumed_reserved_bytes']+(1<<20)<=RES['new_HF_bytes'],'worker output limit');available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024;need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=RES['max_RSS_bytes'] and available>=RES['host_available_min_bytes'] and shutil.disk_usage(out).free>=RES['free_disk_min_bytes'] and time.monotonic()-started<=RES['wall_seconds'],'host/disk/wall limit')
  if 'torch' in sys.modules:need(sys.modules['torch'].cuda.max_memory_reserved()<=RES['GPU_reserved_max_bytes'],'GPU memory limit')
 persist();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('wall limit')));signal.signal(signal.SIGUSR1,lambda *_:(_ for _ in ()).throw(RuntimeError('output monitor')));signal.alarm(RES['wall_seconds']);out_monitor=support.UsageMonitor(out.parent,RES['new_HF_bytes'],RES['free_disk_min_bytes']);out_monitor.start();finder=None;no_hif4=native.NoHiF4();import e0_teal_wan_fullm_capture as fullm_guard
 file_guard=fullm_guard.single_file_bound(RES['single_file_bytes_max']);file_guard_entered=False
 try:
  from e0_b_final_block_capture_local_package import CheckedPackageFinder
  local,rows=charged_assets(p,a.local_root,report,persist);finder=CheckedPackageFinder(local,rows);file_guard.__enter__();file_guard_entered=True;sys.dont_write_bytecode=True;sys.path[:0]=[str(local),str(ROOT/'.runtime/code/LightX2V')];sys.meta_path[:0]=[no_hif4,finder];os.environ.update(DTYPE='BF16',SENSITIVE_LAYER_DTYPE='BF16',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
  import torch
  from safetensors.torch import save_file
  kernels=None
  from ub1_consumer_runtime import tensor_sha
  from run_ub1_consumer_videos import inspect_video
  reservation,monitor,report['CUDA_reservation']=vm.reserved_startup.reserve_before_assets(torch);need(torch.cuda.device_count()==1 and torch.cuda.get_device_capability(0)==(8,0) and 'A100' in torch.cuda.get_device_name(0),'A100 required');torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
  need(torch.__version__=='2.10.0+cu128' and torch.version.cuda=='12.8' and torch.cuda.get_allocator_backend()=='native','bound native allocator/recipe differs')
  report['precision']={'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'gpu_name':torch.cuda.get_device_name(0),'allow_tf32':torch.backends.cuda.matmul.allow_tf32,'allow_bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,'float32_matmul_precision':torch.get_float32_matmul_precision()};persist()
  with native.OriginalGuard(torch,p,ledger) as guard:
   begin=time.perf_counter();pipe=native.build_pipeline(p);torch.cuda.synchronize();report.update(raw_model_setup_seconds=time.perf_counter()-begin,original_loader_paths=guard.loaded);persist();check()
   def progress(event,data):events.write(json.dumps({'event':event,'run_id':data['run_id']},separators=(',',':'))+'\n');prior.update_progress(report,event,data,persist)
   mode_for_id=dict(zip(p['run_ids'],driver.MODES));report['prefix_byte_controls']=[]
   def capture_factory(rid):
    folder=out/rid;folder.mkdir();states={}
    def save(value,name,key):
     def write():
      prior.mark_state(report,rid,name,'STARTED',persist);dtype=torch.float32 if key in ('raw_FP32_noise_files','native_FP32_flow_files') else torch.bfloat16;need(name not in states and value.dtype==dtype and list(value.shape)==native.VIDEO['latent_shape'],'state differs');path=folder/(name+'.safetensors');counter.reserve_payload(value.numel()*value.element_size())
      def d2h():
       cpu=value.detach().cpu().contiguous();need(bool(torch.isfinite(cpu).all()),'nonfinite state');return cpu
      cpu=ledger.call(rid,{'state_tensor_D2H':1,'state_tensor_D2H_bytes':value.numel()*value.element_size()},d2h)
      try:ledger.call(rid,{'state_output_writes':1},save_file,{'flow' if key=='native_FP32_flow_files' else 'latents':cpu},str(path))
      finally:
       if path.exists():ledger.call(rid,{'state_file_hash_calls':1,'state_file_hash_reserved_bytes':value.numel()*value.element_size()+(1<<20)},counter.observe_external,path)
      states[name]={'root':'hf',**counter.binding(path,HF),'raw_tensor_sha256':ledger.call(rid,{'state_raw_sha_calls':1,'state_raw_sha_bytes':cpu.numel()*cpu.element_size()},tensor_sha,cpu),'shape':list(cpu.shape),'dtype':str(cpu.dtype).removeprefix('torch.')};counter.write_json(folder/'STATES.json',states);report.setdefault('state_index_bindings',{})[rid]=counter.binding(folder/'STATES.json',HF)
      label=mode_for_id[rid];reference=p['source_parent']['dense' if label=='h_cache_dense' else 'student'];refs=[(label,reference['states'][name])]
      for label,old in refs:
       q=(HF/old['path']).resolve();same=ledger.call(rid,{'reference_state_checks':1,'reference_state_read_reservation':18<<20},lambda:compare_reference(q,path,old['bytes'],old['sha256'],states[name]['sha256']));report['prefix_byte_controls'].append({'mode':label,'state':name,'equal':same});persist();need(same,'reference state byte differs')
      prior.mark_state(report,rid,name,'COMPLETE',persist);return states[name]
     return ledger.call(rid,{key:1},write)
    return save,states
   def inspect(rid,path):
    def hash_and_inspect():
     video_hash_started.add(rid);return inspect_video(path,native.VIDEO)
    value=ledger.call(rid,{'video_geometry_checks':1,'video_file_hash_calls':1,'video_file_hash_reserved_bytes':RES['video_bytes_max']},hash_and_inspect);counter.observe_external(path,hold=str(counter.path(path)),known=value);mode=mode_for_id[rid]
    refs=[(mode,p['source_parent']['dense' if mode=='h_cache_dense' else 'student']['video'])]
    for label,old in refs:
     q=Path(old['path']);q=q.resolve() if q.is_absolute() else (HF/q).resolve();same=ledger.call(rid,{'reference_video_checks':1,'reference_video_read_reservation':64<<20},lambda:compare_video_reference(q,path,old['bytes'],old['sha256'],value['sha256']));report.setdefault('neutral_video_checks',{})[label]=same;persist();need(same,'reference video byte differs')
    return value
   def auth(rid):return ledger.call(rid,{'authority_checks':1},authorize,p,digest)
   def emit_action(row):
    events.write(json.dumps(row,separators=(',',':'))+'\n')
    if row.get('event')=='cache_boundary_memory':report.setdefault('cache_boundary_memory',[]).append(row);persist()
    if row.get('event')=='candidate_packed_action':report.setdefault('mask_observations',[]).append(row);persist()
    if row.get('event') in ('phase_memory_before','phase_memory_after','expert_boundary_pre_cleanup','expert_boundary_cleanup'):report.setdefault('memory_observations',[]).append(row);persist()
   from chipmunk_student_observation_core import Observer
   from chipmunk_student_observation_sink import Sink
   from chipmunk_student_observation_write import native_bf16_write
   def host_guard(amount):
    check(); need(type(amount) is int and 0 < amount <= (1<<30),'host copy budget differs')
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    need(rss+amount <= RES['max_RSS_bytes'] and available-amount >= RES['host_available_min_bytes'],'host copy headroom differs')
   def observer_factory(mode,rows,selector_list):
    rid=p['run_ids'][driver.MODES.index(mode)]
    sink=Sink(torch=torch,counter=counter,out=out/rid/'observer',hf_root=HF,save_file=save_file,max_bytes=RES['new_HF_bytes'],host_guard=host_guard,max_host_copy_bytes=1<<30)
    return Observer(write_formula=lambda value,gate,start,end,residual,attention:native_bf16_write(torch,value,gate,start,end,residual,attention),sink=sink,selector_indices=selector_list)
   result=driver.execute(observer_factory=observer_factory,torch=torch,pipe=pipe,guard=guard,vm=vm,native=native,run_ids=p['run_ids'],image_path=HF/p['case']['image']['path'],prompt=p['case']['prompt'],output_paths=[out/r/'output.mp4' for r in p['run_ids']],capture_factory=capture_factory,inspect_video=inspect,check=check,authorize=auth,gate=lambda rid,cost,fn:ledger.call(rid,cost,fn),progress=progress,kernels=kernels,emit=emit_action)
   report.update(phases=result['phases'],historical_reference_matches=True,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved());ledger.finish();check();report['status']='COMPLETE_CHIPMUNK_STUDENT_OBSERVATION_EXIT_PENDING';persist()
 except BaseException as e:report.update(status='FAILED_CHIPMUNK_STUDENT_OBSERVATION_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e)[:500],error_notes=[str(x)[:1000] for x in getattr(e,'__notes__',[])]);raise
 finally:
  signal.alarm(0);monitor_error=None
  if 'torch' in sys.modules:report['final_memory']=final_memory(sys.modules['torch'])
  if monitor:
   try:monitor.close();report['GPU_monitor']=monitor.report();need(report['GPU_monitor']['sampled_isolation_pass'] and not report['GPU_monitor']['violations'],'GPU monitor violation')
   except BaseException as e:monitor_error=e;report.update(status='FAILED_MONITOR',resource_error=str(e))
  for x in (finder,no_hif4):
   if x in sys.meta_path:sys.meta_path.remove(x)
  for rid in p['run_ids']:
   path=counter.path(out/rid/'output.mp4')
   if path not in counter.files and path.exists():
    try:
     observe_failed_video(counter,rid,path,video_hash_started,report,persist)
    except BaseException as e:counter.accounting_errors.append(str(e))
  status=out_monitor.stop();report.update(wall_seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,output_monitor=status,output_accounting=counter.summary(),operation_journal=counter.binding(out/'OPERATIONS.jsonl',HF),action_journal=counter.binding(out/'ACTIONS.jsonl',HF))
  if status['fault'] or not status['thread_stopped']:report['status']='FAILED_MONITOR';monitor_error=RuntimeError('output monitor failure')
  try:persist()
  except BaseException as e:support.emergency_failure(emergency,{'id':p['id'],'status':report['status']},'final RUN persist failed '+type(e).__name__);raise
  finally:
   journal.close();events.close()
   if file_guard_entered:file_guard.__exit__(None,None,None)
  if monitor_error:raise monitor_error

def authorize_verification(p,d,a):
 vid=p['id']+'_VERIFY';active=native.read(ROOT/'research/ACTIVE_RESEARCH.json');rows=[x for x in active['experiment_authority']['current_user_authorized_scopes'] if x.get('id')==vid];need(len(rows)==1,'independent verifier ACTIVE required');s=rows[0];need(s.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') and s.get('protocol_sha256')==d and s.get('resources')==VERIFY and s.get('max_gpu_cards')==0 and s.get('parent_scope_id')==p['id'],'verifier scope differs');r=native.read(native.bound(ROOT,s['review']));need(r.get('scope_id')==vid and r.get('review_schema')=='FFNDS_PREEXEC_REVIEW_V1' and r.get('state')=='REVIEWER_RECHECK_APPROVED' and r.get('recheck_disposition')=='APPROVE' and r.get('author_agent_id') and r.get('reviewer_agent_id') and r['author_agent_id']!=r['reviewer_agent_id'] and r.get('approval_scope_fingerprint')==fingerprint(p) and r.get('verification_spec_sha256')==s.get('verification_spec_sha256'),'verifier review differs')
 for k in ('author_review','independent_review','reviewer_recheck'):native.bound(ROOT,r[k])
 need(r['reviewer_agent_id'] not in AUTHORS, 'verifier reviewer must be independent of all authors')
 need(a.job_spec and native.sha(a.job_spec)==s['verification_spec_sha256'],'verifier spec differs');j=native.read(a.job_spec);v=j['argv'];need(len(v)>1 and v[0]=='env' and v[1].startswith('FFNDS_WORKER_CONFIG='),'profile missing');profile=Path(v[1].split('=',1)[1]);need(native.sha(profile)==p['execution_private_worker_config_sha256'],'verifier profile differs');cpus=sorted(os.sched_getaffinity(0));need(len(cpus)==1 and cpus[0] in p['cpu_affinity'] and sys.flags.isolated and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU isolation differs');out=HF/p['output_HF_relative'];dest=out.parent.parent/'verification/worker/results';expected=['env',v[1],'OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PATH=/usr/bin:/bin','CUDA_VISIBLE_DEVICES=','taskset','-c',str(cpus[0]),str(PYTHON),'-I','-B',str(ROOT/SELF),'verify-closed','--protocol',str(a.protocol.resolve()),'--hf-root',str(HF),'--job-spec',str(a.job_spec.resolve()),'--run',str(out/'RUN.json'),'--exit',str(out.parent/'EXIT_RECEIPT.json'),'--verify-out',str(dest/'RUN.json')];need(v==expected and j['id']==vid and j['protocol_sha256']==d and Path(j['protocol_path']).resolve()==a.protocol.resolve() and Path(j['cwd']).resolve()==ROOT and Path(j['hf_root']).resolve()==HF and Path(j['output_dir']).resolve()==dest.parent and Path(j['run_json']).resolve()==dest/'RUN.json' and j['external_timeout_seconds']==VERIFY['outer_seconds'] and a.verify_out.resolve()==dest/'RUN.json','verifier exact job differs');import worker_identity;worker_identity.require_worker();return s

def observation_assets(phases):
    """Validate expected per-site tensor manifest and enumerate each output once."""
    assets = []
    fields = ('V1', 'old_h', 'current_h', 'old_Y', 'current_dense_Y')
    for phase in phases:
        mode = phase['mode']
        manifest = phase['observer_manifest']
        need(manifest['sites'] == 40 and manifest['quality_acceptance'] is False and len(manifest['records'])==40 and
             set(map(int, manifest['records'])) == set(range(40)), 'observation coverage differs')
        expected_fields = fields + (('candidate_Y',) if mode == MODES[1] else ())
        for key, record in manifest['records'].items():
            block = int(key)
            prefix = f"/{phase['run_id']}/observer/block{block:02d}_"
            need(record['metadata']['path'].endswith(prefix+'metadata.json'), 'metadata location differs')
            assets.append({'kind': 'metadata', 'binding': record['metadata'], 'mode': mode, 'block': block})
            sample = record['samples']
            need(sample['file']['path'].endswith(prefix+'samples.safetensors'), 'sample location differs')
            need(set(sample['tensors']) == set(expected_fields), 'sample tensor names differ')
            typed = {}
            for name, tensor in sample['tensors'].items():
                shape = [320, 13824 if name in ('old_h','current_h') else 5120]
                need(tensor['dtype'] == 'bfloat16' and tensor['shape'] == shape, 'sample tensor type differs')
                typed[name] = {'dtype': 'BF16', 'shape': shape, 'raw_sha256': tensor['raw_sha256']}
            assets.append({'kind': 'tensor', 'binding': sample['file'], 'tensors': typed})
            snapshot = record.get('snapshot', {})
            need(set(snapshot) == (set(expected_fields) if block in contract.SNAPSHOT_BLOCKS else set()),
                 'snapshot coverage differs')
            for name, item in snapshot.items():
                need(item['file']['path'].endswith(prefix+'full_'+name+'.safetensors'), 'snapshot location differs')
                shape = [31668, 13824 if name in ('old_h','current_h') else 5120]
                need(set(item['tensors']) == {name}, 'snapshot tensor names differ')
                tensor = item['tensors'][name]
                need(tensor['dtype'] == 'bfloat16' and tensor['shape'] == shape, 'snapshot type differs')
                assets.append({'kind': 'tensor', 'binding': item['file'], 'tensors':
                               {name: {'dtype': 'BF16', 'shape': shape, 'raw_sha256': tensor['raw_sha256']}}})
            if mode == MODES[1]:
                expected_counts = [14*26,14*26,15*26,15*26]*21
                required_stats={'count','pre_squared','gated_squared','write_squared','ideal_total','ideal_diagonal',
                    'ideal_cross','ideal_vs_actual_gated_squared','dense_pre_squared','dense_gated_squared',
                    'dense_write_squared','signed_pre_sum','signed_gated_sum','signed_write_sum'}
                need(record['chunks'] == 124 and set(record['partitions'])==required_stats and
                     record['partitions']['count'] == expected_counts and
                     all(len(values)==84 and all(isinstance(x,(int,float)) and __import__('math').isfinite(x) for x in values)
                         for values in record['partitions'].values()),
                     'student partition coverage differs')
                for name in ('stats', 'stats_progress'):
                    need(name in record, 'missing bounded stats binding')
                    need(record[name]['path'].endswith(prefix+name+'.json'), 'stats location differs')
                    assets.append({'kind': name, 'binding': record[name], 'block': block,
                                   'partitions':record['partitions'], 'chunks':record['chunks']})
            else:
                need(record['chunks'] == 0, 'dense partition coverage differs')
    return assets

def validate_site_metadata(meta, mode, block, expected_selector, expected_mask_sha=None):
    """Validate persisted real-gate identity, frozen sampling and actual mask."""
    from chipmunk_student_observation_coordinates import coordinates
    import math
    need(mode in MODES and meta['block']==block, 'metadata site identity differs')
    selector=meta['selector_rows']; gate=meta['gate']
    need(len(selector)==64 and len(set(selector))==64 and selector==expected_selector and
         meta['evaluation_rows']==coordinates(selector), 'coordinate sampling differs')
    need(meta['gate_shape']==[1,1,5120] and meta['gate_stride']==[30720,5120,1] and
         meta['gate_dtype']=='bfloat16' and len(gate)==5120 and
         all(isinstance(x,(int,float)) and math.isfinite(x) for x in gate), 'gate identity differs')
    if mode==MODES[1]:
        mask=meta['mask']; score=meta['selector_score']
        need(len(mask)==5120 and mask==sorted(set(mask)) and all(type(x) is int and 0<=x<13824 for x in mask) and
             len(score)==13824 and all(isinstance(x,(int,float)) and math.isfinite(x) for x in score) and
             isinstance(expected_mask_sha,str) and
             hashlib.sha256(struct.pack('<'+'q'*len(mask),*mask)).hexdigest()==expected_mask_sha,
             'actual mask metadata differs')
        selected=sorted(sorted(range(len(score)),key=lambda index:(-score[index],index))[:5120])
        need(mask==selected, 'mask does not match stable selector scores')
    return True

def verify_closed(a):
    p = native.read(a.protocol); validate(p, True); digest = native.sha(a.protocol)
    scope = authorize_verification(p, digest, a); out = HF / p['output_HF_relative']
    need(a.run.resolve() == (out/'RUN.json').resolve() and a.exit.resolve() == (out.parent/'EXIT_RECEIPT.json').resolve(), 'RUN/EXIT paths differ')
    report = native.read(a.run); ex = native.read(a.exit); rb = native.binding(a.run, HF)
    vm.exited(ex, rb, digest, p['id'])
    need(ex.get('spec_sha256') == report.get('job_spec_sha256') and
         report.get('status') == 'COMPLETE_CHIPMUNK_STUDENT_OBSERVATION_EXIT_PENDING' and
         report.get('historical_reference_matches') is True and report.get('source_ordinal') == SOURCES[p['part']] and
         report.get('neutral_video_checks') == {mode: True for mode in MODES} and
         report.get('current_phase',{}).get('status') == 'EVIDENCE_COMPLETE' and report.get('current_state') is None,
         'execution identity differs')
    ledger = report['operation_ledger']; expected = {k:v for k,v in limits(p['part']).items() if v}
    need(ledger['predebited'] == ledger['observed'] == expected and not ledger['forbidden_attempts'], 'ledger totals differ')
    phases = report['phases']
    verify_reference_metadata(p['source_parent'], phases, report['prefix_byte_controls'],
        [{'mode': k, 'equal': v} for k,v in report['neutral_video_checks'].items()])
    need(all(z['actual_full_FFN_sites']==160 and z['logical_FFN_linear_dispatches']==320 and
         z['native_transfer_observations']==162 and z['cache_creations']==z['cache_releases']==40 for z in phases) and
         [z['sparse_sites'] for z in phases] == [0,40] and [z['actual_dense_FFN_GEMMs'] for z in phases] == [320,240],
         'native coverage differs')
    need([x['run_id'] for x in phases] == p['run_ids'], 'phase run identity differs')
    cache=report['cache_boundary_memory']; masks=report['mask_observations']; memory=report['memory_observations']
    need(len(cache)==80 and {(x['run_id'],x['block']) for x in cache}=={(rid,b) for rid in p['run_ids'] for b in range(40)} and
         all(x['step']==0 and x['expert']==0 and x['mode']==MODES[p['run_ids'].index(x['run_id'])] and
             x['cache_bytes']==(x['block']+1)*31668*(13824+5120)*2 and
             x['memory']['peak_stats_reset'] is False for x in cache), 'cache identity differs')
    need(len(masks)==40 and {(x['run_id'],x['block']) for x in masks}=={(p['run_ids'][1],b) for b in range(40)} and
         all(x['step']==1 and x['expert']==0 and x['mode']==MODES[1] and len(x['mask_indices_sha256'])==64 for x in masks),
         'mask event identity differs')
    need(sum(x['event']=='expert_boundary_pre_cleanup' for x in memory)==2 and
         sum(x['event']=='expert_boundary_cleanup' for x in memory)==2 and
         sum(x['event'] in ('phase_memory_before','phase_memory_after') for x in memory)==4, 'memory event coverage differs')
    need(report['wall_seconds'] <= RES['wall_seconds'] and report['peak_RSS_bytes'] <= RES['max_RSS_bytes'] and
         report['peak_GPU_reserved_bytes'] <= RES['GPU_reserved_max_bytes'] and
         report['GPU_monitor']['sampled_isolation_pass'] and not report['GPU_monitor']['violations'] and
         report['output_monitor']['thread_stopped'] and not report['output_monitor']['fault'] and
         report['output_accounting']['peak_tracked_bytes_plus_reserved_and_atomic_temp'] <= RES['new_HF_bytes'] and
         report['output_accounting']['unconsumed_reserved_bytes'] == 0 and not report['output_accounting']['accounting_errors'] and
         report['final_memory']['peak_stats_reset'] is False, 'resource evidence differs')
    assets = observation_assets(phases)
    for phase in phases:
        need(phase['run_id'] in report['state_index_bindings'], 'state index missing')
        assets.append({'kind':'state_index','binding':report['state_index_bindings'][phase['run_id']], 'states':phase['states']})
        for name, binding in phase['states'].items():
            dtype = 'F32' if name == 'raw_noise_FP32' or name.startswith('flow_step') else 'BF16'
            tensor_name = 'flow' if name.startswith('flow_step') else 'latents'
            need(binding['shape'] == native.VIDEO['latent_shape'] and binding['dtype'] == ('float32' if dtype == 'F32' else 'bfloat16'), 'state shape differs')
            assets.append({'kind':'tensor', 'binding':binding, 'tensors':{tensor_name:{'dtype':dtype,'shape':binding['shape'],'raw_sha256':binding['raw_tensor_sha256']}}})
        video = phase['video']; vp = Path(video['path']).resolve(); need(vp.is_relative_to(HF), 'video escaped HF')
        assets.append({'kind':'file','binding':{'path':str(vp.relative_to(HF)), 'bytes':video['bytes'], 'sha256':video['sha256']}})
    assets.extend({'kind':k,'binding':report[k]} for k in ('operation_journal','action_journal'))
    need(len({x['binding']['path'] for x in assets}) == len(assets) and
         sum(x['binding']['bytes'] for x in assets) <= VERIFY['read_payload_bytes_max'] and not a.verify_out.exists(), 'artifact budget differs')
    a.verify_out.parent.mkdir(parents=True, exist_ok=True); started = time.monotonic()
    result = {'id': p['id']+'_VERIFY', 'protocol_sha256':digest, 'job_spec_sha256':scope['verification_spec_sha256'],
        'parent_scope_id':p['id'], 'status':'STARTED_READ_ONLY_VERIFY','predebited_bytes':0,'completed_bytes':0,
        'inflight':'NONE','automatic_retry':False,
        'read_accounting':'artifact payload reads predebited; protocol, parent, receipt and output-monitor metadata I/O separately UNKNOWN'}
    def persist_verify():
        data=(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')
        used=support.worker_usage(a.verify_out.parent.parent)['apparent_bytes']
        need(used+len(data)<=VERIFY['new_HF_bytes'], 'verification output budget exceeded')
        native.write(a.verify_out,result)
    persist_verify(); signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('verification wall'))); signal.alarm(VERIFY['wall_seconds'])
    signal.signal(signal.SIGUSR1,lambda *_:(_ for _ in ()).throw(RuntimeError('verification output monitor failure')))
    monitor=support.UsageMonitor(a.verify_out.parent.parent,VERIFY['new_HF_bytes'],0); monitor.start()
    try:
        selectors = None
        action_masks = {x['block']:x['mask_indices_sha256'] for x in report['mask_observations']}
        need(len(action_masks) == 40 and set(action_masks) == set(range(40)), 'mask event coverage differs')
        for item in assets:
            b=item['binding']; path=(HF/b['path']).resolve()
            need(path.is_relative_to(out.resolve()) and path.is_file() and path.stat().st_size==b['bytes'], 'artifact path/size differs')
            result['predebited_bytes']+=b['bytes']; result['inflight']={'path':b['path'],'bytes':b['bytes'],'completed':'UNKNOWN'}; persist_verify()
            if item['kind']=='tensor':
                inspect_typed_payload(path,b['sha256'],item['tensors'])
            elif item['kind'] in ('operation_journal','action_journal'):
                sha=hashlib.sha256(); n=0; rows=[]
                with path.open('rb') as stream:
                    for line in stream:
                        n+=len(line); sha.update(line); rows.append(json.loads(line))
                need(n==b['bytes'] and sha.hexdigest()==b['sha256'], 'journal identity differs')
                if item['kind']=='operation_journal':
                    need(deferred.replay(rows)=={'predebited':ledger['predebited'],'observed':ledger['observed'],'pending_count':0}, 'journal replay differs')
                else:
                    need([z for z in rows if z.get('event')=='cache_boundary_memory']==report['cache_boundary_memory'] and
                         [z for z in rows if z.get('event')=='candidate_packed_action']==report['mask_observations'], 'action journal differs')
            elif item['kind'] == 'state_index':
                with path.open('rb') as stream: data=stream.read()
                need(len(data)==b['bytes'] and hashlib.sha256(data).hexdigest()==b['sha256'] and json.loads(data)==item['states'], 'state index differs')
            elif item['kind'] in ('stats', 'stats_progress'):
                with path.open('rb') as stream: data=stream.read()
                value=json.loads(data)
                need(len(data)==b['bytes'] and hashlib.sha256(data).hexdigest()==b['sha256'] and
                     value['chunks']==item['chunks'] and value['partitions']==item['partitions'], 'partition output differs')
            elif item['kind'] == 'metadata':
                with path.open('rb') as stream: data=stream.read()
                need(len(data)==b['bytes'] and hashlib.sha256(data).hexdigest()==b['sha256'], 'metadata hash differs')
                meta=json.loads(data)
                if selectors is None: selectors=meta['selector_rows']
                validate_site_metadata(meta,item['mode'],item['block'],selectors,action_masks.get(item['block']))
            else:
                native.bound(HF,b)
            result['completed_bytes']+=b['bytes']; result['inflight']='NONE'; persist_verify()
        result.update(status='PASS_CLOSED_CHIPMUNK_STUDENT_OBSERVATION', RUN=rb,
                      exit=native.binding(a.exit,HF),artifact_count=len(assets),quality_acceptance=False)
    except BaseException as e:
        result.update(status='FAILED_READ_ONLY_VERIFY_NO_RETRY',error_type=type(e).__name__,error=str(e)[:500]); raise
    finally:
        signal.alarm(0); result['wall_seconds']=time.monotonic()-started; result['output_monitor']=monitor.stop()
        if result['output_monitor']['fault'] or not result['output_monitor']['thread_stopped']:
            result['status']='FAILED_VERIFY_MONITOR'; persist_verify(); raise RuntimeError('verification output monitor failure')
        persist_verify()
    return result

def inspect_verification(a):
    p=native.read(a.protocol); validate(p,True); digest=native.sha(a.protocol); out=HF/p['output_HF_relative']
    run=out.parent.parent/'verification/worker/results/RUN.json'; ex=run.parent.parent/'EXIT_RECEIPT.json'
    r=native.read(run); e=native.read(ex)
    need(r['id']==p['id']+'_VERIFY' and r['status']=='PASS_CLOSED_CHIPMUNK_STUDENT_OBSERVATION' and
         r['protocol_sha256']==digest and e['id']==r['id'] and e['process_exited'] is True and
         e['child_returncode']==0 and e['timeout'] is False and e['automatic_retry'] is False and
         e['protocol_sha256']==digest and e['run_sha256']==native.sha(run) and e['spec_sha256']==r['job_spec_sha256'] and
         r['predebited_bytes']==r['completed_bytes'] and r['inflight']=='NONE' and
         r['wall_seconds']<=VERIFY['wall_seconds'] and r['output_monitor']['thread_stopped'] and
         not r['output_monitor']['fault'] and r['output_monitor']['peak_apparent_bytes']<=VERIFY['new_HF_bytes'],
         'actual verifier EXIT0 required')
    return {'status':'PASS_ACTUAL_EXIT_CLOSED_CHIPMUNK_STUDENT_OBSERVATION','id':p['id'],
            'verification_RUN':native.binding(run,HF),'verification_exit':native.binding(ex,HF),'quality_acceptance':False}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=('draft', 'inspect', 'preflight', 'run', 'verify-closed', 'inspect-verification', 'freeze'))
    ap.add_argument('--part', type=int)
    ap.add_argument('--worker-profile', type=Path)
    ap.add_argument('--cpu-affinity', default='0,1,2,3')
    ap.add_argument('--protocol', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--verify-out', type=Path)
    ap.add_argument('--hf-root', type=Path, default=HF)
    ap.add_argument('--local-root', type=Path)
    ap.add_argument('--job-spec', type=Path)
    ap.add_argument('--run', type=Path)
    ap.add_argument('--exit', type=Path)
    a = ap.parse_args()
    need(a.hf_root.resolve() == HF, 'HF root differs')
    if a.action == 'draft':
        need(a.out and a.worker_profile and not a.out.exists(), 'fresh output and profile required')
        p = draft(a.part, a.worker_profile, [int(x) for x in a.cpu_affinity.split(',')])
        native.write(a.out, p)
    elif a.action == 'inspect':
        p = native.read(a.protocol)
        validate(p, p['status'] == 'FROZEN_PENDING_REGISTRATION')
        print(json.dumps({'id': p['id'], 'status': p['status'], 'fingerprint': fingerprint(p), 'requires':'independent review and ACTIVE'}))
    elif a.action == 'freeze':
        need(a.out and not a.out.exists(), 'fresh frozen output required')
        p = native.read(a.protocol); validate(p)
        p['status']='FROZEN_PENDING_REGISTRATION'; p['assets_finalized']=True; native.write(a.out,p)
    elif a.action == 'preflight': preflight(a)
    elif a.action == 'run': run(a)
    elif a.action == 'verify-closed': print(json.dumps(verify_closed(a)))
    else: print(json.dumps(inspect_verification(a)))

if __name__ == '__main__':
    main()
