"""Four saved-latent VAE renders; default is a read-only, torch-free preflight.

The coordinator freezes a separate protocol and registers its exact SHA before
--execute-decode. No denoiser, text/image encoder, fitting or trajectory update.
The immutable DECODE_RUN precedes metrics; an independent inventory binds every
actual clip, its source tensor, hybrid identity and the unchanged core gate.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import traceback

from ffnds_observation_common import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
PASS = 'PASS_VAE_ONLY_RENDER_AND_CORE_METRICS_NOT_QUALITY'
FLOW_PASS = 'PASS_NATIVE_LATENT_IDENTITY_AND_FLOW_SPLIT_NOT_QUALITY'
SHAPE = [16, 21, 58, 104]
VAE_SHA = '38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981'
VAE_BYTES = 507609880
ROLES = [('dense_control', 'z_dense_after'), ('late_dense', 'z_dense_at_student_after')]
DECODE_CONFIG = {'z_dim': 16, 'dtype': 'torch.bfloat16', 'device': 'cuda:0', 'parallel': False,
    'use_tiling': False, 'cpu_offload': False, 'use_lightvae': False,
    'load_from_rank0': False, 'dummy_model': False, 'use_cache_vae': False}
MANDATORY_DRIVER_SOURCES = {'tools/' + f for f in [
    'run_e0_saved_flow_decode.py', 'ffnds_observation_common.py', 'ffnds_video_metrics.py',
    'ffnds_video_metric_gate.py', 'ffnds_paired_video_strict_gate.py',
    'startup_gpu_reservation.py', 'assigned_gpu_resources.py', 'worker_identity.py']}
MANDATORY_NATIVE_SOURCES = {'.runtime/code/LightX2V/lightx2v/' + f for f in [
    'models/video_encoders/hf/wan/vae.py', 'utils/utils.py', 'utils/envs.py',
    'models/runners/default_runner.py', 'models/runners/wan/wan_runner.py',
    'models/runners/wan/wan_distill_runner.py']}


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def within(base, relative):
    base = Path(base).resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        raise ValueError('Artifact path escapes its bound root')
    return path


def bound_json(path, expected):
    if sha256(path) != expected:
        raise ValueError('Frozen JSON digest mismatch: ' + Path(path).name)
    return json.loads(Path(path).read_text())


def latent_header(path, key):
    """Read safetensors metadata, never import torch or materialize a tensor."""
    with Path(path).open('rb') as stream:
        size = stream.read(8)
        if len(size) != 8:
            raise ValueError('Truncated safetensors length')
        n = struct.unpack('<Q', size)[0]
        if not 0 < n <= 1 << 20:
            raise ValueError('Unexpected safetensors header size')
        raw = stream.read(n)
        if len(raw) != n:
            raise ValueError('Truncated safetensors header')
        entry = json.loads(raw)[key]
        start, end = entry['data_offsets']
        if (entry['dtype'] != 'BF16' or entry['shape'] != SHAPE
                or end - start != 2 * 16 * 21 * 58 * 104
                or start < 0 or end > Path(path).stat().st_size - 8 - n):
            raise ValueError('Expected a complete original BF16 C,F,H,W latent')
        # Safetensors stores little-endian contiguous values. This exact raw-key
        # digest independently reconstructs tensor_identity without torch.
        if sys.byteorder != 'little':
            raise ValueError('Native identity requires the original little-endian host')
        header = {'dtype': 'torch.bfloat16', 'shape': SHAPE,
                  'storage': 'contiguous_little_endian'}
        digest = hashlib.sha256(json.dumps(header, sort_keys=True, separators=(',', ':')).encode() + b'\0')
        stream.seek(8 + n + start)
        remaining = end - start
        while remaining:
            chunk = stream.read(min(1 << 20, remaining))
            if not chunk:
                raise ValueError('Truncated latent payload')
            digest.update(chunk)
            remaining -= len(chunk)
    return {'dtype': entry['dtype'], 'shape': entry['shape'], 'native_identity': digest.hexdigest()}


def prepare_pair(pair, derived, samples):
    """Reconstruct the reference from authoritative receipts, not metric text."""
    from ffnds_paired_video_strict_gate import inspect as strict_inspect, model_digest
    path = within(derived, pair['whole_flow_RUN_HF_relative'])
    flow = bound_json(path, pair['whole_flow_RUN_sha256'])
    if (flow['status'] != FLOW_PASS or flow['pair']['source_index'] != pair['source_index']
            or flow['source_id'] != pair['source_id'] or flow['seed'] != pair['seed']
            or flow['pair']['modes'] != ['dense', 'candidate']):
        raise ValueError('Completed whole-flow source/seed/operator identity mismatch')
    saved = [v for v in flow['counterfactual_records'] if v['step'] == 3]
    if len(saved) != 1 or saved[0]['status'] != 'PASS_FLOW_AND_NATIVE_MAP_IDENTITIES':
        raise ValueError('One verified final-step whole-flow record required')
    record = saved[0]
    sidecar = path.parent / 'flow_s3.json'
    if json.loads(sidecar.read_text()) != record:
        raise ValueError('Final-step sidecar disagrees with completed flow receipt')
    tensor = within(path.parent, record['saved_tensors']['path'])
    if sha256(tensor) != record['saved_tensors']['sha256']:
        raise ValueError('Saved whole-flow tensor changed')
    latent_identities = {key: latent_header(tensor, key)['native_identity'] for _, key in ROLES}
    retained = flow['retained_evidence']
    parent_path = within(derived, retained['RUN_HF_relative'])
    parent = bound_json(parent_path, retained['RUN_sha256'])
    if (sha256(parent_path.parent / 'STRICT_BUNDLE.json') != retained['STRICT_BUNDLE_sha256']
            or sha256(parent_path.parent / 'TRACE.json') != retained['TRACE_sha256']
            or strict_inspect(parent_path.parent / 'STRICT_BUNDLE.json')):
        raise ValueError('Original paired video inventory changed')
    native_path = parent_path.parent / 'NATIVE_RUN.json'
    native = bound_json(native_path, parent['native_RUN_sha256'])
    if latent_identities['z_dense_after'] != native['trajectory']['teacher'][-1]['native_identity']:
        raise ValueError('Saved dense latent differs from original native teacher tensor')
    for branch in ['teacher', 'student']:
        fields = ['step', 'sha256', 'shape', 'native_dtype', 'native_identity']
        if ([{k: r[k] for k in fields} for r in flow['trajectory'][branch]] !=
                [{k: r[k] for k in fields} for r in native['trajectory'][branch]]):
            raise ValueError('Whole-flow replay differs from original native trajectory')
    sample = samples[pair['source_index']]
    if sample.get('seed', sample.get('calibration_seed')) != pair['seed']:
        raise ValueError('Canonical calibration seed changed')
    prompt_digest = object_sha({'image_sha256': pair['source_id'], 'prompt': sample['prompt'],
                                'negative_prompt': '', 'contract': native['assets']['contract']})
    if sample['input_provenance']['image_sha256'] != pair['source_id']:
        raise ValueError('Canonical source image identity changed')
    video = within(native_path.parent, native['videos']['teacher']['path'])
    reference = {'video': str(video), 'video_sha256': native['videos']['teacher']['sha256'],
                 'receipt': str(native_path), 'receipt_sha256': sha256(native_path),
                 'identity': {'source_id': pair['source_id'], 'seed': pair['seed'], 'expected_frames': 81,
                    'input_prompt_digest': prompt_digest, 'model_digest': model_digest(native, 'dense'),
                    'protocol_sha256': native['protocol_sha256'], 'run_sha256': sha256(native_path)}}
    if sha256(video) != reference['video_sha256']:
        raise ValueError('Original dense video changed')
    return {'pair': pair, 'tensor': str(tensor), 'tensor_sha256': sha256(tensor),
            'flow_RUN_sha256': sha256(path), 'flow_sidecar_sha256': sha256(sidecar),
            'dense_native_identity': native['trajectory']['teacher'][-1]['native_identity'],
            'latent_native_identities': latent_identities,
            'assets': flow['assets'], 'bank_RUN_sha256': flow['bank_RUN_sha256'], 'reference': reference}


def preflight(protocol, hf_root, root=ROOT):
    plan = json.loads(Path(protocol).read_text())
    if (plan.get('schema') != 'ffnds.saved_flow_decode.protocol.v1'
            or plan.get('max_vae_decodes') != 4 or plan.get('max_new_videos') != 4
            or plan.get('max_HF_output_gib') != .1 or plan.get('planned_simultaneous_GPU_cards') != 1
            or plan.get('automatic_retry') is not False
            or [(p['source_index'], p['seed']) for p in plan['pairs']] != [(0, 42), (12, 54)]
            or len({p['id'] for p in plan['pairs']}) != 2):
        raise ValueError('Unsupported fixed four-decode scope')
    for group, required in [('driver_sources_sha256', MANDATORY_DRIVER_SOURCES),
                            ('native_source_sha256', MANDATORY_NATIVE_SOURCES)]:
        if not required <= set(plan[group]):
            raise ValueError('Missing decoder/evaluator source binding')
        for name, expected in plan[group].items():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Source snapshot names must be repository relative')
            if sha256(within(root, name)) != expected:
                raise ValueError('Frozen decode source drift: ' + name)
    if (type(plan['cudnn_allow_tf32']) is not bool or plan['disk_reserve_gib'] <= 0
            or plan['minimum_available_host_gib'] <= 0):
        raise ValueError('Freeze cuDNN TF32 and positive reserve limits')
    logical = bound_json(root / '.runtime/model_view.json', plan['model_view_sha256'])
    local = bound_json(root / '.runtime/node_local_model_view.json', plan['local_model_view_sha256'])
    if (logical['status'] != 'PASS' or local['status'] != 'PASS'
            or local['logical_model_view_sha256'] != plan['model_view_sha256']):
        raise ValueError('Model view provenance mismatch')
    selected = []
    for view in [logical, local]:
        rows = [r for r in view['files'] if r['filename'] == 'Wan2.1_VAE.pth']
        if len(rows) != 1 or rows[0]['bytes'] != VAE_BYTES or rows[0]['sha256'] != VAE_SHA:
            raise ValueError('Wrong original Wan2.1 VAE asset')
        selected.append(rows[0])
    keys = ['repo', 'revision', 'filename', 'bytes', 'sha256']
    if any(selected[0][k] != selected[1][k] for k in keys):
        raise ValueError('Logical/local VAE identities differ')
    vae = Path(selected[1]['view_path'])
    if vae.stat().st_size != VAE_BYTES or sha256(vae) != VAE_SHA:
        raise ValueError('Actual VAE checkpoint bytes changed')
    samples = bound_json(within(root, plan['input_default']), plan['input_default_sha256'])
    prepared = [prepare_pair(p, Path(hf_root) / 'ffnds-derived', samples) for p in plan['pairs']]
    if any(p['assets']['model_view_sha256'] != plan['model_view_sha256'] for p in prepared):
        raise ValueError('Decoder asset view differs from parent native model view')
    active = json.loads((root / 'research/ACTIVE_RESEARCH.json').read_text())
    authorized = any(s['id'] == plan['id'] and s['protocol_sha256'] == sha256(protocol)
        and s.get('execution_status') in ['AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE']
        for s in active['experiment_authority']['current_user_authorized_scopes'])
    return plan, prepared, vae, {'status': 'STATIC_PROVENANCE_PASS_NOT_NATIVE_DECODE_VALIDATION',
        'model_scope_authorized': authorized, 'GPU_initialized': False, 'VAE_decodes_started': 0}


@contextmanager
def ledger_lock(study):
    study.mkdir(parents=True, exist_ok=True)
    with (study / 'LEDGER.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def reserve_attempt(study, plan, protocol_sha):
    with ledger_lock(study):
        path = study / 'STUDY.json'
        if path.exists():
            raise RuntimeError('Four-decode scope already consumed; no automatic retry')
        write_json(path, {'id': plan['id'], 'protocol_sha256': protocol_sha, 'status': 'STARTED',
                          'VAE_decodes_reserved': 4, 'new_videos_reserved': 4})


def close_attempt(study, status, summary):
    with ledger_lock(study):
        state = json.loads((study / 'STUDY.json').read_text())
        state.update(status=status, RUN_sha256=sha256(summary))
        write_json(study / 'STUDY.json', state)


def clip_identity(entry, run_sha, protocol_sha):
    return {'source_id': entry['source_id'], 'seed': entry['seed'], 'expected_frames': 81,
            'input_prompt_digest': entry['input_prompt_digest'], 'model_digest': entry['model_digest'],
            'protocol_sha256': protocol_sha, 'run_sha256': run_sha,
            'decoder_digest': entry['decoder_digest']}


def decoder_binding(plan):
    return {'VAE_sha256': VAE_SHA, 'configuration': DECODE_CONFIG,
        'sources_sha256': {name: plan['native_source_sha256'][name] for name in sorted(MANDATORY_NATIVE_SOURCES)},
        'environment': {'DTYPE': 'BF16', 'SENSITIVE_LAYER_DTYPE': 'None', 'USE_CHANNELS_LAST_3D': '0',
            'cudnn_benchmark': False, 'cudnn_deterministic': True,
            'cudnn_allow_tf32': plan['cudnn_allow_tf32'], 'cuda_matmul_allow_tf32': False},
        'media': {'fps': 16, 'method': 'ffmpeg', 'lossless': False, 'output_pix_fmt': 'yuv420p'}}


def output_name(source_index, role):
    return f'source{source_index:02d}_{role}.mp4'


def source_model_digest(source, role):
    if role == 'dense_control':
        return source['reference']['identity']['model_digest']
    if role != 'late_dense':
        raise ValueError('Unknown macro operator identity')
    return object_sha({'prototype_family': 'affine54_first3steps_dense_laststep',
        'checkpoint_sha256': source['assets']['checkpoint_sha256'],
        'cache_manifest_sha256': source['assets']['cache_manifest_sha256'],
        'bank_RUN_sha256': source['bank_RUN_sha256'], 'VAE_sha256': VAE_SHA,
        'step_operators': ['candidate', 'candidate', 'candidate', 'dense']})


def media_encoder_identity():
    import imageio_ffmpeg
    from importlib.metadata import version
    executable = Path(imageio_ffmpeg.get_ffmpeg_exe())
    info = subprocess.run([str(executable), '-version'], text=True, capture_output=True, timeout=15)
    return {'imageio_ffmpeg_version': version('imageio-ffmpeg'), 'imageio_version': version('imageio'),
        'executable_basename': executable.name, 'executable_sha256': sha256(executable),
        'version_returncode': info.returncode,
        'version_first_line': (info.stdout or info.stderr).splitlines()[0] if info.stdout or info.stderr else None}


def render_inventory(out, run, prepared, renderer, checkpoint, budget):
    """Renderer is injected so failure accounting can be tested without torch."""
    for source in prepared:
        pair = source['pair']
        for role, key in ROLES:
            budget()
            entry = {'source_index': pair['source_index'], 'source_id': pair['source_id'], 'seed': pair['seed'],
                'variant': f"source{pair['source_index']:02d}_{role}", 'role': role, 'tensor_key': key,
                'path': output_name(pair['source_index'], role), 'status': 'DECODE_STARTED',
                'parent_flow_RUN_sha256': source['flow_RUN_sha256'],
                'parent_flow_sidecar_sha256': source['flow_sidecar_sha256'],
                'tensor_path': source['tensor'], 'tensor_sha256': source['tensor_sha256'],
                'input_prompt_digest': source['reference']['identity']['input_prompt_digest'],
                'reference': source['reference'], 'decoder_digest': run['decoder_digest']}
            entry['model_digest'] = source_model_digest(source, role)
            run['videos'].append(entry)
            run['render_attempts_started'] += 1
            checkpoint()
            def mark(stage, **details):
                counter, marker = {'decode_started': ('VAE_decodes_started', 'decode_started'),
                    'decode_completed': ('VAE_decodes_completed', 'decode_completed'),
                    'encoding_started': ('video_encodes_started', 'encoding_started'),
                    'encoding_completed': ('video_encodes_completed', 'encoding_completed')}[stage]
                if entry.get(marker):
                    raise RuntimeError('Repeated renderer accounting event')
                if stage == 'decode_completed' and not entry.get('decode_started'):
                    raise RuntimeError('Decode completion without start')
                if stage == 'encoding_started' and not entry.get('decode_completed'):
                    raise RuntimeError('Encoding before completed decode')
                if stage == 'encoding_completed' and not entry.get('encoding_started'):
                    raise RuntimeError('Encoding completion without start')
                entry.update(details)
                entry[marker] = True
                run[counter] += 1
                checkpoint()
            started = time.perf_counter()
            try:
                details = renderer(source, key, out / entry['path'], mark)
                if not entry.get('encoding_completed'):
                    raise RuntimeError('Renderer omitted completed native stages')
                if not (out / entry['path']).is_file() or not (out / entry['path']).stat().st_size:
                    raise RuntimeError('Completed encoding produced no actual clip')
                entry.update(details, status='RENDER_COMPLETE')
            except BaseException as error:
                entry.update(status='FAILED_RENDER_AFTER_START', error=repr(error))
                raise
            finally:
                entry['render_wall_seconds'] = time.perf_counter() - started
                video = out / entry['path']
                if video.exists():
                    entry.update(sha256=sha256(video), bytes=video.stat().st_size)
                checkpoint()
            if role == 'dense_control' and entry['sha256'] != source['reference']['video_sha256']:
                entry['status'] = 'FAILED_DENSE_DECODE_NEUTRALITY'
                checkpoint()
                raise RuntimeError('Neutral dense VAE render differs from historical video bytes')
            checkpoint()
            budget()


def evaluate_inventory(out, receipt):
    from ffnds_video_metrics import evaluate, validate
    from ffnds_video_metric_gate import inspect as core_inspect
    native = json.loads(receipt.read_text())
    receipt_sha = sha256(receipt)
    bundle = {'schema': 'ffnds.video.bundle.v1', 'expected_video_count': len(native['videos']), 'videos': []}
    failures = []
    declared = {within(out, entry['path']) for entry in native['videos']}
    for entry in native['videos']:
        video = out / entry['path']
        if not video.exists():
            continue
        ref = entry['reference']
        metadata = {'identity': clip_identity(entry, receipt_sha, native['protocol_sha256']),
                    'reference_identity': ref['identity'], 'cost_scope': native['cost_scope'],
                    'cost': {'generation_wall_seconds': entry.get('render_wall_seconds', 0.),
                        'ffn_boundary_seconds': 0., 'setup_seconds': native.get('setup_seconds', 0.),
                        'peak_gpu_memory_mib': native.get('peak_gpu_memory_mib', 0.)}}
        report_path = out / (video.stem + '_metrics.json')
        try:
            report = evaluate(video, Path(ref['video']), metadata)
            report['cost_scope'] = native['cost_scope']
            report['cost_deduplication_key'] = receipt_sha + '/' + entry['variant']
            report['shared_setup_cost_deduplication_key'] = receipt_sha
            report['latent_lineage'] = {k: entry[k] for k in ['parent_flow_RUN_sha256',
                'parent_flow_sidecar_sha256', 'tensor_sha256', 'tensor_key', 'role']}
            errors = validate(report)
            if errors:
                failures.extend(errors)
            write_json(report_path, report)
        except BaseException as error:
            write_json(report_path, {'status': 'METRIC_DECODE_FAILED', 'error': str(error),
                'video_sha256': sha256(video), 'identity': metadata['identity'], 'quality_accepted': False})
            failures.append(video.name + ': metric decode failed')
        bundle['videos'].append({'source_id': entry['source_id'], 'seed': entry['seed'],
            'variant': entry['variant'], 'video_path': entry['path'], 'video_sha256': sha256(video),
            'run_receipt_path': receipt.name, 'run_sha256': receipt_sha,
            'metric_report_path': report_path.name, 'metric_report_sha256': sha256(report_path),
            'reference_video_path': ref['video'], 'reference_video_sha256': ref['video_sha256'],
            'reference_run_receipt_path': ref['receipt'], 'reference_run_sha256': ref['receipt_sha256']})
    # Orphans are observed as failed forensic evidence, never assigned invented
    # source/seed/model identities or silently omitted from the actual inventory.
    for video in sorted(p for p in out.rglob('*.mp4') if p.resolve() not in declared):
        report_path = out / (video.stem + '_orphan_forensic.json')
        forensic = {'status': 'ORPHAN_MEDIA_SOURCE_IDENTITY_UNKNOWN_NOT_COMPLETE_EVIDENCE',
            'video_path': str(video.relative_to(out)), 'video_sha256': sha256(video),
            'DECODE_RUN_sha256': receipt_sha, 'quality_accepted': False}
        try:
            forensic['metrics_with_unverified_identity'] = evaluate(video, None,
                {'identity': {'expected_frames': 81}, 'cost': {},
                 'identity_status': 'UNVERIFIED_ORPHAN_SOURCE_SEED_MODEL_UNKNOWN'})
        except BaseException as error:
            forensic['metric_decode_error'] = repr(error)
        write_json(report_path, forensic)
        failures.append('Unregistered actual MP4 requires failed forensic evidence: ' + video.name)
    write_json(out / 'BUNDLE.json', bundle)
    if bundle['videos']:
        failures.extend(core_inspect(out / 'BUNDLE.json'))
    else:
        failures.append('No generated media available for core metrics')
    return failures


def inspect_inventory(out, receipt, plan, prepared, protocol_sha):
    """Additional actual-file and hybrid identity checks; no native-pair disguise."""
    out, receipt = Path(out), Path(receipt)
    native = json.loads(receipt.read_text())
    bundle = json.loads((out / 'BUNDLE.json').read_text())
    entries = native['videos']
    expected = {(s, role) for s in [0, 12] for role, _ in ROLES}
    if (len(entries) != 4 or {(e['source_index'], e['role']) for e in entries} != expected
            or native['render_attempts_started'] != 4
            or native['VAE_decodes_started'] != 4 or native['VAE_decodes_completed'] != 4
            or native['video_encodes_started'] != 4 or native['video_encodes_completed'] != 4
            or native['denoiser_NFE'] or native['FFN_calls']):
        return ['Incomplete fixed decode count/operator inventory']
    paths = {within(out, e['path']) for e in entries}
    if len(paths) != 4 or paths != {p.resolve() for p in out.rglob('*.mp4')}:
        return ['Missing, duplicate or unregistered actual MP4']
    actual_media = [{'path': str(p.relative_to(out)), 'sha256': sha256(p), 'bytes': p.stat().st_size}
                    for p in sorted(out.rglob('*.mp4'))]
    if native.get('actual_media_inventory') != actual_media:
        return ['Immutable decode receipt omitted or changed an actual media artifact']
    if len(bundle['videos']) != 4:
        return ['Core bundle omitted a decode']
    errors = []
    expected_decoder = decoder_binding(plan)
    if (native['protocol_sha256'] != protocol_sha or native['id'] != plan['id']
            or native.get('decoder_binding') != expected_decoder
            or native.get('decoder_digest') != object_sha(expected_decoder)
            or native.get('decoder_configuration') != {**DECODE_CONFIG,
                'vae_path': 'SHA_BOUND_Wan2.1_VAE.pth'}):
        errors.append('Actual decode receipt differs from frozen decoder/protocol identity')
    for group in ['driver_sources_sha256', 'native_source_sha256']:
        for name, expected_sha in plan[group].items():
            snapshot = within(out / 'EXECUTED_SOURCES', name)
            if not snapshot.is_file() or sha256(snapshot) != expected_sha:
                errors.append('Executed decoder source snapshot mismatch: ' + name)
    by_source = {source['pair']['source_index']: source for source in prepared}
    for entry in entries:
        matches = [e for e in bundle['videos'] if within(out, e['video_path']) == within(out, entry['path'])]
        if len(matches) != 1:
            errors.append('Decode not uniquely represented by core evidence')
            continue
        bound = matches[0]
        report = json.loads(within(out, bound['metric_report_path']).read_text())
        source = by_source[entry['source_index']]
        expected_entry = {'source_id': source['pair']['source_id'], 'seed': source['pair']['seed'],
            'tensor_path': source['tensor'], 'tensor_sha256': source['tensor_sha256'],
            'tensor_key': dict(ROLES)[entry['role']],
            'parent_flow_RUN_sha256': source['flow_RUN_sha256'],
            'parent_flow_sidecar_sha256': source['flow_sidecar_sha256'],
            'path': output_name(entry['source_index'], entry['role']),
            'model_digest': source_model_digest(source, entry['role']),
            'decoder_digest': object_sha(expected_decoder), 'reference': source['reference'],
            'input_latent_native_identity': source['latent_native_identities'][dict(ROLES)[entry['role']]],
            'input_prompt_digest': source['reference']['identity']['input_prompt_digest']}
        if any(entry.get(k) != v for k, v in expected_entry.items()):
            errors.append('Decode identity differs from independently reconstructed parent/decoder binding')
        if report.get('reference_identity') != source['reference']['identity']:
            errors.append('Metric reference identity differs from original dense receipt')
        if (entry['role'] == 'dense_control' and
                entry.get('input_latent_native_identity') != source['dense_native_identity']):
            errors.append('Neutral input native identity differs from original teacher')
        if entry.get('decoded_shape') != [81, 464, 832, 3] or entry.get('status') != 'RENDER_COMPLETE':
            errors.append('Incomplete or incorrectly shaped native VAE output')
        if (bound['run_sha256'] != sha256(receipt) or report['identity'] !=
                clip_identity(entry, sha256(receipt), native['protocol_sha256'])
                or sha256(within(out, entry['path'])) != entry['sha256']
                or entry['tensor_key'] != dict(ROLES)[entry['role']]
                or sha256(Path(entry['tensor_path'])) != entry['tensor_sha256']):
            errors.append('Decode identity/tensor/media binding mismatch')
        if entry['role'] == 'dense_control' and entry['sha256'] != entry['reference']['video_sha256']:
            errors.append('Dense neutral media mismatch')
    return errors


def execute(args, plan, prepared, vae_path, status):
    if not status['model_scope_authorized']:
        raise RuntimeError('Exact decode scope SHA must be registered by the coordinator')
    if args.study_root.resolve() != Path(plan['canonical_study_root']).resolve():
        raise ValueError('One canonical four-decode ledger required')
    from worker_identity import private_config_path, require_worker
    if sha256(private_config_path()) != plan['private_worker_config_sha256']:
        raise ValueError('Private execution binding changed')
    require_worker()
    protocol_sha = sha256(args.protocol)
    reserve_attempt(args.study_root, plan, protocol_sha)
    out = args.study_root / 'render'
    out.mkdir(exist_ok=False)
    run = {'id': plan['id'], 'status': 'STARTED_NOT_ACCEPTED', 'protocol_sha256': protocol_sha,
        'render_attempts_started': 0,
        'VAE_decodes_started': 0, 'VAE_decodes_completed': 0, 'denoiser_NFE': 0, 'FFN_calls': 0,
        'video_encodes_started': 0, 'video_encodes_completed': 0,
        'training_updates': 0, 'new_weight_fits': 0, 'official_quality_calls': 0,
        'quality_accepted': False, 'videos': [], 'automatic_retry': False,
        'VAE_sha256': VAE_SHA, 'cost_scope': 'VAE_ONLY_RENDER_NOT_FULL_GENERATION',
        'decoder_binding': decoder_binding(plan), 'decoder_digest': object_sha(decoder_binding(plan))}
    monitor = vae = None
    native_errors = []
    checkpoint = lambda: write_json(out / 'PROGRESS.json', run)
    checkpoint()
    execution_setup_started = time.perf_counter()
    try:
        host = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        if int(host['MemAvailable'].split()[0]) < plan['minimum_available_host_gib'] * (1 << 20):
            raise RuntimeError('Host memory reserve unavailable')
        os.environ.update(DTYPE='BF16', SENSITIVE_LAYER_DTYPE='None', USE_CHANNELS_LAST_3D='0')
        sys.path.insert(0, str(ROOT / '.runtime/code/LightX2V'))
        import torch
        from safetensors import safe_open
        from startup_gpu_reservation import reserve_before_assets
        reservation, monitor, reservation_receipt = reserve_before_assets(torch)
        if 'A100' not in torch.cuda.get_device_name(0):
            raise RuntimeError('One idle A100 required')
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = plan['cudnn_allow_tf32']
        from lightx2v.models.video_encoders.hf.wan.vae import WanVAE
        from lightx2v.utils.utils import seed_all, save_to_video, wan_vae_to_comfy
        seed_all(42)
        run['environment'] = {'torch': torch.__version__, 'CUDA': torch.version.cuda,
            'cuDNN': torch.backends.cudnn.version(), 'python': sys.version,
            'DTYPE': 'BF16', 'USE_CHANNELS_LAST_3D': '0', 'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32, 'cuda_matmul_allow_tf32': False,
            'BF16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
        run['CUDA_reservation'] = reservation_receipt
        run['media_encoder'] = media_encoder_identity()
        def budget():
            monitor.check()
            if shutil.disk_usage(out).free < plan['disk_reserve_gib'] * (1 << 30):
                raise RuntimeError('Disk reserve reached')
            if sum(p.stat().st_size for p in args.study_root.rglob('*') if p.is_file()) > .1 * (1 << 30):
                raise RuntimeError('Four-decode output cap reached')
        budget()
        snapshot = out / 'EXECUTED_SOURCES'
        for name in {**plan['driver_sources_sha256'], **plan['native_source_sha256']}:
            target = within(snapshot, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        shutil.copyfile(args.protocol, snapshot / args.protocol.name)
        setup_started = time.perf_counter()
        kwargs = {**DECODE_CONFIG, 'vae_path': str(vae_path),
                  'device': torch.device('cuda', 0), 'dtype': torch.bfloat16}
        vae = WanVAE(**kwargs)
        torch.cuda.synchronize()
        run['VAE_load_seconds'] = time.perf_counter() - setup_started
        run['setup_seconds'] = time.perf_counter() - execution_setup_started
        run['decoder_configuration'] = {**kwargs, 'vae_path': 'SHA_BOUND_Wan2.1_VAE.pth',
            'dtype': 'torch.bfloat16', 'device': 'cuda:0'}
        def renderer(source, key, output, mark):
            seed_all(source['pair']['seed'])
            if sha256(source['tensor']) != source['tensor_sha256']:
                raise ValueError('Saved source tensor changed before decode')
            with safe_open(source['tensor'], framework='pt', device='cpu') as archive:
                latent = archive.get_tensor(key)
            if list(latent.shape) != SHAPE or latent.dtype != torch.bfloat16 or not torch.isfinite(latent).all():
                raise ValueError('Nonfinite or non-native saved latent')
            header = {'dtype': str(latent.dtype), 'shape': list(latent.shape),
                      'storage': 'contiguous_' + sys.byteorder + '_endian'}
            identity = hashlib.sha256(json.dumps(header, sort_keys=True, separators=(',', ':')).encode() +
                b'\0' + latent.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            if identity != source['latent_native_identities'][key]:
                raise ValueError('Decoded latent differs from independently hashed safetensors key')
            started = time.perf_counter()
            torch.cuda.synchronize()
            decode_started = time.perf_counter()
            mark('decode_started')
            images = vae.decode(latent.to(device='cuda', dtype=torch.bfloat16))
            torch.cuda.synchronize()
            decode_seconds = time.perf_counter() - decode_started
            mark('decode_completed', decode_seconds=decode_seconds,
                 input_latent_native_identity=identity)
            images = wan_vae_to_comfy(images)
            if list(images.shape) != [81, 464, 832, 3] or not torch.isfinite(images).all():
                raise ValueError('Unexpected decoded full-clip shape/finite values')
            encode_started = time.perf_counter()
            mark('encoding_started')
            save_to_video(images, str(output), fps=16, method='ffmpeg')
            encode_seconds = time.perf_counter() - encode_started
            mark('encoding_completed', encode_seconds=encode_seconds)
            return {'input_latent_native_identity': identity, 'decoded_shape': list(images.shape),
                'decode_seconds': decode_seconds, 'encode_seconds': encode_seconds,
                'native_render_seconds_after_input_preparation': time.perf_counter() - started}
        with torch.inference_mode():
            render_inventory(out, run, prepared, renderer, checkpoint, budget)
        run['status'] = 'PASS_VAE_RENDER_AND_NEUTRALITY_REQUIRES_METRICS'
    except BaseException as error:
        native_errors.append(repr(error))
        run.update(status='FAILED_PRESERVED_NO_RETRY', error_type=type(error).__name__,
                   error=str(error), traceback=traceback.format_exc())
    finally:
        if vae is not None:
            try:
                vae.to_cpu()
                del vae
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except BaseException as error:
                native_errors.append('VAE cleanup: ' + repr(error))
        if monitor is not None:
            try:
                monitor.close()
                run['assigned_GPU_resources'] = monitor.report()
                run['peak_gpu_memory_mib'] = run['assigned_GPU_resources']['peak_physical_used_memory_MiB']
                monitor.check()
            except BaseException as error:
                native_errors.append('GPU isolation/finalization: ' + repr(error))
        if 'torch' in locals() and torch.cuda.is_initialized():
            run['GPU_allocated_bytes_before_CPU_metrics'] = torch.cuda.memory_allocated()
        if native_errors:
            run['status'] = 'FAILED_PRESERVED_NO_RETRY'
        run['native_errors'] = native_errors
        run['actual_media_inventory'] = [{'path': str(p.relative_to(out)), 'sha256': sha256(p),
            'bytes': p.stat().st_size} for p in sorted(out.rglob('*.mp4'))]
        # No future metrics/bundle fields are ever written into this receipt.
        write_json(out / 'DECODE_RUN.json', run)
    receipt = out / 'DECODE_RUN.json'
    metric_errors = []
    try:
        metric_errors = evaluate_inventory(out, receipt)
        metric_errors.extend(inspect_inventory(out, receipt, plan, prepared, protocol_sha))
        if sum(p.stat().st_size for p in args.study_root.rglob('*') if p.is_file()) > .1 * (1 << 30):
            metric_errors.append('Final complete-study disk cap exceeded')
    except BaseException as error:
        metric_errors.append('Metric/inventory failure: ' + repr(error))
    summary = {'status': PASS if not native_errors and not metric_errors else 'FAILED_PRESERVED_NO_RETRY',
        'id': plan['id'], 'protocol_sha256': protocol_sha, 'DECODE_RUN_sha256': sha256(receipt),
        'BUNDLE_sha256': sha256(out / 'BUNDLE.json') if (out / 'BUNDLE.json').exists() else None,
        'native_errors': native_errors, 'metric_errors': metric_errors, 'quality_accepted': False,
        'automatic_retry': False, 'new_denoiser_NFE': 0}
    write_json(out / 'RUN.json', summary)
    close_attempt(args.study_root, summary['status'], out / 'RUN.json')
    if summary['status'] != PASS:
        raise RuntimeError('Decode attempt failed and was preserved; no automatic retry')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['protocol', 'hf-root', 'study-root']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--execute-decode', action='store_true')
    args = parser.parse_args()
    plan, prepared, vae, status = preflight(args.protocol, args.hf_root)
    print(json.dumps(status), flush=True)
    if args.execute_decode:
        execute(args, plan, prepared, vae, status)


if __name__ == '__main__':
    main()
