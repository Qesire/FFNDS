"""Finite CPU comparison and independent typed mask SHA verification.

Requires exact independent preexecution review and actual three E0 EXIT0s.
No model, tensor arithmetic, CUDA import or video generation occurs here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import signal
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import a5_hwg_matched_capture_20261030 as guard
from a5_hwg_registered_static_fit_20261030 import authorize_common
from a5_hwg_evidence_io_20261030 import bound_json
from a5_hwg_core_summary_math_20261030 import metric_vector, summarize_sources

SELF = 'tools/a5_hwg_registered_core_summary_20261030.py'
RES = {'GPU': 0, 'CPU_threads': 1, 'session_max_GPUs': 4,
       'asset_validation_read_max_bytes': 96 << 30, 'new_HF_bytes': 256 << 20,
       'maximum_single_file_bytes': 128 << 20, 'peak_RSS_max_bytes': 512 << 20,
       'wall_seconds': 7200, 'outer_seconds': 7320, 'automatic_retry': False}
KIND = 'CORE3_SOURCE_EQUAL_STATS_WITH_INDEPENDENT_MASK_SHA_V1'
guard.RES = RES


def verify_masks(hf, ref, schema, budget):
    """One sequential read verifies both full-file and every typed raw SHA."""
    path = (hf / ref['path']).resolve()
    guard.need(path.is_relative_to(hf.resolve()) and path.stat().st_size == ref['bytes'], 'mask file confinement/length')
    before = path.stat()
    identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    budget.before(ref['bytes'], ref['path'])
    whole = hashlib.sha256()
    def read_exact(stream, count):
        raw = stream.read(count)
        guard.need(len(raw) == count, 'short typed mask file')
        whole.update(raw)
        budget.after(len(raw))
        return raw
    with path.open('rb') as stream:
        n = struct.unpack('<Q', read_exact(stream, 8))[0]
        guard.need(0 < n <= 1 << 20, 'mask header bound')
        header = json.loads(read_exact(stream, n))
        rows = sorted((v['data_offsets'][0], k, v) for k, v in header.items() if k != '__metadata__')
        guard.need({k for _, k, _ in rows} == set(schema), 'mask schema field set')
        cursor = 0
        for start, name, item in rows:
            row = schema[name]
            dtype, width = {'uint8': ('U8', 1), 'float32': ('F32', 4)}[row['dtype']]
            end = item['data_offsets'][1]
            guard.need(start == cursor and item['shape'] == row['shape'] and item['dtype'] == dtype
                       and end - start == math.prod(row['shape']) * width, 'mask typed offsets/shape')
            raw_sha, remaining = hashlib.sha256(), end - start
            while remaining:
                data = read_exact(stream, min(8 << 20, remaining))
                raw_sha.update(data)
                remaining -= len(data)
            guard.need(raw_sha.hexdigest() == row['raw_tensor_sha256'], 'mask raw tensor SHA')
            cursor = end
        guard.need(stream.tell() == before.st_size and identity(path.stat()) == identity(before)
                   and identity(os.fstat(stream.fileno())) == identity(before), 'mask file changed or trailing bytes')
    guard.need(whole.hexdigest() == ref['sha256'], 'mask full file SHA')
    budget.finish()


@guard.execution_bootstrap('run')
def run(args):
    started, budget = guard.CURRENT_BOOTSTRAP.started, guard.CURRENT_BOOTSTRAP.budget
    p = guard.read(args.protocol)
    authorize_common(p, args.protocol, args.job_spec, RES, {}, KIND, gpu_cards=0)
    private = guard.private_assets(p, args.private_assets)
    guard.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and len(os.sched_getaffinity(0)) == 1,
               'CPU-only summary isolation')
    guard.need(sorted(os.sched_getaffinity(0)) == p['cpu_affinity'], 'registered summary CPU')
    guard.need([s['source_ordinal'] for s in p['sources']] == [898, 1040, 979], 'three frozen sources and roles')
    hf = Path(private['hf_root']).resolve()
    out = hf / p['output_HF_relative']
    guard.need(not out.exists(), 'fresh finite CPU summary')
    out.mkdir()
    report = {'id': p['id'], 'protocol_sha256': guard.sha(args.protocol), 'status': 'RUNNING',
              'automatic_retry': False, 'models_or_CUDA_or_new_videos': 0,
              'verified_mask_files': 0, 'completed_sources': 0,
              'deployment_or_video_quality_acceptance': False}
    def check():
        guard.need(time.monotonic() - started <= RES['wall_seconds'], 'summary wall budget')
        guard.need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= RES['peak_RSS_max_bytes'], 'summary RSS')
    budget.check = check
    writer = guard.Writer(out, hf, check)
    persist = lambda: writer.json(out / 'RUN.json', report)
    persist()
    try:
        with guard.bounded_files():
            failed = p['failed_source96']
            failed_run = bound_json(hf, failed['RUN'], budget)
            failed_exit = bound_json(hf, failed['EXIT'], budget)
            failed_protocol = guard.read(guard.check_binding(ROOT, failed['protocol']))
            guard.need(failed_exit['id'] == failed_run['id'] == failed['id']
                       and failed_exit['process_exited'] is True and failed_exit['child_returncode'] == 1
                       and failed_exit['timeout'] is False and failed_exit['automatic_retry'] is False
                       and failed_exit['spec_sha256'] == failed['job_spec_sha256']
                       and failed_exit['run_sha256'] == failed['RUN']['sha256']
                       and failed_exit['protocol_sha256'] == failed_run['protocol_sha256'] == failed['protocol']['sha256']
                       and failed_protocol['case']['native_source_ordinal'] == 96
                       and failed_run['phases'] == [], 'source96 actual failed, never numerical completion')
            by_source, csv_rows, costs, reference = {}, [], [], None
            for source in p['sources']:
                run = bound_json(hf, source['RUN'], budget)
                receipt = bound_json(hf, source['EXIT'], budget)
                protocol = guard.read(guard.check_binding(ROOT, source['protocol']))
                guard.need(receipt['id'] == run['id'] == source['id'] and receipt['process_exited'] is True
                           and receipt['child_returncode'] == 0 and receipt['timeout'] is False
                           and receipt['automatic_retry'] is False and receipt['run_sha256'] == source['RUN']['sha256']
                           and receipt['spec_sha256'] == source['job_spec_sha256']
                           and receipt['protocol_sha256'] == run['protocol_sha256'] == source['protocol']['sha256']
                           and run['status'] == 'COMPLETE_FULLM_ALL80_58ARM_DISCOVERY_EXIT_PENDING'
                           and run['completed_pairs'] == len(run['site_manifest']) == len(run['call_record_files']) == 80,
                           'actual E0 closure')
                common = {k: protocol[k] for k in ['kind', 'cache', 'runtime_versions', 'arms', 'static_calibration',
                          'keep_channels', 'grid', 'P64_seed', 'shuffle_seed', 'random_group_seed_base']}
                if reference is None: reference = common
                guard.need(common == reference and run['source']['native_source_ordinal'] == source['source_ordinal'], 'same exact comparison recipe')
                expected_arms = {a['name']: a for a in protocol['arms']}
                sites, arms = set(), {name: [] for name in expected_arms}
                for item in run['site_manifest']:
                    result = bound_json(hf, item['result'], budget)
                    key = (result['expert'], result['block'])
                    guard.need(key not in sites and key == (item['expert'], item['block']), 'duplicate or mismatched result site')
                    sites.add(key)
                    guard.need(len(result['native_sample_checks']) == 2 and all(
                        set(c) == {'V1', 'Z', 'H', 'Q2_input', 'V2', 'Y'} and all(v is True for v in c.values())
                        for c in result['native_sample_checks']), 'actual native sample replay checks')
                    guard.need((hf / result['mask_and_proxy']['path']).resolve()
                               == (hf / protocol['output_HF_relative'] / f'{key[0]}_block{key[1]:02d}_MASKS_AND_PROXY.safetensors').resolve(), 'exact site mask filename')
                    verify_masks(hf, result['mask_and_proxy'], result['mask_schema'], budget)
                    report['verified_mask_files'] += 1
                    guard.need(len(result['arms']) == 58 and {v['spec']['name'] for v in result['arms']} == set(expected_arms), 'all exact arms')
                    for arm in result['arms']:
                        name = arm['spec']['name']
                        guard.need(arm['spec'] == expected_arms[name], 'arm specification changed')
                        vector = metric_vector(arm)
                        arms[name].append(vector)
                        csv_rows.append({'source_ordinal': source['source_ordinal'], 'source_role': source['role'],
                                         'expert': key[0], 'block': key[1], 'arm': name, **vector})
                    persist()
                guard.need(sites == {(e, b) for e in ['high_noise_model', 'low_noise_model'] for b in range(40)}, 'all40/both experts')
                ordinals, measured = {}, Counter()
                for ref in run['call_record_files']:
                    for call in bound_json(hf, ref, budget)['records']:
                        name, ordinal = call['name'], call['ordinal']
                        guard.need(call['status'] == 'COMPLETE' and name in protocol['call_limits']
                                   and ordinal not in ordinals.setdefault(name, set()), 'incomplete/duplicate operation')
                        ordinals[name].add(ordinal)
                        measured[name] += 1
                        costs.append({'source_ordinal': source['source_ordinal'], 'operation': name,
                                      'ordinal': ordinal, 'expert': call['identity'].get('expert'),
                                      'block': call['identity'].get('block'), 'arm': call['identity'].get('arm', 'SHARED_REFERENCE_OR_SELECTOR'),
                                      'CUDA_event_interval_seconds': call['actual_CUDA_seconds'],
                                      'host_wall_seconds': call['actual_host_seconds'], 'native_skip': False})
                guard.need(dict(measured) == protocol['call_limits'] == run['call_counts']
                           and all(v == set(range(1, protocol['call_limits'][k] + 1)) for k, v in ordinals.items()), 'finite complete operation ledger')
                by_source[source['source_ordinal']] = arms
                report['completed_sources'] += 1
                persist()
            summaries = {}
            for arm in reference['arms']:
                name = arm['name']
                metrics = next(iter(by_source.values()))[name][0]
                summaries[name] = {}
                for metric in metrics:
                    vectors = [[row[metric] for row in by_source[s['source_ordinal']][name]] for s in p['sources']]
                    if all(any(v is not None for v in a) for a in vectors):
                        summaries[name][metric] = summarize_sources(vectors)
                    else:
                        summaries[name][metric] = {'status': 'UNDEFINED_IN_AT_LEAST_ONE_SOURCE',
                            'defined_site_counts': [sum(v is not None for v in a) for a in vectors]}
            writer.json(out / 'SOURCE_EQUAL_COMPARISON.json', {'sources': p['sources'], 'arms': summaries,
                'independent_development_image_count': 2, 'calibration_image_count': 1,
                'heldout_blind_confirmation': False, 'failed_source96_not_in_numeric_mean': True,
                'candidate_free_video_and_eight_axis_quality': 'NOT_MEASURED_THIS_TEACHER_REPLAY_SCOPE',
                'camera_motion_flicker_events': 'NOT_MEASURED_THIS_TEACHER_REPLAY_SCOPE',
                'operation_intervals': 'PAID_DIAGNOSTIC_BOUNDARIES_NOT_KERNEL_ACTIVE_TIME_OR_FAIR_SYS_SPEEDUP',
                'evidence': 'PAID_TEACHER_STATE_DIAGNOSTIC_NOT_VIDEO_OR_NATIVE_SKIP'})
            for filename, rows in [('SOURCE_SITE_ARM_METRICS.csv', csv_rows), ('PAID_OPERATION_INTERVALS.csv', costs)]:
                stream = io.StringIO()
                output = csv.DictWriter(stream, fieldnames=list(rows[0]))
                output.writeheader()
                output.writerows(rows)
                raw = stream.getvalue().encode()
                dest = out / filename
                writer.begin_file(dest, len(raw))
                with dest.open('xb') as handle: guard.need(handle.write(raw) == len(raw), 'short comparison CSV')
                writer.account_file(dest, True, hashlib.sha256(raw).hexdigest())
            report.update(status='COMPLETE_CORE3_COMPARISON_AND_INDEPENDENT_MASK_SHA_EXIT_PENDING',
                          numerical_source_count=3, site_count=240, arm_site_rows=len(csv_rows),
                          result_files=[{'path': str(Path(k).relative_to(hf)), 'bytes': v['bytes'], 'sha256': v['sha256']}
                                        for k, v in writer.files.items() if Path(k).name != 'RUN.json'])
    except BaseException as error:
        report.update(status='FAILED_CORE_SUMMARY_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error)[:1000])
        raise
    finally:
        signal.alarm(0)
        guard.CONTROL_READ = None
        writer.check = lambda: None
        writer.finish_pending()
        report.update(read_ledger=budget.summary(), wall_seconds=time.monotonic() - started,
                      peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      output_accounting=writer.summary())
        persist()
        writer.journal.close()


def main():
    ap = argparse.ArgumentParser()
    for name in ['protocol', 'job-spec', 'private-assets']: ap.add_argument('--' + name, type=Path, required=True)
    run(ap.parse_args())


if __name__ == '__main__': main()
