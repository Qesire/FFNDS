"""Finite CPU-only file readahead advice for an already running E0.

No tensor values enter this process. Advice affects the operating-system
cache only; the original reader still checks every input SHA and raw SHA.
No GPU, model, numerical candidate, input file or parent process is changed.
Kernel I/O caused by advice is UNKNOWN and separate from explicit reads.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import signal
import stat
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import a5_hwg_matched_capture_20261030 as guard
from a5_hwg_registered_static_fit_20261030 import authorize_common, validate_parent
from a5_hwg_evidence_io_20261030 import bound_json

SELF = 'tools/a5_hwg_resource_readahead_20261030.py'
KIND = 'CPU_RESOURCE_READAHEAD_ADVICE_ONLY_NO_TENSOR_READ_V1'
RES = {'GPU': 0, 'CPU_threads': 1, 'session_max_GPUs': 4,
       'asset_validation_read_max_bytes': 256 << 20, 'new_HF_bytes': 4 << 20,
       'maximum_single_file_bytes': 1 << 20, 'peak_RSS_max_bytes': 256 << 20,
       'wall_seconds': 7200, 'outer_seconds': 7320, 'automatic_retry': False}
LIMITS = {'maximum_advised_file_bytes': 96 << 30, 'maximum_advice_syscalls': 25000,
          'range_bytes': 4 << 20, 'lookahead_pairs': 2, 'poll_seconds': 5}
guard.RES = RES


def advise_file(path, declared_bytes, range_bytes, can_issue, before_call, after_call):
    """No user-space payload read; stop between every bounded hint syscall."""
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    before = path.stat()
    guard.need(stat.S_ISREG(before.st_mode) and before.st_size == declared_bytes, 'advice regular file/length')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    verified = False
    try:
        opened = os.fstat(fd)
        guard.need(stat.S_ISREG(opened.st_mode) and identity(opened) == identity(before), 'advice opened identity differs')
        verified = True
        for offset in range(0, declared_bytes, range_bytes):
            state = can_issue()
            if state != 'CONTINUE': return state
            length = min(range_bytes, declared_bytes - offset)
            before_call()
            os.posix_fadvise(fd, offset, length, os.POSIX_FADV_WILLNEED)
            after_call(length)
        return 'COMPLETE'
    finally:
        try:
            if verified:
                guard.need(identity(os.fstat(fd)) == identity(before) == identity(path.stat()), 'input changed during advice')
        finally: os.close(fd)


@guard.execution_bootstrap('run')
def run(args):
    started, budget = guard.CURRENT_BOOTSTRAP.started, guard.CURRENT_BOOTSTRAP.budget
    p = guard.read(args.protocol)
    authorize_common(p, args.protocol, args.job_spec, RES, {}, KIND, gpu_cards=0)
    private = guard.private_assets(p, args.private_assets)
    guard.need(p['advice_limits'] == LIMITS and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
               and sorted(os.sched_getaffinity(0)) == p['cpu_affinity']
               and len(p['cpu_affinity']) == 1, 'CPU-only frozen resource scope')
    guard.need(hasattr(os, 'posix_fadvise') and hasattr(os, 'POSIX_FADV_WILLNEED'), 'OS advice API required')
    parent = guard.read(guard.check_binding(ROOT, p['parent_protocol']))
    guard.need(parent['id'] == p['parent_id'] and parent['source'] == p['source']
               and parent['cache'] == p['cache'], 'exact unchanged parent E0')
    hf = Path(private['hf_root']).resolve()
    out = hf / p['output_HF_relative']
    guard.need(not out.exists(), 'one fresh finite resource output')
    out.mkdir()
    report = {'id': p['id'], 'protocol_sha256': guard.sha(args.protocol), 'status': 'RUNNING',
              'parent_id': p['parent_id'], 'automatic_retry': False,
              'advised_files': 0, 'advised_file_bytes': 0, 'advice_syscalls': 0,
              'advice_range_bytes_returned': 0, 'partial_advice_files': 0,
              'tensor_payload_bytes_returned_to_this_process': 0,
              'kernel_physical_IO_bytes': 'UNKNOWN_NOT_AN_EXPLICIT_PAYLOAD_READ',
              'new_GPU_model_or_numerical_calls': 0, 'source_files_changed': 0}
    def check():
        guard.need(time.monotonic() - started <= RES['wall_seconds'], 'resource scope deadline')
        guard.need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= RES['peak_RSS_max_bytes'], 'resource RSS')
    budget.check = check
    writer = guard.Writer(out, hf, check)
    writer.holds['journal'] = 2 << 20
    persist = lambda: writer.json(out / 'RUN.json', report)
    persist()
    try:
        with guard.bounded_files():
            validate_parent(p['source'], hf, budget, p['cache'])
            # Read only the frozen small metadata. No payload open/read occurs
            # until an advice-only read-only fd is opened for an eligible pair.
            pairs = []
            for ei, steps in enumerate([(0, 1), (2, 3)]):
                for block in range(40):
                    refs = []
                    for step in steps:
                        meta = bound_json(hf, p['source']['metadata'][f'step{step}_block{block:02d}'], budget)
                        refs.extend([meta['payload'], meta['full_V1']])
                    pairs.append(refs)
            advised, previous_progress = set(), None
            run_path = hf / parent['output_HF_relative'] / 'RUN.json'
            exit_path = run_path.parent.parent / 'EXIT_RECEIPT.json'
            def can_issue():
                check()
                if exit_path.exists(): return 'PARENT_CLOSED'
                if guard.available_memory() < 128 << 30: return 'LOW_MEMORY'
                return 'CONTINUE'
            def before_call(): report['advice_syscalls'] += 1
            def after_call(length): report['advice_range_bytes_returned'] += length
            parent_closed = False
            while True:
                check()
                if exit_path.exists():
                    report['status'] = 'COMPLETE_RESOURCE_ADVICE_PARENT_EXIT_PRESENT'
                    break
                live = guard.read(run_path)
                guard.need(live['id'] == p['parent_id']
                           and live['protocol_sha256'] == p['parent_protocol']['sha256'], 'live parent identity')
                count = live['completed_pairs']
                guard.need(type(count) is int and 0 <= count <= 80, 'bounded live progress')
                report['parent_completed_pairs_latest'] = count
                if count == 80:
                    report['status'] = 'COMPLETE_RESOURCE_ADVICE_ALL80_PARENT_PAIRS'
                    break
                # Current pair=count is already reading/computing. Warm only
                # the next two distinct pairs, with a 128GiB host reserve.
                if guard.available_memory() >= 128 << 30:
                    for index in range(count + 1, min(80, count + 1 + LIMITS['lookahead_pairs'])):
                        for ref in pairs[index]:
                            if can_issue() != 'CONTINUE': break
                            if ref['path'] in advised: continue
                            latest = guard.read(run_path)
                            guard.need(latest['id'] == p['parent_id'] and latest['protocol_sha256'] == p['parent_protocol']['sha256'], 'live parent identity')
                            if latest['completed_pairs'] >= index: continue
                            path = (hf / ref['path']).resolve()
                            guard.need(path.is_relative_to(hf) and path.stat().st_size == ref['bytes'], 'advice input confinement/length')
                            calls = (ref['bytes'] + LIMITS['range_bytes'] - 1) // LIMITS['range_bytes']
                            guard.need(report['advised_file_bytes'] + ref['bytes'] <= LIMITS['maximum_advised_file_bytes']
                                       and report['advice_syscalls'] + calls <= LIMITS['maximum_advice_syscalls'], 'advice finite budget')
                            advised.add(ref['path'])
                            report['advised_files'] += 1
                            report['advised_file_bytes'] += ref['bytes']
                            writer.note('ADVICE_FILE_PREDEBIT_NO_USER_PAYLOAD_READ', path=ref['path'],
                                        bytes=ref['bytes'], maximum_calls=calls, kernel_IO='UNKNOWN')
                            os.fsync(writer.journal.fileno())
                            prior_calls = report['advice_syscalls']
                            state = advise_file(path, ref['bytes'], LIMITS['range_bytes'], can_issue, before_call, after_call)
                            if state != 'COMPLETE': report['partial_advice_files'] += 1
                            writer.note('ADVICE_FILE_RETURN_NO_USER_PAYLOAD_READ', path=ref['path'],
                                        actual_calls=report['advice_syscalls'] - prior_calls, status=state,
                                        full_file_bytes_are_predebited_not_claimed_physical_IO=True)
                            if state == 'PARENT_CLOSED': parent_closed = True
                            if state != 'COMPLETE': break
                        if can_issue() != 'CONTINUE' or parent_closed: break
                if parent_closed or exit_path.exists():
                    report['status'] = 'COMPLETE_RESOURCE_ADVICE_PARENT_EXIT_PRESENT'
                    break
                if previous_progress != count:
                    persist()
                    previous_progress = count
                time.sleep(LIMITS['poll_seconds'])
    except BaseException as error:
        report.update(status='FAILED_RESOURCE_ADVICE_NO_RETRY', error_type=type(error).__name__, error=str(error)[:1000])
        raise
    finally:
        signal.alarm(0)
        guard.CONTROL_READ = None
        writer.check = lambda: None
        writer.finish_pending()
        report.update(metadata_read_ledger=budget.summary(), wall_seconds=time.monotonic() - started,
                      peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      output_accounting=writer.summary())
        persist()
        writer.journal.close()


def main():
    ap = argparse.ArgumentParser()
    for name in ['protocol', 'job-spec', 'private-assets']: ap.add_argument('--' + name, type=Path, required=True)
    run(ap.parse_args())


if __name__ == '__main__': main()
