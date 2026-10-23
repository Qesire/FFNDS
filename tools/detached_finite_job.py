"""Server-local, detached single-attempt supervisor; stdout never uses SSH.

The launcher exits after a double-forked supervisor has written its receipt.
The supervisor retains an exclusive lock and records the actual child exit.
"""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time

from rollout_contract import atomic, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(); command = args.command
    if command[:1] == ['--']: command = command[1:]
    folder = args.folder.resolve()
    if folder.exists(): raise RuntimeError('Detached job directory already exists; no retry')
    folder.mkdir(parents=True)
    pid = os.fork()
    if pid:
        for _ in range(100):
            receipt = folder / 'JOB.json'
            if receipt.exists():
                print('Detached finite job receipt created', flush=True); return
            time.sleep(.1)
        raise RuntimeError('Supervisor did not register its process; inspect directory before any retry')
    os.setsid()
    if os.fork(): os._exit(0)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    with open(os.devnull, 'rb') as source, (folder / 'SUPERVISOR.log').open('ab', buffering=0) as log:
        os.dup2(source.fileno(), 0); os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
        lock = (folder / 'JOB.lock').open('a'); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        record = {'status': 'RUNNING', 'supervisor_pid': os.getpid(), 'child_pid': child.pid,
                  'child_exit_code': None, 'automatic_retry': False, 'log_file': 'SUPERVISOR.log'}
        atomic(folder / 'JOB.json', record)
        while child.poll() is None:
            record['heartbeat_epoch_seconds'] = time.time(); atomic(folder / 'JOB.json', record); time.sleep(5)
        record.update(status='COMPLETE' if child.returncode == 0 else 'FAILED_NO_RETRY',
                      child_exit_code=child.returncode, heartbeat_epoch_seconds=time.time())
        atomic(folder / 'JOB.json', record)
    os._exit(0)


if __name__ == '__main__': main()
