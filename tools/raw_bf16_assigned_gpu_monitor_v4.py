"""Private startup monitor with slower NVML queries and strict failures."""
import csv
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_ffnds_startup_ord240_v4', ROOT/'tools/startup_gpu_reservation.py')
startup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(startup)
MONITOR = {'query_timeout_seconds': 30, 'sampling_interval_seconds': 5, 'close_join_seconds': 35,
    'stop_checked_between_queries': True, 'foreign_or_query_error': 'ABORT_NO_IGNORE_NO_RETRY'}


def gpu_uuid(index):
    value = subprocess.check_output(['nvidia-smi', '-i', index, '--query-gpu=uuid', '--format=csv,noheader,nounits'],
        text=True, timeout=MONITOR['query_timeout_seconds']).strip()
    if not value.startswith('GPU-') or '\n' in value or ',' in value:
        raise RuntimeError('Require one physical GPU UUID')
    return value


class AssignedMonitorV4(startup.UUIDAssignedGPUResources):
    def query(self, args):
        result = subprocess.run(['nvidia-smi', '-i', self.gpu, *args], text=True, capture_output=True, check=True,
            timeout=MONITOR['query_timeout_seconds'])
        return list(csv.reader(io.StringIO(result.stdout)))

    def sample(self):
        apps = self.query(['--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'])
        foreign = [row for row in apps if row and int(row[0].strip()) != os.getpid()]
        # Preserve detected foreign compute even if shutdown interrupts the rest
        # of this sample. A query timeout always propagates as a real violation.
        if foreign:
            self.violations.append('Foreign compute appeared on the assigned physical GPU')
        if self.stop.is_set():
            return
        hardware = self.query(['--query-gpu=name,driver_version,memory.total,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu', '--format=csv,noheader,nounits'])
        record = {'monotonic_seconds': time.monotonic(), 'hardware': hardware, 'foreign_process_count_on_assigned_GPU': len(foreign)}
        if not self.stop.is_set():
            used = self.query(['--query-gpu=memory.used', '--format=csv,noheader,nounits'])
            record['physical_GPU_used_memory_MiB'] = int(used[0][0].strip())
        self.samples.append(record)

    def start(self):
        self.sample(); self.check()
        def loop():
            while not self.stop.wait(MONITOR['sampling_interval_seconds']):
                try:
                    self.sample()
                except BaseException as error:
                    self.violations.append('Assigned GPU monitor failed: '+type(error).__name__)
        self.thread = threading.Thread(target=loop, daemon=True); self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=MONITOR['close_join_seconds'])
            if self.thread.is_alive():
                self.violations.append('Assigned GPU monitor did not exit')

    def report(self):
        result = super().report()
        result.update(MONITOR)
        return result


# Only this independently loaded module is rebound. Frozen public modules and
# their class definitions remain unchanged.
startup.UUIDAssignedGPUResources = AssignedMonitorV4
startup.gpu_uuid = gpu_uuid
