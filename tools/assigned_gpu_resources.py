"""Monitor only the exclusively assigned physical card; leave others usable."""
import csv
import io
import os
import subprocess
import threading
import time


class AssignedGPUResources:
    def __init__(self):
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if visible not in ('0', '1', '2', '3'):
            raise RuntimeError('One explicit physical GPU index required')
        self.gpu = visible
        self.samples = []
        self.violations = []
        self.stop = threading.Event()
        self.thread = None

    def query(self, args):
        p = subprocess.run(['nvidia-smi', '-i', self.gpu, *args], text=True,
                           capture_output=True, check=True, timeout=10)
        return list(csv.reader(io.StringIO(p.stdout)))

    def sample(self):
        apps = self.query(['--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'])
        foreign = [row for row in apps if row and int(row[0].strip()) != os.getpid()]
        hardware = self.query(['--query-gpu=name,driver_version,memory.total,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu',
                               '--format=csv,noheader,nounits'])
        self.samples.append({'monotonic_seconds': time.monotonic(), 'hardware': hardware,
                             'foreign_process_count_on_assigned_GPU': len(foreign)})
        if foreign:
            self.violations.append('Foreign compute appeared on the assigned physical GPU')

    def start(self):
        self.sample()
        self.check()
        def loop():
            while not self.stop.wait(2):
                try:
                    self.sample()
                except BaseException as error:
                    self.violations.append('Assigned GPU monitor failed: ' + type(error).__name__)
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def check(self):
        if self.violations:
            raise RuntimeError(self.violations[-1])

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=12)
        if self.thread is not None and self.thread.is_alive():
            self.violations.append('Assigned GPU monitor did not exit')

    def report(self):
        return {'physical_GPU_index': int(self.gpu), 'sampling_interval_seconds': 2,
                'other_GPUs_required_idle': False, 'samples': self.samples,
                'violations': self.violations, 'sampled_isolation_pass': not self.violations}
