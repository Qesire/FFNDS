"""Record both worker GPUs and reject competing compute during formal timing."""
import csv
from datetime import datetime,timezone
import io
import os
import socket
import subprocess
import threading
import time


def query(arguments):
    result=subprocess.run(['nvidia-smi',*arguments],capture_output=True,text=True,check=True,timeout=10)
    return list(csv.reader(io.StringIO(result.stdout)))


class ResourceMonitor:
    def __init__(self):
        self.samples=[];self.violations=[];self.stop=threading.Event();self.thread=None

    def sample(self):
        applications=query(['--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'])
        gpus=query(['--query-gpu=index,uuid,name,driver_version,memory.total,clocks.sm,clocks.mem,pstate,power.draw,power.limit,temperature.gpu',
            '--format=csv,noheader,nounits'])
        sample={'utc':datetime.now(timezone.utc).isoformat(),'applications':applications,'gpus':gpus}
        foreign=[row for row in applications if int(row[1].strip())!=os.getpid()]
        if foreign:self.violations.append({'utc':sample['utc'],'foreign_compute':foreign})
        self.samples.append(sample)

    def start(self):
        if socket.gethostname().split('.')[0]!='worker' or os.environ.get('CUDA_VISIBLE_DEVICES')!='0':
            raise RuntimeError('Formal timing requires the declared physical worker GPU0')
        self.sample();self.check()
        def loop():
            while not self.stop.wait(1):
                try:self.sample()
                except BaseException as error:
                    self.violations.append({'monitor_error':str(error)})
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start()

    def check(self):
        if self.violations:raise RuntimeError('Competing compute or resource-monitor failure invalidated timing: '+str(self.violations[-1]))

    def close(self):
        self.stop.set()
        if self.thread is not None:self.thread.join(timeout=12)
        if self.thread is not None and self.thread.is_alive():self.violations.append({'monitor_error':'monitor did not stop'})

    def report(self):
        return {'own_pid':os.getpid(),'sampling_interval_seconds':1,'both_GPUs_required_idle_except_this_process':True,
            'samples':self.samples,'violations':self.violations,'isolation_pass':not self.violations}
