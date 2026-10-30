"""Bounded assigned-GPU monitor; no CUDA initialization on import."""
import csv, io, os, subprocess, threading, time
import startup_gpu_reservation as startup

class Monitor(startup.UUIDAssignedGPUResources):
    def __init__(self):
        super().__init__()
        self.query_diagnostics=[]
    def query(self,args):
        start=time.monotonic(); label=args[0] if args else 'unknown'
        try:
            p=subprocess.run(['nvidia-smi','-i',self.gpu,*args],text=True,capture_output=True,check=True,timeout=30)
            result=list(csv.reader(io.StringIO(p.stdout)))
        except BaseException as error:
            self.query_diagnostics.append({'query':label,'elapsed_seconds':time.monotonic()-start,'error_type':type(error).__name__})
            raise
        self.query_diagnostics.append({'query':label,'elapsed_seconds':time.monotonic()-start,'error_type':None})
        return result
    def sample(self):
        apps=self.query(['--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'])
        foreign=[x for x in apps if x and int(x[0].strip())!=os.getpid()]
        if foreign:self.violations.append('Foreign compute appeared on the assigned physical GPU')
        if self.stop.is_set():return
        hardware=self.query(['--query-gpu=name,driver_version,memory.total,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu','--format=csv,noheader,nounits'])
        record={'monotonic_seconds':time.monotonic(),'hardware':hardware,'foreign_process_count_on_assigned_GPU':len(foreign)}
        if not self.stop.is_set():
            used=self.query(['--query-gpu=memory.used','--format=csv,noheader,nounits'])
            record['physical_GPU_used_memory_MiB']=int(used[0][0].strip())
        self.samples.append(record)
    def start(self):
        self.sample();self.check()
        def loop():
            while not self.stop.wait(5):
                try:self.sample()
                except BaseException as error:
                    self.violations.append('Assigned GPU monitor failed: '+type(error).__name__)
                    break
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start()
    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=35)
            if self.thread.is_alive():self.violations.append('Assigned GPU monitor did not exit')
    def report(self):
        out=super().report();out.update(sampling_interval_seconds=5,query_timeout_seconds=30,close_join_seconds=35,query_diagnostics=self.query_diagnostics)
        return out

def reserve_before_assets(torch,*,timeout_seconds=5):
    monitor=Monitor()
    if torch.cuda.device_count()!=1:raise RuntimeError('One UUID-selected CUDA device required')
    started=time.monotonic();torch.cuda.init();reservation=torch.empty(1,device='cuda',dtype=torch.uint8);torch.cuda.synchronize()
    while True:
        apps=monitor.query(['--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'])
        pids={int(row[0].strip()) for row in apps if row}
        if pids-{os.getpid()}:raise RuntimeError('Foreign compute appeared before CPU asset loading')
        if os.getpid() in pids:break
        if time.monotonic()-started>timeout_seconds:raise RuntimeError('Model CUDA context not visible to NVML; do not load assets')
        time.sleep(.1)
    monitor.start()
    receipt={'before_CPU_asset_loading':True,'same_model_process':True,'NVML_UUID_matches_locked_GPU':True,'own_process_visible_to_NVML':True,'reservation_tensor_logical_bytes':1,'setup_seconds':time.monotonic()-started}
    return reservation,monitor,receipt
