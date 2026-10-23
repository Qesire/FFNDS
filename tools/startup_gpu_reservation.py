"""Claim the model's own CUDA context before long CPU preparation.

No Torch import or GPU operation occurs merely by importing this module.
An authorized caller supplies Torch after its registration guard has passed.
"""
import os
import subprocess
import threading
import time

from assigned_gpu_resources import AssignedGPUResources


def gpu_uuid(index):
    value = subprocess.check_output([
        'nvidia-smi', '-i', index, '--query-gpu=uuid',
        '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
    if not value.startswith('GPU-') or '\n' in value or ',' in value:
        raise RuntimeError('Require one physical GPU UUID')
    return value


class UUIDAssignedGPUResources(AssignedGPUResources):
    """Keep NVML's physical index distinct from CUDA's UUID visibility token."""
    def __init__(self):
        index = os.environ.get('FFNDS_PHYSICAL_GPU', '')
        if index not in ('0', '1', '2', '3'):
            raise RuntimeError('One explicit locked physical GPU required')
        if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu_uuid(index):
            raise RuntimeError('CUDA UUID and locked physical GPU differ')
        self.gpu = index
        self.samples, self.violations = [], []
        self.stop, self.thread = threading.Event(), None

    def sample(self):
        super().sample()
        value = self.query(['--query-gpu=memory.used', '--format=csv,noheader,nounits'])
        self.samples[-1]['physical_GPU_used_memory_MiB'] = int(value[0][0].strip())

    def report(self):
        result = super().report()
        result['peak_physical_used_memory_MiB'] = max(
            (s['physical_GPU_used_memory_MiB'] for s in self.samples if 'physical_GPU_used_memory_MiB' in s),
            default=None)
        return result


def reserve_before_assets(torch, *, timeout_seconds=5):
    """Use a tiny allocation in the actual model PID, then verify its visibility.

    The returned tensor stays alive while assets load. It is not a second
    holder process and executes no FFN, norm, selector or learned model.
    The existing foreign-process guard remains active during preparation.
    """
    monitor = UUIDAssignedGPUResources()
    if torch.cuda.device_count() != 1:
        raise RuntimeError('One UUID-selected CUDA device required')
    started = time.monotonic()
    torch.cuda.init()
    reservation = torch.empty(1, device='cuda', dtype=torch.uint8)
    torch.cuda.synchronize()
    while True:
        apps = monitor.query(['--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'])
        pids = {int(row[0].strip()) for row in apps if row}
        if pids - {os.getpid()}:
            raise RuntimeError('Foreign compute appeared before CPU asset loading')
        if os.getpid() in pids:
            break
        if time.monotonic() - started > timeout_seconds:
            raise RuntimeError('Model CUDA context not visible to NVML; do not load assets')
        time.sleep(.1)
    monitor.start()
    receipt = {'before_CPU_asset_loading': True, 'same_model_process': True,
               'NVML_UUID_matches_locked_GPU': True, 'own_process_visible_to_NVML': True,
               'reservation_tensor_logical_bytes': 1, 'setup_seconds': time.monotonic() - started,
               'own_NVML_process_memory_MiB_at_claim': next(int(row[1].strip()) for row in apps
                                                            if row and int(row[0].strip()) == os.getpid()),
               'new_FFN_norm_selector_or_generation_calls': 0}
    return reservation, monitor, receipt
