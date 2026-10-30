"""Versioned namespace repair with early assigned-GPU checks during reads.

The original native acquisition and numerical observation stay unchanged.
An actual foreign-compute conflict motivated checking the existing monitor
after every charged CPU-asset read, rather than waiting for model setup.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a5_hwg_matched_capture_namespace_v2_20261030 as repaired

SELF = 'tools/a5_hwg_matched_capture_relocation_v3_20261030.py'


def attach_monitor_to_reads(budget, monitor):
    original_check = budget.check
    def check():
        original_check()
        monitor.check()
    budget.check = check


def main():
    import startup_gpu_reservation
    original = startup_gpu_reservation.reserve_before_assets
    def reserve(torch, *, timeout_seconds=5):
        result = original(torch, timeout_seconds=timeout_seconds)
        boot = repaired.owner.CURRENT_BOOTSTRAP
        repaired.owner.need(boot is not None, 'reservation requires durable bootstrap')
        attach_monitor_to_reads(boot.budget, result[1])
        return result
    startup_gpu_reservation.reserve_before_assets = reserve
    repaired.owner.SELF = SELF
    repaired.owner.validate_import_origins = repaired.validate_import_origins
    try:
        repaired.owner.main()
    finally:
        startup_gpu_reservation.reserve_before_assets = original


if __name__ == '__main__': main()
