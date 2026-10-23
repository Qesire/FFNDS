"""CPU-only review/export of a separately authorized, finished C2N1 attempt."""
import os
from pathlib import Path
from c2_reserved_contract import METHOD, ROOT, read


def main():
    from worker_identity import require_worker
    require_worker()
    relative = 'derived/ffnds/C2N1_reserved_norm_scaled_smoke'
    root = Path(os.environ['HF_HOME']) / relative
    run = read(root / 'RUN.json')
    if run['id'] != METHOD: raise RuntimeError('Wrong experiment; preserve original C2N0')
    if run['status'] == 'COMPLETE_C2N1_NATIVE_VIDEOS_PENDING_FULL_REVIEW':
        from review_c2_native_videos import main as review
        review(root_relative=relative, complete_status=run['status'], method_id=METHOD)
    from export_c2_native_results import main as export
    export(root_relative=relative, prefix='C2N1', supervisor_folder='runs/c2_reserved_execution')
    from check_c2_native_results import main as check
    check(prefix='C2N1', registration_key='registered_c2_reserved_smoke',
          complete_status='COMPLETE_C2N1_NATIVE_VIDEOS_PENDING_FULL_REVIEW')


if __name__ == '__main__': main()
