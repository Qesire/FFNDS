"""CPU-only full media preparation and exact export after C2N2 finishes."""
import os
from pathlib import Path
from c2_environment_contract import METHOD, ROOT, read


def main():
    from worker_identity import require_worker
    require_worker()
    relative = 'derived/ffnds/C2N2_environment_preserved_smoke'
    root = Path(os.environ['HF_HOME']) / relative
    run = read(root / 'RUN.json')
    if run['id'] != METHOD:
        raise RuntimeError('Wrong fixed attempt')
    if run['status'] == 'COMPLETE_C2N2_NATIVE_VIDEOS_PENDING_FULL_REVIEW':
        from review_c2_native_videos import main as review
        review(root_relative=relative, complete_status=run['status'], method_id=METHOD)
    from export_c2_environment_results import main as export
    export()
    from check_c2_environment_results import main as check
    check()


if __name__ == '__main__':
    main()
