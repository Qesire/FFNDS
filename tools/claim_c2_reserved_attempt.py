"""Allow one of two CPU idle queues to enter the fixed C2N1 model attempt.

The persistent shared mkdir claim is never released or retried. Losing queues
exit before dependency/model imports or CUDA. Actual GPU isolation remains
with the frozen UUID queue and same-process reservation implementation.
"""
import argparse
import json
import os
from pathlib import Path
import time


def claim_once(root, supervisor_folder, worker_slot, protocol_sha256):
    allowed = {
        'executor': 'runs/c2_reserved_wait_executor',
        'original': 'runs/c2_reserved_wait_original',
    }
    allowed.update({f'slot{i}': f'runs/c2_reserved_wait_slot{i}' for i in range(7)})
    if allowed.get(worker_slot) != supervisor_folder:
        raise RuntimeError('Unexpected CPU worker slot or supervisor folder')
    folder = root / supervisor_folder
    if not folder.is_dir():
        raise RuntimeError('Missing independently supervised CPU queue')
    claim = root / 'runs/C2N1_SINGLE_MODEL_CLAIM'
    try:
        claim.mkdir()
    except FileExistsError:
        return False
    # A failed winner never gives the scope to the other queue automatically.
    receipt = {
        'status': 'CLAIMED_BEFORE_DEPENDENCY_MODEL_OR_CUDA',
        'worker_slot': worker_slot, 'supervisor_folder': supervisor_folder,
        'protocol_sha256': protocol_sha256, 'claimed_epoch_seconds': time.time(),
        'automatic_retry': False, 'max_concurrent_model_GPUs': 1,
        'new_model_calls_at_claim': 0,
    }
    (claim / 'CLAIM.json').write_text(json.dumps(receipt, indent=2) + '\n')
    canonical = root / 'runs/c2_reserved_execution'
    canonical.symlink_to(folder, target_is_directory=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-slot', required=True, choices=['executor', 'original'] + [f'slot{i}' for i in range(7)])
    parser.add_argument('--supervisor-folder', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    from c2_reserved_contract import ROOT, require_registration, sha
    from worker_identity import require_worker
    require_worker()
    protocol = require_registration()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('One fixed source-view/model child command required')
    if not claim_once(ROOT, args.supervisor_folder, args.worker_slot, sha(ROOT / protocol['self_path'])):
        print('CPU queue lost the shared once-only claim; no model or CUDA started.', flush=True)
        return
    print('Won the shared once-only C2N1 claim; starting the fixed child.', flush=True)
    os.execvpe(command[0], command, dict(os.environ))


if __name__ == '__main__':
    main()
