"""Record an exact test count and source inventory for a CPU preflight."""
import argparse
from pathlib import Path
import sys
import unittest

import torch

from ar2_checkpoint import atomic_json,digest

ROOT = Path(__file__).resolve().parents[1]
# Running a file under tools/ does not add the repository root as `python -m`
# does. Some existing tests import the tools namespace rather than flat modules.
sys.path.insert(0,str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--expected',type=int,required=True)
    ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--log',type=Path,required=True)
    args = ap.parse_args()
    args.log.parent.mkdir(parents=True,exist_ok=True)
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    with args.log.open('w') as log:
        result = unittest.TextTestRunner(stream=log,verbosity=1).run(suite)
    passed = result.wasSuccessful() and result.testsRun == args.expected and not result.skipped and not torch.cuda.is_initialized()
    value = {'status':'PASS_COMPLETE_CPU_UNIT_TEST_PREFLIGHT' if passed else 'FAILED_CPU_UNIT_TEST_PREFLIGHT',
             'tests_run':result.testsRun,'expected':args.expected,'skipped':len(result.skipped),
             'failures':len(result.failures),'errors':len(result.errors),'CUDA_initialized':torch.cuda.is_initialized(),
             'torch':torch.__version__,'source_sha256':digest(Path(__file__)),'log_sha256':digest(args.log),
             'test_sources':{str(p):digest(p) for p in sorted((ROOT/'tests').glob('test_*.py'))}}
    atomic_json(args.report,value)
    if not passed:
        raise RuntimeError('Exact complete CPU test preflight failed; inspect the preserved log')


if __name__ == '__main__':
    main()
