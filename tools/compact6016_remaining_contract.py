"""Pure metadata contract for the 66 remaining compact6016 cells.

The 54 parent RUN cells are candidates for reuse; their payload hashes and
process termination are independently audited before any native driver runs.
No partial training parameters or optimizer state may be loaded.
"""
import hashlib
import json
from pathlib import Path

import e0_compact6016_full80_completion as completion

ROOT = Path(__file__).resolve().parents[1]
FROZEN_PATH = ROOT / 'config/e0_compact6016_full80_completion_20261028_v1.json'
PROTOCOL_SHA256 = 'b9e98842b22d5edb3e8277c62e7d7b8dc9514606f257a231f01821b42413e265'
ID = 'E0_COMPACT6016_REMAINING66_20261028'
EXPERTS = completion.EXPERTS
INITIALIZATIONS = completion.INITIALIZATIONS
FUNCTIONS = completion.FUNCTIONS
OLD_CELLS = frozenset((e, b, i) for e, b in completion.COMPLETED_FUNCTIONS for i in INITIALIZATIONS)
PARENT_CELLS = frozenset(
    (e, b, init)
    for part in range(4)
    for n, (e, b) in enumerate(completion.functions_for(part))
    for init in INITIALIZATIONS[:(2 if n < (7 if part < 2 else 6) else 1 if n == 6 and part >= 2 else 0)]
)
REUSED_CELLS = OLD_CELLS | PARENT_CELLS
ALL_CELLS = frozenset((e, b, i) for e, b in FUNCTIONS for i in INITIALIZATIONS)
MISSING_CELLS = ALL_CELLS - REUSED_CELLS
REMAINING_FUNCTIONS = tuple(f for f in FUNCTIONS if any((*f, i) in MISSING_CELLS for i in INITIALIZATIONS))
PART_SIZES = (9, 9, 8, 8)
RESOURCE_LIMITS = {'GPU_per_part': 1, 'global_max_GPU': 2, 'CPU_threads_per_part': 2,
                   'compute_wall_seconds_per_part': 3600, 'external_wall_seconds_per_part': 3900,
                   'new_HF_bytes_per_part': 3 << 30, 'new_HF_bytes_total': 12 << 30}
INITIALIZATION_CONTRACT = {'initialization': 'original_native_QDQ_slices_and_original_seed42_factors_only',
                           'load_partial_parameters': False, 'load_partial_optimizer': False}


def functions_for(part):
    if type(part) is not int or part not in range(4):
        raise ValueError('One of four fixed parts required')
    start = sum(PART_SIZES[:part])
    return REMAINING_FUNCTIONS[start:start + PART_SIZES[part]]


def initializations_for(expert, block):
    if (expert, block) not in FUNCTIONS or type(block) is not int:
        raise ValueError('Unknown native function')
    return tuple(i for i in INITIALIZATIONS if (expert, block, i) in MISSING_CELLS)


def run_ids(part=None):
    functions = REMAINING_FUNCTIONS if part is None else functions_for(part)
    return [f"compact6016remain_{'high' if e == EXPERTS[0] else 'low'}_b{b:02d}_{i}"
            for e, b in functions for i in initializations_for(e, b)]


def part_limits(part):
    functions = functions_for(part)
    cells = len(run_ids(part))
    result = {}
    for key, value in completion.PART_LIMITS.items():
        divisor = 15 if key.startswith('support_') else 30
        if value % divisor:
            raise ValueError('Nonintegral frozen cost coefficient')
        result[key] = value // divisor * (len(functions) if key.startswith('support_') else cells)
    return result


TOTAL_LIMITS = {key: sum(part_limits(p)[key] for p in range(4)) for key in completion.PART_LIMITS}


def validate_budget(part, predebited, observed):
    limits = part_limits(part)
    if set(predebited) != set(limits) or set(observed) != set(limits):
        raise ValueError('Budget keys differ')
    for key, limit in limits.items():
        pre, obs = predebited[key], observed[key]
        if type(pre) is not int or type(obs) is not int or not 0 <= obs <= pre <= limit:
            raise ValueError('Budget exceeds fixed part limit: ' + key)


def verify_partition(frozen_completion_plan, parent_runs):
    """Verify frozen design and RUN metadata; payload/termination audit is separate."""
    raw = FROZEN_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_SHA256:
        raise ValueError('Frozen completion protocol hash differs')
    if frozen_completion_plan != json.loads(raw):
        raise ValueError('Frozen completion design differs')
    if len(parent_runs) != 4 or [r.get('part') for r in parent_runs] != list(range(4)):
        raise ValueError('Four ordered parent RUNs required')
    observed = []
    for part, run in enumerate(parent_runs):
        if (run.get('id') != completion.ID or run.get('protocol_sha256') != PROTOCOL_SHA256
                or run.get('status') != 'FAILED_NO_RETRY' or run.get('error_type') != 'BrokenPipeError'
                or run.get('source_split') != frozen_completion_plan['source_split']):
            raise ValueError('Parent scope, protocol, failure or source split differs')
        cells = run.get('cells', [])
        if len(cells) != (14, 14, 13, 13)[part]:
            raise ValueError('Parent cell count differs')
        for cell in cells:
            e, b, init = cell.get('expert'), cell.get('block'), cell.get('initialization')
            key = (e, b, init)
            if (key not in PARENT_CELLS or (e, b) not in completion.functions_for(part)
                    or cell.get('status') != completion.CELL_STATUS
                    or cell.get('run_id') != completion.function_id(e, b) + '_' + init):
                raise ValueError('Parent completed cell identity differs')
            observed.append(key)
    if len(set(observed)) != 54 or set(observed) != PARENT_CELLS:
        raise ValueError('Parent cells duplicated or missing')
    new = {(e, b, i) for e, b in REMAINING_FUNCTIONS for i in initializations_for(e, b)}
    if (len(OLD_CELLS) != 40 or len(PARENT_CELLS) != 54 or len(REUSED_CELLS) != 94
            or len(new) != 66 or len(REMAINING_FUNCTIONS) != 34
            or new & REUSED_CELLS or new | REUSED_CELLS != ALL_CELLS
            or len(ALL_CELLS) != 160):
        raise ValueError('94 reused and 66 complement do not partition 160 cells')
    return {'reused_cells': len(REUSED_CELLS), 'missing_cells': len(new),
            'missing_functions': len(REMAINING_FUNCTIONS), 'overlap': 0}
