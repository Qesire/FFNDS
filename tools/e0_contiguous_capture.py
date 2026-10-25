"""Passive adjacent-token capture; all native operations remain original.

The original random32 rows are also retained for accepted-run identity checks.
"""
import math

import ffnds_observation_capture as original
from ffnds_observation_common import sha256, write_json


def contiguous_plan(grid, source_id, uniform_count=32):
    from ffnds_observation_common import row_plan
    plan = row_plan(grid, source_id, uniform_count)
    m = math.prod(grid)
    starts = [128 * ((m // 128) * k // 6) for k in (1, 3, 5)]
    tiles = [list(range(start, start + 128)) for start in starts]
    rows = sorted(set(plan['rows']).union(*map(set, tiles)))
    plan.update(rows=rows, coordinates=[[r // (grid[1]*grid[2]),
                 r // grid[2] % grid[1], r % grid[2]] for r in rows],
                contiguous_tiles=tiles,
                tile_sampling='Three deterministic original M128 tiles at 1/6,3/6,5/6 of flattened sequence; not IID or tracked objects')
    return plan


class ContiguousCapture(original.DiscoveryCapture):
    fields = {'q1_post', 'z', 'gelu', 'q2_post', 'ffn_output'}

    def __enter__(self):
        self.original_plan = original.row_plan
        original.row_plan = contiguous_plan
        try:
            return super().__enter__()
        except BaseException:
            original.row_plan = self.original_plan
            raise

    def put(self, state, name, value):
        if name in self.fields:
            super().put(state, name, value)

    def finish(self, state):
        if state['seen'] != [0, 2] or state['qseen'] != [0, 2]:
            raise RuntimeError('Missing original projection/QDQ')
        if set(state['samples']) != self.fields | {'gate', 'b1', 'b2'}:
            raise RuntimeError('Unexpected adjacent-token field coverage')
        if any(not bool(self.torch.isfinite(v).all()) for v in state['samples'].values()):
            raise RuntimeError('Nonfinite adjacent operand')
        if any((s['step'], s['block']) == (state['step'], state['block']) for s in self.scopes):
            raise RuntimeError('Duplicate adjacent scope')
        path = self.out / f"step{state['step']}_block{state['block']:02d}.safetensors"
        if path.exists():
            raise FileExistsError(path)
        self.save_file(state['samples'], str(path))
        self.scopes.append({'step':state['step'], 'block':state['block'], 'expert':state['expert'],
                            'row_plan':state['plan'], 'path':path.name, 'sha256':sha256(path),
                            'weight_binding':self.weight_binding(state['expert'],state['block'])})
        write_json(self.out/'SCOPES.json', self.scopes)
        state['samples'].clear()

    def __exit__(self, *error):
        try:
            super().__exit__(*error)
        finally:
            original.row_plan = self.original_plan
