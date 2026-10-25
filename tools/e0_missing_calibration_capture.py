"""Only missing calibration blocks, uniform32, three native fields."""
import ffnds_observation_capture as original
from e0_contiguous_capture import ContiguousCapture
from ffnds_observation_common import row_plan


def uniform_plan(grid, source_id, uniform_count=32):
    p = row_plan(grid, source_id, uniform_count)
    rows = sorted(p['uniform_rows'])
    p.update(rows=rows, coordinates=[[r // (grid[1] * grid[2]), r // grid[2] % grid[1], r % grid[2]] for r in rows],
             neighbor_pairs=[], contiguous_tiles=[],
             sampling='Original source-specific uniform32 only; no repeated five-block fields')
    return p


class MissingCalibrationCapture(ContiguousCapture):
    fields = {'q1_post', 'q2_post', 'ffn_output'}

    def __enter__(self):
        self.original_plan = original.row_plan
        original.row_plan = uniform_plan
        try:
            return original.DiscoveryCapture.__enter__(self)
        except BaseException:
            original.row_plan = self.original_plan
            raise

    def weight_binding(self, expert, block):
        return {'expert': expert, 'block': block, 'cache_identity': self.cache_identity,
                'scope': 'Original decoded checkpoint; no duplicate panels or fitted weights'}
