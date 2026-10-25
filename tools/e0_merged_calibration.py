"""Combine original five-block data with independently accepted missing blocks."""
import json

from e0_saved_function_common import ROOT
from ffnds_observation_common import sha256
from run_e0_nonlinear_reconstruction import CalibrationStore


class MergedCalibrationStore(CalibrationStore):
    def __init__(self, root, sources, torch):
        super().__init__(root, sources, torch)
        manifest = json.loads((ROOT / 'research/20261026/E0_MISSING_CALIBRATION_OBSERVATIONS.json').read_text())
        self.missing = {}
        for index in sources:
            binding = next(v for v in manifest['raw_bindings'] if v['source_index'] == index)
            path = root.parent / binding['HF_relative']
            if sha256(path) != binding['sha256']:
                raise ValueError('Missing-block observation changed')
            self.missing[index] = path, json.loads(path.read_text())

    def load(self, source, step, block):
        if block in [0, 10, 20, 30, 39]:
            return super().load(source, step, block)
        original = self.runs[source]
        self.runs[source] = self.missing[source]
        try:
            return super().load(source, step, block)
        finally:
            self.runs[source] = original
