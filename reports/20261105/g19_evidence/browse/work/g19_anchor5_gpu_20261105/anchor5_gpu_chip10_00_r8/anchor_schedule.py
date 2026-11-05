"""Pure scheduling state for original-model anchor experiments; integration pending review."""
class AnchorSchedule:
    def __init__(self):
        self.anchors = {}
        self.last_expert = None
        self.step = -1
        self.switch = False

    def begin_step(self, step, expert):
        if step != self.step + 1 or not 0 <= step < 40:
            raise ValueError('non-sequential sampling step')
        self.switch = self.last_expert is not None and expert != self.last_expert
        self.step, self.last_expert = step, expert

    def action(self, expert, branch, layer):
        if expert != self.last_expert or branch not in ('conditional', 'unconditional') or not 0 <= layer < 40:
            raise ValueError('invalid actual dispatch')
        key = (expert, branch, layer)
        if self.step % 5 == 0 or self.switch:
            return 'anchor'
        if key not in self.anchors: raise ValueError('missing anchor')
        age = self.step - self.anchors[key]
        if not 1 <= age <= 4:
            raise ValueError('stale or duplicate cache')
        return 'hit'

    def record_anchor(self, expert, branch, layer):
        key = (expert, branch, layer)
        if self.action(*key) != 'anchor':
            raise ValueError('unexpected refresh')
        self.anchors[key] = self.step
