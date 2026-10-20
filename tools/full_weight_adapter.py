"""Full retained matrices with original native-QDQ forward and an anchored penalty."""
import torch

from folded_adapter import straight_through


class FullMainWeights(torch.nn.Module):
    def __init__(self, w0, w2):
        super().__init__()
        for slot, value in [('0', w0), ('2', w2)]:
            prior = value.detach().float().clone()
            self.register_buffer('prior' + slot, prior)
            self.register_parameter('weight' + slot, torch.nn.Parameter(prior.clone()))
        if any(not bool(torch.isfinite(getattr(self, 'prior' + s)).all())
                or float(getattr(self, 'prior' + s).square().mean()) <= 0 for s in ['0', '2']):
            raise ValueError('Nonzero finite main priors required')

    def weights(self, quantizer):
        return tuple(straight_through(getattr(self, 'weight' + s), quantizer).bfloat16() for s in ['0', '2'])

    def relative_deviation(self):
        return .5 * sum((getattr(self, 'weight' + s) - getattr(self, 'prior' + s)).square().mean()
            / getattr(self, 'prior' + s).square().mean() for s in ['0', '2'])
