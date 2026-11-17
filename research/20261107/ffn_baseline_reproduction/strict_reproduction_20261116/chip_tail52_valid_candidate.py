"""Executable valid-count 52-row Chipmunk tail candidate for bounded E6 review.

Inputs and caches are BF16. Selection uses exactly 52 real rows; only the
unchanged upstream Triton MM1/MM2 invocation is padded to 128 rows. The caller
must load the SHA-bound author provider with its reviewed safe-selector wrapper,
configure the same global MLP settings, and run inside an independently
reviewed resource guard. This module does not dispatch any jobs.
"""
import torch
import torch.nn.functional as F

VALID = 52
PADDED = 128
WIDTH = 5120
HIDDEN = 13824


def _shape(x, shape, label):
    if tuple(x.shape) != shape or x.dtype != torch.bfloat16 or x.device.type != 'cuda':
        raise ValueError(f'invalid {label}')


def _padded(valid, *, pad_value=0.0):
    result = torch.full((PADDED, valid.shape[1]), pad_value,
                        dtype=valid.dtype, device=valid.device)
    result[:VALID].copy_(valid)
    return result


def _apply_author(ops, x, w1, b1, w2t, indices, counts, h, y):
    # h is hidden x padded rows, exactly the layout used by author MM1.
    ops.mlp(x=x, fc1w=w1, fc1b=b1, fc2w_T=w2t, indices=indices,
            counts=counts, sparse_act_T=h, cached_out=y,
            num_sms_scatter_add=6, mm1_scale_a=None, mm1_scale_b=None)


class ValidCountTail52:
    """A single layer/request. Caller supplies the actual inference step clock."""
    def __init__(self, first, second, ops, config, *, w2t=None):
        self.first, self.second, self.ops, self.config = first, second, ops, config
        self.w1, self.b1 = first.weight, first.bias
        self.w2, self.b2 = second.weight, second.bias
        _shape(self.w1, (HIDDEN, WIDTH), 'W1')
        _shape(self.b1, (HIDDEN,), 'b1')
        _shape(self.w2, (WIDTH, HIDDEN), 'W2')
        _shape(self.b2, (WIDTH,), 'b2')
        if config['provider'] != 'triton' or not config['is_enabled']:
            raise ValueError('enabled author BF16 Triton provider required')
        self.w2t = self.w2.transpose(0, 1).contiguous() if w2t is None else w2t
        _shape(self.w2t, (HIDDEN, WIDTH), 'transposed W2')
        if not self.w2t.is_contiguous(): raise ValueError('transposed W2 must be contiguous')
        self.h = self.y = self.mid = self.indices = self.counts = None
        self.step = -1

    def forward(self, x, step, *, padding_probe=False):
        _shape(x, (VALID, WIDTH), 'tail input')
        if step != self.step + 1:
            raise RuntimeError('nonconsecutive tail clock')
        self.step = step
        full = step % self.config['full_step_every'] == 0
        if full:
            pre = self.first(x)
            act = F.gelu(pre, approximate='tanh')
            out = self.second(act)
            _shape(pre, (VALID, HIDDEN), 'anchor preactivation')
            _shape(act, (VALID, HIDDEN), 'anchor activation')
            _shape(out, (VALID, WIDTH), 'anchor output')
            self.mid = pre.mean(dim=0, keepdim=True).unsqueeze(0).contiguous()
            self.h = _padded(act).T.contiguous()
            self.y = _padded(out)
            # As in upstream, a full step updates caches but keeps any old mask.
            return out, {'kind': 'full', 'valid_rows': VALID}
        if self.mid is None: raise RuntimeError('first step must seed dense caches')
        recompute = not (step % self.config['block_mask_cache'] != 0 and
                         self.indices is not None and step >= 10)
        if recompute:
            # Torch mean preserves BF16 input/output like upstream block_mean.
            bm = self.first(x.mean(dim=0, keepdim=True)).unsqueeze(0).contiguous()
            mdiff = (bm - self.mid).abs()
            inds = torch.empty_like(mdiff, dtype=torch.int32)
            counts = torch.empty((1, 1), dtype=torch.int32, device=x.device)
            self.ops.topk_indices(mdiff, inds, counts, 1 - self.config['top_keys'],
                                  256, self.config['random_keys'])
            # Wrapper validates prefix and supplies MM2's 64-element lookahead.
            self.ops.copy_indices(bm, self.mid, inds, counts)
            self.indices, self.counts = inds, counts
        indices, counts = self.indices[0], self.counts[0]
        xpad = _padded(x)
        probe = None
        if padding_probe:
            # Identical selected mask and valid state, arbitrary invalid rows.
            alt_x = xpad.clone(); alt_x[VALID:] = 3.0
            alt_h, alt_y = self.h.clone(), self.y.clone()
            alt_h[:, VALID:] = -2.0; alt_y[VALID:] = 5.0
            _apply_author(self.ops, alt_x, self.w1, self.b1, self.w2t,
                          indices, counts, alt_h, alt_y)
        _apply_author(self.ops, xpad, self.w1, self.b1, self.w2t,
                      indices, counts, self.h, self.y)
        if padding_probe:
            if not torch.equal(alt_h[:, :VALID], self.h[:, :VALID]):
                raise AssertionError('padding changed valid hidden cache')
            if not torch.equal(alt_y[:VALID], self.y[:VALID]):
                raise AssertionError('padding changed valid output cache')
            probe = {'mask_shared': True, 'valid_h_bitwise_equal': True,
                     'valid_y_bitwise_equal': True}
        return self.y[:VALID].clone(), {
            'kind': 'sparse', 'valid_rows': VALID, 'mask_refreshed': recompute,
            'count': int(counts[0].item()), 'padding_invariance': probe,
        }
