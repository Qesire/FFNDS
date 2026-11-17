"""Pre-gate author Chipmunk with safe selector, true raster voxel order and dense tail."""

import torch
import torch.nn.functional as F

from wan_ffn_port import dense


class NativeLinear:
    """Expose the currently active native MMWeight with the author Linear layout."""

    def __init__(self, operation):
        self.op = operation
        actual = operation._get_actual_weight()
        self.weight, self.bias = actual.T.contiguous(), operation._get_actual_bias()
        if self.bias is None or actual.data_ptr() != operation.weight_cuda_buffer.data_ptr():
            raise RuntimeError('native Chip encountered invalid live weights')

    def __call__(self, inputs):
        rows = inputs.reshape(-1, inputs.shape[-1])
        result = self.op.apply(rows)
        return result.reshape(*inputs.shape[:-1], result.shape[-1])


class NativeAuthorChipPort:
    """The author MLP owns active proxy, masks and outputs; tail stays native Dense."""

    def __init__(self, *, force_dense, provider=None, voxel=None):
        if not force_dense and (provider is None or voxel is None):
            raise ValueError('missing approved author provider or voxel map')
        self.force_dense, self.provider = force_dense, provider
        self.seen, self.storages = [], {}
        self.voxel = voxel
        if voxel is not None:
            perm, inv = self.voxel
            index = torch.arange(31668, device=perm.device)
            if perm.shape != index.shape or inv.shape != index.shape or not torch.equal(perm.index_select(0, inv), index):
                raise RuntimeError('invalid native author voxel mapping')

    def __call__(self, model, phase, u):
        step, layer = int(model.scheduler.step_index), int(model.block_idx)
        if (step, layer) != divmod(len(self.seen), 40) or u.shape != (31668, 5120) or u.dtype != torch.bfloat16:
            raise RuntimeError('invalid native Chip site')
        expert = 'high' if step < 2 else 'low'
        if self.force_dense or layer < 2:
            y = dense(u, phase.ffn_0.apply, phase.ffn_2.apply)[2]
            kind = 'forced_dense' if self.force_dense else 'fixed_dense'
            counts = None
        else:
            y, counts = self._sparse(step, layer, expert, phase, u)
            kind = 'anchor' if step in (0, 2) else 'hit'
        if y.dtype != u.dtype or y.shape != u.shape:
            raise RuntimeError('invalid Chipmunk pre-gate result')
        self.seen.append({'step': step, 'layer': layer, 'expert': expert, 'kind': kind,
                          'block_counts': counts, 'dense_tail_rows': 52 if not self.force_dense and layer >= 2 else 0})
        return y

    def _sparse(self, step, layer, expert, phase, u):
        sparse_type, counter_type = self.provider
        key = expert, layer
        counter = counter_type(40, 1)
        counter.cur_inference_step, counter.cur_layer = step % 2, layer
        first, second = NativeLinear(phase.ffn_0), NativeLinear(phase.ffn_2)
        activation = lambda x: F.gelu(x, approximate='tanh')
        module = sparse_type(layer, counter, first, activation, second)
        perm, inv = (x.to(u.device) for x in self.voxel)
        arranged = u.index_select(0, perm).unsqueeze(0)
        head = arranged[:, :31616].contiguous()
        tail = arranged[:, 31616:]
        if step % 2 == 1:
            module.storage = self.storages.pop(key)
        result = module(head)
        counts = module.storage.get_counts()
        if step % 2 == 0:
            if key in self.storages or counts is not None:
                raise RuntimeError('Chip anchor unexpectedly inherits another request/step state')
            self.storages[key] = module.storage
        else:
            if counts is None or counts.shape != (1, 247):
                raise RuntimeError('missing 247 author support counts')
            counts = counts[0].tolist()
        tail_result = second(activation(first(tail)))
        return torch.cat((result, tail_result), dim=1)[0].index_select(0, inv), counts

    def finish(self):
        if len(self.seen) != 160 or self.storages:
            raise RuntimeError('incomplete author Chip request or retained state')
        return self.seen
