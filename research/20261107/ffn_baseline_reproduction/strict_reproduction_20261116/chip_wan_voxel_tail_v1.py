"""Explicit Wan voxel/full-block/dense-tail adaptation of upstream SparseDiffMlp.

Caller binds the frozen upstream imports, the independently reviewed safe selector
shared library, its global config and counter. This adapter is not a deployment gate.
"""
import torch

TOKENS = 31668
FULL = 247 * 128
TAIL = 52
HIDDEN = 13824
WIDTH = 5120


def install_safe_selector_boundary(ops):
    """Preserve safe selector results and add backing storage for Triton lookahead.

    Upstream mm2 loads another 64 indices after its final loop iteration. Backing
    storage is enlarged without changing the tensor's visible shape or row stride.
    This function must be installed before SparseDiffMlp starts allocating masks.
    The separately reviewed safe selector must already be registered in torch.ops.
    """
    original_topk, original_copy, original_mlp = ops.topk_indices, ops.copy_indices, ops.mlp
    def validate(indices, counts):
        if indices.shape[-1] != HIDDEN: raise ValueError('unexpected hidden width')
        for row, count in zip(indices.reshape(-1, HIDDEN), counts.reshape(-1)):
            n = int(count.item())
            if not (0 <= n <= HIDDEN and n % 256 == 0): raise RuntimeError('unsafe selector count')
            prefix = row[:n]
            if not bool(((prefix >= 0) & (prefix < HIDDEN)).all().item()): raise RuntimeError('invalid selector prefix')
            if prefix.unique().numel() != n: raise RuntimeError('duplicate selector index')
    def topk(activation, indices, counts, *rest):
        indices.fill_(-1); counts.fill_(-1)
        original_topk(activation, indices, counts, *rest)
        validate(indices, counts)
        size = indices.numel()
        backing = torch.zeros(size + 64, device=indices.device, dtype=indices.dtype)
        visible = backing[:size].view(indices.shape)
        visible.copy_(indices)
        indices.set_(visible)
    def copy(x, cache, indices, counts):
        validate(indices, counts)
        return original_copy(x, cache, indices, counts)
    def mlp(**kwargs):
        indices, counts = kwargs['indices'], kwargs['counts']
        validate(indices, counts)
        if indices.untyped_storage().nbytes() < (indices.storage_offset() + indices.numel() + 64) * indices.element_size():
            raise RuntimeError('missing MM2 lookahead backing')
        return original_mlp(**kwargs)
    ops.topk_indices, ops.copy_indices, ops.mlp = topk, copy, mlp


class WanVoxelSparseTail:
    def __init__(self, sparse_module, voxel_to_raster, raster_to_voxel):
        self.module = sparse_module
        self.perm = voxel_to_raster
        self.inverse = raster_to_voxel
        if self.perm.shape != (TOKENS,) or self.inverse.shape != (TOKENS,): raise ValueError('wrong voxel mapping')
        ordered = torch.arange(TOKENS, device=self.perm.device)
        if not torch.equal(self.perm[self.inverse], ordered): raise ValueError('voxel mapping not bijective')
        if sparse_module.fc1[0].weight.shape != (HIDDEN, WIDTH) or sparse_module.fc2[0].weight.shape != (WIDTH, HIDDEN):
            raise ValueError('not Wan 5120/13824 MLP')
    def __call__(self, raster):
        if raster.shape != (1, TOKENS, WIDTH): raise ValueError('unexpected raster input')
        voxel = raster.index_select(1, self.perm)
        head = self.module(voxel[:, :FULL].contiguous())
        tail = self.module.fc2[0](self.module.activation(self.module.fc1[0](voxel[:, FULL:])))
        if tail.shape[1] != TAIL: raise AssertionError('tail length')
        return torch.cat((head, tail), dim=1).index_select(1, self.inverse)
