"""Readout-separated attention operators for the frozen centered/OUT-ABS graph.

External Q/K/V are token-first [token, head, channel]. Attention values,
features and history are head-first [head, token, channel]. Every subset API
accepts original token IDs; neither routes nor probe positions are renumbered.
The Torch partition/selected backends are streaming correctness references.
The optional Triton backends are separate real-arithmetic-equivalent kernels.
"""
from dataclasses import dataclass
from typing import Optional

import torch


def _work_dtype(x):
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def _ids(ids, device):
    return torch.as_tensor(ids, device=device, dtype=torch.long)


def _validate_qkv(q, k, v, route_indices, block_size, token_ids=None):
    if q.ndim != 3 or k.ndim != 3 or k.shape != v.shape or q.shape[1:] != k.shape[1:]:
        raise ValueError('Expected q[Q,H,D] and matching k,v[N,H,D]')
    if q.device != k.device or q.device != v.device or q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError('Q/K/V device and dtype must match')
    n, heads, _ = k.shape
    if block_size < 1 or route_indices.ndim != 3:
        raise ValueError('Expected a positive block size and route[H,QB,TOP]')
    blocks = (n + block_size - 1) // block_size
    if route_indices.shape[:2] != (heads, blocks) or not 1 <= route_indices.shape[2] <= blocks:
        raise ValueError('Route extent does not match global key/query blocks')
    if route_indices.device != q.device:
        raise ValueError('Routes and operands must share device')
    if token_ids is None and len(q) != n:
        raise ValueError('A full partition requires the full query axis')
    if token_ids is not None and (token_ids.ndim != 1 or len(token_ids) != len(q)):
        raise ValueError('token_ids must identify every query row')
    # Layout/route validation is reusable by adapters; kernel calls need not
    # silently reinterpret an out-of-range or repeated selected key block.
    torch._assert_async(((route_indices >= 0) & (route_indices < blocks)).all(), 'Route block out of range')
    sorted_route = route_indices.sort(-1).values
    torch._assert_async((sorted_route[..., 1:] != sorted_route[..., :-1]).all(), 'Repeated selected key block')
    if token_ids is not None:
        torch._assert_async(((token_ids >= 0) & (token_ids < n)).all(), 'Global query ID out of range')


@dataclass
class PartitionOutput:
    full: torch.Tensor                 # H,N,D, original value dtype
    selected: torch.Tensor             # H,N,D, original value dtype
    rho: torch.Tensor                  # H,N,1, FP32 (FP64 CPU reference)
    full_lse: torch.Tensor              # H,N
    selected_lse: torch.Tensor          # H,N
    centered: Optional[torch.Tensor] = None  # H,N,D, optional direct complement encoding


def value_moments(value, floor=1e-12):
    """Frozen population variance semantics; value is [N,H,D]."""
    variance, mean = torch.var_mean(value.to(_work_dtype(value)), dim=0, correction=0)
    scale = variance.clamp_min(0).sqrt()
    scale = torch.where(scale > floor, scale, 0)
    inverse = torch.where(scale > 0, scale.reciprocal(), 0)
    return mean[:, None], scale[:, None], inverse[:, None]


def centered_history(native, selected, rho, mean, inverse):
    dtype = _work_dtype(native)
    raw = native.to(dtype) - (1 - rho) * selected.to(dtype)
    return ((raw - rho * mean) * inverse).to(native.dtype)


def partition_attention(q, k, v, route_indices, *, block_size=128, backend='torch',
                        query_tile=32, key_tile=64, mean=None, inverse=None,
                        validate=True):
    """Single QK/PV traversal, separately accumulating selected/complement.

    Pass mean/inverse together to write centered history directly from the
    complement accumulator. This avoids the full-minus-selected cancellation.
    The real target is identical; reduction order differs from frozen kernels.
    """
    if (mean is None) != (inverse is None):
        raise ValueError('mean and inverse must be supplied together')
    if validate:
        _validate_qkv(q, k, v, route_indices, block_size)
    if backend == 'triton':
        from .partition_triton import partition_triton
        return partition_triton(q, k, v, route_indices, block_size=block_size,
                                query_tile=query_tile, key_tile=key_tile,
                                mean=mean, inverse=inverse)
    if backend != 'torch':
        raise ValueError(f'Unknown attention backend: {backend}')
    n, heads, dim = k.shape
    work = _work_dtype(q)
    full = torch.empty((heads, n, dim), device=q.device, dtype=q.dtype)
    selected = torch.empty_like(full)
    rho = torch.empty((heads, n, 1), device=q.device, dtype=work)
    full_lse = torch.empty((heads, n), device=q.device, dtype=work)
    selected_lse = torch.empty_like(full_lse)
    centered = torch.empty_like(full) if mean is not None else None
    # A query tile stays inside one frozen query block. Each key-block QK/PV
    # appears once and its sufficient statistics enter exactly one partition.
    for start in range(0, n, block_size):
        end = min(n, start + block_size)
        query = q[start:end].transpose(0, 1).to(work)
        maxima = [torch.full((heads, end-start), -torch.inf, device=q.device, dtype=work) for _ in range(2)]
        den = [torch.zeros_like(maxima[0]) for _ in range(2)]
        num = [torch.zeros((heads, end-start, dim), device=q.device, dtype=work) for _ in range(2)]
        for kb, ks in enumerate(range(0, n, block_size)):
            ke = min(n, ks + block_size)
            logits = (query @ k[ks:ke].permute(1, 2, 0).to(work)) * dim ** -.5
            mb = logits.amax(-1)
            weights = (logits - mb[..., None]).exp()
            lb = weights.sum(-1)
            zb = weights @ v[ks:ke].transpose(0, 1).to(work)
            flag = (route_indices[:, start // block_size] == kb).any(-1)[:, None]
            for part in range(2):
                active = flag if part == 0 else ~flag
                merged = torch.maximum(maxima[part], mb)
                old = torch.exp(maxima[part] - merged)
                new = torch.exp(mb - merged)
                den[part] = torch.where(active, old * den[part] + new * lb, den[part])
                num[part] = torch.where(active[..., None], old[..., None] * num[part] + new[..., None] * zb, num[part])
                maxima[part] = torch.where(active, merged, maxima[part])
        maximum = torch.maximum(maxima[0], maxima[1])
        w0 = torch.where(den[0] > 0, (maxima[0]-maximum).exp(), 0)
        w1 = torch.where(den[1] > 0, (maxima[1]-maximum).exp(), 0)
        total = w0 * den[0] + w1 * den[1]
        rr = (w1 * den[1] / total)[..., None]
        native = (w0[..., None]*num[0] + w1[..., None]*num[1]) / total[..., None]
        small = num[0] / den[0][..., None]
        full[:, start:end] = native.to(q.dtype)
        selected[:, start:end] = small.to(q.dtype)
        rho[:, start:end] = rr
        full_lse[:, start:end] = maximum + total.log()
        selected_lse[:, start:end] = maxima[0] + den[0].log()
        if centered is not None:
            omitted = num[1] / den[1].clamp_min(torch.finfo(work).tiny)[..., None]
            centered[:, start:end] = (rr * (omitted-mean) * inverse).to(q.dtype)
    return PartitionOutput(full, selected, rho, full_lse, selected_lse, centered)


@dataclass
class SelectedLayout:
    token_ids: torch.Tensor
    row_map: torch.Tensor
    active_tiles: torch.Tensor
    total_tokens: int
    block_size: int
    query_tile: int


def prepare_selected_layout(token_ids, total_tokens, *, block_size=128, query_tile=32):
    """Cache this once for P64 and each recurring query tile, then reuse by site."""
    if query_tile < 1 or block_size % query_tile:
        raise ValueError('query_tile must divide the frozen route block size')
    ids = _ids(token_ids, token_ids.device if isinstance(token_ids, torch.Tensor) else 'cpu')
    torch._assert_async(((ids >= 0) & (ids < total_tokens)).all(), 'Global query ID out of range')
    if len(ids) != len(torch.unique(ids)):
        raise ValueError('Repeated query IDs do not define an unambiguous row layout')
    padded = ((total_tokens + block_size - 1)//block_size)*block_size
    row_map = torch.full((padded,), -1, device=ids.device, dtype=torch.int32)
    row_map[ids] = torch.arange(len(ids), device=ids.device, dtype=torch.int32)
    active = torch.unique(ids // query_tile, sorted=True).to(torch.int32)
    return SelectedLayout(ids, row_map, active, total_tokens, block_size, query_tile)


def selected_attention(q, k, v, route_indices, token_ids, *, block_size=128,
                       backend='torch', layout=None, query_tile=32, key_tile=64,
                       validate=True):
    """Selected attention for arbitrary original query IDs, returning H,Q,D."""
    ids = _ids(token_ids, q.device)
    if validate:
        _validate_qkv(q, k, v, route_indices, block_size, ids)
    if backend == 'triton':
        from .partition_triton import selected_triton
        if layout is None:
            layout = prepare_selected_layout(ids, len(k), block_size=block_size, query_tile=query_tile)
        if layout.total_tokens != len(k) or layout.block_size != block_size:
            raise ValueError('Selected layout has a different global axis')
        if validate:
            torch._assert_async((layout.token_ids == ids).all(), 'Selected layout row order differs')
        return selected_triton(q, k, v, route_indices, layout, key_tile=key_tile)
    if backend != 'torch':
        raise ValueError(f'Unknown attention backend: {backend}')
    work = _work_dtype(q)
    nq, heads, dim = q.shape
    out = torch.empty((heads, nq, dim), device=q.device, dtype=q.dtype)
    logsum = torch.empty((heads, nq), device=q.device, dtype=work)
    offsets = torch.arange(block_size, device=q.device)
    # Each group's keys are shared by all its queries, retaining GEMM shape.
    for query_block in torch.unique(ids // block_size, sorted=True).tolist():
        local = torch.nonzero(ids // block_size == query_block).flatten()
        for head in range(heads):
            keys = (route_indices[head, query_block, :, None].long()*block_size + offsets).flatten()
            keys = keys[keys < len(k)]
            logits = q.index_select(0, local)[:, head].to(work) @ k.index_select(0, keys)[:, head].to(work).T
            logits = logits * dim ** -.5
            out[head, local] = (logits.softmax(-1) @ v.index_select(0, keys)[:, head].to(work)).to(q.dtype)
            logsum[head, local] = torch.logsumexp(logits, -1)
    return out, logsum


def _probe_locations(ids, probes):
    """Return (local row, probe production row), with no dense Q-by-P mask."""
    sorted_ids, order = torch.sort(probes)
    pos = torch.searchsorted(sorted_ids, ids)
    safe = pos.clamp_max(max(len(probes)-1, 0))
    matched = (pos < len(probes)) & (sorted_ids[safe] == ids)
    local = torch.nonzero(matched).flatten()
    return local, order[safe[local]]


@dataclass
class LowHistory:
    rho: torch.Tensor
    probe_ids: torch.Tensor
    history: Optional[torch.Tensor]
    anchor_design: Optional[torch.Tensor]
    probe_history: torch.Tensor
    probe_design: torch.Tensor
    code: Optional[torch.Tensor] = None
    basis: Optional[torch.Tensor] = None

    @classmethod
    def from_fields(cls, history, anchor_design, rho, probes):
        if history.shape[:2] != anchor_design.shape[:2] or rho.shape != (*history.shape[:2], 1):
            raise ValueError('History, design and rho axes do not match')
        ids = _ids(probes, history.device)
        if len(ids) < 1 or len(torch.unique(ids)) != len(ids):
            raise ValueError('Probe IDs must be nonempty and unique')
        return cls(rho, ids, history, anchor_design,
                   history.index_select(1, ids).contiguous(),
                   anchor_design.index_select(1, ids).contiguous())

    @property
    def compressed(self):
        return self.code is not None

    @property
    def total_tokens(self):
        return self.rho.shape[1]

    def encode_joint(self, basis, *, tile_rows=8192, code_dtype=torch.bfloat16,
                     validate_orthogonal=True):
        """Consume supplied basis; never infer a model-safe rank from dimensions.

        Encoding uses the exact stored basis bytes. A BF16 basis is accepted;
        it is then a quantized orthogonal-basis representation, accounted for
        in Z_tilde-Z. No full H*N*159 concatenation is materialized.
        """
        if self.compressed:
            raise ValueError('History is already jointly encoded')
        heads, _, channels = self.history.shape
        width = channels + self.anchor_design.shape[-1]
        if basis.ndim == 2:
            basis = basis.unsqueeze(0).expand(heads, -1, -1)
        if basis.ndim != 3 or basis.shape[:2] != (heads, width) or not 0 < basis.shape[2] <= width:
            raise ValueError('basis must be [H, D+feature_width, rank]')
        if basis.device != self.rho.device or tile_rows < 1:
            raise ValueError('Basis device and positive tile_rows must match history')
        work = _work_dtype(self.history)
        if validate_orthogonal:
            gram = basis.to(work).transpose(-2, -1) @ basis.to(work)
            tol = .04 if basis.dtype in (torch.bfloat16, torch.float16) else (1e-8 if basis.dtype == torch.float64 else 2e-4)
            eye = torch.eye(basis.shape[-1], device=basis.device, dtype=work)
            torch._assert_async(((gram-eye).abs().amax() < tol), 'Encoding columns are not orthonormal within storage precision')
        probe_sorted = self.probe_ids.sort().values
        code = torch.empty((heads, self.total_tokens-len(self.probe_ids), basis.shape[-1]), device=basis.device, dtype=code_dtype)
        for start in range(0, self.total_tokens, tile_rows):
            ids = torch.arange(start, min(start+tile_rows, self.total_tokens), device=basis.device)
            pos = torch.searchsorted(probe_sorted, ids)
            is_probe = (pos < len(probe_sorted)) & (probe_sorted[pos.clamp_max(len(probe_sorted)-1)] == ids)
            keep = ids[~is_probe]
            dest = keep - torch.searchsorted(probe_sorted, keep)
            values = torch.cat((self.history.index_select(1, keep).to(work), self.anchor_design.index_select(1, keep).to(work)), -1)
            code[:, dest] = (values @ basis.to(work)).to(code_dtype)
        return LowHistory(self.rho, self.probe_ids, None, None, self.probe_history,
                          self.probe_design, code, basis.contiguous())

    def _code_rows(self, ids):
        sorted_probes = self.probe_ids.sort().values
        pos = torch.searchsorted(sorted_probes, ids)
        probe = (pos < len(sorted_probes)) & (sorted_probes[pos.clamp_max(len(sorted_probes)-1)] == ids)
        local = torch.nonzero(~probe).flatten()
        return local, ids[local]-pos[local]

    def fields(self, token_ids):
        """Decode only requested rows, with exact probe sidecars restored."""
        ids = _ids(token_ids, self.rho.device)
        if not self.compressed:
            return self.history.index_select(1, ids), self.anchor_design.index_select(1, ids)
        work = _work_dtype(self.probe_history)
        width = self.basis.shape[1]
        z = torch.empty((self.rho.shape[0], len(ids), width), device=ids.device, dtype=work)
        local, code_rows = self._code_rows(ids)
        z[:, local] = self.code.index_select(1, code_rows).to(work) @ self.basis.to(work).transpose(-2, -1)
        local_p, probe_p = _probe_locations(ids, self.probe_ids)
        z[:, local_p] = torch.cat((self.probe_history.index_select(1, probe_p).to(work), self.probe_design.index_select(1, probe_p).to(work)), -1)
        dim = self.probe_history.shape[-1]
        return z[..., :dim], z[..., dim:]

    def dynamic_reader(self, gamma1, scale):
        """D G_t for the orthogonal code E=D^T; returned matrix is k-by-D."""
        if not self.compressed:
            raise ValueError('A joint code is required')
        work = gamma1.dtype
        g = torch.cat((torch.diag_embed(scale[:, 0].to(work)), -gamma1), dim=1)
        return self.basis.to(work).transpose(-2, -1) @ g

    def historical_action(self, token_ids, decoder, gamma1, scale):
        ids = _ids(token_ids, self.rho.device)
        work = gamma1.dtype
        out = torch.empty((self.rho.shape[0], len(ids), self.probe_history.shape[-1]), device=ids.device, dtype=work)
        local, code_rows = self._code_rows(ids)
        out[:, local] = self.code.index_select(1, code_rows).to(work) @ decoder
        local_p, probe_p = _probe_locations(ids, self.probe_ids)
        out[:, local_p] = self.probe_history.index_select(1, probe_p).to(work)*scale - self.probe_design.index_select(1, probe_p).to(work)@gamma1
        return out

    def storage_bytes(self):
        """Owned tensor payloads; avoids counting aliases/shared storage twice."""
        seen, total = set(), 0
        for value in (self.rho, self.probe_ids, self.history, self.anchor_design,
                      self.probe_history, self.probe_design, self.code, self.basis):
            if value is not None:
                key = (value.device, value.untyped_storage().data_ptr())
                if key not in seen:
                    seen.add(key)
                    total += value.untyped_storage().nbytes()
        return total


@dataclass
class PreparedReadout:
    coefficient: torch.Tensor           # unabsorbed frozen scale_r semantics
    truth: torch.Tensor                 # H,P,D native teacher
    probe_ids: torch.Tensor
    scale_r: float
    low: bool
    joint_decoder: Optional[torch.Tensor] = None

    @property
    def gamma(self):
        return self.coefficient*self.scale_r


class AttentionReadout:
    """Prepare P64 fit first, then read arbitrary nonprobe query tiles."""
    def __init__(self, pv, basis, probes, *, ridge_lambda=.1, scale_r=1.,
                 feature_rounding=True, precompose=False):
        if pv.ndim != 2 or basis.ndim != 2 or pv.shape[1] != basis.shape[0]:
            raise ValueError('Expected pv[D,F] and basis[F,R]')
        if ridge_lambda <= 0 or scale_r <= 0:
            raise ValueError('Positive ridge_lambda and scale_r are required')
        self.pv, self.basis = pv, basis
        self.probes = _ids(probes, pv.device)
        if not len(self.probes) or len(torch.unique(self.probes)) != len(self.probes):
            raise ValueError('Probe IDs must be nonempty and unique')
        self.ridge_lambda, self.scale_r = float(ridge_lambda), float(scale_r)
        self.feature_rounding, self.precompose = bool(feature_rounding), bool(precompose)
        self.composed = pv @ basis if precompose else None

    def features(self, selected):
        dtype = _work_dtype(selected)
        selected = selected.to(dtype)
        if self.precompose:
            mapped = selected @ self.composed.to(dtype)
        else:
            mapped = (selected @ self.pv.to(dtype)) @ self.basis.to(dtype)
        return mapped.to(torch.bfloat16).to(dtype) if self.feature_rounding else mapped

    def fit(self, selected_p, truth_p, *, cache=None, mean=None, scale=None):
        if selected_p.shape != truth_p.shape or selected_p.shape[1] != len(self.probes):
            raise ValueError('selected_p/truth_p must use the original probe production order')
        work = _work_dtype(selected_p)
        selected = selected_p.to(work)
        phi = self.features(selected)
        if cache is None:
            base, diff = selected, phi
        else:
            if mean is None or scale is None:
                raise ValueError('Low readout requires current value mean and scale')
            torch._assert_async((cache.probe_ids == self.probes).all(), 'Cache probe order differs')
            rho = cache.rho.index_select(1, self.probes).to(work)
            base = ((1-rho)*selected + rho*mean) + cache.probe_history.to(work)*scale
            diff = phi-cache.probe_design.to(work)
        xp = torch.cat((torch.ones_like(diff[..., :1]), diff), -1)
        target = (truth_p.to(work)-base)/self.scale_r
        xt = xp.transpose(-2, -1)
        eye = torch.eye(xp.shape[-1], device=xp.device, dtype=work)
        chol = torch.linalg.cholesky(xt@xp + len(self.probes)*self.ridge_lambda*eye)
        pinv = torch.cholesky_solve(xt, chol)
        coefficient = pinv @ target
        prepared = PreparedReadout(coefficient, truth_p, self.probes, self.scale_r, cache is not None)
        if cache is not None and cache.compressed:
            prepared.joint_decoder = cache.dynamic_reader(prepared.gamma[:, 1:], scale)
        return prepared

    def read(self, selected, token_ids, prepared, *, cache=None, mean=None,
             scale=None, output_dtype=None, joint_mode='direct', overwrite_probes=True):
        ids = _ids(token_ids, selected.device)
        if selected.ndim != 3 or selected.shape[1] != len(ids):
            raise ValueError('selected must be head-first and match token_ids')
        if prepared.low != (cache is not None):
            raise ValueError('Prepared fit and history kind disagree')
        work = _work_dtype(selected)
        values, phi = selected.to(work), self.features(selected)
        if cache is None:
            x = torch.cat((torch.ones_like(phi[..., :1]), phi), -1)
            estimate = values + (x@prepared.coefficient)*self.scale_r
        else:
            if mean is None or scale is None:
                raise ValueError('Low readout requires mean and scale')
            rho = cache.rho.index_select(1, ids).to(work)
            if cache.compressed and joint_mode == 'direct':
                gamma = prepared.gamma
                history = cache.historical_action(ids, prepared.joint_decoder, gamma[:, 1:], scale)
                current = ((1-rho)*values + rho*mean)
                estimate = (current + gamma[:, :1]) + phi@gamma[:, 1:] + history
            elif joint_mode in ('direct', 'source_order'):
                old_c, old_a = cache.fields(ids)
                base = ((1-rho)*values + rho*mean) + old_c.to(work)*scale
                diff = phi-old_a.to(work)
                x = torch.cat((torch.ones_like(diff[..., :1]), diff), -1)
                estimate = base + (x@prepared.coefficient)*self.scale_r
            else:
                raise ValueError('joint_mode must be direct or source_order')
        out = estimate.to(output_dtype or selected.dtype)
        if overwrite_probes:
            local, source = _probe_locations(ids, prepared.probe_ids)
            out.index_copy_(1, local, prepared.truth.index_select(1, source).to(out.dtype))
        return out


@dataclass
class FoldedHighProjection:
    weight: torch.Tensor                # H*D,O, row-vector convention
    bias: torch.Tensor                  # O
    original_weight: torch.Tensor
    original_bias: Optional[torch.Tensor]
    prepared: PreparedReadout

    def apply(self, selected, token_ids, *, output_dtype=None, overwrite_probes=True):
        """Bypasses feature/correction materialization and intermediate cast.

        Native probe rows use the original projection. The folded path is a
        separate real-equivalent option and deliberately has no bitwise claim.
        """
        ids = _ids(token_ids, selected.device)
        q = selected.shape[1]
        current = selected.transpose(0, 1).reshape(q, -1).to(self.weight.dtype)
        out = torch.addmm(self.bias, current, self.weight)
        if overwrite_probes:
            local, source = _probe_locations(ids, self.prepared.probe_ids)
            if len(source):
                native_input = self.prepared.truth.index_select(1, source).transpose(0, 1).reshape(len(source), self.original_weight.shape[0]).to(self.original_weight.dtype)
                native = (native_input@self.original_weight if self.original_bias is None
                          else torch.addmm(self.original_bias, native_input, self.original_weight))
                out.index_copy_(0, local, native)
        return out.to(output_dtype or selected.dtype)


def fold_high_output(reader, prepared, weight_in_out, *, bias=None, storage_dtype=None):
    """Fold high OUT-ABS into W_o under an unrounded-feature real graph.

    For source-rounded features, the exact dropped defect is
    (BF16(S B)-S B) Gamma_1 W_o plus the intermediate output cast.
    Callers must select this as a precision-changing optimization explicitly.
    Set storage_dtype=BF16 on the production BF16 path to retain tensor-core
    output projection; quantizing the dynamic weight/bias adds its own defect.
    None retains working precision for algebraic CPU checks.
    """
    if prepared.low:
        raise ValueError('High output folding only applies to OUT-ABS')
    heads, _, dim = prepared.truth.shape
    if weight_in_out.ndim != 2 or weight_in_out.shape[0] != heads*dim:
        raise ValueError('Output weight must have row-vector layout [H*D,O]')
    dtype = _work_dtype(weight_in_out)
    original = weight_in_out.to(dtype)
    weights = original.view(heads, dim, -1)
    projection = reader.pv.to(dtype)@reader.basis.to(dtype)
    gamma = prepared.gamma.to(dtype)
    effective = weights + projection[None]@(gamma[:, 1:]@weights)
    shift = (gamma[:, :1]@weights).sum(0).squeeze(0)
    if bias is not None:
        shift = shift+bias.to(dtype)
    storage_dtype = dtype if storage_dtype is None else storage_dtype
    return FoldedHighProjection(effective.reshape_as(original).to(storage_dtype),
                                shift.to(storage_dtype), weight_in_out, bias, prepared)
