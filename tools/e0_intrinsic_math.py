"""Full-hidden signed-value diagnostics on saved operands, never an FFN policy.

Imports do not initialize CUDA. All projections below are charged diagnostics.
"""
import math


def top_support(score, count):
    import torch
    if not 0 < count <= score.shape[-1]:
        raise ValueError("Invalid retained count")
    ids = torch.topk(score, count, dim=-1, sorted=False).indices
    return torch.zeros_like(score, dtype=torch.bool).scatter_(-1, ids, True)


def group_values(q, weight, gate, width=64):
    """[R,H] coefficients × [D,H] actual cast weights → [R,H/64,D]."""
    import torch
    r, h = q.shape
    if h % width or weight.shape[1] != h or gate.shape != (1, weight.shape[0]):
        raise ValueError("Wrong coefficient/value/gate layout")
    values = torch.bmm(q.reshape(r, h//width, width).permute(1, 0, 2),
                       weight.reshape(weight.shape[0], h//width, width).permute(1, 2, 0))
    return values.permute(1, 0, 2) * gate[:, None, :]


def coefficient_scores(q, weight, gate):
    norms = ((weight * gate.T)**2).sum(dim=0)
    return {"coefficient": q*q, "value_diagonal": q*q*norms}


def source_fixed_score(scores, held_source):
    import torch
    other = [v for source, v in scores.items() if source != held_source]
    if len(other) != 5:
        raise ValueError("Exactly five other sources required")
    # Equal source weight; large raw source energy cannot dominate this prior.
    return torch.stack([v.sum(dim=0)/v.sum() for v in other]).mean(dim=0)


def energy(x):
    return float((x*x).sum())


def projection_description(q, weight, gate, fixed_neuron, fixed_group, fractions, seed):
    import torch
    r, h = q.shape
    groups = h//64
    c = group_values(q, weight, gate)
    target = c.sum(dim=1)
    norm = energy(target)
    scores = coefficient_scores(q, weight, gate)
    diagonal, group_diagonal = scores["value_diagonal"], (c*c).sum(dim=2)
    if min(norm, energy(q), float(diagonal.sum())) <= 0:
        raise ValueError("Degenerate diagnostic target")
    flat = c.permute(1, 0, 2).reshape(groups, -1)
    gram = flat @ flat.T
    triangle = torch.triu_indices(groups, groups, 1, device=q.device)
    cross = gram[triangle[0], triangle[1]]
    group_sum = float(group_diagonal.sum())
    if abs(float(gram.sum())-norm) > 1e-10*max(1., norm):
        raise ValueError("Signed group Gram identity failed")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_neuron = torch.randperm(h, generator=generator).to(q.device)
    random_group = torch.randperm(groups, generator=generator).to(q.device)
    out = {"target_energy_excluding_bias": norm,
           "sum_neuron_diagonal_energy": float(diagonal.sum()),
           "sum_group_diagonal_energy": group_sum,
           "net_over_neuron_diagonal": norm/float(diagonal.sum()),
           "net_over_group_diagonal": norm/group_sum,
           "group_cross_energy_over_net": (norm-group_sum)/norm,
           "negative_aggregate_group_pair_fraction": float((cross<0).double().mean()),
           "coefficient_nonzero_fraction": float((q!=0).double().mean()),
           "neuron": {}, "group": {}}
    for fraction in fractions:
        count, group_count = int(round(h*fraction)), int(round(groups*fraction))
        if count != 64*group_count:
            raise ValueError("Neuron/group budgets must match exactly")
        key = str(fraction)
        ns = {**scores, "source_fixed": fixed_neuron.expand(r, -1)}
        gs = {"coefficient": scores["coefficient"].reshape(r, groups, 64).sum(-1),
              "value_diagonal": diagonal.reshape(r, groups, 64).sum(-1),
              "actual_group_energy": group_diagonal,
              "source_fixed": fixed_group.expand(r, -1)}
        out["neuron"][key], out["group"][key] = {}, {}
        for name in ["coefficient", "value_diagonal", "source_fixed", "random_fixed"]:
            if name == "random_fixed":
                mask = torch.zeros((r,h), dtype=torch.bool, device=q.device)
                mask[:,random_neuron[:count]] = True
            else:
                mask = top_support(ns[name], count)
            approximation = (q*mask) @ weight.T * gate
            sse = energy(approximation-target)
            out["neuron"][key][name] = {
                "retained_channels": count, "relative_SSE": sse/norm,
                "retained_diagonal_fraction": float(diagonal[mask].sum()/diagonal.sum()),
                "union_column_fraction_in_32_rows": float(mask.any(dim=0).double().mean()),
                "nonempty_K64_fraction_in_32_rows": float(mask.reshape(r,groups,64).any(-1).any(0).double().mean())}
        for name in [*gs, "random_fixed"]:
            if name == "random_fixed":
                mask = torch.zeros((r,groups), dtype=torch.bool, device=q.device)
                mask[:,random_group[:group_count]] = True
            else:
                mask = top_support(gs[name], group_count)
            error = (c * (~mask)[:, :, None]).sum(dim=1)
            sse = energy(error)
            discarded_independent = float(group_diagonal[~mask].sum())
            out["group"][key][name] = {
                "retained_channels": 64*group_count, "relative_SSE": sse/norm,
                "discarded_group_diagonal_over_net": discarded_independent/norm,
                "discarded_signed_cross_over_net": (sse-discarded_independent)/norm,
                "union_group_fraction_in_32_rows": float(mask.any(dim=0).double().mean())}
    return out


def support_tile_statistics(activation, fractions, row_tiles=(1,16,64,128)):
    """Full contiguous-token coverage, not a sampled-row occupancy estimate.

    This observes top-|GELU| supports only; the passed activation is unchanged.
    Returned occupancy is an operation-count bound, not measured sparse time.
    """
    import torch
    m, h = activation.shape
    if h % 64 or 128 % max(row_tiles):
        raise ValueError("Unexpected layout")
    groups = h//64
    out = {str(f): {"kept_nonzero_elements":0, "hot_columns":torch.zeros(h,dtype=torch.bool,device=activation.device),
                   "row_tiles":{str(t):{"hot_group_tiles":0,"total_group_tiles":0,"retained_row_group_work":0}
                                for t in row_tiles}} for f in fractions}
    for start in range(0, m, 128):
        x = activation[start:start+128]
        for fraction in fractions:
            item = out[str(fraction)]
            mask = top_support(x.abs(), int(round(h*fraction))) & (x!=0)
            item["kept_nonzero_elements"] += int(mask.sum())
            item["hot_columns"] |= mask.any(dim=0)
            row_groups = mask.reshape(len(x),groups,64).any(-1)
            for tile in row_tiles:
                counts = item["row_tiles"][str(tile)]
                number = math.ceil(len(x)/tile)
                padded = torch.zeros((number*tile,groups), dtype=torch.bool, device=x.device)
                padded[:len(x)] = row_groups
                hot = padded.reshape(number,tile,groups).any(dim=1).sum(dim=1)
                real_rows = (len(x)-tile*torch.arange(number,device=x.device)).clamp(max=tile)
                counts["hot_group_tiles"] += int(hot.sum())
                counts["total_group_tiles"] += number*groups
                counts["retained_row_group_work"] += int((real_rows*hot).sum())
    for item in out.values():
        item["retained_nonzero_element_fraction"] = item.pop("kept_nonzero_elements")/(m*h)
        item["union_column_fraction_full_M"] = float(item.pop("hot_columns").double().mean())
        for counts in item["row_tiles"].values():
            counts["hot_group_tile_fraction"] = counts["hot_group_tiles"]/counts["total_group_tiles"]
            counts["retained_dense_tile_work_fraction"] = counts.pop("retained_row_group_work")/(m*groups)
    return {"M":m,"H":h,"K_group":64,"fractions":out}


def iid_nonempty_group_probability(hidden, retained, group_width, rows):
    """Descriptive independence comparator with fixed-count sampling per row."""
    if retained > hidden-group_width:
        return 1.
    log_cold = sum(math.log1p(-group_width/(hidden-j)) for j in range(retained))
    return -math.expm1(rows*log_cold)
