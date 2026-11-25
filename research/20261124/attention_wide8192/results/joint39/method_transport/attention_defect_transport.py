"""Torch implementation of the channel-standardized attention defect transport."""
from dataclasses import dataclass
import math

import torch


@dataclass
class Anchor:
    defect: torch.Tensor
    labels: torch.Tensor
    support: object
    selected: torch.Tensor
    raw_panel: torch.Tensor | None
    normalized_panel: torch.Tensor | None


def group_catalog(k, v, direction, block, labels=None):
    """Use [H,N,D] keys/values. Include only valid keys in a group's count."""
    heads, keys, channels = k.shape
    direction = direction / direction.norm()
    assignments, counts, key_means, value_means = [], [], [], []
    batch = 16 * block
    offsets = torch.arange(block, device=k.device)
    for start in range(0, keys, batch):
        stop = min(start + batch, keys)
        length = stop - start
        blocks = (length + block - 1) // block
        padding = blocks * block - length
        valid = (torch.arange(blocks, device=k.device)[:, None] * block + offsets) < length
        key = torch.nn.functional.pad(k[:, start:stop].float(), (0, 0, 0, padding))
        key = key.reshape(heads, blocks, block, channels)
        if labels is None:
            moment = key.sum(2) / valid.sum(1)[None, :, None]
            positive = (((key - moment[:, :, None]) @ direction) >= 0)
            assignments.append(positive.flatten(1)[:, :length])
        else:
            positive = torch.nn.functional.pad(labels[:, start:stop].bool(), (0, padding))
            positive = positive.reshape(heads, blocks, block)
        membership = (torch.stack((~positive, positive), dim=2) & valid[None, :, None]).float()
        count = membership.sum(-1)
        denominator = count.clamp_min(1)[..., None]
        value = torch.nn.functional.pad(v[:, start:stop].float(), (0, 0, 0, padding))
        value = value.reshape(heads, blocks, block, v.shape[-1])
        counts.append(count)
        key_means.append((membership @ key) / denominator)
        value_means.append((membership @ value) / denominator)
    labels = torch.cat(assignments, dim=1).to(torch.uint8) if labels is None else labels
    return (labels, torch.cat(counts, dim=1), torch.cat(key_means, dim=1),
            torch.cat(value_means, dim=1))


def channel_scale(v):
    variance = v.float().var(dim=1, correction=0)
    positive = variance > 0
    scale = torch.where(positive, variance.sqrt().clamp_min(1e-12), 0)
    inverse = torch.where(positive, scale.reciprocal(), 0)
    return scale, inverse


class CurrentBase:
    def __init__(self, q, k, v, selected, catalog, reference, block, lse):
        self.block = block
        self.q, self.selected, self.lse = q, reference, lse
        num_blocks = (k.shape[1] + block - 1) // block
        assert selected.shape[:2] == (q.shape[0], (q.shape[1] + block - 1) // block)
        self.mask = torch.zeros((*selected.shape[:2], num_blocks), dtype=torch.bool, device=q.device)
        self.mask.scatter_(2, selected.long(), True)
        self.log_count = catalog[1].flatten(1).log()
        self.key_mean = catalog[2].flatten(1, 2).transpose(1, 2).contiguous()
        self.value_mean = catalog[3].flatten(1, 2).contiguous()
        self.scale, self.inverse = channel_scale(v)
        self.factor = 1.0 / math.sqrt(q.shape[-1])

    def evaluate(self, rows):
        query = self.q.index_select(1, rows).float()
        selected = self.selected.index_select(1, rows).float()
        lse = self.lse.index_select(1, rows).float()
        logits = (query @ self.key_mean).mul_(self.factor).add_(self.log_count[:, None])
        excluded = self.mask.index_select(1, rows // self.block).repeat_interleave(2, dim=-1)
        logits.masked_fill_(excluded, -torch.inf)
        shift = torch.maximum(lse, logits.amax(dim=-1))
        coarse = logits.sub_(shift[..., None]).exp_()
        exact = (lse - shift).exp()
        denominator = exact + coarse.sum(-1)
        grouped = (coarse @ self.value_mean + exact[..., None] * selected) / denominator[..., None]
        return query, selected, grouped, exact / denominator

    def features(self, query, selected, grouped, mass, pk, pv):
        response = ((grouped - selected) * self.inverse[:, None]) @ pv
        return torch.cat((torch.ones_like(mass[..., None]), (query @ pk) * self.factor,
                          response, mass[..., None]), dim=-1)


def fit(features, label):
    gram = features.transpose(1, 2) @ features
    gram.diagonal(dim1=-2, dim2=-1).add_(features.shape[1] * 0.1)
    return torch.linalg.solve_ex(gram, features.transpose(1, 2) @ label).result


def energy(a, b):
    return float((a.float() - b).double().square().sum())


class TransportAttention:
    def __init__(self, online, block=128, query_chunk=512):
        self.online = online
        self.block = block
        self.query_chunk = query_chunk
        self.pk, self.pv = (x[:, :15].contiguous() for x in (online.pk, online.pv))
        self.probes = online.probes

    def current(self, q, k, v, support, selected, labels):
        catalog = group_catalog(k.transpose(0, 1), v.transpose(0, 1), self.pk[:, 0], self.block, labels)
        b, lse, _ = self.online.selected(q, k, v, support, False)
        base = CurrentBase(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), selected,
                           catalog, b, self.block, lse)
        return base, catalog[0]

    def anchor(self, q, k, v, native, audit_rows=None):
        support, selected = self.online.route(q, k)
        base, labels = self.current(q, k, v, support, selected, None)
        truth = native.reshape_as(q).transpose(0, 1)
        defect = torch.empty_like(truth, memory_format=torch.contiguous_format)
        for rows in torch.arange(q.shape[0], device=q.device).split(self.query_chunk):
            grouped = base.evaluate(rows)[2]
            value = (truth.index_select(1, rows).float() - grouped) * base.inverse[:, None]
            defect.index_copy_(1, rows, value.to(q.dtype))
        raw = None if audit_rows is None else truth.index_select(1, audit_rows).float() - base.evaluate(audit_rows)[2]
        normalized = None if raw is None else raw * base.inverse[:, None]
        return Anchor(defect, labels, support, selected, raw, normalized)

    def recover(self, q, k, v, state, native=None, panel=None, panel_probes=None, panel_held=None):
        base, _ = self.current(q, k, v, state.support, state.selected, state.labels)
        probe = self.probes
        if not bool(base.inverse.any()):
            output = v[:1].expand(q.shape).contiguous().reshape(q.shape[0], -1)
            return output, None, 0
        args = [x.transpose(0, 1).unsqueeze(0) for x in (q.index_select(0, probe), k, v)]
        short = torch.nn.functional.scaled_dot_product_attention(*args, dropout_p=0, is_causal=False,
                                                                 scale=base.factor)[0]
        query, b, g, mass = base.evaluate(probe)
        xp = base.features(query, b, g, mass, self.pk, self.pv)
        previous_p = state.defect.index_select(1, probe).float() * (base.inverse > 0)[:, None]
        teacher = short if native is None else native.reshape_as(q).transpose(0, 1).index_select(1, probe)
        innovation_p = (teacher.float() - g) * base.inverse[:, None] - previous_p
        coefficients = fit(xp, innovation_p)
        if native is not None:
            detail = self.audit(base, state, native.reshape_as(q).transpose(0, 1), coefficients,
                                xp, g, teacher, short, panel, panel_probes, panel_held)
            return native, detail, len(probe)
        predicted = torch.empty_like(q)
        for rows in torch.arange(q.shape[0], device=q.device).split(self.query_chunk):
            query, b, g, mass = base.evaluate(rows)
            x = base.features(query, b, g, mass, self.pk, self.pv)
            normalized = state.defect.index_select(1, rows).float() + x @ coefficients
            y = g + normalized * base.scale[:, None]
            predicted.index_copy_(0, rows, y.transpose(0, 1).to(q.dtype))
        predicted.index_copy_(0, probe, teacher.transpose(0, 1))
        return predicted.reshape(q.shape[0], -1), None, len(probe)

    def audit(self, base, state, truth, coefficients, xp, gp, teacher, short, panel, p, h):
        query, b, g, mass = base.evaluate(panel)
        x = base.features(query, b, g, mass, self.pk, self.pv)
        target = truth.index_select(1, panel)
        cache = state.defect.index_select(1, panel).float()
        mask = (base.inverse > 0)[:, None]
        raw, transported = state.raw_panel * mask, state.normalized_panel * base.scale[:, None]
        common = {"native": target,
                  "transport_bf16": (g + (cache + x @ coefficients) * base.scale[:, None]).to(truth.dtype)}
        for name, past in (("raw", raw), ("transport", transported)):
            beta = fit(xp, (teacher.float() - gp - past.index_select(1, p)) * base.inverse[:, None])
            common[name] = (g + past + (x @ beta) * base.scale[:, None]).to(truth.dtype)
        for pred in common.values():
            pred.index_copy_(1, p, teacher)
        held = lambda value: value.index_select(1, h)
        report = {
            "short_teacher_relative_l2": math.sqrt(energy(short, teacher) / max(energy(teacher, 0), 1e-30)),
            "base_energy": {"raw": energy(held(g + raw), held(target)),
                            "transport": energy(held(g + transported), held(target))},
            "fit_energy": {name: energy(held(pred), held(target)) for name, pred in common.items() if name != "native"},
        }
        return report, common
