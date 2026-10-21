"""Small, source-fitted C-profile banks and a folded native BF16 routing probe."""
import math

import numpy as np
import torch
import torch.nn.functional as F


class DegenerateBanks(ValueError):
    pass


def normalized_profiles(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != 216 or not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError('Finite nonnegative216-group C profiles required')
    energy = scores.sum(axis=1, keepdims=True)
    if np.any(energy <= 0):
        raise ValueError('Every profile must have positive total C-score')
    return scores / energy


def squared_distances(x, centers):
    return np.maximum((x * x).sum(1)[:, None] + (centers * centers).sum(1)[None] - 2 * x @ centers.T, 0.)


def fit_banks(profiles, keep, seed, *, min_bank=128):
    x = np.asarray(profiles, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 216 or not 0 < keep < 216 or len(x) < 4 * min_bank:
        raise ValueError('Invalid training bank geometry/budget')
    if not np.isfinite(x).all() or not np.allclose(x.sum(1), 1., atol=1e-12, rtol=0):
        raise ValueError('Normalized training profiles required')
    rng = np.random.default_rng(seed)
    centers = [x[int(rng.integers(len(x)))].copy()]
    for _ in range(3):
        distance = squared_distances(x, np.asarray(centers)).min(1)
        if distance.sum() <= 0:
            raise DegenerateBanks('All remaining profiles coincide with initialized centers')
        centers.append(x[int(rng.choice(len(x), p=distance / distance.sum()))].copy())
    centers = np.asarray(centers)
    for _ in range(20):
        assignments = squared_distances(x, centers).argmin(1)
        counts = np.bincount(assignments, minlength=4)
        if np.any(counts == 0):
            raise DegenerateBanks('Empty bank; do not reseed or replace')
        centers = np.stack([x[assignments == i].mean(0) for i in range(4)])
    assignments = squared_distances(x, centers).argmin(1)
    counts = np.bincount(assignments, minlength=4)
    if np.any(counts < min_bank):
        raise DegenerateBanks(f'Training bank below fixed minimum: {counts.tolist()}')
    scores = np.stack([x[assignments == i].mean(0) for i in range(4)])
    masks = np.sort(np.argsort(-scores, axis=1, kind='stable')[:, :keep], axis=1)
    static = np.sort(np.argsort(-x.mean(0), kind='stable')[:keep])
    return {'centers': centers, 'assignments': assignments, 'counts': counts,
            'masks': masks, 'static': static}


def bank_energy(profiles, masks):
    x, masks = np.asarray(profiles, np.float64), np.asarray(masks)
    if masks.ndim != 2 or masks.shape[0] != 4 or np.any(masks < 0) or np.any(masks >= 216):
        raise ValueError('Four valid whole-group masks required')
    if any(len(np.unique(row)) != len(row) for row in masks):
        raise ValueError('Duplicate original group in a bank')
    return np.stack([x[:, row].sum(1) for row in masks], axis=1)


def describe_proxy(profiles, banks, predicted):
    energy = bank_energy(profiles, banks['masks'])
    predicted = np.asarray(predicted)
    if (predicted.dtype.kind not in 'iu' or predicted.shape != (len(profiles),)
            or np.any(predicted < 0) or np.any(predicted >= 4)):
        raise ValueError('One valid bank prediction per row required')
    static = float(profiles[:, banks['static']].sum(1).mean())
    oracle = float(energy.max(1).mean())
    learned = float(energy[np.arange(len(predicted)), predicted].mean())
    gap = oracle - static
    captured = (learned - static) / gap if gap > 0 else None
    return {'static_retained_C_fraction': static, 'oracle_retained_C_fraction': oracle,
            'native_router_retained_C_fraction': learned, 'oracle_minus_static': gap,
            'native_fraction_of_oracle_gain': captured,
            'oracle_label_accuracy': float((predicted == energy.argmax(1)).mean()),
            'gate_pass': bool(gap >= .02 and captured is not None and captured >= .5)}


def fit_router(x, labels, seed):
    if x.shape != (3072, 5120) or labels.shape != (3072,):
        raise RuntimeError('Fixed12-source256-row router training geometry required')
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(5120, 32), torch.nn.ReLU(), torch.nn.Linear(32, 4)).to(device=x.device, dtype=torch.float32)
    mean = x.float().mean(0)
    scale = x.float().std(0, unbiased=False).clamp_min(1e-4)
    standardized = (x.float() - mean) / scale
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    generator = torch.Generator(device=x.device).manual_seed(seed)
    history, updates = [], 0
    for epoch in range(20):
        order = torch.randperm(len(x), generator=generator, device=x.device)
        total = 0.
        for ids in order.split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(standardized[ids]), labels[ids])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite router training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
            updates += 1
        history.append(total / len(x))
        print(f'RB1 epoch{epoch + 1}: CE={history[-1]:.8g}', flush=True)
    if updates != 240:
        raise RuntimeError('Fixed router budget changed')
    return model, mean, scale, {'epoch_training_CE': history, 'updates': updates}


def folded_parameters(model, mean, scale):
    weight = model[0].weight.detach() / scale[None]
    bias = model[0].bias.detach() - F.linear(mean, weight, None)
    parameters = {'w1': weight.bfloat16().contiguous(), 'b1': bias.bfloat16().contiguous(),
                  'w2': model[2].weight.detach().bfloat16().contiguous(),
                  'b2': model[2].bias.detach().bfloat16().contiguous()}
    if any(not torch.isfinite(v).all() for v in parameters.values()):
        raise RuntimeError('Nonfinite native folded router parameter')
    return parameters


@torch.no_grad()
def native_predict_rows(case, parameters):
    x, scope = case['V1'], case['scope']
    shape, stride = tuple(scope['native_V1_shape']), tuple(scope['native_V1_stride'])
    if (x.dtype != torch.bfloat16 or shape[-1] != 5120 or len(shape) not in [2, 3]
            or (len(shape) == 3 and shape[0] != 1) or math.prod(shape[:-1]) != scope['full_rows']
            or len(stride) != len(shape) or any(v <= 0 for v in stride)):
        raise RuntimeError('Original postQ1 native geometry required')
    ids = torch.tensor(scope['selected_rows'], device=x.device, dtype=torch.long)
    if len(ids) != len(x) or len(ids.unique()) != len(ids) or bool((ids < 0).any()) or bool((ids >= scope['full_rows']).any()):
        raise RuntimeError('Invalid original sampled row positions')
    full = torch.empty_strided(shape, stride, device=x.device, dtype=x.dtype).zero_()
    rows = full if full.ndim == 2 else full[0]
    rows.index_copy_(0, ids, x)
    logits = F.linear(F.relu(F.linear(full, parameters['w1'], parameters['b1'])), parameters['w2'], parameters['b2'])
    selected = logits if logits.ndim == 2 else logits[0]
    sampled = selected.index_select(0, ids).float()
    if not torch.isfinite(sampled).all():
        raise RuntimeError('Nonfinite native routing logits')
    return sampled.argmax(1).cpu().numpy()
