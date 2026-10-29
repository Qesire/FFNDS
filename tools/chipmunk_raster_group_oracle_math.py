"""Post-hoc same-budget group oracle; no online or sparse-performance claim."""
import hashlib
import torch
import torch.nn.functional as F

def selection_debits(m, k, kind, size=128):
    if m <= 0 or k <= 0 or size <= 0 or kind not in ("global", "raster", "random"):
        raise ValueError("invalid selection budget")
    n = 1 if kind == "global" else (m + size - 1) // size
    return {"selection":1,"score_sort_groups":n,"indices_D2H_logical_bytes":n*k*8,"group_rows_H2D_logical_bytes":m*8,"group_rows_host_hash_logical_bytes":m*8}

def groups(m, kind, size=128, seed=101):
    if m <= 0 or size <= 0 or kind not in ('global', 'raster', 'random'):
        raise ValueError('invalid grouping')
    if kind == 'global':
        order = torch.arange(m, dtype=torch.int64)
        size = m
    elif kind == 'random':
        order = torch.randperm(m, generator=torch.Generator(device='cpu').manual_seed(seed))
    else:
        order = torch.arange(m, dtype=torch.int64)
    return list(order.split(size))

def selected_delta(old_h, new_h, k, kind, size=128, seed=101):
    if old_h.ndim != 2 or old_h.shape != new_h.shape or old_h.dtype != torch.bfloat16 or new_h.dtype != torch.bfloat16 or not 0 < k <= old_h.shape[1]:
        raise ValueError('bad h or K')
    m, width = old_h.shape
    delta = new_h.float() - old_h.float()
    masked = torch.zeros_like(old_h)
    index_hash = hashlib.sha256(); group_hash = hashlib.sha256()
    selected_energy = torch.zeros((), device=old_h.device, dtype=torch.float64)
    for cpu_rows in groups(m, kind, size, seed):
        rows = cpu_rows.to(old_h.device)
        values = delta.index_select(0, rows)
        score = values.square().mean(dim=0)
        cols = torch.sort(torch.argsort(score, descending=True, stable=True)[:k]).values
        masked[rows[:, None], cols[None, :]] = values[:, cols].to(torch.bfloat16)
        selected_energy += values[:, cols].double().square().sum()
        group_hash.update(cpu_rows.numpy().tobytes())
        index_hash.update(cols.to(device='cpu', dtype=torch.int64).numpy().tobytes())
    return masked, {'group_count':len(groups(m,kind,size,seed)), 'selected_elements':m*k,
                    'row_groups_sha256':group_hash.hexdigest(), 'indices_sha256':index_hash.hexdigest(),
                    'selected_energy':float(selected_energy)}

def relative(x, ref):
    diff = x.float() - ref.float()
    if not bool(torch.isfinite(diff).all()): raise ValueError('nonfinite result')
    norm = torch.linalg.vector_norm(ref.float())
    if float(norm) == 0: raise ValueError('zero reference')
    return float(torch.linalg.vector_norm(diff) / norm)

def project(old_y, delta, w2, gate):
    """BF16 delta, BF16 dense W2, BF16 cache add, BF16 gate multiplication."""
    y = (old_y.float() + F.linear(delta, w2).float()).to(torch.bfloat16)
    return (y * gate).to(torch.bfloat16)
