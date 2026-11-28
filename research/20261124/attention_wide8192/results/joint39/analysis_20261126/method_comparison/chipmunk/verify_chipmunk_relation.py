"""CPU algebra checks; synthetic arrays are not Wan model measurements."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
rng = np.random.default_rng(20261126)
n, d, k, group = 73, 11, 17, 8
logits_a = rng.normal(size=(n, n))
logits_t = logits_a + 0.2 * rng.normal(size=(n, n))
v_a = rng.normal(size=(n, d))
v_t = rng.normal(size=(n, d))

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

p_a = softmax(logits_a)
mask = np.zeros((n, n), dtype=bool)
for begin in range(0, n, group):
    end = min(begin + group, n)
    idx = np.argsort(p_a[begin:end].sum(axis=0))[-k:]
    mask[begin:end, idx] = True

def responses(logits, value):
    p = softmax(logits)
    selected = softmax(np.where(mask, logits, -np.inf)) @ value
    full = p @ value
    rho = 1 - (p * mask).sum(axis=-1, keepdims=True)
    mean = value.mean(axis=0, keepdims=True)
    std = value.std(axis=0, keepdims=True)
    return full, selected, rho, mean, std

a_a, s_a, rho, mu_a, std_a = responses(logits_a, v_a)
z = (a_a - (1-rho)*s_a - rho*mu_a) / std_a

def evaluate(logits, value):
    target, selected, rho_t, mean, std = responses(logits, value)
    chip = a_a + selected - s_a
    centered = (1-rho)*selected + rho*mean + z*std
    difference = -rho*(selected-s_a) + rho*(mean-mu_a) + z*(std-std_a)
    return {
        "exact_base_difference_max_abs": float(np.max(np.abs(centered-chip-difference))),
        "chip_relative_l2": float(np.linalg.norm(target-chip)/np.linalg.norm(target)),
        "centered_relative_l2": float(np.linalg.norm(target-centered)/np.linalg.norm(target)),
        "rho_change_max_abs": float(np.max(np.abs(rho-rho_t))),
    }

out = {
    "scope": "Synthetic float64 conditional algebra, not empirical quality or speed",
    "seed": 20261126,
    "shape": {"tokens": n, "value_dim": d, "selected_per_group": k, "query_group": group},
    "general_change": evaluate(logits_t, v_t),
    "pure_value_translation": evaluate(logits_a, v_a+rng.normal(size=(1,d))),
    "positive_channel_scale_and_translation": evaluate(logits_a, v_a*rng.uniform(0.4,2.3,size=(1,d))+rng.normal(size=(1,d))),
}

N, D, M, K, L = 31668, 5120, 13824, 2560, 40
out["wan_shape_bf16_static_ffn_cache"] = {
    "N": N,"D": D,"M": M,"K": K,"layers_retained":L,
    "ours_three_main_slots_bytes": 2*N*(2*D+K)*L,
    "chipmunk_two_main_slots_bytes": 2*N*(M+D)*L,
    "chipmunk_blockmean_bytes": 2*((N+127)//128)*M*L,
    "ours_three_main_slots_gib": 2*N*(2*D+K)*L/2**30,
    "chipmunk_two_main_slots_gib": 2*N*(M+D)*L/2**30,
    "chipmunk_blockmean_gib": 2*((N+127)//128)*M*L/2**30,
    "scope": "All 40 layers' indicated BF16 logical slots resident, one active expert; excludes weights, packing, indices, workspaces, offloading. Chipmunk video main tables disable MLP, so this is only a shape-matched adaptation budget.",
}
for case in ("general_change", "pure_value_translation", "positive_channel_scale_and_translation"):
    assert out[case]["exact_base_difference_max_abs"] < 1e-12
assert out["pure_value_translation"]["chip_relative_l2"] < 1e-12
assert out["pure_value_translation"]["centered_relative_l2"] < 1e-12
assert out["positive_channel_scale_and_translation"]["centered_relative_l2"] < 1e-12
assert out["positive_channel_scale_and_translation"]["chip_relative_l2"] > 1e-3
(ROOT/"CHIPMUNK_RELATION_VERIFIED.json").write_text(json.dumps(out, indent=2)+"\n")
print(json.dumps(out, indent=2))
