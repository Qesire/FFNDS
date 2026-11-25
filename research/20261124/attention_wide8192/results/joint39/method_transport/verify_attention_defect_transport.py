"""Compare the Torch transport with the pinned NumPy oracle on the same inputs."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from attention_defect_transport import TransportAttention, group_catalog

sys.path.insert(0, str(Path(__file__).with_name("transport_reference")))
import reference_attention as oracle


class Selected:
    def __init__(self, pk, pv, rows, support, config):
        self.pk, self.pv, self.probes = pk, pv, rows
        self.support, self.config = support, config

    def route(self, q, k):
        support = torch.as_tensor(self.support, device=q.device)
        return support, support.nonzero()[:, 2].reshape(*support.shape[:2], -1)

    def selected(self, q, k, v, support, finite):
        arrays = [z.transpose(0, 1).float().cpu().numpy() for z in (q, k, v)]
        flags = np.zeros(arrays[1].shape[:2], dtype=np.uint8)
        base = oracle.grouped_attention(*arrays, self.support, flags, self.config)
        return (torch.as_tensor(base.selected, device=q.device, dtype=q.dtype),
                torch.as_tensor(base.selected_lse, device=q.device, dtype=torch.float32), None)


@torch.inference_mode()
def verify(device, assets, dtype):
    config = oracle.Config(key_block=64, query_block=64)
    rng = np.random.default_rng(250911)
    heads, rows, channels = 2, 190, 128
    pk, pv = (np.load(assets / name, allow_pickle=False) for name in ("P_k.npy", "P_v.npy"))
    support = np.ones((heads, 3, 3), dtype=bool)
    support[:, np.arange(3), np.arange(3)] = False
    p = rng.choice(rows, 64, replace=False)
    tensor = lambda x: torch.as_tensor(x, device=device, dtype=dtype).transpose(0, 1).contiguous()
    array = lambda t: t.transpose(0, 1).float().cpu().numpy()
    layer = Selected(torch.as_tensor(pk, dtype=torch.float32, device=device),
                     torch.as_tensor(pv, dtype=torch.float32, device=device),
                     torch.as_tensor(p, device=device), support, config)
    method = TransportAttention(layer, block=64, query_chunk=47)
    q, k, v = (rng.normal(size=(heads, rows, channels)) * 0.4 for _ in range(3))
    k[0, :64] = 0.25
    v[0, :, 0] = 0.125
    first = tuple(map(tensor, (q, k, v)))
    anchor_values = oracle.full_attention(*map(array, first))[0]
    state = method.anchor(*first, tensor(anchor_values))
    ref_state, _, _ = oracle.build_anchor(*map(array, first), support, pk, pv, config,
                                          native_output=array(tensor(anchor_values)))
    assert np.array_equal(state.labels.cpu(), ref_state.labels)
    assert torch.count_nonzero(state.defect[0, :, 0]) == 0
    catalog = group_catalog(first[1].transpose(0, 1), first[2].transpose(0, 1), layer.pk[:, 0], 64, state.labels)
    assert torch.equal(catalog[1].sum(-1).cpu(), torch.tensor([[64, 64, 62], [64, 64, 62]]))
    assert catalog[1][0, 0, 0] == 0
    first_hit = tuple(map(tensor, (q + rng.normal(size=q.shape) * 0.03,
                                  k + rng.normal(size=k.shape) * 0.03, v * 1.5 + 0.25)))
    expected = oracle.recover_hit(*map(array, first_hit), pk, pv, p, state=ref_state, config=config)
    out, extra, probe_count = method.recover(*first_hit, state)
    assert extra is None and probe_count == 64
    output = array(out.reshape(rows, heads, channels))
    assert np.isfinite(output).all()
    assert np.max(np.abs(output[0, :, 0] - array(first_hit[2])[0, 0, 0])) < 1e-6
    limit = 2e-6 if dtype == torch.float32 else 0.008
    max_error = float(np.max(np.abs(output - expected.output)))
    assert max_error < limit, (device, dtype, max_error)
    qt, kt, vt = first_hit
    probe = layer.probes
    args = [x.transpose(0, 1).unsqueeze(0) for x in (qt.index_select(0, probe), kt, vt)]
    teacher = torch.nn.functional.scaled_dot_product_attention(*args, dropout_p=0, is_causal=False)[0]
    assert torch.equal(out.reshape(rows, heads, channels).index_select(0, probe), teacher.transpose(0, 1))
    const = torch.ones_like(vt) * 0.375
    constant, _, probes = method.recover(qt, kt, const, state)
    assert probes == 0 and torch.equal(constant, const.flatten(1))
    return dict(dtype=str(dtype), device=device, max_abs_error=max_error, tolerance=limit,
                probe_overwrite_exact=True, empty_group=True, partial_key_block_count=62,
                all_constant_skips_teacher=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    result = [verify(args.device, args.assets, dtype) for dtype in (torch.float32, torch.bfloat16)]
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result, indent=2))
