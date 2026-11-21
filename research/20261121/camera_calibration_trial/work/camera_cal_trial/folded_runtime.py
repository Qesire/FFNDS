"""Compile per-site CAL3 secant means and replace two existing fast512 weights."""
import hashlib
import json
import os
from pathlib import Path
from time import monotonic

import numpy as np
import torch
import torch.nn.functional as F


CAL = ("vbi_0032", "vbi_0304", "vbi_0456")
MANIFEST_SHA = "de1dd6b00171a587a5b05bb05c47496ceea273aa08ba8cf12e14d538a605169b"
FIELDS = ("A0", "w2k")


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy()).hexdigest()


def verified(ref):
    path = Path(ref["path"])
    assert file_sha(path) == ref["sha256"], path
    return path


def save_json(path, value):
    tmp = path.with_suffix(".pending")
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False))
    os.replace(tmp, path)


def joint_operators(x, r, ids, coords):
    """Same normalization as the real layer-19 point+space fit."""
    edges = set()
    for i in range(len(ids)):
        js = np.flatnonzero((coords[:, 0] == coords[i, 0]) & (ids != ids[i]))
        if len(js):
            distance = np.square(coords[js, 1:] - coords[i, 1:]).sum(1)
            j = int(js[np.lexsort((ids[js], distance))[0]])
            edges.add(tuple(sorted((i, j))))
    i, j = torch.as_tensor(sorted(edges), device=x.device).T
    s = (len(ids) / len(i))**.5
    design = torch.cat((x, (x[j] - x[i]) * s))
    target = torch.cat((r, (r[j] - r[i]) * s))
    return design, target


def moments(anchor, hit, pack, w1, b1, w2k):
    assert all(torch.equal(anchor[k], hit[k]) for k in ("token_indices", "token_thw"))
    ids = hit["token_indices"].numpy()
    coords = hit["token_thw"].numpy()
    assert len(np.unique(ids)) == 512 and np.array_equal(ids, np.ravel_multi_index(coords.T, (21, 29, 52)))
    u0, u1 = anchor["U"].cuda(), hit["U"].cuda()
    y0, y1 = anchor["Y_pre_gate"].cuda(), hit["Y_pre_gate"].cuda()
    assert all(v.dtype == torch.bfloat16 and v.shape == (512, 5120) for v in (u0,u1,y0,y1))
    sel = pack["sel"]
    z0 = F.linear(u0, w1, b1).index_select(1, sel)
    z1 = F.linear(u1, w1.index_select(0,sel), b1.index_select(0,sel))
    dh = (F.gelu(z1, approximate="tanh").float() - F.gelu(z0, approximate="tanh").float()).bfloat16().float()
    dz = z1.float() - z0.float()
    du = u1.float() - u0.float()
    base = y0.float() + (du.bfloat16() @ pack["A0"]).float() + dh @ w2k.float()
    q = dh @ pack["zmap"] - du @ pack["jmap"]
    x = (dh - dz * pack["kernel_beta"]) / pack["kernel_scale"] / sel.numel()**.5
    design, target = joint_operators(x, y1.float() - base - q @ pack["v"], ids, coords)
    return design.T @ design, design.T @ target


def folded_weights(samples, pack, w1, w2k):
    def fit(rows):
        gram = sum(s[0] for s in rows)
        gram.diagonal().add_(1.)
        return torch.cholesky_solve(sum(s[1] for s in rows), torch.linalg.cholesky(gram))
    cross = power = 0.
    for i, (gram, xy) in enumerate(samples):
        w = fit(samples[:i] + samples[i+1:])
        cross += float((w.double() * xy).sum())
        power += float((w.double() * (gram @ w)).sum())
    assert np.isfinite([cross, power]).all() and power > 0
    alpha = float(np.clip(cross / power, 0., 1.))
    delta = alpha * fit(samples) / pack["sel"].numel()**.5 / pack["kernel_scale"][:, None]
    projected = (w1[pack["sel"]].float().T * pack["kernel_beta"]) @ delta
    pair = {"A0": (pack["A0"].float() - projected).bfloat16(), "w2k": (w2k.float() + delta).bfloat16()}
    assert all(t.isfinite().all() for t in pair.values())
    return {k:v.cpu().contiguous() for k,v in pair.items()}, dict(alpha=alpha, unbounded=cross/power)


def compile_site(layer, ck, record, pairs, model, destination, code_sha):
    expert = record["expert"]
    name = f"{expert}{layer:02d}"
    file = destination / (name + ".pt")
    receipt = file.with_suffix(".json")
    if receipt.exists():
        saved = json.loads(receipt.read_bytes())
        assert saved["compiler"] == code_sha and saved["base"] == record["sha256"] and file_sha(file) == saved["sha256"]
        return saved
    start = monotonic()
    assert record["layer"] == layer and tuple(record["cal_sources"]) == CAL and ck["sha256"] == record["checkpoint_sha256"]
    pack = torch.load(verified(record), map_location="cuda:0", weights_only=True)
    assert tensor_sha(pack["sel"]) == record["selection_sha256"]
    native = []
    for suffix in ("0.weight", "0.bias", "2.weight"):
        key = f"blocks.{layer}.ffn.{suffix}"
        weight = model.get_tensor(key)
        assert tensor_sha(weight) == record["ffn_tensor_sha256"][key]
        native.append(weight.cuda())
    w1, b1, w2 = native
    w2k = w2.index_select(1, pack["sel"]).T.contiguous()
    samples = []
    for index, source in enumerate(CAL):
        row = pairs[source, expert, layer]
        refs = [row[phase] for phase in ("anchor", "hit")]
        assert [r["step"] for r in refs] == ([0,1] if expert == "high" else [2,3])
        assert all(r["seed"] == 42 and r["source"] == source for r in refs)
        assert [r["panel"]["sha256"] for r in refs] == record["input_panel_sha256"][index]
        a, h = [torch.load(verified(r["panel"]), map_location="cpu", weights_only=True) for r in refs]
        samples.append(moments(a, h, pack, w1, b1, w2k))
    result, alpha = folded_weights(samples, pack, w1, w2k)
    tmp = file.with_suffix(".pt.tmp")
    torch.save(result, tmp)
    os.replace(tmp, file)
    note = dict(name=name, compiler=code_sha, base=record["sha256"], panels=record["input_panel_sha256"],
                bytes=file.stat().st_size, sha256=file_sha(file), seconds=monotonic()-start, **alpha)
    save_json(receipt, note)
    print("CAL3 FOLD", name, round(alpha["alpha"],4), round(note["seconds"],2), flush=True)
    return note


@torch.inference_mode()
def compile_fold(base, destination, experts=("high","low")):
    from safetensors import safe_open
    start = monotonic()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    assert experts and set(experts) <= {"high", "low"}
    assert base["cal_manifest"]["sha256"] == MANIFEST_SHA
    manifest = verified(base["cal_manifest"])
    archive = json.loads(manifest.read_bytes())
    assert archive["status"] == "PASS"
    bindings_file = manifest.parent / "src/BINDINGS.json"
    assert file_sha(bindings_file) == archive["source_bindings_sha256"]
    bindings = json.loads(bindings_file.read_bytes())
    ref = bindings["input_manifests"]["cal_pairs"]
    assert ref["sha256"] == archive["cal_pairs_sha256"]
    pairs = {(v["source"],v["expert"],v["layer"]):v for v in json.loads(verified(ref).read_bytes())["input_pairs"]}
    torch.backends.cuda.matmul.allow_tf32 = False
    code_sha = file_sha(__file__)
    sites = {}
    for expert in experts:
        ck = bindings["checkpoints"][expert]
        assert Path(ck["path"]).stat().st_size == ck["bytes"]
        with safe_open(ck["path"], framework="pt", device="cpu") as model:
            for layer in range(40):
                name = f"{expert}{layer:02d}"
                sites[name] = compile_site(layer, ck, archive["sites"][name], pairs, model, destination, code_sha)
    torch.cuda.empty_cache()
    report = dict(sites=sites, cal=list(CAL), base=MANIFEST_SHA, seconds=monotonic()-start,
                  bytes=sum(v["bytes"] for v in sites.values()), solver="CAL3 point+space ridge=1, source-holdout shrink per site",
                  nfe=0, compiler=code_sha)
    save_json(destination/"FOLD.json", report)
    return report


class FoldSwitch:
    """For a warmed fast512 port. Switch only at synchronized trajectory boundaries."""
    def __init__(self, method, destination):
        self.method = method
        self.destination = Path(destination)
        self.metadata = json.loads((self.destination/"FOLD.json").read_bytes())
        assert self.metadata["base"] == MANIFEST_SHA and self.metadata["compiler"] == file_sha(__file__)
        self.verified = set()
        self.originals = {}
        self.active = set()

    @torch.inference_mode()
    def activate(self, experts):
        wanted = set(experts)
        assert not self.method.busy and wanted <= {"high","low"} and self.method.mode == "fast512"
        torch.cuda.synchronize()
        start = monotonic()
        updates = []
        for expert in sorted(self.active ^ wanted):
            for layer in range(40):
                key = expert, layer
                current = self.method.weights[key]
                if expert in wanted:
                    name = f"{expert}{layer:02d}"
                    file = self.destination/(name+".pt")
                    if name not in self.verified:
                        assert file_sha(file) == self.metadata["sites"][name]["sha256"]
                        self.verified.add(name)
                    alternate = torch.load(file, mmap=True, map_location="cpu", weights_only=True)
                    if key not in self.originals:
                        self.originals[key] = {k:current[k].cpu() for k in FIELDS}
                else:
                    alternate = self.originals[key]
                for k in FIELDS:
                    assert alternate[k].dtype == current[k].dtype == torch.bfloat16
                    assert alternate[k].shape == current[k].shape
                updates.append((current,alternate))
        for current, alternate in updates:
            for k in FIELDS:
                current[k].copy_(alternate[k])
        torch.cuda.synchronize()
        self.active = wanted
        return dict(experts=sorted(wanted), changed_sites=len(updates), seconds=monotonic()-start)
