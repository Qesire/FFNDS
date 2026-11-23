"""Fit CAL3 source-blocked folds and replay the frozen 64-probe readout on new sources."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write_json(path, obj):
    temp = path.with_suffix(path.suffix + ".pending")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf8")
    temp.replace(path)


def groups(protocol):
    sites = protocol["sources"]
    assert len({s["source_cluster_id"] for s in sites}) == len(sites)
    assert len({s["image_sha256"] for s in sites}) == len(sites)
    sources = {s["record_id"]: s for s in sites}
    cal = {s for s, m in sources.items() if m["split"] == "calibration"}
    assert cal == {"vbi_0032", "vbi_0304", "vbi_0456"} and "vbi_0951" not in sources
    rows = protocol["trajectories"]
    assert len({r["trajectory_id"] for r in rows}) == len(rows)
    for r in rows:
        s = sources[r["source_id"]]
        assert r["split"] == s["split"] and r["image_sha256"] == s["image_sha256"]
        assert hashlib.sha256(r["prompt"].encode()).hexdigest() == r["prompt_sha256"]
        if r["split"] == "validation":
            assert not r["arms"]
    result = {}
    for arm in ("A", "B"):
        result[arm] = {}
        for s in sorted(cal):
            sel = [r for r in rows if r["source_id"] == s and arm in r["arms"]]
            assert sorted((r["seed"], r["condition"]) for r in sel) == [(42, "base"), (43, "base" if arm == "A" else "camera")]
            result[arm][s] = [r["trajectory_id"] for r in sel]
    return result


def directional(pred, target, anchor, gate, at):
    # Channel gate at the hit applies to the estimated and true cross-step increment.
    true = (target.float() - anchor.float()) * gate.float()
    error = (pred.float() - target.float()) * gate.float()
    if at.ndim == 1:
        true, error = true[at].double(), error[at].double()
    else:
        i, j = at.T
        true = (true[j] - true[i]).double()
        error = (error[j] - error[i]).double()
    norm2 = float(true.square().sum())
    assert norm2 > 0
    err2 = float(error.square().sum())
    dot = float((error * true).sum())
    along = dot / norm2
    ortho = max(0., err2 / norm2 - along ** 2)
    prednorm = 1 + 2 * along + err2 / norm2
    cosine = (1 + along) / max(prednorm, 1e-30) ** .5
    return dict(rows=len(at), energy=norm2, error_energy=err2, error_target_dot=dot,
                relative_rms=(err2 / norm2) ** .5, projected_gain=1 + along,
                orthogonal_rms=ortho ** .5, direction_cosine=cosine)


def evaluate_sets(panel, probe_ids):
    ids = torch.as_tensor(panel["token_indices"]).long().cpu()
    thw = torch.as_tensor(panel["token_thw"]).long().cpu()
    assert ids.shape == (512,) and thw.shape == (512, 3)
    assert len(ids.unique()) == len(ids) and torch.equal(ids, thw @ torch.tensor([29*52, 52, 1]))
    assert (thw >= 0).all() and (thw < torch.tensor([21, 29, 52])).all()
    locate = {v: i for i, v in enumerate(ids.tolist())}
    probes = torch.tensor([locate[i] for i in probe_ids])
    assert len(probes.unique()) == 64
    hold = ~torch.isin(torch.arange(len(ids)), probes)
    sets = {"point": torch.where(hold)[0]}
    for k, name in (("spatial_edges", "space"), ("temporal_edges", "time")):
        edge = torch.as_tensor(panel[k], dtype=torch.long).cpu()
        assert edge.ndim == 2 and edge.shape[1] == 2 and len(edge)
        assert (edge >= 0).all() and (edge < len(ids)).all() and hold[edge].all()
        delta = thw[edge[:, 1]] - thw[edge[:, 0]]
        if name == "time":
            assert (delta == torch.tensor([1, 0, 0])).all()
        else:
            assert (delta[:, 0] == 0).all() and (delta[:, 1:].abs().sum(1) == 1).all()
        sets[name] = edge
    return probes, sets


class Linear:
    """Expose matched checkpoint projections through the existing fast-kernel interface."""
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def _get_actual_weight(self):
        return self.weight.T

    def _get_actual_bias(self):
        return self.bias

    def apply(self, u):
        return F.linear(u, self.weight, self.bias)


def panel_at(stage, ref):
    path = stage / ref["path"]
    assert digest(path) == ref["sha256"], path
    panel = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("U0", "U1", "Y0", "Y1"):
        assert panel[key].shape == (512, 5120) and panel[key].dtype == torch.bfloat16
    assert panel["gate1"].numel() == 5120
    assert panel["anchor_z0k"].shape == (512, 2560) and panel["anchor_z0k"].dtype == torch.bfloat16
    assert panel["probe_Y1"].shape == (64, 5120) and panel["probe_Y1"].dtype == torch.bfloat16
    return panel


def at_phase(panel, step):
    return dict(token_indices=panel["token_indices"], token_thw=panel["token_thw"],
                U=panel["U" + step], Y_pre_gate=panel["Y" + step])


def metrics_for(site, panel, probe_ids, weights):
    import exact_port
    from native_joint import probe_metric_inverse
    probes, sets = evaluate_sets(panel, probe_ids)
    inv, _ = probe_metric_inverse(probe_ids, panel["token_thw"][probes].numpy(), "cuda")
    u, u0, y0, target = [panel[k].cuda() for k in ("U1", "U0", "Y0", "Y1")]
    gate = panel["gate1"].reshape(-1).cuda()
    cached = dict(u0=u0, y0=y0, z0k=panel["anchor_z0k"].cuda())
    prepared = site.prepared
    assert torch.equal(probes.cpu(), torch.as_tensor(panel["probe_rows"]).long().cpu())
    probes = probes.cuda()
    result = {}
    # The literal native 64-row output from capture is substituted only at the
    # observed branch. The teacher values at holdout rows never enter fast_hit.
    observed = panel["probe_Y1"].cuda()
    phase = SimpleNamespace(ffn_0=site.phase.ffn_0, ffn_2=SimpleNamespace(apply=lambda h: observed))
    for label, pair in weights.items():
        tensors = dict(prepared, **{k: v.cuda() for k, v in pair.items()})
        pred, _ = exact_port.fast_hit(u, phase, cached, tensors, probes, inv)
        assert torch.isfinite(pred).all()
        result[label] = {k: directional(pred, target, y0, gate, p.cuda()) for k, p in sets.items()}
        result[label]["joint_relative_energy"] = sum(v["relative_rms"] ** 2 for v in result[label].values()) / 3
    return result, "captured-64"


def run_site(stage, out, layer, protocol, arms, site_refs, base, record, model):
    import folded_runtime as fold
    import exact_port
    name = f"high{layer:02d}"
    pack = torch.load(fold.verified(record), map_location="cuda", weights_only=True)
    assert fold.tensor_sha(pack["sel"]) == record["selection_sha256"]
    native = []
    for suffix in ("0.weight", "0.bias", "2.weight", "2.bias"):
        key = f"blocks.{layer}.ffn.{suffix}"
        tensor = model.get_tensor(key)
        assert fold.tensor_sha(tensor) == record["ffn_tensor_sha256"][key]
        native.append(tensor.cuda())
    w1, b1, w2, b2 = native
    w2k = w2.index_select(1, pack["sel"]).T.contiguous()
    phase = SimpleNamespace(ffn_0=Linear(w1, b1), ffn_2=Linear(w2, b2))
    site = SimpleNamespace(pack=pack, phase=phase, prepared=exact_port.prepare(phase, pack, w1.device))
    weights = {"frozen": {"A0": pack["A0"], "w2k": w2k}}
    selections, compiled = {}, {}
    coordinates = None
    for arm, sources in arms.items():
        selections[arm] = []
        for source in sources:
            tables = []
            for traj in sources[source]:
                panel = panel_at(stage, site_refs[traj, layer])
                layout = fold.tensor_sha(panel["token_indices"])
                if coordinates is None:
                    coordinates = layout
                assert layout == coordinates
                tables.append(fold.moments(at_phase(panel, "0"), at_phase(panel, "1"), pack, w1, b1, w2k))
            # Unit effective 512-row budget per source, inherited ridge strength,
            # two equally weighted trajectories. Each Gram block is source-local.
            selections[arm].append(tuple((a + b) * .5 for a, b in zip(*tables)))
        weights[arm], receipt = fold.folded_weights(selections.pop(arm), pack, w1, w2k)
        path = out / arm / (name + ".pt")
        path.parent.mkdir(exist_ok=True)
        temp = path.with_suffix(".pt.pending")
        torch.save(weights[arm], temp)
        temp.replace(path)
        compiled[arm] = dict(**receipt, sha256=digest(path), source_groups=sources, bytes=path.stat().st_size)
    validation = []
    for r in protocol["trajectories"]:
        if r["split"] != "validation":
            continue
        traj = r["trajectory_id"]
        panel = panel_at(stage, site_refs[traj, layer])
        assert fold.tensor_sha(panel["token_indices"]) == coordinates
        metrics, obs_kind = metrics_for(site, panel, base["observer_ids"], weights)
        validation.append(dict(trajectory_id=traj, source=r["source_id"], condition=r["condition"],
                               seed=r["seed"], layer=layer, probe_labels=obs_kind, arms=metrics))
    result = dict(site=name, input_coordinates_sha256=coordinates, fold=compiled, validation=validation)
    write_json(out / (name + ".json"), result)
    return result


@torch.inference_mode()
def run(stage, out):
    stage, out = stage.resolve(), out.resolve()
    sys.path.append(str(stage))
    import folded_runtime as fold
    from safetensors import safe_open
    protocol, index, base = [read(stage / n) for n in ("protocol.json", "output/CAPTURE_INDEX.json", "base.json")]
    assert digest(stage / "protocol.json") == index["protocol_sha256"]
    assert index["status"] == "complete" and len(index["trajectories"]) == 13 and index["nfe"] == 52
    assert protocol["layers"] == [0, 13, 26, 39] and protocol["expert"] == "high"
    arms = groups(protocol)
    refs = {(r["trajectory_id"], r["layer"]): r for r in index["sites"]}
    expected = {(r["trajectory_id"], layer) for r in protocol["trajectories"] for layer in protocol["layers"]}
    assert refs.keys() == expected and len(index["sites"]) == len(expected)
    assert all(refs[r["trajectory_id"], layer]["source_id"] == r["source_id"] for r in protocol["trajectories"] for layer in protocol["layers"])
    archive_file = fold.verified(base["cal_manifest"])
    assert digest(archive_file) == fold.MANIFEST_SHA
    archive = read(archive_file)
    bindings_path = archive_file.parent / "src/BINDINGS.json"
    assert digest(bindings_path) == archive["source_bindings_sha256"]
    checkpoint = read(bindings_path)["checkpoints"]["high"]
    assert Path(checkpoint["path"]).stat().st_size == checkpoint["bytes"]
    out.mkdir(exist_ok=True, parents=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    results = []
    with safe_open(checkpoint["path"], framework="pt", device="cpu") as model:
        for layer in protocol["layers"]:
            record = archive["sites"][f"high{layer:02d}"]
            assert record["checkpoint_sha256"] == checkpoint["sha256"]
            results.append(run_site(stage, out, layer, protocol, arms, refs, base, record, model))
            print("completed high layer", layer, flush=True)
    summary = dict(status="PASS", trial=protocol["trial"], job=os.environ.get("SLURM_JOB_ID"),
                   protocol_sha256=index["protocol_sha256"], capture_index_sha256=digest(stage / "output/CAPTURE_INDEX.json"),
                   evaluator_sha256=digest(__file__), frozen_manifest_sha256=digest(archive_file),
                   fold_implementation_sha256=digest(Path(fold.__file__)),
                   scope="Same-input replay on Dense anchor/hit trajectories, true cached anchor hidden, actual hit gate, panel-sized 512-row tile for writeback, native captured 64-row observations. No closed-loop video quality or speed measurement.",
                   joint="Equal-weight mean of three normalized gated error energies (point, spatial first difference, temporal first difference). Report each component and signed projections.",
                   training="Frozen prior/features; unit-weight point plus same-frame nearest neighbor; two trajectories averaged within source; three source blocks for held-source shrinkage at fixed ridge.",
                   layers=results)
    write_json(out / "RESULT.json", summary)
    print("complete", out / "RESULT.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-stage", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    run(args.capture_stage, args.output)
