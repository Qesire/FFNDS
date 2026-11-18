"""Fit exactly four endpoint arms on CAL and evaluate on the paired DEV panel."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from .analysis import metrics, prepare_pair, structure, verify_features
from .features import ARMS, arm_features, extra_basis, select_hidden, source_rms

SITES = tuple((expert, layer) for expert in ("high", "low") for layer in (4, 19, 34))
RIDGE_EPS = 1e-6
BINDINGS_SCHEMA = "FFNDS_STATE_OPERATOR_BINDINGS_V1"
FIT_SCHEMA = "FFNDS_STATE_OPERATOR_FIT_V1"
KEYS = {"L+": "Lplus", "Q32": "Q32", "N32": "N32"}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified(record):
    path = Path(record["path"])
    if sha256(path) != record["sha256"]:
        raise ValueError(f"Asset SHA mismatch for {path}")
    return path


def load_q1_math(site_binding, policy, device, old=None):
    if policy == "REUSE_ALL_SIX":
        if site_binding["bf16_bitwise_equal"] is not True:
            raise ValueError("Old L64 requires exact strict Q1 parity")
        if old is None:
            old = torch.load(verified(site_binding["old_params"]), map_location=device, weights_only=True)
        matrix = old["A_q1"]
        if matrix.shape != (5120, 5120) or matrix.dtype != torch.float64:
            raise ValueError("Expected the old kit's real-arithmetic Q1 matrix")
        return matrix
    if policy == "REFIT_ALL_SIX":
        value = torch.load(verified(site_binding["strict_q1"]), map_location=device, weights_only=True)
        matrix = value["A_q1_f32"]
        if matrix.shape != (5120, 5120) or matrix.dtype != torch.float32:
            raise ValueError("Expected the strict runtime's FP32 source Q1 matrix")
        return matrix.double()
    raise ValueError(f"Unknown L64 policy {policy}")


def fit_readout(cal, features, q, *, curvature=False):
    """Gate-aware normal equations, using equal source weight and exact gate vectors."""
    gamma = source_rms(features)
    size, out = gamma.numel(), q.shape[1]
    dim = size * out
    gram = q.new_zeros((dim, dim))
    rhs = q.new_zeros((size, out))
    for pair, feature in zip(cal, features, strict=True):
        phi = feature / gamma
        weight = 1 / (len(cal) * len(phi))
        g2 = pair["g"].square()
        gram += torch.kron((phi.T @ phi) * weight, q.T @ (g2[:, None] * q))
        rhs += phi.T @ ((pair["R"] * g2) @ q) * weight

    rho = float(gram.diagonal().mean()) * RIDGE_EPS
    gram.diagonal().add_(rho)
    rhs = rhs.flatten()
    coeff = torch.linalg.solve(gram, rhs) if rho else torch.zeros_like(rhs)
    info = {"ridge": rho, "normal_relative_residual":
            float((gram @ coeff - rhs).norm() / rhs.norm().clamp_min(1e-300))}
    linear = None
    if curvature:
        cut = dim // 2
        normal_d = gram[:cut, :cut]
        linear = torch.linalg.solve(normal_d, rhs[:cut]) if rho else torch.zeros_like(rhs[:cut])
        # Reduced RHS of the Schur system. c_n equals Schur^-1 rhs_n.
        reduced_rhs = rhs[cut:] - gram[cut:, :cut] @ linear
        info["cal_schur_gain_mse"] = float(reduced_rhs @ coeff[cut:]) / q.shape[0]
        linear = linear.reshape(-1, out)
    return gamma, coeff.reshape(size, out), linear, info


def _prepare_base(cal, old, a_q1, policy, legacy):
    fields = ("P64", "Q32", "b", "sigma", "gamma_L", "C_L")
    if policy == "REUSE_ALL_SIX":
        params = {key: old[key] for key in fields}
        shapes = ((5120, 64), (5120, 32), (32,), (32,), (64,), (64, 32))
        for key, shape in zip(fields, shapes, strict=True):
            if params[key].dtype != torch.float64 or params[key].shape != shape:
                raise ValueError(f"Invalid old L64 parameter {key}")
        if not (params["sigma"].min() > 0 and params["gamma_L"].min() > 0):
            raise ValueError("Old L64 must provide positive RMS and endpoint scales")
        return params
    p, q = legacy.select_bases(cal, a_q1)
    b, sigma = legacy.center_scale(cal, p[:, :32])
    for pair in cal:
        pair["R"] = pair["Y"] - pair["X"] @ a_q1
    gamma, coeff, _, _ = fit_readout(cal, [pair["X"] @ p for pair in cal], q)
    return dict(zip(fields, (p, q, b, sigma, gamma, coeff), strict=True))


def fit_site(cal, dev, old, a_q1, weights, policy, legacy=None):
    """Consume a validated site and return compact parameters plus paired evidence."""
    w1, b1, w2, _ = [tensor.double() for tensor in weights]
    for pair in (*cal, *dev):
        prepare_pair(pair, w1, b1, w2)
    params = _prepare_base(cal, old, a_q1, policy, legacy)
    p, q = params["P64"], params["Q32"]
    a_l = a_q1 + (p / params["gamma_L"]) @ params["C_L"] @ q.T
    for pair in (*cal, *dev):
        pair["base"] = pair["X"] @ a_l
        pair["R"] = pair["Y"] - pair["base"]

    params["P_extra64"] = extra_basis(cal, p, p.shape[1])
    params["hidden_indices"], params["hidden_alpha"], scores = select_hidden(cal, params, w1, b1, q.shape[1])
    fit_info = {"hidden_greedy_surrogate_gains": scores,
                "p_extra_max_overlap": float((p.T @ params["P_extra64"]).abs().max())}
    if fit_info["p_extra_max_overlap"] > 1e-10:
        raise ValueError("L+ complement overlaps the old P64 span")
    restricted = {}
    for arm in ARMS[1:]:
        key = KEYS[arm]
        raw = [arm_features(arm, pair, params, w1, b1) for pair in cal]
        gamma, coeff, linear, fit_info[arm] = fit_readout(cal, raw, q, curvature=arm != "L+")
        params[f"gamma_{key}"], params[f"C_{key}"] = gamma, coeff
        if linear is not None:
            restricted[arm] = linear

    readout, diagnostics = [], []
    for split, pack in (("CAL", cal), ("DEV", dev)):
        for pair in pack:
            identity = {"split": split, "source": pair["source"]}
            verify_features(pair, params, w1, b1)
            baseline = metrics(pair, pair["base"])
            diagnostics.append({**identity, **structure(pair, pair["base"], q)})
            captured = dict(pair, Y=pair["y1"] - pair["y0"])
            for arm in ARMS:
                prediction, result, d_result = pair["base"], baseline, None
                if arm != "L0":
                    key = KEYS[arm]
                    phi = arm_features(arm, pair, params, w1, b1) / params[f"gamma_{key}"]
                    prediction = pair["base"] + phi @ params[f"C_{key}"] @ q.T
                    result = metrics(pair, prediction)
                    if arm in restricted:
                        d_result = metrics(pair, pair["base"] + phi[:, :q.shape[1]] @ restricted[arm] @ q.T)
                mse = result["gated_mse"]
                observed = metrics(captured, prediction)
                readout.append({**identity, "arm": arm, **result,
                                "captured_target_gated_mse_fp64_prediction": observed["gated_mse"],
                                "captured_target_relative_risk_fp64_prediction": observed["relative_risk"],
                                "improvement_mse": baseline["gated_mse"] - mse,
                                "d_only_gated_mse": d_result["gated_mse"] if d_result else None,
                                "d_only_relative_risk": d_result["relative_risk"] if d_result else None,
                                "curvature_improvement_mse": d_result["gated_mse"] - mse if d_result else None})
    return params, fit_info, diagnostics, readout


def load_pair(record, device):
    a, u = [torch.load(verified(record[key]), map_location="cpu", weights_only=True)
            for key in ("anchor", "hit")]
    pair = {"source": record["source"]}
    for i, data in enumerate((a, u)):
        endpoint = record[("anchor", "hit")[i]]
        if data["metadata"]["token_id_sha256_le_int64"] != endpoint["token_sha256"]:
            raise ValueError("Native panel token hash differs from the P0 binding")
        for name, key in (("u", "U"), ("y", "Y_pre_gate")):
            value = data[key]
            if value.shape != (512, 5120) or value.dtype != torch.bfloat16:
                raise ValueError(f"Invalid native panel tensor {key} shape/dtype {value.shape}/{value.dtype}")
            pair[f"{name}{i}"] = value.to(device=device, dtype=torch.float64)
    if u["gate"].shape != (1, 1, 5120) or u["gate"].dtype != torch.bfloat16:
        raise ValueError("Invalid native current gate shape/dtype")
    pair["g"] = u["gate"].reshape(-1).to(device=device, dtype=torch.float64)
    if pair["g"].shape != (5120,):
        raise ValueError("Expected one actual current gate vector shared by panel rows")
    pair["X"] = pair["u1"] - pair["u0"]
    pair["n"] = len(pair["X"])
    if not all(torch.isfinite(x).all() for x in pair.values() if isinstance(x, torch.Tensor)):
        raise ValueError("Non-finite stored panel value")
    return pair


def load_weights(path, layer, device):
    from safetensors import safe_open

    names = ("0.weight", "0.bias", "2.weight", "2.bias")
    shapes = ((13824, 5120), (13824,), (5120, 13824), (5120,))
    with safe_open(path, framework="pt", device="cpu") as model:
        values = [model.get_tensor(f"blocks.{layer}.ffn.{name}") for name in names]
    if any(t.dtype != torch.bfloat16 or t.shape != s for t, s in zip(values, shapes, strict=True)):
        raise ValueError(f"Unexpected native FFN tensors in {path} layer {layer}")
    return [t.to(device) for t in values]


def aggregate(rows, sources, *, expert=None, layer=None):
    selected = [row for row in rows if row["split"] == "DEV"
                and (expert is None or row["expert"] == expert)
                and (layer is None or row["layer"] == layer)]
    result = {}
    names = ("gated_mse", "relative_risk", "cosine", "norm_ratio", "improvement_mse",
             "curvature_improvement_mse", "captured_target_gated_mse_fp64_prediction",
             "captured_target_relative_risk_fp64_prediction")

    def mean_metrics(records):
        result = {}
        for key in names:
            values = [r[key] for r in records if r[key] is not None]
            result[key] = sum(values) / len(values) if values else None
        return result

    for arm in ARMS:
        by_source = {}
        for source in sources:
            segment = [r for r in selected if r["arm"] == arm and r["source"] == source]
            by_source[source] = mean_metrics(segment)
        deltas = {s: row["improvement_mse"] for s, row in by_source.items()}
        result[arm] = {"by_source": by_source,
                       "mean_metrics": mean_metrics(by_source.values()),
                       "worst_source": min(deltas, key=deltas.get),
                       "leave_one_out_improvement_mse": {s: (sum(deltas.values()) - value) / (len(sources) - 1)
                                                         for s, value in deltas.items()}}
    return result


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def run(bindings_path, out, device):
    import sys

    bind = json.loads(bindings_path.read_text(encoding="utf-8-sig"))
    if bind["schema"] != BINDINGS_SCHEMA or bind["status"] != "PASS":
        raise ValueError("Expected an approved P0 asset binding with PASS")
    policy = bind["l64_policy"]
    if set(bind["sites"]) != {f"{expert}{layer:02d}" for expert, layer in SITES}:
        raise ValueError("Unexpected site records for the fixed six-site fit")
    cohorts = {split: bind[f"{split.lower()}_sources"] for split in ("CAL", "DEV")}
    if any(len(src) != 11 or len(set(src)) != 11 for src in cohorts.values()):
        raise ValueError("The frozen panel has exactly 11 unique sources per split")
    if set(cohorts["CAL"]) & set(cohorts["DEV"]):
        raise ValueError("CAL and DEV must be disjoint")
    expected = {(expert, layer, split, source) for expert, layer in SITES
                for split, sources in cohorts.items() for source in sources}
    pairs = {(r["expert"], r["layer"], r["split"], r["source"]): r for r in bind["pairs"]}
    if set(pairs) != expected or len(bind["pairs"]) != len(expected):
        raise ValueError("Missing, extra or duplicate panel binding")
    verified(bind["bank_manifest"])
    checks = {expert: verified(bind["checkpoints"][expert]) for expert in ("high", "low")}
    legacy = None
    if policy == "REFIT_ALL_SIX":
        legacy_dir = (Path(__file__).resolve().parents[2] /
                      "reports/20261114/g20_two_track_core/A_Q1_RESIDUAL/v1/source")
        sys.path.insert(0, str(legacy_dir))
        import a_q1_residual_repr as legacy

    out.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": FIT_SCHEMA, "bindings": {"path": str(bindings_path.resolve()),
                "sha256": sha256(bindings_path)}, "sites": {}, "l64_policy": policy,
                "device": str(device), "torch_version": torch.__version__, "ridge_eps": RIDGE_EPS,
                "arms": ARMS, "cal_sources": cohorts["CAL"], "dev_sources": cohorts["DEV"],
                "producer": {name: sha256(Path(__file__).with_name(name))
                             for name in ("fit.py", "features.py", "analysis.py")},
                "numeric": "FP64 Dense tanh GELU and FP64 endpoint fit; native BF16 is separately replayed",
                "hidden_selection": "CAL sequential conditional gain using trace-mean Q gate Gram surrogate, exact gate-aware final fit",
                "ridge": "Ridge epsilon times mean diagonal of the RMS-standardized full normal matrix, shared with the D-only fit",
                "p_extra64": "Top CAL SVD in P64-orthogonal complement, QR and max-absolute-entry positive orientation"}
    if legacy is not None:
        manifest["producer"]["legacy_basis_and_center"] = sha256(legacy.__file__)
    structures, readouts = [], []
    for expert, layer in SITES:
        tag = f"{expert}{layer:02d}"
        print(f"Fitting {tag} on CAL and scoring frozen DEV", flush=True)
        site = bind["sites"][tag]
        if (site["expert"], site["layer"]) != (expert, layer):
            raise ValueError("Site record does not match the frozen six-site panel")
        old = (torch.load(verified(site["old_params"]), map_location=device, weights_only=True)
               if policy == "REUSE_ALL_SIX" else None)
        a_q1 = load_q1_math(site, policy, device, old)
        pack = {split: [load_pair(pairs[expert, layer, split, src], device) for src in sources]
                for split, sources in cohorts.items()}
        weights = load_weights(checks[expert], layer, device)
        params, info, diagnostic, readout = fit_site(pack["CAL"], pack["DEV"], old, a_q1, weights, policy, legacy)
        if not all(torch.isfinite(x).all() for x in params.values()):
            raise ValueError(f"Non-finite fitted parameters for {tag}")
        path = out / f"{tag}.pt"
        torch.save({key: val.cpu().contiguous() for key, val in params.items()}, path)
        manifest["sites"][tag] = {"expert": expert, "layer": layer, "fit_info": info,
                                   "params": {"path": str(path.resolve()), "sha256": sha256(path)}}
        identities = {"expert": expert, "layer": layer}
        structures.extend({**identities, **row} for row in diagnostic)
        readouts.extend({**identities, **row} for row in readout)
        del pack, old, a_q1, weights, params
        print(f"Wrote compact parameters for {tag}", flush=True)
    write_csv(out / "STRUCTURE.csv", structures)
    write_csv(out / "READOUT.csv", readouts)
    dev_sources = cohorts["DEV"]
    summary = {"all_six": aggregate(readouts, dev_sources)}
    for expert in ("high", "low"):
        summary[expert] = aggregate(readouts, dev_sources, expert=expert)
    for expert, layer in SITES:
        summary[f"{expert}{layer:02d}"] = aggregate(readouts, dev_sources, expert=expert, layer=layer)
    (out / "DEV_SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    manifest["derived"] = {name: sha256(out / name) for name in ("STRUCTURE.csv", "READOUT.csv", "DEV_SUMMARY.json")}
    (out / "FIT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--bindings", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--device", default="cuda")
    args = cli.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    run(args.bindings, args.out, torch.device(args.device))


if __name__ == "__main__":
    main()
