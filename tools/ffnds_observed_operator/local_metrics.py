"""Common deployment risk and same-query FP64 direction diagnostics."""

from collections import defaultdict
from math import isfinite
from statistics import mean

import torch
from torch.nn import functional as F

from .local_data import require


def observation_gap(actual, archive, gate):
    difference = actual.float() - archive.float()
    return {"observer_capture_bf16_value_equal": int((actual == archive).sum()),
            "observer_capture_count": difference.numel(),
            "observer_capture_max_abs": float(difference.abs().max()),
            "observer_capture_gated_mse": float((difference * gate).square().mean())}


def measure(prediction, anchor, target, gate):
    predicted = (prediction.double() - anchor.double()) * gate.double()
    truth = (target.double() - anchor.double()) * gate.double()
    error = float((prediction.double() - target.double()).mul(gate.double()).square().sum())
    true_norm = float(truth.norm())
    pred_norm = float(predicted.norm())
    return {"mse": error / target.numel(), "relative_risk": error / true_norm**2 if true_norm else None,
            "increment_cosine": float((predicted * truth).sum()) / (pred_norm * true_norm)
            if pred_norm and true_norm else None, "increment_norm_ratio": pred_norm / true_norm if true_norm else None}


def baseline(pair, a):
    c0 = (pair["y0"].float() - (pair["u0"] @ a).float()).bfloat16()
    return ((pair["u1"] @ a).float() + c0.float()).bfloat16()


def common_rows(site, panels):
    rows, observers = [], []
    for pair in panels:
        common = baseline(pair, site["A"])
        require(site["gate_witness"](common, pair), f"{site['key']}/{pair['source']}: common-A current gate differs")
        for fold, observed in zip(pair["folds"], pair["probe_y1"], strict=True):
            q = fold["query"]
            identity = {"site": site["key"], "source": pair["source"], "split": pair["split"], "fold": fold["fold"]}
            rows.append({**identity, "family": "shared", "arm": "common_a", **measure(
                common[q], pair["y0"][q], pair["y1"][q], pair["gate"])})
            observers.append({**identity, "rows": 64, "probe_global_ids": fold["probe_ids"], **observation_gap(
                observed, pair["y1"][fold["probe"]], pair["gate"])})
    return rows, observers


class PanelDirections:
    def __init__(self, site):
        self.site_key = site["key"]
        self.w1, self.w2, self.b1 = (site[key].double() for key in ("w1", "w2", "b1"))
        self.a = site["A"].double()
        self.q32 = site["q32"].double()
        self.cache = {}
        self.rows = []

    def _prepare(self, pair):
        site_key = self.site_key
        u0, u1 = pair["u0"].double(), pair["u1"].double()
        h0 = F.gelu(F.linear(u0, self.w1, self.b1), approximate="tanh")
        h1 = F.gelu(F.linear(u1, self.w1, self.b1), approximate="tanh")
        delta_h = h1 - h0
        delta_a = (u1 - u0) @ self.a
        gate = pair["gate"].double()
        residual = (delta_h @ self.w2.T - delta_a) * gate
        basis, singular, _ = torch.linalg.svd(gate[:, None] * self.q32, full_matrices=False)
        basis = basis[:, singular > singular.max() * 1e-10]
        for fold in pair["folds"]:
            q = fold["query"]
            score2 = delta_h[fold["probe"]].square().mean(dim=0)
            chosen = torch.argsort(score2, descending=True, stable=True)[:5120].sort().values
            require(chosen.numel() == 5120, f"{site_key}: panel-P64 exact K failure")
            panel_delta = delta_h[q][:, chosen] @ self.w2[:, chosen].T
            panel = (panel_delta - delta_a[q]) * gate
            cache_key = pair["source"], fold["fold"]
            identity = {"source": pair["source"], "site": site_key, "split": pair["split"], "fold": fold["fold"],
                        "target_space": "fp64_mathematical_ffn_gated_residual"}
            self.cache[cache_key] = (identity, residual[q], panel, gate, basis)
            self(pair, fold, "panel_p64_fp64", None)

    def __call__(self, pair, fold, arm, correction):
        cache_key = pair["source"], fold["fold"]
        if cache_key not in self.cache:
            self._prepare(pair)
        identity, residual, panel, gate, basis = self.cache[cache_key]
        candidate = panel if correction is None else correction.double() * gate
        require(candidate.shape == residual.shape, f"{self.site_key} {arm}: direction must use same query")
        full_gain = 2 * (residual * candidate).sum() - candidate.square().sum()
        residual_in, candidate_in = residual @ basis, candidate @ basis
        gain_in = 2 * (residual_in * candidate_in).sum() - candidate_in.square().sum()
        panel_norm = float(panel.norm())
        candidate_norm = float(candidate.norm())
        row = {**identity, "arm": arm, "q32_gated_rank": basis.shape[1], "gain": float(full_gain),
               "gain_in_gated_q32": float(gain_in), "gain_outside_gated_q32": float(full_gain-gain_in),
               "candidate_correction_norm": candidate_norm, "panel_p64_correction_norm": panel_norm,
               "candidate_residual_norm": float((residual - candidate).norm()), "base_residual_norm": float(residual.norm()),
               "candidate_panel_p64_cosine": float((candidate * panel).sum()) / (panel_norm * candidate_norm)
               if candidate_norm and panel_norm else None}
        if correction is None:
            row["selected_k"] = 5120
        self.rows.append(row)
        return row


FAMILIES = {
    "observed_readout": {"static": "readout_static", "online": "readout_fixed64",
                         "continuous_oracle": "readout_oracle_continuous", "compiled_oracle": "readout_oracle_quantized"},
    "native_modes": {"static": "static", "online": "fixed", "continuous_oracle": "oracle_continuous",
                     "compiled_oracle": "oracle_compiled"},
}


def relative_decrease(before, after):
    return (before - after) / before if before > 0 else (0.0 if after == 0 else None)


def select_families(rows, acceptance, sources, sites, criteria):
    # Fold averages are formed first, then site averages within each source.
    scored_arms = {"common_a"} | {arm for arms in FAMILIES.values() for arm in arms.values()}
    grouped = defaultdict(dict)
    for row in rows:
        if row["split"] == "DEV" and row["arm"] in scored_arms:
            key = row["arm"], row["source"], row["site"]
            require(row["fold"] not in grouped[key], f"Duplicate risk row for {key}/{row['fold']}")
            grouped[key][row["fold"]] = row["mse"]

    def source_risks(arm, source):
        folds = [grouped.get((arm, source, site), {}) for site in sites]
        if any(item.keys() != {1, 2} or any(not isfinite(value) or value < 0 for value in item.values())
               for item in folds):
            return None
        return mean(mean(item.values()) for item in folds)

    families = {}
    for key, arms in FAMILIES.items():
        source_results = {source: {name: source_risks(arm, source) for name, arm in {"common_a": "common_a", **arms}.items()}
                          for source in sources}
        invalid = [site for site in sites if not acceptance[site][key]["valid"]]
        missing = [source for source, means in source_results.items() if any(v is None for v in means.values())]
        site_status = {site: acceptance[site][key]["status"] for site in sites}
        result = {"passes": False, "sources": source_results, "sites_invalid": invalid,
                  "sites_status": site_status, "sources_incomplete": missing, "evaluated_for_admission": False}
        families[key] = result
        if invalid or missing:
            result["decision"] = "INCOMPLETE_OR_INVALID_EXECUTION"
            continue

        means = {name: mean(source_results[s][name] for s in sources) for name in source_results[sources[0]]}
        online_gain = relative_decrease(means["static"], means["online"])
        gain_floor = criteria["minimum_online_over_static_relative_mse_gain"]
        wins = 0
        no_large_regression = True
        for detail in source_results.values():
            wins += detail["online"] < detail["static"]
            regression = relative_decrease(detail["static"], detail["online"])
            detail["relative_mse_regression"] = -regression if regression is not None else None
            no_large_regression &= detail["online"] <= (1 + criteria["maximum_single_source_relative_mse_regression"]) * detail["static"]
        tests = {"mean_online_gain": online_gain is not None and online_gain >= gain_floor,
                 "source_wins": wins >= criteria["minimum_source_wins"], "source_regression": no_large_regression,
                 "online_no_worse_than_common_a": means["online"] <= means["common_a"], "native_bf16_current_gate": True}
        continuous = relative_decrease(means["static"], means["continuous_oracle"])
        compiled = relative_decrease(means["static"], means["compiled_oracle"])
        static_gain = relative_decrease(means["common_a"], means["static"])
        result.update(means=means, relative_online_gain=online_gain, source_wins=wins, checks=tests, evaluated_for_admission=True,
                      continuous_oracle_gain=continuous, compiled_oracle_gain=compiled,
                      static_over_common_a_gain=static_gain,
                      static_representation_gain_at_two_percent=static_gain is not None and static_gain >= gain_floor,
                      passes=all(tests.values()))
        if result["passes"]:
            result["decision"] = "ADMITTED"
        elif tests["mean_online_gain"]:
            result["decision"] = "ADMISSION_CONSTRAINT_NOT_MET"
        elif continuous is None or continuous < gain_floor:
            result["decision"] = "CONTINUOUS_REPRESENTATION_LIMITED"
        elif compiled is None or compiled < gain_floor:
            result["decision"] = "QUANTIZATION_OR_COMPILATION_LOSS"
        else:
            result["decision"] = "IDENTIFICATION_LIMITED"

    admitted = [key for key in FAMILIES if families[key]["passes"]]
    unresolved_preference = admitted == ["native_modes"] and not families["observed_readout"]["evaluated_for_admission"]
    if unresolved_preference:
        admitted = []
    if admitted:
        next_stage = "PRIMARY_VIDEO_PREPARATION"
    elif any(item["sites_invalid"] or item["sources_incomplete"] for item in families.values()):
        next_stage = "REVIEW_INCOMPLETE_LOCAL_EVIDENCE"
    else:
        next_stage = "STOP_NEW_VIDEOS"
    return {"families": families, "primary": admitted[0] if admitted else None,
            "secondary": admitted[1] if len(admitted) == 2 else None, "next_stage": next_stage,
            "criteria": criteria, "nfe": 0, "d4": 0, "statistical_unit": "source_image_cluster"}
