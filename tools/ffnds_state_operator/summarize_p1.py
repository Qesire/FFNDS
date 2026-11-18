"""Summarize the frozen P1 outputs with source-equal six-site aggregation."""

import argparse
from collections import defaultdict
import csv
from hashlib import file_digest
import json
from pathlib import Path
from statistics import mean, median


def load(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def read_summary(root):
    fitdir = root / "output" / "fit"
    manifest = load(fitdir / "FIT_MANIFEST.json")
    accepted = load(root / "output" / "BF16_ACCEPTANCE.json")
    end = load(root / "receipt" / "EXIT.json")
    if end["exit_code"] != 0 or accepted["status"] != "PASS" or accepted["d4"] or accepted["nfe"]:
        raise ValueError("P1 does not have a successful finite-scope acceptance")
    if accepted["fit"]["sha256"] != sha(fitdir / "FIT_MANIFEST.json"):
        raise ValueError("BF16 acceptance bound a different fit manifest")
    for name, digest in manifest["derived"].items():
        if sha(fitdir / name) != digest:
            raise ValueError(f"fit metric digest mismatch for {name}")
    result = {"job": end["slurm_job_id"], "fit_sha256": accepted["fit"]["sha256"],
              "bf16_sha256": sha(root / "output" / "BF16_ACCEPTANCE.json")}
    base = load(fitdir / "DEV_SUMMARY.json")["all_six"]
    check = defaultdict(lambda: defaultdict(list))
    for row in accepted["sites"]:
        for pair in row["pairs"]:
            check[row["arm"]][pair["source"]].append(pair["current_gate_relative_risk"])
    result["arms"] = {}
    for arm, stored in base.items():
        if not all(len(r) == 6 for r in check[arm].values()) or len(check[arm]) != 11:
            raise ValueError("BF16 summary requires all six sites and DEV11")
        risk = {s: mean(rows) for s, rows in check[arm].items()}
        med = median(risk.values())
        worst = max(risk, key=risk.get)
        row = stored["mean_metrics"]
        result["arms"][arm] = {
            "math_mse": row["gated_mse"],
            "math_risk": row["relative_risk"],
            "math_gain_beyond_d_mse": row["curvature_improvement_mse"],
            "math_sources_better_than_l0": sum(x["improvement_mse"] > 0 for x in stored["by_source"].values()),
            "math_sources_better_than_own_d": sum((x["curvature_improvement_mse"] or 0) > 0
                                                  for x in stored["by_source"].values()),
            "bf16_mean_risk": mean(risk.values()), "bf16_median_risk": med,
            "bf16_worst_source": worst, "bf16_worst_risk": risk[worst],
            "bf16_risk_by_source": risk,
        }
    with (fitdir / "STRUCTURE.csv").open(newline="", encoding="utf-8") as f:
        dev = [r for r in csv.DictReader(f) if r["split"] == "DEV"]
    if len(dev) != 66:
        raise ValueError("FP64 structure table source/site coverage mismatch")
    energies = ["state_energy", "chord_energy", "cross_energy", "residual_energy", "output_missed_energy"]
    result["structure_means"] = {key: mean(float(r[key]) for r in dev) for key in energies}
    eta = [float(r["eta_out"]) for r in dev if r["eta_out"]]
    result["structure_means"]["eta_out"] = mean(eta) if eta else None
    result["structure_means"]["closure_max_abs"] = max(float(r["closure_max_abs"]) for r in dev)
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("p1", type=Path)
    args = cli.parse_args()
    print(json.dumps(read_summary(args.p1), ensure_ascii=False, indent=2, allow_nan=False))

