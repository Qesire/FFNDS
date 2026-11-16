"""Bounded video-feedback workflow with an explicit external evaluator contract.

The external command receives one candidate and the entire fixed CAL cohort.
It must rebuild anchor caches, generate complete videos, and return per-video
metrics. This module never imports a model or silently reuses another recipe.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable

import numpy as np

from .metrics import score_records, validate_metric_specification
from .optimizer import Evaluation, OptimizerConfig, optimize, validate_optimizer_config


def json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return json_value(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def digest(value: Any) -> str:
    text = json.dumps(json_value(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_value(value), ensure_ascii=False,
                                   indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _resolve(value: str, directory: Path) -> str:
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else directory / path).resolve())


def optimizer_configuration(config: dict) -> OptimizerConfig:
    parameters = config["parameters"]
    options = dict(config.get("optimizer", {}))
    options.setdefault("max_iterations", 1)
    options.setdefault("max_evaluations", 2 * len(parameters) + 2)
    options.update(lower_bounds=[p["lower"] for p in parameters],
                   upper_bounds=[p["upper"] for p in parameters],
                   finite_difference_step=[p["finite_difference_step"] for p in parameters],
                   trust_radius=[p["trust_radius"] for p in parameters])
    result = OptimizerConfig(**options)
    validate_optimizer_config([p["initial"] for p in parameters], result)
    return result


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    config = read_json(path)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("configuration must be a schema_version=1 object")
    directory = path.parent
    cohort = config.get("cohort")
    cohort_directory = directory
    if isinstance(cohort, str):
        cohort_path = Path(_resolve(cohort, directory))
        cohort_directory = cohort_path.parent
        cohort = read_json(cohort_path)
        if isinstance(cohort, dict):
            cohort = cohort.get("records")
    if not isinstance(cohort, list) or not cohort:
        raise ValueError("cohort must be a nonempty row list or a JSON list path")
    config["cohort"] = [dict(row) for row in cohort]
    for row in config["cohort"]:
        for key in ("image_path", "reference_video_path"):
            if key in row:
                row[key] = _resolve(row[key], cohort_directory)
    matrices = config.get("matrices")
    if matrices is not None:
        matrices = dict(matrices)
        matrices["base_manifest_path"] = _resolve(matrices["base_manifest_path"], directory)
        matrices["sites"] = [dict(site) for site in matrices["sites"]]
        for site in matrices["sites"]:
            site["kit_path"] = _resolve(site["kit_path"], directory)
        config["matrices"] = matrices
    oracle = dict(config.get("oracle", {}))
    if "cwd" in oracle:
        oracle["cwd"] = _resolve(oracle["cwd"], directory)
    config["oracle"] = oracle
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    mode = config.get("mode", "video")
    if mode not in ("video", "synthetic_unit_fixture"):
        raise ValueError("mode must be video or synthetic_unit_fixture")
    parameters = config.get("parameters")
    if not isinstance(parameters, list) or not parameters:
        raise ValueError("parameters must be nonempty")
    names = [p["name"] for p in parameters]
    if any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
        raise ValueError("parameter names must be unique nonempty strings")
    for p in parameters:
        for field in ("initial", "lower", "upper", "finite_difference_step", "trust_radius"):
            if isinstance(p[field], bool) or not math.isfinite(float(p[field])):
                raise ValueError(f"invalid {field} for parameter {p['name']}")
        if not float(p["lower"]) <= float(p["initial"]) <= float(p["upper"]):
            raise ValueError("initial parameters must lie within bounds")
        if float(p["lower"]) >= float(p["upper"]):
            raise ValueError("parameter lower bound must be smaller than upper bound")
        if min(float(p["finite_difference_step"]), float(p["trust_radius"])) <= 0:
            raise ValueError("finite-difference steps and trust radii must be positive")
    rows = config.get("cohort")
    if not isinstance(rows, list) or not rows:
        raise ValueError("cohort is empty")
    source_images, keys = {}, set()
    for row in rows:
        if row.get("split", "").upper() not in ("CAL", "CALIBRATION"):
            raise ValueError("parameter updates accept CAL rows only; score DEV separately")
        if not row.get("source_id") or not row.get("image_id"):
            raise ValueError("source_id and image_id are required")
        if not isinstance(row.get("seed"), int) or isinstance(row["seed"], bool):
            raise ValueError("seed must be an integer")
        sample = row.get("sample_index", 0)
        if not isinstance(sample, int) or isinstance(sample, bool) or sample < 0:
            raise ValueError("sample_index must be a nonnegative integer")
        key = (row["source_id"], row["seed"], sample)
        if key in keys:
            raise ValueError(f"duplicate cohort sample: {key}")
        keys.add(key)
        if source_images.setdefault(row["source_id"], row["image_id"]) != row["image_id"]:
            raise ValueError("one source_id maps to multiple image_id values")
        if mode == "video" and (not row.get("image_path") or not isinstance(row.get("prompt"), str)):
            raise ValueError("video cohort rows need image_path and prompt")
    specification = config.get("metrics", {})
    if not specification.get("objectives"):
        raise ValueError("at least one metric objective is required")
    validate_metric_specification(specification)
    oracle = config.get("oracle", {})
    argv = oracle.get("command")
    if not isinstance(argv, list) or not argv or any(not isinstance(a, str) or not a for a in argv):
        raise ValueError("oracle.command must be a nonempty argv array, never a shell string")
    if not any("{request}" in a for a in argv) or not any("{result}" in a for a in argv):
        raise ValueError("oracle.command must include {request} and {result} placeholders")
    if not isinstance(oracle.get("identity"), dict) or not oracle["identity"]:
        raise ValueError("oracle.identity must bind the generator, evaluator and recipe")
    timeout = float(oracle.get("timeout_seconds", 21600))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("oracle timeout must be positive and finite")
    if mode == "video":
        if not isinstance(config.get("matrices"), dict):
            raise ValueError("video mode requires matrix export configuration")
        used_names = {site["parameter"] for site in config["matrices"]["sites"]}
        if used_names != set(names):
            raise ValueError("every parameter must control a configured site, and every site must bind a parameter")
        if not oracle.get("require_initial_noise_identity", True):
            raise ValueError("video search requires per-sample initial_noise_sha256 identities")
    options = config.get("optimizer", {})
    for field in ("max_iterations", "max_evaluations"):
        value = options.get(field, 1 if field == "max_iterations" else 2 * len(names) + 2)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    optimizer_configuration(config)


def plan(config: dict) -> dict:
    validate_config(config)
    p = len(config["parameters"])
    rows = config["cohort"]
    options = config.get("optimizer", {})
    budget = options.get("max_evaluations", 2 * p + 2)
    return {
        "mode": config.get("mode", "video"),
        "parameter_names": [x["name"] for x in config["parameters"]],
        "independent_image_clusters": len({x["image_id"] for x in rows}),
        "source_count": len({x["source_id"] for x in rows}),
        "videos_per_evaluation": len(rows),
        "first_full_iteration_evaluations": 2 * p + 2,
        "subsequent_full_iteration_evaluations": 2 * p + 1,
        "max_evaluations": budget,
        "max_candidate_videos": budget * len(rows),
        "reference_videos": "existing fixed Dense references are additional assets, not generated by this optimizer",
        "objectives": config["metrics"]["objectives"],
        "guards": config["metrics"].get("guards", []),
        "model_calls_made": 0,
    }


class ExternalOracle:
    """Content-keyed metric cache and argv-only invocation of a video producer."""

    def __init__(self, config: dict, output_dir: str | Path,
                 exporter: Callable | None = None):
        self.config = config
        self.directory = Path(output_dir).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.exporter = exporter
        input_payloads = {}
        if config.get("mode", "video") == "video":
            from .matrices import preflight_manifest, sha256_file
            matrix_config = config["matrices"]
            preflight_manifest(matrix_config["base_manifest_path"],
                               expected_width=matrix_config.get("expected_width", 5120),
                               expected_layers=matrix_config.get("expected_layers", 40),
                               verify_hashes=True)
            for row in config["cohort"]:
                for field in ("image_path", "reference_video_path"):
                    if field in row and row[field] not in input_payloads:
                        input_payloads[row[field]] = sha256_file(row[field])
                expected = row.get("image_sha256")
                if expected is not None and input_payloads[row["image_path"]] != expected:
                    raise ValueError("cohort image content differs from image_sha256")
        self.context_digest = digest({"configuration": config, "input_payloads": input_payloads})
        context_path = self.directory / "context.json"
        if context_path.exists() and read_json(context_path).get("context_digest") != self.context_digest:
            raise ValueError("output directory belongs to a different cohort, recipe or configuration")
        write_json(context_path, {"context_digest": self.context_digest,
                                  "configuration": config, "input_payloads": input_payloads})
        self.memory = {}
        self.external_calls = 0
        self.cache_hits = 0
        self.noise_reference = None
        self.attempt_path = self.directory / "oracle_attempts.jsonl"
        self.attempts_total = 0
        if self.attempt_path.exists():
            for line in self.attempt_path.read_text(encoding="utf-8").splitlines():
                if line.strip() and json.loads(line).get("phase") == "started":
                    self.attempts_total += 1

    def _attempt_event(self, event: dict) -> None:
        with self.attempt_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({**event, "unix_time": time.time()}, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _validate_bundle(self, bundle: dict, evaluation_key: str, fingerprint: str) -> None:
        if bundle.get("evaluation_key") != evaluation_key:
            raise ValueError("metric cache/result evaluation identity mismatch")
        if bundle.get("execution_fingerprint") != fingerprint:
            raise ValueError("oracle result/cache used different execution matrices")
        if bundle.get("schema_version") != 1 or not bundle.get("candidate_id"):
            raise ValueError("oracle result schema_version=1 and candidate_id are required")
        if self.config.get("mode", "video") == "video" and bundle.get("synthetic", False):
            raise ValueError("synthetic records cannot enter a video search")
        score_records(bundle, bundle, self.config["cohort"], self.config["metrics"])
        if self.config["oracle"].get("require_initial_noise_identity", True):
            noise = self._noise(bundle)
            if self.noise_reference is None:
                self.noise_reference = noise
            elif noise != self.noise_reference:
                raise ValueError("paired candidates have different initial noise identities")

    def _noise(self, bundle: dict) -> dict:
        identities = {}
        for row in bundle.get("records", []):
            key = (row["source_id"], int(row["seed"]), int(row.get("sample_index", 0)))
            identity = row.get("initial_noise_sha256")
            if not isinstance(identity, str) or len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
                raise ValueError("oracle must return a lowercase initial_noise_sha256 for every sample")
            if key in identities:
                raise ValueError("duplicate oracle noise sample")
            identities[key] = identity
        expected = {(r["source_id"], r["seed"], r.get("sample_index", 0)) for r in self.config["cohort"]}
        if set(identities) != expected:
            raise ValueError("oracle returned an incomplete or different cohort")
        return identities

    def __call__(self, theta: np.ndarray) -> dict:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (len(self.config["parameters"]),) or not np.isfinite(theta).all():
            raise ValueError("invalid candidate parameter vector")
        values = {p["name"]: float(t) for p, t in zip(self.config["parameters"], theta)}
        parameter_key = digest(values)
        if parameter_key in self.memory:
            self.cache_hits += 1
            return self.memory[parameter_key]
        candidate_id = "candidate_" + parameter_key[:20]
        candidate_dir = self.directory / "candidates" / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        if self.config.get("mode", "video") == "video":
            if self.exporter is None:
                from .matrices import export_candidate
                self.exporter = export_candidate
            matrix_options = dict(self.config["matrices"])
            exported = self.exporter(theta=values, output_dir=candidate_dir / "matrices", **matrix_options)
            fingerprint = exported["fingerprint"]
            matrix_manifest = exported["manifest_path"]
        else:
            exported = {"mode": "synthetic_unit_fixture"}
            fingerprint = digest({"synthetic_parameters": values})
            matrix_manifest = None
        evaluation_key = digest({"context": self.context_digest, "execution": fingerprint})
        cache_dir = self.directory / "evaluations" / evaluation_key
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / "metrics.json"
        if cached.exists():
            bundle = read_json(cached)
            self._validate_bundle(bundle, evaluation_key, fingerprint)
            self.cache_hits += 1
        else:
            request = {
                "schema_version": 1, "mode": self.config.get("mode", "video"),
                "candidate_id": candidate_id, "parameters": values,
                "execution_fingerprint": fingerprint,
                "evaluation_key": evaluation_key,
                "matrix_manifest_path": matrix_manifest,
                "matrix_export": exported,
                "cohort": self.config["cohort"],
                "metrics": self.config["metrics"],
                "oracle_identity": self.config["oracle"]["identity"],
                "required_lifecycle": "new complete rollout and fresh anchor caches for every source/seed",
            }
            request_path = cache_dir / "request.json"
            result_path = cache_dir / "result.json"
            write_json(request_path, request)
            # A failed invocation may leave a partial result. A retry must produce it anew.
            if result_path.exists():
                result_path.unlink()
            substitutions = {"request": str(request_path), "result": str(result_path),
                             "output_dir": str(cache_dir), "matrix_manifest": matrix_manifest or ""}
            argv = [arg.format_map(substitutions) for arg in self.config["oracle"]["command"]]
            budget = self.config.get("optimizer", {}).get("max_evaluations", 2 * len(theta) + 2)
            if self.attempts_total >= budget:
                raise RuntimeError("external evaluation attempt budget exhausted, including earlier failed attempts")
            self.external_calls += 1
            self.attempts_total += 1
            attempt = {"evaluation_key": evaluation_key, "candidate_id": candidate_id,
                       "attempt_index": self.attempts_total}
            self._attempt_event({**attempt, "phase": "started"})
            try:
                with (cache_dir / "oracle.log").open("w", encoding="utf-8") as log:
                    subprocess.run(argv, check=True, shell=False,
                                   cwd=self.config["oracle"].get("cwd"), stdout=log,
                                   stderr=subprocess.STDOUT,
                                   timeout=float(self.config["oracle"].get("timeout_seconds", 21600)))
            except (subprocess.SubprocessError, OSError) as error:
                self._attempt_event({**attempt, "phase": "failed", "error_type": type(error).__name__})
                raise
            self._attempt_event({**attempt, "phase": "command_completed"})
            bundle = read_json(result_path)
            if bundle.get("candidate_id") != candidate_id or bundle.get("evaluation_key") != evaluation_key:
                raise ValueError("oracle result does not match its candidate request")
            # Validate pairing before making the result eligible for cache reuse.
            self._validate_bundle(bundle, evaluation_key, fingerprint)
            write_json(cached, bundle)
        result = dict(bundle)
        result["evaluated_candidate_id"] = bundle["candidate_id"]
        result["candidate_id"] = candidate_id
        result["candidate_directory"] = str(candidate_dir)
        self.memory[parameter_key] = result
        return result


def run_search(config: dict, output_dir: str | Path, *, oracle: Callable | None = None) -> dict:
    validate_config(config)
    optimizer_config = optimizer_configuration(config)
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    external = oracle if oracle is not None else ExternalOracle(config, directory)
    raw_by_theta = {}

    def raw_evaluation(theta):
        key = tuple(float(value) for value in theta)
        if key not in raw_by_theta:
            raw_by_theta[key] = external(np.asarray(theta, dtype=np.float64).copy())
        return raw_by_theta[key]

    initial = np.array([p["initial"] for p in config["parameters"]], dtype=np.float64)
    reference = raw_evaluation(initial)
    # Guards are always relative to this original reference, never a moving center.
    write_json(directory / "reference_metrics.json", reference)

    def evaluate(theta):
        raw = raw_evaluation(theta)
        scored = score_records(raw, reference, config["cohort"], config["metrics"])
        return Evaluation(loss=scored["loss"], unit_losses=scored["unit_losses"],
                          guard_margins=scored["guard_margins"],
                          fingerprint=raw.get("execution_fingerprint"),
                          metadata={"candidate_id": raw.get("candidate_id"),
                                    "evaluation_key": raw.get("evaluation_key"),
                                    "score": scored})

    parameters = config["parameters"]
    trace_path = directory / "optimization_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as trace:
        def on_event(event):
            trace.write(json.dumps(json_value(event), ensure_ascii=False, allow_nan=False) + "\n")
            trace.flush()
        result = optimize(initial, evaluate, optimizer_config, on_event=on_event)
    selected = raw_evaluation(result.theta)
    summary = {
        "schema_version": 1, "mode": config.get("mode", "video"),
        "parameters": {p["name"]: float(t) for p, t in zip(parameters, result.theta)},
        "plan": plan(config), "optimization": json_value(result),
        "selected_candidate_id": selected.get("candidate_id"),
        "selected_execution_fingerprint": selected.get("execution_fingerprint"),
        "selected_candidate_directory": selected.get("candidate_directory"),
        "external_calls": getattr(external, "external_calls", None),
        "external_attempts_total": getattr(external, "attempts_total", None),
        "cache_hits": getattr(external, "cache_hits", None),
        "assessment": "CAL parameter selection; independent video benefit is not established by this run",
    }
    write_json(directory / "optimization_result.json", summary)
    write_json(directory / "selected_metrics.json", selected)
    return summary
