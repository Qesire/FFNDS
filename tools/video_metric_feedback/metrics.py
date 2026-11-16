"""Import video scores and evaluate paired, image-cluster-weighted objectives.

The VBench adapter follows Vchitect/VBench at
fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490. In particular, camera_motion
returns three objects, and per-video MUSIQ values are NOT divided by 100
by VBench (only its aggregate is). Local FFNDS diagnostics remain labelled
as diagnostics; this module does not run or approximate a benchmark model.
"""

from __future__ import annotations

import copy
import json
import math
import os
from collections import defaultdict
from statistics import fmean


SCHEMA_VERSION = 1
LOCAL_SCHEMA = "ffnds.video.metrics.v1"
LOCAL_FIELDS = {
    "flow_magnitude_ratio": (
        "paired_motion", "candidate_to_reference_flow_magnitude_ratio", "mean"
    ),
    "temporal_delta_rmse": (
        "paired_temporal", "adjacent_delta_RGB_RMSE_0to1", "mean"
    ),
    "spatial_rgb_rmse": ("paired_spatial", "RGB_RMSE_0to1", "mean"),
    "spatial_rgb_mae": ("paired_spatial", "RGB_MAE_0to1", "mean"),
    "flow_l2": ("paired_motion", "downsampled_Farneback_flow_L2_px", "mean"),
    "luma_rmse": ("paired_spatial", "encoded_luma_RMSE_0to1", "mean"),
    "laplacian_energy_ratio": (
        "paired_spatial", "candidate_to_reference_laplacian_energy_ratio", "mean"
    ),
    "mean_flow_px": ("temporal_motion", "downsampled_Farneback_mean_flow_px", "mean"),
    "adjacent_rgb_mae": ("temporal_motion", "adjacent_RGB_MAE_0to1", "mean"),
}


def _finite(value, label, *, allow_bool=False):
    if isinstance(value, bool) and not allow_bool:
        raise ValueError(f"{label} must be a finite number, not bool")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _text(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _path(value):
    # Explicit paths are matched after lexical normalization only. In particular,
    # no basename lookup, filesystem search, or cross-candidate path guessing.
    return os.path.normpath(_text(value, "video_path"))


def _identity(row):
    if not isinstance(row, dict):
        raise ValueError("a sample must be an object")
    source = _text(row.get("source_id"), "source_id")
    image = _text(row.get("image_id"), "image_id")
    seed = row.get("seed")
    index = row.get("sample_index", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("sample_index must be a nonnegative integer")
    return (source, seed, index), image


def _index(rows, label, *, check_metrics=False):
    if not isinstance(rows, list):
        raise ValueError(f"{label} must be a list")
    indexed, images = {}, {}
    for row in rows:
        key, image = _identity(row)
        if key in indexed:
            raise ValueError(f"duplicate sample in {label}: {key!r}")
        if key[0] in images and images[key[0]] != image:
            raise ValueError(f"source {key[0]!r} maps to different image_id values")
        images[key[0]] = image
        if "eligible_metrics" in row:
            eligible = row["eligible_metrics"]
            if (not isinstance(eligible, list)
                    or any(not isinstance(x, str) or not x for x in eligible)
                    or len(set(eligible)) != len(eligible)):
                raise ValueError("eligible_metrics must contain unique metric names")
        if check_metrics:
            values = row.get("metrics")
            if not isinstance(values, dict):
                raise ValueError(f"metrics missing for sample {key!r}")
            for name, value in values.items():
                _text(name, "metric name")
                _finite(value, f"metric {name} at {key!r}", allow_bool=True)
            if "raw_metrics" in row:
                if not isinstance(row["raw_metrics"], dict):
                    raise ValueError("raw_metrics must be an object")
                for name, value in row["raw_metrics"].items():
                    _finite(value, f"raw metric {name} at {key!r}", allow_bool=True)
        indexed[key] = row
    return indexed


def _base_record(video):
    key, image = _identity(video)
    record = {
        "source_id": key[0], "image_id": image, "seed": key[1],
        "sample_index": key[2], "video_path": _text(video.get("video_path"), "video_path"),
        "metrics": {}, "raw_metrics": {}, "metric_origins": {},
    }
    if "eligible_metrics" in video:
        record["eligible_metrics"] = list(video["eligible_metrics"])
    if "initial_noise_sha256" in video:
        record["initial_noise_sha256"] = _text(video["initial_noise_sha256"], "initial_noise_sha256")
    return record


def import_vbench_results(results: dict, videos: list[dict], candidate_id: str,
                          evaluator_revision: str) -> dict:
    """Normalize an official ``*_eval_results.json`` using an explicit manifest.

    A manifest record is returned even when this import contains no metric for
    it, allowing independent dimension files to be merged. Completeness for the
    selected objective is checked by :func:`score_records`.
    """
    if not isinstance(results, dict):
        raise ValueError("VBench results must be an object")
    _text(candidate_id, "candidate_id")
    _text(evaluator_revision, "evaluator_revision")
    manifest = _index(videos, "videos")
    records = {key: _base_record(video) for key, video in manifest.items()}
    by_path = {}
    for key, video in manifest.items():
        path = _path(video["video_path"])
        if path in by_path:
            raise ValueError(f"duplicate video_path in manifest: {path}")
        by_path[path] = key
    for metric, payload in results.items():
        _text(metric, "metric name")
        count = 3 if metric == "camera_motion" else 2
        if not isinstance(payload, (list, tuple)) or len(payload) != count:
            raise ValueError(f"invalid VBench payload for {metric}")
        _finite(payload[0], f"aggregate {metric}", allow_bool=True)
        rows = payload[-1]
        if not isinstance(rows, list):
            raise ValueError(f"per-video records for {metric} must be a list")
        if metric == "camera_motion" and not isinstance(payload[1], dict):
            raise ValueError("camera_motion requires a per-type object")
        seen = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"invalid per-video record for {metric}")
            path = _path(row.get("video_path"))
            if path not in by_path:
                raise ValueError(f"unmapped video_path for {metric}: {path}")
            key = by_path[path]
            if key in seen:
                raise ValueError(f"duplicate {metric} result for {key!r}")
            seen.add(key)
            raw = _finite(row.get("video_results"), f"{metric} score", allow_bool=True)
            divisor = 100.0 if metric == "imaging_quality" else 1.0
            record = records[key]
            record["metrics"][metric] = raw / divisor
            record["raw_metrics"][metric] = raw
            record["metric_origins"][metric] = {
                "kind": "official_vbench", "revision": evaluator_revision,
                "normalization_divisor": divisor,
            }
            if "image_path" in row:
                image_path = _text(row["image_path"], "image_path")
                if ("image_path" in record
                        and os.path.normpath(record["image_path"]) != os.path.normpath(image_path)):
                    raise ValueError(f"conflicting image_path for {key!r}")
                record["image_path"] = image_path
            if metric == "camera_motion":
                record["camera_motion_details"] = {
                    key: copy.deepcopy(row[key])
                    for key in ("prompt_type", "predict_type") if key in row
                }
    return {
        "schema_version": SCHEMA_VERSION, "candidate_id": candidate_id,
        "evaluator": {"name": "vbench_i2v", "revision": evaluator_revision},
        "records": list(records.values()),
    }


def import_local_diagnostics(report: dict, video: dict) -> dict:
    """Map one existing FFNDS report into a record of labelled diagnostics.

    Optional families are imported when present. An absent paired quantity is
    left absent; the objective validator will reject it if it is required.
    These names are custom diagnostics, never official VBench dimensions.
    """
    if not isinstance(report, dict) or report.get("schema") != LOCAL_SCHEMA:
        raise ValueError(f"local report must use schema {LOCAL_SCHEMA}")
    _index([video], "video manifest")
    record = _base_record(video)
    identity = report.get("identity", {})
    if not isinstance(identity, dict):
        raise ValueError("local report identity must be an object")
    for field in ("source_id", "seed"):
        if identity.get(field) != record[field]:
            raise ValueError(f"local report {field} does not match video manifest")
    for field in ("image_id", "sample_index"):
        if field in identity and identity[field] != record[field]:
            raise ValueError(f"local report {field} does not match video manifest")
    reference_identity = report.get("reference_identity")
    if reference_identity is not None:
        if not isinstance(reference_identity, dict):
            raise ValueError("local reference_identity must be an object")
        for field in ("source_id", "seed"):
            if reference_identity.get(field) != identity[field]:
                raise ValueError(f"local diagnostic reference has mismatched {field}")
        if ("input_prompt_digest" in identity
                and reference_identity.get("input_prompt_digest") != identity["input_prompt_digest"]):
            raise ValueError("local diagnostic reference has mismatched input_prompt_digest")
    candidate = report.get("candidate", {})
    if not isinstance(candidate, dict) or _path(candidate.get("path")) != _path(video["video_path"]):
        raise ValueError("local report candidate path does not match video manifest")
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("local report metrics must be an object")
    for name, path in LOCAL_FIELDS.items():
        node = metrics
        for field in path:
            if not isinstance(node, dict) or field not in node:
                node = None
                break
            node = node[field]
        if node is None:
            continue
        value = _finite(node, f"local diagnostic {name}")
        record["metrics"][name] = value
        record["raw_metrics"][name] = value
        record["metric_origins"][name] = {
            "kind": "ffnds_local_diagnostic", "schema": LOCAL_SCHEMA,
            "implementation": report.get("implementation"),
            "field": "metrics." + ".".join(path),
        }
    record["diagnostic_schema"] = LOCAL_SCHEMA
    record["diagnostic_reference"] = copy.deepcopy(report.get("reference"))
    return record


def merge_record_sets(*record_sets: dict) -> dict:
    """Merge score envelopes and/or single local records without overwriting.

    At least one envelope supplies ``candidate_id``. Overlapping sample keys are
    expected across dimension imports; a duplicate within one envelope is an
    error, as is any conflicting value, identity, path, or metric provenance.
    """
    candidate_id, evaluators, merged = None, [], {}
    request_identity = {}
    synthetic = None
    for data in record_sets:
        if not isinstance(data, dict):
            raise ValueError("a record set must be an object")
        if "records" in data:
            if data.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("unsupported score schema_version")
            current_id = _text(data.get("candidate_id"), "candidate_id")
            if candidate_id is not None and current_id != candidate_id:
                raise ValueError("cannot merge different candidate_id values")
            candidate_id = current_id
            for field in ("execution_fingerprint", "evaluation_key"):
                if field in data:
                    value = _text(data[field], field)
                    if field in request_identity and request_identity[field] != value:
                        raise ValueError(f"cannot merge conflicting {field} values")
                    request_identity[field] = value
            if "synthetic" in data:
                if not isinstance(data["synthetic"], bool):
                    raise ValueError("synthetic must be a boolean")
                synthetic = bool(synthetic) or data["synthetic"]
            rows = data["records"]
            if data.get("evaluator") not in evaluators:
                evaluators.append(copy.deepcopy(data.get("evaluator")))
        else:
            rows = [data]
            if data.get("diagnostic_schema"):
                evaluator = {"name": "ffnds_local_diagnostics", "schema": data["diagnostic_schema"]}
                if evaluator not in evaluators:
                    evaluators.append(evaluator)
        indexed = _index(rows, "record set", check_metrics=True)
        for key, incoming in indexed.items():
            if key not in merged:
                merged[key] = copy.deepcopy(incoming)
                merged[key].setdefault("sample_index", 0)
                continue
            dest = merged[key]
            if incoming["image_id"] != dest["image_id"]:
                raise ValueError(f"conflicting image_id for {key!r}")
            for field, value in incoming.items():
                if field in ("metrics", "raw_metrics", "metric_origins"):
                    target = dest.setdefault(field, {})
                    if not isinstance(value, dict):
                        raise ValueError(f"{field} must be an object")
                    for name, item in value.items():
                        if name in target and target[name] != item:
                            raise ValueError(f"conflicting {field}.{name} for {key!r}")
                        target[name] = copy.deepcopy(item)
                elif field in dest:
                    same = (os.path.normpath(dest[field]) == os.path.normpath(value)
                            if field in ("video_path", "image_path") else dest[field] == value)
                    if not same:
                        raise ValueError(f"conflicting {field} for {key!r}")
                else:
                    dest[field] = copy.deepcopy(value)
    if candidate_id is None:
        raise ValueError("merge requires an envelope with candidate_id")
    _index(list(merged.values()), "merged records", check_metrics=True)
    evaluator = evaluators[0] if len(evaluators) == 1 else {"name": "combined", "components": evaluators}
    result = {
        "schema_version": SCHEMA_VERSION, "candidate_id": candidate_id,
        "evaluator": evaluator, "records": list(merged.values()),
    }
    result.update(request_identity)
    if synthetic is not None:
        result["synthetic"] = synthetic
    return result


def _direction(item):
    direction = item.get("direction")
    if direction not in ("max", "min"):
        raise ValueError(f"unknown direction {direction!r}; use 'max' or 'min'")
    return 1.0 if direction == "max" else -1.0


def _specification(specification):
    if not isinstance(specification, dict):
        raise ValueError("specification must be an object")
    objectives = specification.get("objectives", [])
    guards = specification.get("guards", [])
    if not isinstance(objectives, list) or not isinstance(guards, list):
        raise ValueError("objectives and guards must be lists")
    if not objectives:
        raise ValueError("at least one objective is required")
    required = set()
    seen = set()
    for item in objectives:
        if not isinstance(item, dict):
            raise ValueError("an objective must be an object")
        name = _text(item.get("name"), "objective name")
        if name in seen:
            raise ValueError(f"duplicate objective {name}")
        seen.add(name)
        required.add(name)
        _direction(item)
        if _finite(item.get("scale"), f"{name}.scale") <= 0:
            raise ValueError("objective scale must be positive")
        if _finite(item.get("weight"), f"{name}.weight") < 0:
            raise ValueError("objective weight must be nonnegative")
    for item in guards:
        if not isinstance(item, dict):
            raise ValueError("a guard must be an object")
        name = _text(item.get("name"), "guard name")
        required.add(name)
        _direction(item)
        if item.get("level", "cluster") not in ("cluster", "aggregate"):
            raise ValueError("guard level must be 'cluster' or 'aggregate'")
        bounds = [field for field in ("min_value", "max_value", "max_drop") if field in item]
        if not bounds:
            raise ValueError(f"guard {name} has no bound")
        for field in bounds:
            value = _finite(item[field], f"{name}.{field}")
            if field == "max_drop" and value < 0:
                raise ValueError("max_drop must be nonnegative")
        if ("min_value" in item and "max_value" in item
                and item["min_value"] > item["max_value"]):
            raise ValueError("guard min_value exceeds max_value")
    return objectives, guards, sorted(required)


def validate_metric_specification(specification: dict) -> None:
    """Validate directions, weights, scales and guard bounds before evaluation."""
    _specification(specification)


def _metric_clusters(indexed, cohort, name):
    by_source = defaultdict(list)
    for key, row in cohort.items():
        if "eligible_metrics" in row and name not in row["eligible_metrics"]:
            continue
        values = indexed[key]["metrics"]
        if name not in values:
            raise ValueError(f"missing eligible metric {name} at {key!r}")
        by_source[key[0]].append(_finite(values[name], f"metric {name}", allow_bool=True))
    by_image = defaultdict(list)
    images = {key[0]: row["image_id"] for key, row in cohort.items()}
    for source, values in by_source.items():
        by_image[images[source]].append(fmean(values))
    if not by_image:
        raise ValueError(f"metric {name} has no eligible image clusters")
    return {image: fmean(values) for image, values in sorted(by_image.items())}


def score_records(candidate: dict, reference: dict, cohort: list,
                  specification: dict) -> dict:
    """Evaluate a paired loss and guards with sample -> source -> image means.

    ``cohort`` fixes eligibility. Candidate and reference must have exactly its
    sample keys; paths may differ across candidates. A guard passes iff every
    returned margin is nonnegative. ``max_drop`` uses the fixed reference and
    is in that metric's units, independently of objective scaling.
    """
    fixed = _index(cohort, "cohort")
    if not fixed:
        raise ValueError("cohort must not be empty")
    panels = []
    for label, data in (("candidate", candidate), ("reference", reference)):
        if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported {label} schema_version")
        panel = _index(data.get("records"), label, check_metrics=True)
        if set(panel) != set(fixed):
            missing, extra = set(fixed) - set(panel), set(panel) - set(fixed)
            raise ValueError(f"{label} sample keys differ: missing={sorted(missing)!r}, extra={sorted(extra)!r}")
        for key, row in panel.items():
            if row["image_id"] != fixed[key]["image_id"]:
                raise ValueError(f"{label} image_id differs from cohort for {key!r}")
        panels.append(panel)
    objectives, guards, required = _specification(specification)
    candidate_clusters, reference_clusters = {}, {}
    for name in required:
        candidate_clusters[name] = _metric_clusters(panels[0], fixed, name)
        reference_clusters[name] = _metric_clusters(panels[1], fixed, name)
    clusters = sorted({row["image_id"] for row in fixed.values()})
    unit_losses = dict.fromkeys(clusters, 0.0)
    terms = {}
    for item in objectives:
        name = item["name"]
        values, baseline = candidate_clusters[name], reference_clusters[name]
        factor = -item["weight"] * _direction(item) / item["scale"]
        terms[name] = factor * fmean(values[image] - baseline[image] for image in values)
        correction = len(clusters) / len(values)
        for image in values:
            unit_losses[image] += correction * factor * (values[image] - baseline[image])
    margins = {}
    for index, guard in enumerate(guards):
        name = guard["name"]
        level = guard.get("level", "cluster")
        values, baseline = candidate_clusters[name], reference_clusters[name]
        if level == "aggregate":
            units = {"all": (fmean(values.values()), fmean(baseline.values()))}
        else:
            units = {image: (value, baseline[image]) for image, value in values.items()}
        for unit, (value, previous) in units.items():
            checks = {}
            if "min_value" in guard:
                checks["min_value"] = value - guard["min_value"]
            if "max_value" in guard:
                checks["max_value"] = guard["max_value"] - value
            if "max_drop" in guard:
                checks["max_drop"] = guard["max_drop"] + _direction(guard) * (value - previous)
            for bound, margin in checks.items():
                stable_id = json.dumps([index, name, level, unit, bound], separators=(",", ":"), ensure_ascii=False)
                margins[stable_id] = _finite(margin, f"computed guard margin {stable_id}")
    means = {name: fmean(values.values()) for name, values in candidate_clusters.items()}
    reference_means = {name: fmean(values.values()) for name, values in reference_clusters.items()}
    for image, value in unit_losses.items():
        _finite(value, f"computed unit loss {image}")
    return {
        "loss": _finite(fmean(unit_losses.values()), "computed loss"), "unit_losses": unit_losses,
        "guard_margins": margins, "metrics": means,
        "reference_metrics": reference_means,
        "metric_deltas": {name: means[name] - reference_means[name] for name in required},
        "metric_cluster_counts": {name: len(values) for name, values in candidate_clusters.items()},
        "objective_terms": terms,
    }
